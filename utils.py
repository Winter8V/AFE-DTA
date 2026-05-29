

from __future__ import annotations

import os
import re
from math import sqrt

import numpy as np
import torch
from scipy import stats
from torch.nn.utils.rnn import pad_sequence
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset
from tqdm.auto import tqdm


class Tokenizer:
    """SMILES tokenizer retained for compatibility with existing preprocessing scripts."""

    NUM_RESERVED_TOKENS = 32
    SPECIAL_TOKENS = ("<sos>", "<eos>", "<pad>", "<mask>", "<sep>", "<unk>")
    SPECIAL_TOKENS += tuple([f"<t_{i}>" for i in range(len(SPECIAL_TOKENS), 32)])

    PATTERN = re.compile(
        r"\[[^\]]+\]"
        r"|B[r]?|C[l]?|N|O|P|S|F|I"
        r"|[bcnops]"
        r"|@@|@"
        r"|%\d{2}"
        r"|."
    )
    ATOM_PATTERN = re.compile(r"\[[^\]]+\]|B[r]?|C[l]?|N|O|P|S|F|I|[bcnops]")

    # Backward-compatible aliases for older preprocessing scripts.
    PATTEN = PATTERN
    ATOM_PATTEN = ATOM_PATTERN

    @staticmethod
    def gen_vocabs(smiles_list):
        vocabs = set()
        for smiles in tqdm(set(smiles_list), desc="Building SMILES vocabulary"):
            vocabs.update(re.findall(Tokenizer.PATTERN, smiles))
        return vocabs

    def __init__(self, vocabs):
        special_tokens = list(Tokenizer.SPECIAL_TOKENS)
        vocabs = special_tokens + sorted(set(vocabs) - set(special_tokens), key=lambda item: (len(item), item))
        self.vocabs = vocabs
        self.i2s = {i: token for i, token in enumerate(vocabs)}
        self.s2i = {token: i for i, token in self.i2s.items()}

    def __len__(self):
        return len(self.vocabs)

    def parse(self, smiles, return_atom_idx=False):
        token_ids = []
        atom_idx = []
        tokens = ("<sos>", *re.findall(Tokenizer.PATTERN, smiles), "<eos>")
        for idx, token in enumerate(tokens):
            token_ids.append(self.s2i.get(token, self.s2i.get("<unk>", 5)))
            if return_atom_idx and re.fullmatch(Tokenizer.ATOM_PATTERN, token) is not None:
                atom_idx.append(idx)
        return (token_ids, atom_idx) if return_atom_idx else token_ids

    def get_text(self, predictions):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.tolist()
        smiles_list = []
        for prediction in predictions:
            tokens = []
            for token_id in prediction:
                token = self.i2s[int(token_id)]
                if token == "<eos>":
                    break
                if token not in {"<sos>", "<pad>"}:
                    tokens.append(token)
            smiles_list.append("".join(tokens))
        return smiles_list


class TestbedDataset(InMemoryDataset):
    """PyTorch Geometric dataset used by Davis, KIBA, and BindingDB benchmarks."""

    def __init__(
        self,
        root="data",
        dataset="davis",
        xd=None,
        xdt=None,
        xt=None,
        y=None,
        transform=None,
        pre_transform=None,
        smile_graph=None,
    ):
        super().__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.pad_token = Tokenizer.SPECIAL_TOKENS.index("<pad>")
        if os.path.isfile(self.processed_paths[0]):
            print(f"Pre-processed data found: {self.processed_paths[0]}, loading ...")
            try:
                self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
            except TypeError:
                self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            if any(item is None for item in [xd, xdt, xt, y, smile_graph]):
                raise FileNotFoundError(
                    f"Processed data {self.processed_paths[0]} not found and raw inputs were not provided."
                )
            print(f"Pre-processed data {self.processed_paths[0]} not found, preprocessing ...")
            self.process(xd, xdt, xt, y, smile_graph)
            try:
                self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
            except TypeError:
                self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.dataset + ".pt"]

    def download(self):
        pass

    def process(self, xd, xdt, xt, y, smile_graph):
        assert len(xd) == len(xt) == len(y) == len(xdt), "Input lists must have the same length."
        smiles_tokens = pad_sequence(xdt, batch_first=True, padding_value=self.pad_token)
        data_list = []
        for idx, smiles in enumerate(tqdm(xd, desc="Preparing PyG data")):
            target = xt[idx]
            label = y[idx]
            tokenized_smiles = smiles_tokens[idx].tolist()
            c_size, features, edge_index, edge_feats = smile_graph[smiles]
            graph_data = DATA.Data(
                x=torch.tensor(features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_feats, dtype=torch.float),
                y=torch.tensor([label], dtype=torch.float),
            )
            graph_data.target = torch.tensor([target], dtype=torch.long)
            graph_data.target_seq = torch.tensor([tokenized_smiles], dtype=torch.long)
            graph_data.c_size = torch.tensor([c_size], dtype=torch.long)
            data_list.append(graph_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        os.makedirs(self.processed_dir, exist_ok=True)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def logging(msg, flags):
    os.makedirs(flags.log_dir, exist_ok=True)
    with open(os.path.join(flags.log_dir, f"log_{flags.dataset_name}.txt"), "a", encoding="utf-8") as file:
        file.write(f"{msg}\n")


def rmse(y, f):
    y = np.asarray(y).reshape(-1)
    f = np.asarray(f).reshape(-1)
    return sqrt(np.mean((y - f) ** 2))


def mse(y, f):
    y = np.asarray(y).reshape(-1)
    f = np.asarray(f).reshape(-1)
    return np.mean((y - f) ** 2)


def pearson(y, f):
    y = np.asarray(y).reshape(-1)
    f = np.asarray(f).reshape(-1)
    if y.size < 2:
        return 0.0
    return np.corrcoef(y, f)[0, 1]


def spearman(y, f):
    y = np.asarray(y).reshape(-1)
    f = np.asarray(f).reshape(-1)
    if y.size < 2:
        return 0.0
    return stats.spearmanr(y, f)[0]


def get_cindex(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    pred_matrix = y_pred[:, None] - y_pred[None, :]
    pred_order = (pred_matrix > 0).astype(np.float32) + 0.5 * (pred_matrix == 0).astype(np.float32)
    true_matrix = y_true[:, None] - y_true[None, :]
    true_order = np.tril((true_matrix > 0).astype(np.float32), 0)
    denominator = np.sum(true_order)
    if denominator == 0:
        return 0.0
    return float(np.sum(pred_order * true_order) / denominator)


def r_squared_error(y_obs, y_pred):
    y_obs = np.asarray(y_obs).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_obs_centered = y_obs - np.mean(y_obs)
    y_pred_centered = y_pred - np.mean(y_pred)
    numerator = np.sum(y_obs_centered * y_pred_centered) ** 2
    denominator = np.sum(y_obs_centered ** 2) * np.sum(y_pred_centered ** 2)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def get_k(y_obs, y_pred):
    y_obs = np.asarray(y_obs).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denominator = np.sum(y_pred * y_pred)
    if denominator == 0:
        return 0.0
    return float(np.sum(y_obs * y_pred) / denominator)


def squared_error_zero(y_obs, y_pred):
    y_obs = np.asarray(y_obs).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    k = get_k(y_obs, y_pred)
    numerator = np.sum((y_obs - k * y_pred) ** 2)
    denominator = np.sum((y_obs - np.mean(y_obs)) ** 2)
    if denominator == 0:
        return 0.0
    return float(1 - numerator / denominator)


def get_rm2(y_obs, y_pred):
    r2 = r_squared_error(y_obs, y_pred)
    r02 = squared_error_zero(y_obs, y_pred)
    return float(r2 * (1 - np.sqrt(abs((r2 ** 2) - (r02 ** 2)))))
