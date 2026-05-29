

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model import AFEDTA
from utils import TestbedDataset, get_cindex, get_rm2, mse, pearson, rmse, spearman


@torch.no_grad()
def evaluate(model, device, loader):
    model.eval()
    total_true = []
    total_pred = []
    for data in tqdm(loader, desc="Testing"):
        data = data.to(device)
        prediction = model(data)
        total_true.append(data.y.view(-1, 1).detach().cpu())
        total_pred.append(prediction.detach().cpu())

    y_true = torch.cat(total_true, dim=0).numpy().flatten()
    y_pred = torch.cat(total_pred, dim=0).numpy().flatten()
    return y_true, y_pred


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    model_path = args.model_path or os.path.join(args.save_dir, f"afedta_model_{args.dataset}_seed{args.seed}.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    test_data = TestbedDataset(root=args.data_root, dataset=f"{args.dataset}_test")
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = AFEDTA().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    y_true, y_pred = evaluate(model, device, test_loader)
    metrics = {
        "MSE": float(mse(y_true, y_pred)),
        "CI": float(get_cindex(y_true, y_pred)),
        "r_m2": float(get_rm2(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "Pearson": float(pearson(y_true, y_pred)),
        "Spearman": float(spearman(y_true, y_pred)),
    }

    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {model_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        np.savetxt(os.path.join(args.output_dir, f"true_labels_{args.dataset}_seed{args.seed}.txt"), y_true)
        np.savetxt(os.path.join(args.output_dir, f"estimated_labels_{args.dataset}_seed{args.seed}.txt"), y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AFE-DTA on a benchmark dataset.")
    parser.add_argument("--dataset", required=True, choices=["davis", "kiba", "bindingdb"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--save_dir", default="saved_models")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="Affinities")
    main(parser.parse_args())
