import argparse
import pickle
from pathlib import Path
import pandas as pd
import rdkit
import torch
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
from tqdm.auto import tqdm
from model import AFE_DTA
from torch.utils.data import DataLoader
from rdkit import Chem

from utils import *


RDLogger.DisableLog('rdApp.*')

def load_model(model_path, tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    model = AFE_DTA(tokenizer)
    states = torch.load(model_path, map_location='cpu')
    print(model.load_state_dict(states, strict=False))

    return model, tokenizer

def format_smiles(smiles):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None

    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    return smiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str, choices=['kiba'], help='the dataset name (kiba or davis)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='device to use (cpu or cuda)')

    args = parser.parse_args()

    dataset = args.dataset
    device = args.device

    config = {
        'input_path': 'your_input_path',
        'output_dir': 'your_output_directory',
        'model_path': 'your_model_weights_path.pth',
        'tokenizer_path': 'your_tokenizer_path.pkl',
        'n_mol': 40000,
        'filter': True,
        'batch_size': 1,
        'seed': -1,
        'device': device
    }

    model_path = f'/saved_models/afe_dta_model_{dataset}.pth'
    tokenizer_path = f'data/{dataset}_tokenizer.pkl'


    test_data = TestbedDataset(root="data", dataset=f"{dataset}_test")
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    output_dir = Path(f'generated_results/{dataset}')
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(model_path, tokenizer_path)
    model.eval()
    model.to(device)


    results = []
    generated_smiles_list = []


    for i, data in enumerate(tqdm(test_loader)):

        if len(generated_smiles_list) >= config['n_mol']:
            break
            
        data.to(config['device'])

        generated_smiles = tokenizer.get_text(model.generate(data))[0]  
        

        input_info = {
            'index': i,
            'input_compound': data.compound[0] if hasattr(data, 'compound') else None,
            'input_target': data.target[0] if hasattr(data, 'target') else None,
            'generated_smiles_raw': generated_smiles
        }
        
        results.append(input_info)
        generated_smiles_list.append(generated_smiles)

