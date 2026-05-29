

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model import AFEDTA
from utils import TestbedDataset, get_cindex, get_rm2, logging, mse


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch, flags):
    model.train()
    running_loss = 0.0

    with tqdm(train_loader, desc=f"Epoch {epoch}", leave=False) as progress:
        for data in progress:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(data)
            target = data.y.view(-1, 1).float().to(device)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.num_graphs
            progress.set_postfix(MSE=f"{loss.item():.4f}")

    mean_loss = running_loss / len(train_loader.dataset)
    logging(f"Epoch {epoch}, train_MSE={mean_loss:.6f}", flags)
    return mean_loss


@torch.no_grad()
def evaluate(model, device, data_loader):
    model.eval()
    y_true = []
    y_pred = []

    for data in tqdm(data_loader, desc="Evaluating", leave=False):
        data = data.to(device)
        prediction = model(data)
        y_true.append(data.y.view(-1, 1).detach().cpu())
        y_pred.append(prediction.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy().flatten()
    y_pred = torch.cat(y_pred, dim=0).numpy().flatten()

    return {
        "mse": float(mse(y_true, y_pred)),
        "ci": float(get_cindex(y_true, y_pred)),
        "rm2": float(get_rm2(y_true, y_pred)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def run_experiment(args, dataset: str, seed: int, device: torch.device):
    set_seed(seed)
    flags = argparse.Namespace(log_dir=args.log_dir, dataset_name=f"afedta_{dataset}_seed{seed}")
    os.makedirs(flags.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.affinity_dir, exist_ok=True)

    train_file = os.path.join(args.data_root, "processed", f"{dataset}_train.pt")
    test_file = os.path.join(args.data_root, "processed", f"{dataset}_test.pt")
    if not (os.path.isfile(train_file) and os.path.isfile(test_file)):
        raise FileNotFoundError(
            f"Processed files not found: {train_file} and/or {test_file}. "
            "Please prepare the PyTorch Geometric datasets first."
        )

    train_data = TestbedDataset(root=args.data_root, dataset=f"{dataset}_train")
    test_data = TestbedDataset(root=args.data_root, dataset=f"{dataset}_test")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = AFEDTA().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best = {"mse": float("inf"), "ci": 0.0, "rm2": 0.0, "epoch": 0}
    model_path = os.path.join(args.save_dir, f"afedta_model_{dataset}_seed{seed}.pth")

    header = (
        f"Dataset={dataset}, Seed={seed}, Device={device}, Batch={args.batch_size}, "
        f"LR={args.lr}, Epochs={args.epochs}, WeightDecay={args.weight_decay}"
    )
    print(header)
    logging(header, flags)

    last_metrics = None
    for epoch in range(1, args.epochs + 1):
        train_mse = train_one_epoch(model, device, train_loader, optimizer, criterion, epoch, flags)

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            metrics = evaluate(model, device, test_loader)
            last_metrics = metrics
            is_best = metrics["mse"] < best["mse"]
            if is_best:
                best.update({"mse": metrics["mse"], "ci": metrics["ci"], "rm2": metrics["rm2"], "epoch": epoch})
                torch.save(model.state_dict(), model_path)

            msg = (
                f"{datetime.now():%Y-%m-%d %H:%M:%S} | Epoch={epoch} | "
                f"train_MSE={train_mse:.6f} | test_MSE={metrics['mse']:.6f} | "
                f"CI={metrics['ci']:.6f} | r_m2={metrics['rm2']:.6f} | "
                f"best_MSE={best['mse']:.6f} at epoch {best['epoch']}"
            )
            print(msg)
            logging(msg, flags)

    if last_metrics is None:
        last_metrics = evaluate(model, device, test_loader)

    np.savetxt(os.path.join(args.affinity_dir, f"true_labels_{dataset}_seed{seed}.txt"), last_metrics["y_true"])
    np.savetxt(os.path.join(args.affinity_dir, f"estimated_labels_{dataset}_seed{seed}.txt"), last_metrics["y_pred"])

    return {"dataset": dataset, "seed": seed, **best}


def parse_args():
    parser = argparse.ArgumentParser(description="Train AFE-DTA.")
    parser.add_argument("--datasets", nargs="+", default=["davis"], choices=["davis", "kiba", "bindingdb"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--save_dir", default="saved_models")
    parser.add_argument("--log_dir", default="training_logs")
    parser.add_argument("--affinity_dir", default="Affinities")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[4221])
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    results = []
    for dataset_name in args.datasets:
        for seed_value in args.seeds:
            results.append(run_experiment(args, dataset_name, seed_value, device))

    print("\nSummary of best results:")
    for item in results:
        print(
            f"{item['dataset']} | seed={item['seed']} | "
            f"best_MSE={item['mse']:.6f} | CI={item['ci']:.6f} | "
            f"r_m2={item['rm2']:.6f} | epoch={item['epoch']}"
        )
