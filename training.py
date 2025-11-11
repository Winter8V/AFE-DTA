
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data.storage import GlobalStorage
from utils import *
from model import AEF_DTA
from FetterGrad import FetterGrad
import sys, os
import pickle
import random

# Set random seeds for reproducibility
seed = 4221
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if torch.cuda.is_available():
  generator = torch.Generator('cuda').manual_seed(seed)
else:
  generator = torch.Generator().manual_seed(seed)

def train(model, device, train_loader, optimizer, mse_f, epoch):
    """Train the model for one epoch"""
    model.train()

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}") as t:
        for i, data in enumerate(t):
            # Zero gradients
            optimizer.zero_grad()
            batch = data.batch.to(device)
            
            # Forward pass
            Pridection, new_drug, lm_loss, kl_loss = model(data.to(device))

            # Calculate MSE loss between predictions and true values
            mse_loss = mse_f(Pridection, data.y.view(-1, 1).float().to(device))

            # Calculate training concordance index
            train_ci = get_cindex(Pridection.cpu().detach().numpy(), data.y.view(-1, 1).float().cpu().detach().numpy())

            # Combined loss with KL divergence and language modeling loss
            loss = kl_loss * 0.001 + mse_loss + lm_loss

            # Backward pass with FetterGrad optimizer
            losses = [loss, mse_loss] 
            optimizer.ft_backward(losses)
            optimizer.step()
            
            # Update progress bar with current metrics
            t.set_postfix(MSE=mse_loss.item(), Train_cindex=train_ci, KL=kl_loss.item(), LM=lm_loss.item())
    return model

def test(model, device, test_loader, dataset):
    """Evaluate the model on test data"""
    print('Testing on {} samples...'.format(len(test_loader.dataset)))
    model.eval()
    total_true = torch.Tensor()
    total_predict = torch.Tensor()
    total_loss = 0 

    # Set different thresholds for different datasets
    if dataset == "kiba":
        thresholds = [10.0, 10.50, 11.0, 11.50, 12.0, 12.50]
    else:
        thresholds = [5.0, 5.50, 6.0, 6.50, 7.0, 7.50, 8.0, 8.50]  

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            # Forward pass
            Pridection, new_drug, lm_loss, kl_loss = model(data.to(device))

            # Collect predictions and true values
            total_true = torch.cat((total_true, data.y.view(-1, 1).cpu()), 0)
            total_predict = torch.cat((total_predict, Pridection.cpu()), 0)   
            
            # Convert to numpy for metric calculation
            G = total_true.numpy().flatten()
            P = total_predict.numpy().flatten()
            
            # Calculate evaluation metrics
            mse_loss = mse(G, P)
            test_ci = get_cindex(G, P)      
            rm2 = get_rm2(G, P)   
            
            # Calculate AUC values for different thresholds
            auc_values = []
            for t in thresholds:
                auc = get_aupr(np.int32(G > t), P)
                auc_values.append(auc) 
            
            loss = lm_loss + kl_loss
            total_loss += loss.item() * data.num_graphs
            
    return total_loss, mse_loss, test_ci, rm2, auc_values, G, P

def experiment(dataset, device):
    """Main experiment function for training and evaluation"""
    # Hyperparameters
    BATCH_SIZE = 128
    LR = 0.0002
    NUM_EPOCHS = 500

    # Track best performance metrics
    best_mse = float('inf')
    best_ci = 0.0
    best_rm2 = 0.0
    best_mse_epoch = 0
    best_ci_epoch = 0
    best_rm2_epoch = 0

    # Print experiment configuration
    print(f"Dataset: {dataset}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Epochs: {NUM_EPOCHS}")

    # Load tokenizer for sequence processing
    with open(f'data/{dataset}_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Check if processed data files exist
    processed_data_file_train = f"data/processed/{dataset}_train.pt"
    processed_data_file_test = f"data/processed/{dataset}_test.pt"
    
    if not (os.path.isfile(processed_data_file_train) and os.path.isfile(processed_data_file_test)):
        print("Please run create_data.py to prepare data in PyTorch format!")
    else:
        # Load training and test datasets
        train_data = TestbedDataset(root="data", dataset=f"{dataset}_train")
        test_data = TestbedDataset(root="data", dataset=f"{dataset}_test")

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model, optimizer and loss function
        model = AEF_DTA(tokenizer).to(device)
        optimizer = FetterGrad(optim.Adam(model.parameters(), lr=LR))
        mse_f = nn.MSELoss()

        # Training loop
        for epoch in range(NUM_EPOCHS):
            model = train(model, device, train_loader, optimizer, mse_f, epoch)

            # Evaluate every 20 epochs
            if (epoch + 1) % 20 == 0:
                total_loss, mse_loss, test_ci, rm2, auc_values, G, P = test(model, device, test_loader, dataset)
                filename = f"saved_models/aef_dta_model_{dataset}.pth"

                if mse_loss < best_mse:
                    best_mse = mse_loss
                    torch.save(model.state_dict(), filename)
                    print('model saved')

                print(f"MSE: {mse_loss.item():.4f}")
                print(f"CI: {test_ci:.4f}")
                print(f"RM2: {rm2:.4f}")
                print(f"AUCs: {', '.join([f'{auc:.4f}' for auc in auc_values])}")

        # Save estimated and true labels
        folder_path = "Affinities/"
        np.savetxt(folder_path + f"estimated_labels_{dataset}.txt", P)
        np.savetxt(folder_path + f"true_labels_{dataset}.txt", G)

        logging('Program finished', FLAGS)

if __name__ == "__main__":
    # Available datasets
    datasets = ['davis', 'kiba']
    
    # Get dataset index from command line argument or use default
    dataset_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    dataset = datasets[dataset_idx]

    # Set device (GPU if available, else CPU)
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:" + str(int(sys.argv[2])) if len(sys.argv) > 2 and torch.cuda.is_available() else default_device)

    # Create directories for saving results
    os.makedirs('Affinities', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    # Run experiment
    experiment(dataset, device)
