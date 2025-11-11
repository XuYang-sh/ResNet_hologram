# -*- coding: utf-8 -*-
"""
A scalable PyTorch script to train a Hologram-generating CNN.

This version is MODIFIED to use a pre-processed dataset of individual .pt
files, which provides the fastest possible data loading.

It also includes options to limit the number of samples for quick tests.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import warnings

# <<< ADDED >>>
# Suppress the torch.load warning, as we trust our own pre-processed files
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`", FutureWarning)

# --- 1. Configuration ---
CONFIG = {
    "experiment_name": "HologramRun128_0",
    "img_dim": 128,
    "batch_size": 128, 
    "epochs": 100,
    "lr": 1e-4,
    "patience": 10,
    
    "dataset_base_dir": "hologram_dataset_processed_128x128",
    "resume_from_dir": None,
    "num_workers": 8,

    # <<< ADDED: Data Limiting >>>
    # Set to a number (e.g., 10000) to limit samples, or None to use all.
    "max_train_samples": None,  # e.g., 50000
    "max_val_samples": None,    # e.g., 5000
    "max_test_samples": None,   # e.g., 5000
}

# --- 1a. Adjust Paths based on Environment ---
SCRATCH_DIR = os.getenv('SCRATCH')
if not SCRATCH_DIR:
    print("Warning: $SCRATCH environment variable not set. Using current directory for dataset.")
    SCRATCH_DIR = "."  # Fallback to current directory
CONFIG["dataset_base_dir"] = os.path.join(SCRATCH_DIR, CONFIG["dataset_base_dir"])
print(f"Dataset will be loaded from: {CONFIG['dataset_base_dir']}")
print(f"Logs/models will be saved to local './runs' directory.")


# --- 2. Custom Dataset for Scalable On-the-Fly Loading ---
class HologramPreprocessedDataset(Dataset):
    """
    A simple Dataset that loads pre-processed .pt files.
    """
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # We trust these files since we created them ourselves.
        return torch.load(self.file_paths[idx])

# --- 3. Model and Loss Function Definitions (Unchanged) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels=16):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.bn1(out)
        out = F.relu(self.conv2(out))
        return out + residual

class HologramCNN(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(HologramCNN, self).__init__()
        self.initial_convs = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(16), ResidualBlock(16), ResidualBlock(16)
        )
        self.output_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
    def forward(self, x):
        return self.output_conv(self.res_blocks(self.initial_convs(x)))

def create_gaussian_kernel(size=5, sigma=1.0, device='cpu'):
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= (size - 1) / 2.0
    g = coords**2
    g = (-g / (2 * sigma**2)).exp()
    kernel = torch.outer(g, g)
    return kernel / kernel.sum()

def hologram_loss(y_pred, y_true, x_input):
    A_input_weights = x_input[:, 0:1, :, :]
    kernel_size = 15
    sigma = 7
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma, device=x_input.device)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    padding = (kernel_size - 1) // 2
    neighborhood_weights = F.conv2d(A_input_weights, gaussian_kernel, padding=padding)
    
    max_val = torch.max(neighborhood_weights)
    if max_val > 0:
        neighborhood_weights = neighborhood_weights / max_val
    
    y_pred_amp, y_pred_phase = y_pred[:, 0, :, :], y_pred[:, 1, :, :]
    y_true_amp, y_true_phase = y_true[:, 0, :, :], y_true[:, 1, :, :]
    
    l1_diff = torch.abs(y_true_amp - y_pred_amp)
    
    weighted_local_loss = torch.mean(neighborhood_weights.squeeze(1) * l1_diff)
    global_loss = torch.mean(l1_diff)
    # amplitude_loss = 10.0 * (weighted_local_loss + 0.01 * global_loss)
    amplitude_loss = 10.0 * (global_loss)
    
    phase_diff = y_pred_phase - y_true_phase
    mapped_diff = (phase_diff + torch.pi) % (2 * torch.pi) - torch.pi
    mapped_l2_loss = torch.mean(torch.square(mapped_diff))
    raw_l2_loss = torch.mean(torch.square(phase_diff))
    phase_loss = mapped_l2_loss + 0.1 * raw_l2_loss
    
    return amplitude_loss + phase_loss


# --- 4. Main Training and Evaluation Function ---
def main():
    # --- Setup Experiment ---
    start_epoch = 0
    best_val_loss = float('inf')
    
    if CONFIG["resume_from_dir"] and os.path.isdir(CONFIG["resume_from_dir"]):
        log_dir = CONFIG["resume_from_dir"]
        model_path = os.path.join(log_dir, "best_model.pt")
        print(f"Attempting to resume experiment from: {log_dir}")
        if not os.path.exists(model_path):
            print(f"Warning: Checkpoint file not found at {model_path}. Starting a new run.")
            log_dir = None
    else:
        log_dir = None

    if log_dir is None:
        timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
        run_name = f"{timestamp}_{CONFIG['experiment_name']}"
        log_dir = os.path.join("runs", run_name)
        model_path = os.path.join(log_dir, "best_model.pt")
        print(f"Starting new experiment: {run_name}")

    writer = SummaryWriter(log_dir)
    print(f"Logs and models will be saved to: {log_dir}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Data Loading using Custom Dataset and DataLoader ---
    base_dir = CONFIG['dataset_base_dir']
    if not os.path.exists(base_dir):
        print(f"Error: Dataset directory '{base_dir}' not found."); return

    print("Loading file paths...")
    train_files = sorted(glob.glob(os.path.join(base_dir, "train", "*.pt")))
    val_files = sorted(glob.glob(os.path.join(base_dir, "val", "*.pt")))
    test_files = sorted(glob.glob(os.path.join(base_dir, "test", "*.pt")))

    # <<< MODIFIED: Apply sample limits based on CONFIG >>>
    if CONFIG["max_train_samples"] is not None:
        print(f"--- Limiting training data to {CONFIG['max_train_samples']} samples (out of {len(train_files)}) ---")
        train_files = train_files[:CONFIG["max_train_samples"]]
    
    if CONFIG["max_val_samples"] is not None:
        print(f"--- Limiting validation data to {CONFIG['max_val_samples']} samples (out of {len(val_files)}) ---")
        val_files = val_files[:CONFIG["max_val_samples"]]
    
    if CONFIG["max_test_samples"] is not None:
        print(f"--- Limiting test data to {CONFIG['max_test_samples']} samples (out of {len(test_files)}) ---")
        test_files = test_files[:CONFIG["max_test_samples"]]
    # <<< END OF MODIFICATION >>>

    if not train_files:
        print(f"Error: Could not find training files in {os.path.join(base_dir, 'train')}")
        return
    if not val_files:
        print(f"Warning: Could not find validation files in {os.path.join(base_dir, 'val')}")
    if not test_files:
        print(f"Warning: Could not find test files in {os.path.join(base_dir, 'test')}")

    
    train_dataset = HologramPreprocessedDataset(train_files)
    val_dataset = HologramPreprocessedDataset(val_files)
    test_dataset = HologramPreprocessedDataset(test_files)
    
    use_persistent_workers = CONFIG['num_workers'] > 0
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=CONFIG['num_workers'], pin_memory=True, 
                              persistent_workers=use_persistent_workers)
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, 
                                num_workers=CONFIG['num_workers'], pin_memory=True, 
                                persistent_workers=use_persistent_workers)
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, 
                                 num_workers=CONFIG['num_workers'], pin_memory=True, 
                                 persistent_workers=use_persistent_workers)
    
    print(f"\nDatasets initialized. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)} samples.")

    # --- Model, Optimizer, and Scheduler ---
    model = HologramCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # --- Load Checkpoint if Resuming ---
    if CONFIG["resume_from_dir"] and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
        print(f"Resumed from checkpoint. Starting at epoch {start_epoch}.")

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=CONFIG['patience']//2)
    
    epochs_no_improve = 0

    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, CONFIG['epochs']):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
        for x_batch, y_batch in train_pbar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = hologram_loss(y_pred, y_batch, x_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.5f}")
        avg_train_loss = running_train_loss / len(train_loader)

        # --- Validation Phase ---
        avg_val_loss = 0.0
        if val_loader:
            model.eval()
            running_val_loss = 0.0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]")
            with torch.no_grad():
                for i, (x_batch, y_batch) in enumerate(val_pbar):
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    y_pred = model(x_batch)
                    loss = hologram_loss(y_pred, y_batch, x_batch)
                    running_val_loss += loss.item()
                    val_pbar.set_postfix(loss=f"{loss.item():.5f}")
                    
                    if i == 0:
                        n_images = min(x_batch.size(0), 4)
                        grid = make_grid(
                            torch.cat([
                                y_batch[:n_images, 0:1], y_pred[:n_images, 0:1],
                                y_batch[:n_images, 1:2], y_pred[:n_images, 1:2]
                            ]), nrow=n_images, normalize=True, scale_each=True
                        )
                        writer.add_image('Validation/Images (True-Pred Amp/Phase)', grid, epoch)
            
            avg_val_loss = running_val_loss / len(val_loader)
        else:
            print("No validation loader. Skipping validation.")
            avg_val_loss = avg_train_loss 
        
        # --- Logging and Checkpointing ---
        writer.add_scalars('Loss', {'Train': avg_train_loss, 'Validation': avg_val_loss}, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'loss': best_val_loss,
            }, model_path)
            epochs_no_improve = 0
            print(f"New best model saved with loss: {best_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG['patience']:
                print(f"Early stopping triggered after {CONFIG['patience']} epochs.")
                break
    
    writer.close()
    print("--- Training Finished ---")

    # --- Final Evaluation and Visualization ---
    if test_loader:
        print(f"\n--- Evaluating Best Model from {model_path} ---")
        checkpoint = torch.load(model_path)
        best_model = HologramCNN().to(device)
        best_model.load_state_dict(checkpoint['model_state_dict'])
        best_model.eval()
        
        print("\n--- Generating Final Visualization Plot ---")
        x_sample_batch, y_true_batch = next(iter(test_loader))
        x_sample_dev = x_sample_batch[0:1].to(device)
        y_true_sample = y_true_batch[0]
        
        with torch.no_grad():
            y_pred_sample = best_model(x_sample_dev).squeeze(0)
        
        y_true_np = y_true_sample.cpu().numpy().transpose(1, 2, 0)
        y_pred_np = y_pred_sample.cpu().numpy().transpose(1, 2, 0)

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f"Final Model Performance - {CONFIG['experiment_name']}")
        im1 = axs[0, 0].imshow(y_true_np[..., 0], cmap='gray'); axs[0, 0].set_title("True Amplitude"); fig.colorbar(im1, ax=axs[0, 0], shrink=0.8)
        im2 = axs[0, 1].imshow(y_pred_np[..., 0], cmap='gray'); axs[0, 1].set_title("Predicted Amplitude"); fig.colorbar(im2, ax=axs[0, 1], shrink=0.8)
        im3 = axs[1, 0].imshow(y_true_np[..., 1], cmap='twilight'); axs[1, 0].set_title("True Phase"); fig.colorbar(im3, ax=axs[1, 0], shrink=0.8)
        im4 = axs[1, 1].imshow(y_pred_np[..., 1], cmap='twilight'); axs[1, 1].set_title("Predicted Phase"); fig.colorbar(im4, ax=axs[1, 1], shrink=0.8)
        
        for ax_row in axs:
            for ax in ax_row: ax.set_xticks([]); ax.set_yticks([])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        final_plot_path = os.path.join(log_dir, "final_prediction.png")
        plt.savefig(final_plot_path)
        print(f"Saved final visualization to '{final_plot_path}'")
    else:
        print("No test loader. Skipping final evaluation.")


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    main()