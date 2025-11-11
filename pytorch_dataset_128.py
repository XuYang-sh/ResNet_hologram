# -*- coding: utf-8 -*-
"""
This script generates the training data for the AI model and saves it
DIRECTLY into the final pre-processed .pt format.

Each *step* of a rearrangement path is calculated and saved as its own
.pt file, which is automatically sorted into train/val/test folders.

This eliminates the need for an intermediate .npz format and a
separate pre-processing script.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import torch
import torch.nn.functional as F
import cupy as cp
import gc

# Try to import slmsuite, provide a helpful error if it's not installed.
try:
    from slmsuite.holography.algorithms import SpotHologram
except ImportError:
    print("slmsuite not found. Please install it with: pip install slmsuite")
    exit()

# --- 1. Main Configuration ---
PLOT_DEBUG_FIGURES = True  # Set to True to see plots for the first step of the first path
NUM_SIMULATIONS_TO_PROCESS = -1 # Process all available paths (-1)
PATH_BIAS = 0000  # Add an offset to saved path file indices (e.g. start at path_01000)

AI_GRID_DIM = 128
WGS_GRID_DIM = 1024
GRID_RATIO = WGS_GRID_DIM // AI_GRID_DIM

WGS_MAX_ITERATIONS = 10
WGS_FIXED_PHASE_INTERATIONS = 5
PADDING_FACTOR = 8  # PADDING_FACTOR for 128 is 8, (128*4 = 1024)

# --- Configuration for Scalable Output ---

# Get $SCRATCH directory for input and output
SCRATCH_DIR = os.getenv('SCRATCH')
if not SCRATCH_DIR:
    print("Warning: $SCRATCH environment variable not set. Using current directory.")
    SCRATCH_DIR = "."  # Fallback to current directory

# <<< MODIFIED >>>
# Output directory now points to the "processed" folder
OUTPUT_BASE_DIR = os.path.join(SCRATCH_DIR, f"hologram_dataset_processed_{AI_GRID_DIM}x{AI_GRID_DIM}")
TRAIN_DIR = os.path.join(OUTPUT_BASE_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_BASE_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_BASE_DIR, "test")

# <<< MODIFIED >>>
# Updated split, focusing on train and val as requested.
SPLIT_RATIOS = {'train': 0.89, 'val': 0.10, 'test': 0.01}

INPUT_DATA_FILE = os.path.join(SCRATCH_DIR, f'path_data_grid_{WGS_GRID_DIM}x{WGS_GRID_DIM}.npz')
if not os.path.exists(INPUT_DATA_FILE):
    print(f"Error: Input file '{INPUT_DATA_FILE}' not found.")
    print("Please run 'generate_paths.py' first.")
    exit()

# --- 2. Helper Functions for Input Encoding (Unchanged) ---
def create_interpolated_image(coords, values, grid_dim):
    grid = np.zeros((grid_dim, grid_dim), dtype=np.float32)
    x, y = coords[:, 0], coords[:, 1]
    x_int, y_int = np.floor(x).astype(int), np.floor(y).astype(int)
    x_frac, y_frac = x - x_int, y - y_int
    valid_mask = (x_int >= 0) & (x_int < grid_dim - 1) & (y_int >= 0) & (y_int < grid_dim - 1)
    x_int, y_int, x_frac, y_frac, values = (arr[valid_mask] for arr in [x_int, y_int, x_frac, y_frac, values])
    w_tl, w_tr, w_bl, w_br = (1 - x_frac) * (1 - y_frac), x_frac * (1 - y_frac), (1 - x_frac) * y_frac, x_frac * y_frac
    np.add.at(grid, (y_int, x_int), values * w_tl)
    np.add.at(grid, (y_int, x_int + 1), values * w_tr)
    np.add.at(grid, (y_int + 1, x_int), values * w_bl)
    np.add.at(grid, (y_int + 1, x_int + 1), values * w_br)
    return grid


def create_phase_image_no_interpolation(coords, values, grid_dim):
    """
    Creates a phase image by assigning the phase value to the four
    neighboring pixels of a float coordinate, without interpolation.
    """
    grid = np.zeros((grid_dim, grid_dim), dtype=np.float32)
    x, y = coords[:, 0], coords[:, 1]
    x_int, y_int = np.floor(x).astype(int), np.floor(y).astype(int)
    valid_mask = (x_int >= 0) & (x_int < grid_dim - 1) & (y_int >= 0) & (y_int < grid_dim - 1)
    x_int, y_int, values = x_int[valid_mask], y_int[valid_mask], values[valid_mask]
    grid[y_int, x_int] = values
    grid[y_int, x_int + 1] = values
    grid[y_int + 1, x_int] = values
    grid[y_int + 1, x_int + 1] = values
    return grid

# --- 3. Main Logic ---
def main():
    print("--- Setting up directories and loading path data ---")
    os.makedirs(TRAIN_DIR, exist_ok=True); os.makedirs(VAL_DIR, exist_ok=True); os.makedirs(TEST_DIR, exist_ok=True)
    print(f"Dataset will be saved in: '{OUTPUT_BASE_DIR}'")

    print(f"Loading coordinate data from '{INPUT_DATA_FILE}'...")
    with np.load(INPUT_DATA_FILE) as data:
        tweezer_paths = data['tweezer_paths']

    total_available = tweezer_paths.shape[0]
    start_idx = PATH_BIAS
    if start_idx < 0 or start_idx >= total_available:
        print(f"Error: PATH_BIAS={PATH_BIAS} is out of range for available paths (0..{total_available - 1}).")
        exit()

    if NUM_SIMULATIONS_TO_PROCESS == -1:
        end_idx = total_available
    else:
        end_idx = min(start_idx + NUM_SIMULATIONS_TO_PROCESS, total_available)

    tweezer_paths = tweezer_paths[start_idx:end_idx]
    num_simulations, num_steps, num_targets, _ = tweezer_paths.shape
    print(f"Loaded {num_simulations} simulations, each with {num_steps} steps.")

    print(f"\n--- Processing {num_simulations} paths and saving each *step* as a .pt file ---")
    progress_bar = tqdm(range(num_simulations), unit="path")

    # Determine split indices based on paths
    train_end_idx = int(num_simulations * SPLIT_RATIOS['train'])
    val_end_idx = train_end_idx + int(num_simulations * SPLIT_RATIOS['val'])

    # <<< MODIFIED >>>
    # Counters for individual samples in each split
    train_sample_counter = 0
    val_sample_counter = 0
    test_sample_counter = 0

    mempool = cp.get_default_memory_pool()

    for sim_idx in progress_bar:
        
        # <<< MODIFIED >>>
        # Determine output directory and counter *per path*
        # All steps from one path go to the same split
        if sim_idx < train_end_idx:
            output_dir = TRAIN_DIR
        elif sim_idx < val_end_idx:
            output_dir = VAL_DIR
        else:
            output_dir = TEST_DIR
            
        # <<< REMOVED >>>
        # No longer need to aggregate data for the whole path
        # path_A_input, path_phi_input, path_A_label, path_phi_label = [], [], [], []

        for step_idx in range(num_steps):
            coords_WGS = tweezer_paths[sim_idx, step_idx]

            # a) Calculate hologram using slm_shape for high accuracy
            holo = SpotHologram(shape=(WGS_GRID_DIM, WGS_GRID_DIM), slm_shape=(AI_GRID_DIM, AI_GRID_DIM), spot_vectors=coords_WGS.T, basis="knm")
            holo.optimize(method="WGS-Kim", maxiter=WGS_MAX_ITERATIONS, fix_phase_iteration=WGS_FIXED_PHASE_INTERATIONS, verbose=False)
            slm_hologram = holo.get_phase() - np.pi  # Shift phase to [-pi, pi]

            # b) Get far-field to extract tweezer phases for the input
            farfield = holo.get_farfield()
            phases_at_tweezers = np.angle(farfield[coords_WGS[:, 1], coords_WGS[:, 0]])

            # c) Generate AI inputs
            coords_AI_float = coords_WGS / GRID_RATIO
            A_input = create_interpolated_image(coords_AI_float, np.ones(num_targets), AI_GRID_DIM)
            phi_input = create_phase_image_no_interpolation(coords_AI_float, phases_at_tweezers, AI_GRID_DIM)

            # d) Generate AI labels
            hologram_AI_gpu = cp.asarray(slm_hologram)
            label_field_AI = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.exp(1j * hologram_AI_gpu)), norm="ortho"))
            A_label_gpu = cp.abs(label_field_AI)
            max_val = A_label_gpu.max()
            if max_val > 0: A_label_gpu /= max_val
            phi_label_gpu = cp.angle(label_field_AI)

            # <<< REMOVED >>>
            # No longer appending to lists
            
            # <<< MODIFIED >>>
            # Convert directly to Tensors and save
            
            # 1. Create x_tensor (input)
            x_tensor = torch.from_numpy(
                np.stack([A_input, phi_input])
            ).float() # Shape: (2, 1024, 1024)
            
            # 2. Create y_tensor (label)
            # Move from CuPy -> NumPy -> Torch Tensor
            A_label_np = cp.asnumpy(A_label_gpu).astype(np.float32)
            phi_label_np = cp.asnumpy(phi_label_gpu).astype(np.float32)
            y_tensor = torch.from_numpy(
                np.stack([A_label_np, phi_label_np])
            ).float() # Shape: (2, 1024, 1024)

            # 3. Determine filename and save
            if sim_idx < train_end_idx:
                output_filename = os.path.join(output_dir, f"sample_{train_sample_counter:08d}.pt")
                train_sample_counter += 1
            elif sim_idx < val_end_idx:
                output_filename = os.path.join(output_dir, f"sample_{val_sample_counter:08d}.pt")
                val_sample_counter += 1
            else:
                output_filename = os.path.join(output_dir, f"sample_{test_sample_counter:08d}.pt")
                test_sample_counter += 1
                
            # Save the (x_tensor, y_tensor) tuple
            torch.save((x_tensor, y_tensor), output_filename)
            
            # --- Debug Plots for the very first step (Unchanged) ---
            if sim_idx == 0 and step_idx == 0 and PLOT_DEBUG_FIGURES:
                print("\nGenerating debug plots for the first data sample...")

                # Plot 1: Tweezer Intensity from Padded Hologram
                padded_shape = (AI_GRID_DIM * PADDING_FACTOR, AI_GRID_DIM * PADDING_FACTOR)
                slm_field_torch = torch.exp(1j * torch.from_numpy(slm_hologram).float())
                padded_slm_field = F.pad(slm_field_torch, [(s - AI_GRID_DIM) // 2 for s in reversed(padded_shape) for _ in range(2)])
                far_field_torch = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(padded_slm_field), norm="ortho"))
                tweezer_power = (torch.abs(far_field_torch)**2).cpu().numpy()
                true_positions_padded = coords_WGS * (padded_shape[0] / WGS_GRID_DIM)
                intensities_at_targets = tweezer_power[true_positions_padded[:, 1].astype(int), true_positions_padded[:, 0].astype(int)]
                mean_intensity = np.mean(intensities_at_targets)
                std_intensity = np.std(intensities_at_targets)
                homogeneity = std_intensity / mean_intensity if mean_intensity > 0 else 0

                fig_ff, ax_ff = plt.subplots(figsize=(8, 8))
                im_ff = ax_ff.imshow(tweezer_power, cmap='hot')
                ax_ff.plot(true_positions_padded[:, 0], true_positions_padded[:, 1], 'c+', markersize=5, alpha=0.7, label="Target Locations")
                if true_positions_padded.shape[0] > 0:
                    min_x, min_y = np.min(true_positions_padded, axis=0); max_x, max_y = np.max(true_positions_padded, axis=0)
                    padding = 50 * (padded_shape[0] / WGS_GRID_DIM)
                    ax_ff.set_xlim(min_x - padding, max_x + padding)
                    ax_ff.set_ylim(max_y + padding, min_y - padding)
                ax_ff.set_title(f"Debug: Tweezer Intensity from Padded Hologram")
                ax_ff.set_xlabel(f"{padded_shape[0]} Grid X (pixels)"); ax_ff.set_ylabel(f"{padded_shape[0]} Grid Y (pixels)")
                ax_ff.text(0.02, 0.98, f"Homogeneity (std/mean): {homogeneity:.4f}",
                           transform=ax_ff.transAxes, color='white', fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))
                ax_ff.legend()
                fig_ff.colorbar(im_ff, ax=ax_ff, fraction=0.046, pad=0.04)
                plot_filename_ff = "debug_tweezer_intensity.png"
                fig_ff.savefig(plot_filename_ff)
                print(f"Saved debug plot to '{plot_filename_ff}'")
                if PLOT_DEBUG_FIGURES: plt.show()
                plt.close(fig_ff)

                # Plot 2: AI Input/Label Pair
                fig_ai, axs_ai = plt.subplots(2, 2, figsize=(8, 8))
                fig_ai.suptitle(f"Debug: AI Input/Label Pair")
                axs_ai[0, 0].imshow(A_input, cmap='gray'); axs_ai[0, 0].set_title("A_input")
                axs_ai[0, 1].imshow(phi_input, cmap='twilight'); axs_ai[0, 1].set_title("phi_input")
                axs_ai[1, 0].imshow(cp.asnumpy(A_label_gpu), cmap='gray'); axs_ai[1, 0].set_title("A_label")
                axs_ai[1, 1].imshow(cp.asnumpy(phi_label_gpu), cmap='twilight'); axs_ai[1, 1].set_title("phi_label")
                for ax in axs_ai.ravel(): ax.set_xticks([]); ax.set_yticks([])
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plot_filename_ai = "debug_ai_input_label_pair.png"
                fig_ai.savefig(plot_filename_ai)
                print(f"Saved debug plot to '{plot_filename_ai}'")
                if PLOT_DEBUG_FIGURES: plt.show()
                plt.close(fig_ai)

        # --- End of step_idx loop ---
        
        # <<< MODIFIED >>>
        # Clean up GPU memory *after each path*
        mempool.free_all_blocks()
        gc.collect()

    # --- End of sim_idx loop ---
    progress_bar.close()

    print(f"\n--- Generation Complete ---")
    print(f"Successfully generated and saved {train_sample_counter} training samples.")
    print(f"Successfully generated and saved {val_sample_counter} validation samples.")
    print(f"Successfully generated and saved {test_sample_counter} test samples.")
    print(f"\nTraining data is in: '{TRAIN_DIR}'")
    print(f"Validation data is in: '{VAL_DIR}'")
    print(f"Test data is in: '{TEST_DIR}'")

if __name__ == '__main__':
    main()