# -*- coding: utf-8 -*-
"""
This script evaluates the positional accuracy and quality of a trained PyTorch model.
For a number of simulated rearrangement sequences, it does the following for each of the 21 steps:
1.  Generates the hologram from the model.
2.  Computes the resulting far-field tweezer power map and phase map.
3.  Finds tweezer centroids and fits them with a 2D Gaussian profile.
4.  Calculates displacement error by comparing to ground truth positions.
5.  Calculates quality metrics (intensity, spot size, ellipticity) from the Gaussian fits.
6.  Calculates the weighted average phase of each spot and compares it to the target phase.
Finally, it plots the overall distributions and step-wise evolution of all metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment, curve_fit
from scipy.spatial import KDTree
from tqdm import tqdm
import glob
import cv2
from matplotlib.patches import Circle, Rectangle
from matplotlib.gridspec import GridSpec

# Try to import slmsuite, provide a helpful error if it's not installed.
try:
    from slmsuite.holography.algorithms import SpotHologram
except ImportError:
    print("Error: slmsuite is not installed. Please install it with 'pip install slmsuite'")
    exit()

# --- 1. Configuration (Aligned with prepare_training_data.py) ---

EXPERIMENT_NAME = "HologramRun128_0"  # Must match the training experiment name

# --- Analysis Parameters ---
NUM_ANALYSIS_CASES = 20 # How many random rearrangements to simulate.
PADDING_FACTOR = 8     # Factor to pad the SLM grid for high-res FFT.
ROI_SIZE = 2*PADDING_FACTOR + 1           # Size of the box (ROI_SIZE x ROI_SIZE) to extract for fitting.
PLOT_ROI_DEBUG = False   # Set to True to generate a plot visualizing the ROIs for the first step of the first case.
CENTROID_FINDING_METHOD = 'ground_truth_guess' # Options: 'opencv' or 'ground_truth_guess'

# --- Physical Array Parameters (in micrometers) ---
initial_dim = 19
initial_spacing = 3.0
target_dim = 13
target_spacing = 4.5
loading_rate = 0.5

# --- SLM Computational Grid Parameters ---
GRID_DIMENSION = 1024
slm_pixel_number = 128

# Define a field of view that comfortably contains the physical arrays.
wavelength_um = 0.532
objective_NA = 0.6
FIELD_OF_VIEW_UM = wavelength_um / 2 / objective_NA * slm_pixel_number
SCALE_FACTOR = GRID_DIMENSION / FIELD_OF_VIEW_UM
GRID_RATIO = GRID_DIMENSION // slm_pixel_number

# WGS algorithm parameters for phase extraction
WGS_MAX_ITERATIONS = 15
WGS_FIXED_PHASE_INTERATIONS = 5


# --- 2. Model Definition ---
# This must be IDENTICAL to the model definition in your training script.
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

# --- 3. Helper Functions ---
def create_interpolated_image(coords, values, grid_dim):
    grid = np.zeros((grid_dim, grid_dim), dtype=np.float32)
    x, y = coords[:, 0], coords[:, 1]
    x_int, y_int = np.floor(x).astype(int), np.floor(y).astype(int)
    x_frac, y_frac = x - x_int, y - y_int
    valid_mask = (x_int >= 0) & (x_int < grid_dim - 1) & (y_int >= 0) & (y_int < grid_dim - 1)
    x_int, y_int, x_frac, y_frac, values = (arr[valid_mask] for arr in [x_int, y_int, x_frac, y_frac, values])
    w_tl = (1 - x_frac) * (1 - y_frac); w_tr = x_frac * (1 - y_frac)
    w_bl = (1 - x_frac) * y_frac;   w_br = x_frac * y_frac
    np.add.at(grid, (y_int, x_int), values * w_tl)
    np.add.at(grid, (y_int, x_int + 1), values * w_tr)
    np.add.at(grid, (y_int + 1, x_int), values * w_bl)
    np.add.at(grid, (y_int + 1, x_int + 1), values * w_br)
    return grid

def create_phase_image_no_interpolation(coords, values, grid_dim):
    grid = np.zeros((grid_dim, grid_dim), dtype=np.float32)
    x, y = coords[:, 0], coords[:, 1]
    x_int, y_int = np.floor(x).astype(int), np.floor(y).astype(int)
    valid_mask = (x_int >= 0) & (x_int < grid_dim - 1) & (y_int >= 0) & (y_int < grid_dim - 1)
    x_int, y_int, values = x_int[valid_mask], y_int[valid_mask], values[valid_mask]
    grid[y_int, x_int] = values; grid[y_int, x_int + 1] = values
    grid[y_int + 1, x_int] = values; grid[y_int + 1, x_int + 1] = values
    return grid

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    xo, yo = float(x0), float(y0)
    g = offset + amplitude * np.exp(-((x - xo)**2 / (2 * sigma_x**2) + (y - yo)**2 / (2 * sigma_y**2)))
    return g.ravel()

def fit_gaussian_to_roi(roi):
    h, w = roi.shape
    x, y = np.arange(w), np.arange(h)
    xx, yy = np.meshgrid(x, y)
    amp_guess = np.max(roi) - np.min(roi)
    if amp_guess < 0: amp_guess = 0
    offset_guess = np.min(roi)
    y0_guess, x0_guess = np.unravel_index(np.argmax(roi), roi.shape)
    sigma_guess = (h + w) / 8.0
    initial_guess = [amp_guess, x0_guess, y0_guess, sigma_guess, sigma_guess, offset_guess]
    bounds = ([0, 0, 0, 0.1, 0.1, -np.inf], [np.inf, w, h, w, h, np.inf])
    try:
        popt, _ = curve_fit(gaussian_2d, (xx, yy), roi.ravel(), p0=initial_guess, bounds=bounds)
        return popt
    except (RuntimeError, ValueError):
        return None

# --- 4. Main Analysis Logic ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model ---
    list_of_dirs = glob.glob(f"runs/*_{EXPERIMENT_NAME}")
    if not list_of_dirs: print(f"Error: No experiment directories for '{EXPERIMENT_NAME}'"); return
    latest_dir = max(list_of_dirs, key=os.path.getctime)
    model_path = os.path.join(latest_dir, "best_model.pt")
    if not os.path.exists(model_path): print(f"Error: Model file not found at '{model_path}'"); return
    
    print(f"--- Loading best model from '{model_path}' ---")
    checkpoint = torch.load(model_path, map_location=device)
    model = HologramCNN().to(device); model.load_state_dict(checkpoint['model_state_dict']); model.eval()

    # --- Data Structures for Step-wise Analysis ---
    num_steps = 21
    all_step_displacement_errors = [[] for _ in range(num_steps)]
    all_step_intensities = [[] for _ in range(num_steps)]
    all_step_waists_x = [[] for _ in range(num_steps)]
    all_step_waists_y = [[] for _ in range(num_steps)]
    all_step_ellipticities = [[] for _ in range(num_steps)]
    all_step_phase_errors = [[] for _ in range(num_steps)] # New

    padded_shape = (slm_pixel_number * PADDING_FACTOR, slm_pixel_number * PADDING_FACTOR)
    pixel_size_nm = (FIELD_OF_VIEW_UM * 1000) / padded_shape[0]

    for case_num in tqdm(range(NUM_ANALYSIS_CASES), desc="Analyzing Cases"):
        # --- Simulate a path ---
        # ... (path simulation code is unchanged)
        x_init = np.arange(initial_dim) * initial_spacing
        y_init = np.arange(initial_dim) * initial_spacing
        xx_i, yy_i = np.meshgrid(x_init, y_init)
        all_initial_sites = np.vstack([xx_i.ravel(), yy_i.ravel()]).T; all_initial_sites -= all_initial_sites.mean(axis=0)
        x_targ = np.arange(target_dim) * target_spacing
        y_targ = np.arange(target_dim) * target_spacing
        xx_t, yy_t = np.meshgrid(x_targ, y_targ)
        target_sites = np.vstack([xx_t.ravel(), yy_t.ravel()]).T; target_sites -= target_sites.mean(axis=0)
        num_to_load = int(all_initial_sites.shape[0] * loading_rate)
        loaded_indices = np.random.choice(all_initial_sites.shape[0], size=num_to_load, replace=False)
        source_atoms = all_initial_sites[loaded_indices]
        cost_matrix = np.sum((source_atoms[:, None, :] - target_sites[None, :, :])**2, axis=2)
        atom_idx, targ_idx = linear_sum_assignment(cost_matrix)
        start_points, end_points = source_atoms[atom_idx], target_sites[targ_idx]

        def get_phases(points_um):
            coords_wgs = (points_um + FIELD_OF_VIEW_UM / 2) * SCALE_FACTOR
            coords_wgs_int = np.round(coords_wgs).astype(int)
            hologram = SpotHologram(
                shape=(GRID_DIMENSION, GRID_DIMENSION),
                slm_shape=(slm_pixel_number, slm_pixel_number),
                spot_vectors=coords_wgs_int.T, basis="knm")
            hologram.optimize(method="WGS-Kim", maxiter=WGS_MAX_ITERATIONS, fix_phase_iteration=WGS_FIXED_PHASE_INTERATIONS, verbose=False)
            farfield = hologram.get_farfield()
            return np.angle(farfield[coords_wgs_int[:, 1], coords_wgs_int[:, 0]])

        start_phases, end_phases = get_phases(start_points), get_phases(end_points)
        positions_sequence = np.linspace(start_points, end_points, num_steps)
        phases_sequence = np.linspace(start_phases, end_phases, num_steps)
        
        for i in range(num_steps):
            coords_um, target_phases = positions_sequence[i], phases_sequence[i]
            
            # --- Generate Hologram and Far-field ---
            coords_wgs = (coords_um + FIELD_OF_VIEW_UM / 2) * SCALE_FACTOR
            coords_AI_float = coords_wgs / GRID_RATIO
            
            A_input = create_interpolated_image(coords_AI_float, np.ones_like(target_phases), slm_pixel_number)
            phi_input = create_phase_image_no_interpolation(coords_AI_float, target_phases, slm_pixel_number)
            model_input = torch.from_numpy(np.stack([A_input, phi_input])).unsqueeze(0).to(device)
            
            with torch.no_grad():
                A_pred, phi_pred = model(model_input).squeeze(0)
                position_domain_field = A_pred * torch.exp(1j * phi_pred)
                hologram_complex = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(position_domain_field), norm="ortho"))
                slm_field = torch.exp(1j * torch.angle(hologram_complex))
                padded_slm_field = F.pad(slm_field, [ (s - slm_pixel_number) // 2 for s in reversed(padded_shape) for _ in range(2) ])
                far_field = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(padded_slm_field), norm="ortho"))
                tweezer_power = (torch.abs(far_field)**2).cpu().numpy()
                tweezer_phase = torch.angle(far_field).cpu().numpy()

            # --- Find, Match, and Analyze Tweezers ---
            true_positions_padded = (coords_um + FIELD_OF_VIEW_UM / 2) / FIELD_OF_VIEW_UM * padded_shape[0]
            
            final_found_positions = []
            
            if CENTROID_FINDING_METHOD == 'ground_truth_guess':
                positions_to_fit = true_positions_padded
                true_positions_to_compare = true_positions_padded
            else:
                raise ValueError(f"Unknown method. Use 'ground_truth_guess'")

            # --- Unified Fitting and Analysis Loop ---
            for idx, (true_pos, guess_pos) in enumerate(zip(true_positions_to_compare, positions_to_fit)):
                x_c, y_c = int(round(guess_pos[0])), int(round(guess_pos[1]))
                half_roi = ROI_SIZE // 2
                
                x_start, x_end = max(0, x_c - half_roi), min(padded_shape[1], x_c + half_roi + 1)
                y_start, y_end = max(0, y_c - half_roi), min(padded_shape[0], y_c + half_roi + 1)
                power_roi = tweezer_power[y_start:y_end, x_start:x_end]

                if power_roi.size == 0: continue

                fit_params = fit_gaussian_to_roi(power_roi)

                if fit_params is not None:
                    amplitude, x0, y0, sigma_x, sigma_y, offset = fit_params
                    if sigma_x <= 0 or sigma_y <= 0: continue
                    
                    final_pos = np.array([x_start + x0, y_start + y0])
                    final_found_positions.append(final_pos)
                    
                    displacement_vector = final_pos - true_pos
                    all_step_displacement_errors[i].append(displacement_vector)
                    
                    intensity = amplitude
                    ellipticity = 1 - (min(sigma_x, sigma_y) / max(sigma_x, sigma_y))
                    all_step_intensities[i].append(intensity)
                    all_step_waists_x[i].append(sigma_x * pixel_size_nm * 2)
                    all_step_waists_y[i].append(sigma_y * pixel_size_nm * 2)
                    all_step_ellipticities[i].append(ellipticity)

                    # --- New Phase Analysis ---
                    phase_roi = tweezer_phase[y_start:y_end, x_start:x_end]
                    h, w = phase_roi.shape
                    x, y = np.arange(w), np.arange(h)
                    xx, yy = np.meshgrid(x, y)

                    # Create Gaussian weights from the intensity fit
                    weights = gaussian_2d((xx,yy), amplitude, x0, y0, sigma_x, sigma_y, 0).reshape(h,w)
                    
                    # Calculate weighted average of the raw phase values (no complex phasors / angle)
                    denom = np.sum(weights)
                    if denom == 0:
                        continue
                    measured_phase = np.sum(phase_roi * weights) / denom

                    # Compare with target and store wrapped error
                    target_phase = target_phases[idx]
                    # phase_error = measured_phase - target_phase + np.pi
                    phase_error = measured_phase - target_phase
                    wrapped_error = (phase_error + np.pi) % (2 * np.pi) - np.pi
                    # wrapped_error = phase_error
                    all_step_phase_errors[i].append(wrapped_error)

            # --- ROI Debug Plot ---
            if i == 2 and case_num == 0 and PLOT_ROI_DEBUG:
                fig_roi, (ax_power, ax_phase) = plt.subplots(1, 2, figsize=(16, 8))
                
                # --- Left Plot: Power ---
                im_power = ax_power.imshow(tweezer_power, cmap='hot', origin='upper')
                cbar_power = fig_roi.colorbar(im_power, ax=ax_power, fraction=0.046, pad=0.04); cbar_power.set_label('Power (a.u.)')
                
                debug_positions = final_found_positions
                
                for centroid in debug_positions:
                    x_c, y_c = centroid[0], centroid[1]
                    half_roi = ROI_SIZE // 2
                    rect = Rectangle((x_c - half_roi, y_c - half_roi), ROI_SIZE, ROI_SIZE, linewidth=1, edgecolor='cyan', facecolor='none', linestyle='--')
                    ax_power.add_patch(rect)
                
                num_found = len(debug_positions)
                ax_power.set_title(f"Far-Field Power (Found {num_found} Tweezers)"); ax_power.set_xlabel("Padded Grid X (pixels)"); ax_power.set_ylabel("Padded Grid Y (pixels)")

                # --- Right Plot: Phase ---
                im_phase = ax_phase.imshow(tweezer_phase, cmap='twilight', origin='upper')
                cbar_phase = fig_roi.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04); cbar_phase.set_label('Phase (radians)')
                
                # Draw ROI boxes on phase plot as well
                for centroid in debug_positions:
                    x_c, y_c = centroid[0], centroid[1]
                    half_roi = ROI_SIZE // 2
                    rect = Rectangle((x_c - half_roi, y_c - half_roi), ROI_SIZE, ROI_SIZE, linewidth=1, edgecolor='lime', facecolor='none', linestyle='--')
                    ax_phase.add_patch(rect)

                ax_phase.set_title("Far-Field Phase"); ax_phase.set_xlabel("Padded Grid X (pixels)")
                
                # Zoom both plots
                if len(debug_positions) > 0:
                    min_x, min_y = np.min(debug_positions, axis=0); max_x, max_y = np.max(debug_positions, axis=0)
                    padding = ROI_SIZE * 3
                    ax_power.set_xlim(min_x - padding, max_x + padding); ax_power.set_ylim(max_y + padding, min_y - padding)
                    ax_phase.set_xlim(min_x - padding, max_x + padding); ax_phase.set_ylim(max_y + padding, min_y - padding)
                
                fig_roi.suptitle(f"ROI Debug Plot (Case 0, Step 2)"); plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(os.path.join(latest_dir, "roi_debug_plot.png")); print(f"Saved ROI debug plot to '{os.path.join(latest_dir, 'roi_debug_plot.png')}'")
                plt.show()

    # --- 5. Process and Plot Final Results ---
    print(f"\n--- Plotting analysis from {sum(len(e) for e in all_step_displacement_errors)} matched tweezers ---")

    all_errors = np.array([item for sublist in all_step_displacement_errors for item in sublist])
    if all_errors.size == 0:
        print("\nWARNING: No valid tweezers were matched. Cannot generate plots."); return

    # ... (Stats calculation code is expanded)
    errors_nm = all_errors * pixel_size_nm
    dx_nm, dy_nm = errors_nm[:, 0], errors_nm[:, 1]
    magnitudes_nm = np.sqrt(dx_nm**2 + dy_nm**2)
    
    steps = np.arange(num_steps)
    mean_disp_error, std_disp_error = np.zeros(num_steps), np.zeros(num_steps)
    homogeneity = np.zeros(num_steps)
    mean_waist, std_waist = np.zeros(num_steps), np.zeros(num_steps)
    mean_ellipticity, std_ellipticity = np.zeros(num_steps), np.zeros(num_steps)
    mean_phase_error, std_phase_error = np.zeros(num_steps), np.zeros(num_steps) # New

    for i in range(num_steps):
        if all_step_displacement_errors[i]:
            disps = np.array(all_step_displacement_errors[i]) * pixel_size_nm
            mags = np.sqrt(np.sum(disps**2, axis=1))
            mean_disp_error[i], std_disp_error[i] = np.mean(mags), np.std(mags)
        if len(all_step_intensities[i]) > 1:
            intensities = np.array(all_step_intensities[i])
            homogeneity[i] = np.std(intensities) / np.mean(intensities)
        if all_step_waists_x[i] and all_step_waists_y[i]:
            waists = np.concatenate([all_step_waists_x[i], all_step_waists_y[i]])
            mean_waist[i], std_waist[i] = np.mean(waists), np.std(waists)
        if all_step_ellipticities[i]:
            ellipticities = np.array(all_step_ellipticities[i])
            mean_ellipticity[i], std_ellipticity[i] = np.mean(ellipticities), np.std(ellipticities)
        if all_step_phase_errors[i]: # New
            phase_errors = np.array(all_step_phase_errors[i])
            mean_phase_error[i], std_phase_error[i] = np.mean(phase_errors), np.std(phase_errors)

    # ... (Plotting code is updated)

    # Plot 1: Displacement Error Distribution (Unchanged)
    fig_dist = plt.figure(figsize=(10, 8))
    ax_2d = fig_dist.add_subplot(111)
    range_limit_nm = np.ceil(np.std(magnitudes_nm) * 4) if magnitudes_nm.size > 0 else 10
    h_range = [[-range_limit_nm, range_limit_nm], [-range_limit_nm, range_limit_nm]]
    mappable = ax_2d.hist2d(dx_nm, dy_nm, bins=50, cmap='inferno', range=h_range)[-1]
    ax_2d.set_title(f"AI Model - Overall Tweezer Displacement Error Distribution")
    ax_2d.set_xlabel("Displacement in X (nm)"); ax_2d.set_ylabel("Displacement in Y (nm)")
    ax_2d.set_aspect('equal', adjustable='box'); ax_2d.grid(True, linestyle='--', alpha=0.5)
    cbar = fig_dist.colorbar(mappable, ax=ax_2d); cbar.set_label('Counts')
    plt.tight_layout(); plot_path_dist = os.path.join(latest_dir, "displacement_error_distribution.png"); plt.savefig(plot_path_dist)
    print(f"Saved displacement distribution plot to '{plot_path_dist}'")
    plt.show()

    # Plot 2: Tweezer Quality Metrics Distribution (Updated to 2x2)
    fig_qual, axs_qual = plt.subplots(2, 2, figsize=(11, 9))
    axs_qual = axs_qual.ravel()
    all_intensities_flat = np.array([i for step_i in all_step_intensities for i in step_i])
    all_waists_flat = np.array([w for step_w in all_step_waists_x + all_step_waists_y for w in step_w])
    all_ellipticities_flat = np.array([e for step_e in all_step_ellipticities for e in step_e])
    all_phase_errors_flat = np.array([p for step_p in all_step_phase_errors for p in step_p]) # New
    
    if all_intensities_flat.size > 0:
        axs_qual[0].hist(all_intensities_flat / np.mean(all_intensities_flat), bins=50, color='skyblue', edgecolor='black'); axs_qual[0].set_title("Normalized Intensity Distribution"); axs_qual[0].set_xlabel("Intensity (Amplitude) / Mean")
    if all_waists_flat.size > 0:
        axs_qual[1].hist(all_waists_flat, bins=50, color='salmon', edgecolor='black'); axs_qual[1].set_title("Spot Waist (Size) Distribution"); axs_qual[1].set_xlabel("Waist (nm)")
    if all_ellipticities_flat.size > 0:
        axs_qual[2].hist(all_ellipticities_flat, bins=50, color='lightgreen', edgecolor='black'); axs_qual[2].set_title("Ellipticity Distribution"); axs_qual[2].set_xlabel("Ellipticity (1 - min/max waist)")
    if all_phase_errors_flat.size > 0: # New
        axs_qual[3].hist(all_phase_errors_flat, bins=100, color='plum', edgecolor='black'); axs_qual[3].set_title("Phase Error Distribution"); axs_qual[3].set_xlabel("Phase Error (radians)")
    
    for ax in axs_qual: ax.grid(True, linestyle='--', alpha=0.5); ax.set_ylabel("Frequency")
    fig_qual.suptitle("AI Model - Overall Tweezer Quality Metrics"); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); 
    plot_path_qual = os.path.join(latest_dir, "tweezer_quality_distribution.png"); plt.savefig(plot_path_qual)
    print(f"Saved quality distribution plot to '{plot_path_qual}'")
    plt.show()

    # Plot 3: Metrics vs. Rearrangement Step (Updated)
    fig_step, axs_step = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
    axs_step = axs_step.ravel()
    
    axs_step[0].errorbar(steps, mean_disp_error, yerr=std_disp_error, fmt='-o', capsize=4, label='Mean ± 1σ')
    axs_step[0].set_title('Displacement Error vs. Step'); axs_step[0].set_ylabel('Mean Displacement (nm)')

    axs_step[1].plot(steps, homogeneity * 100, '-o', color='C1')
    axs_step[1].set_title('Intensity Homogeneity vs. Step'); axs_step[1].set_ylabel('Std Dev / Mean (%)')

    axs_step[2].errorbar(steps, mean_waist, yerr=std_waist, fmt='-o', capsize=4, color='C2', label='Mean ± 1σ')
    axs_step[2].set_title('Spot Size (Waist) vs. Step'); axs_step[2].set_ylabel('Mean Waist (nm)')

    axs_step[3].errorbar(steps, mean_phase_error, yerr=std_phase_error, fmt='-o', capsize=4, color='C3', label='Mean ± 1σ') # Replaced Ellipticity
    axs_step[3].set_title('Phase Error vs. Step'); axs_step[3].set_ylabel('Mean Phase Error (radians)')

    for ax in axs_step:
        ax.grid(True, linestyle='--'); ax.set_xlabel('Rearrangement Step')
        ax.set_xticks(np.arange(0, num_steps, 2))
        if 'errorbar' in [c.get_label() for c in ax.containers if hasattr(c, 'get_label')]: ax.legend()
    
    fig_step.suptitle("AI Model - Tweezer Metrics Across Rearrangement Steps")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); 
    plot_path_step = os.path.join(latest_dir, "metrics_vs_step.png"); plt.savefig(plot_path_step)
    print(f"Saved metrics vs. step plot to '{plot_path_step}'")
    plt.show()

if __name__ == '__main__':
    main()

