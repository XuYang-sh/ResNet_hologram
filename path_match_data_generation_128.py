import numpy as np
from scipy.optimize import linear_sum_assignment
import time

# --- Configuration ---
NUM_VALID_ITERATIONS = 2500  # The number of valid paths to generate
MAX_DISPLACEMENT_THRESHOLD = 7  # Keep only simulations below this max move (µm)
NUM_STEPS = 21  # Number of steps for the movement, including start and end

# --- Physical Array Parameters (in micrometers) ---
initial_dim = 19
initial_spacing = 3.0
target_dim = 13
target_spacing = 4.5
loading_rate = 0.6

# --- SLM Computational Grid Parameters ---
GRID_DIMENSION = 1024
# Define a field of view that comfortably contains the physical arrays.
wavelength_um = 0.532
pitch_um = 17.0
slm_pixel_number = 128
objective_NA = 0.6
FIELD_OF_VIEW_UM = wavelength_um / 2 / objective_NA * slm_pixel_number # ~56.7um
SCALE_FACTOR = GRID_DIMENSION / FIELD_OF_VIEW_UM

# --- Pre-calculate fixed array geometries ---
print("Pre-calculating array geometries...")
# Initial Array (all possible sites)
x_init = np.arange(initial_dim) * initial_spacing
y_init = np.arange(initial_dim) * initial_spacing
xx_i, yy_i = np.meshgrid(x_init, y_init)
all_initial_sites = np.vstack([xx_i.ravel(), yy_i.ravel()]).T
all_initial_sites -= all_initial_sites.mean(axis=0) # Center around (0,0)

# Target Array
x_targ = np.arange(target_dim) * target_spacing
y_targ = np.arange(target_dim) * target_spacing
xx_t, yy_t = np.meshgrid(x_targ, y_targ)
target_sites = np.vstack([xx_t.ravel(), yy_t.ravel()]).T
target_sites -= target_sites.mean(axis=0) # Center around (0,0)

num_initial_sites = all_initial_sites.shape[0]
num_target_sites = target_sites.shape[0]

# --- Main Data Generation Loop ---
print(f"Starting data generation for {NUM_VALID_ITERATIONS} valid simulations...")
all_valid_paths = []
rng = np.random.default_rng()
total_attempts = 0
start_time = time.time()

while len(all_valid_paths) < NUM_VALID_ITERATIONS:
    total_attempts += 1
    
    # 1. Simulate a new random loading
    loaded_indices = rng.choice(
        num_initial_sites,
        size=int(num_initial_sites * loading_rate),
        replace=False
    )
    source_atoms = all_initial_sites[loaded_indices]

    # 2. Build the cost matrix (squared Euclidean distance)
    diff = source_atoms[:, None, :] - target_sites[None, :, :]
    cost_matrix = np.sum(diff**2, axis=2)

    # 3. Solve the assignment problem with the Hungarian algorithm
    atom_indices, target_indices = linear_sum_assignment(cost_matrix)

    # 4. Filter based on maximum displacement
    assigned_starts = source_atoms[atom_indices]
    assigned_ends = target_sites[target_indices]
    
    distances = np.linalg.norm(assigned_starts - assigned_ends, axis=1)
    max_dist = distances.max()

    if max_dist < MAX_DISPLACEMENT_THRESHOLD:
        # This is a valid path, so we process and save it.
        
        # 5. Interpolate the path into N steps
        # Shape: (num_steps, num_targets, 2)
        steps = np.linspace(0, 1, NUM_STEPS)[:, np.newaxis, np.newaxis]
        path_in_um = assigned_starts + steps * (assigned_ends - assigned_starts)

        # 6. Map physical coordinates (µm) to the 8192x8192 grid
        # We shift the origin from the center to the top-left (0,0) of the grid
        path_in_grid_coords = (path_in_um * SCALE_FACTOR) + (GRID_DIMENSION / 2)
        
        # Round to nearest integer pixel and ensure correct data type
        path_in_grid_coords = np.round(path_in_grid_coords).astype(np.uint16)
        
        all_valid_paths.append(path_in_grid_coords)
        
        # Progress update
        if len(all_valid_paths) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Generated {len(all_valid_paths)}/{NUM_VALID_ITERATIONS} valid paths... (Total attempts: {total_attempts}, Time: {elapsed:.1f}s)")


# --- Finalize and Save Data ---
print("\nData generation complete.")
print(f"Total attempts made: {total_attempts}")
print(f"Acceptance rate: {NUM_VALID_ITERATIONS / total_attempts * 100:.2f}%")

# Stack all paths into a single large numpy array
# Shape: (num_simulations, num_steps, num_targets, 2)
final_data_array = np.stack(all_valid_paths, axis=0)

print(f"\nFinal data array shape: {final_data_array.shape}")
print(f"Data type: {final_data_array.dtype}")

# Save the compressed numpy array to a file
output_filename = f'path_data_grid_{GRID_DIMENSION}x{GRID_DIMENSION}.npz'
np.savez_compressed(output_filename, tweezer_paths=final_data_array)

print(f"\nSuccessfully saved all path data to '{output_filename}'")
