"""
Build dataset consisting of static grating images to eventually merge with natural scenes dataset
"""

import h5py
import numpy as np
import os

# To change if you run this file by itself so it matches the .nwb you aim to convert to a dataset
PATH_TO_NWB = "brain_observatory/ophys_experiment_data/501794235.nwb"

current_file_path = os.path.dirname(os.path.abspath(__file__))
path_to_nwb = os.path.join(current_file_path, PATH_TO_NWB)

def quantize_to_step(y_values, step=50):
    return np.round(y_values / step) * step


def generate_gratings(orientations,
                      spatial_freqs,
                      phases,
                      size=(229, 293),
                      contrast=1.0,
                      max_pixel_value=255.0,
                      stripes_per_unit=75):  # calibrated: sf=0.32 → 24 stripes
    """
    Generate a batch of sinusoidal gratings.

    Based on parameters presented here: https://observatory.brain-map.org/visualcoding/stimulus/static_gratings

    Parameters
    ----------
    orientations     : array-like (N,) — degrees
    spatial_freqs    : array-like (N,) — cycles per degree
    phases           : array-like (N,) — fraction of cycle [0, 1]
    size             : (H, W) in pixels
    contrast         : float
    max_pixel_value  : float
    stripes_per_unit : float — calibration constant (sf=0.32 → 48 stripes)

    Returns
    -------
    imgs : (N, H, W)
    """
    orientations  = np.asarray(orientations)
    spatial_freqs = np.asarray(spatial_freqs)
    phases        = np.asarray(phases)

    N = len(orientations)

    # x axis: 0 to 1, so sf * stripes_per_unit = number of cycles across width
    x = np.linspace(0, 1, size[1])
    y = np.linspace(0, 1, size[0])
    X, Y = np.meshgrid(x, y)

    imgs = np.empty((N, size[0], size[1]), dtype=np.float32)

    for i in range(N):
        theta = np.deg2rad(orientations[i])
        sf    = spatial_freqs[i] * stripes_per_unit  # cycles across image
        phi   = 2 * np.pi * phases[i]

        # rotate coordinates
        Xr = X * np.cos(theta) + Y * np.sin(theta)

        # sinusoidal grating
        img = contrast * np.sin(2 * np.pi * sf * Xr + phi)

        # map [-1, 1] -> [0, max_pixel_value]
        img = (0.5 + 0.5 * img) * max_pixel_value

        imgs[i] = img.astype(np.float32)

    return imgs

def get_full_gratings_dataset(num_samples: int = 500, downsample=True, image_size=(918, 1174)):
    """
    Trial-level dataset with full neural response windows for static gratings.

    Output:
        X : (n_trials, H, W)
        y : (n_trials, n_neurons, n_timesteps)
    """
    if downsample:
        output_path = f"./data/static_gratings_dataset_{num_samples}_downsampled.npz"
    else:
        output_path = f"./data/static_gratings_dataset_{num_samples}.npz"

    PRE_STIM  = 0.1
    POST_STIM = 5.0

    # Downsampling hyperparameters
    N_BINS = 20

    H, W   = image_size

    bin_values = np.linspace(0, 255, N_BINS, dtype=np.float32)

    with h5py.File(path_to_nwb, "r") as f:
        # --- Stimulus ---
        grating_data  = f["stimulus/presentation/static_gratings_stimulus/data"][()]       # (6000, 3)
        grating_times = f["stimulus/presentation/static_gratings_stimulus/timestamps"][()]  # (6000,)

        # --- Neural data ---
        neural_data  = f["processing/brain_observatory_pipeline/Fluorescence/imaging_plane_1/data"][()]
        neural_times = f["processing/brain_observatory_pipeline/Fluorescence/imaging_plane_1/timestamps"][()]

    # --- Filter blank presentations (NaN rows) ---
    valid_mask    = ~np.isnan(grating_data).any(axis=1)
    grating_data  = grating_data[valid_mask]
    grating_times = grating_times[valid_mask]

    orientations  = grating_data[:, 0]
    spatial_freqs = grating_data[:, 1]
    phases        = grating_data[:, 2]

    # --- Window params ---
    dt          = neural_times[1] - neural_times[0]
    n_pre       = int(PRE_STIM / dt)
    n_post      = int(POST_STIM / dt)
    n_timesteps = n_pre + n_post
    n_neurons   = neural_data.shape[0]

    print(f"Sampling interval: {dt:.4f}s (~{1/dt:.1f} Hz)")
    print(f"Window: -{PRE_STIM}s to +{POST_STIM}s -> {n_timesteps} timepoints")
    print(f"Valid grating presentations: {len(grating_times)}")

    # --- Pass 1: find valid trials (no boundary issues) ---
    keep_trial_ids, starts, ends = [], [], []

    for i, t in enumerate(grating_times):
        neural_idx = np.searchsorted(neural_times, t)
        start      = neural_idx - n_pre
        end        = neural_idx + n_post

        if start >= 0 and end <= neural_data.shape[1]:
            keep_trial_ids.append(i)
            starts.append(start)
            ends.append(end)

    keep_trial_ids = np.array(keep_trial_ids, dtype=np.int32)
    starts         = np.array(starts, dtype=np.int32)
    ends           = np.array(ends, dtype=np.int32)

    # --- Subsample ---
    idx            = np.random.choice(len(keep_trial_ids), size=num_samples, replace=False)
    keep_trial_ids = keep_trial_ids[idx]
    starts         = starts[idx]
    ends           = ends[idx]
    n_trials       = len(keep_trial_ids)

    print(f"Trials kept: {n_trials}")

    # --- Generate all grating images at once (no chunking needed) ---
    print("Generating grating images...")

    spatial_freqs_kept = spatial_freqs[keep_trial_ids]
    print(f"3 first samples of kept spatial frequencies:{spatial_freqs_kept[:3]}")

    imgs = generate_gratings(
        orientations  = orientations[keep_trial_ids],
        spatial_freqs = spatial_freqs[keep_trial_ids],
        phases        = phases[keep_trial_ids],
        size          = image_size,   # matches natural scenes size
    )  # (n_trials, H, W)

    # --- quantize X ---
    if downsample:
        print("Quantizing images...")
        bin_idx = np.round(imgs / 255 * (N_BINS - 1)).astype(np.int32)
        X = bin_values[bin_idx]  # (n_trials, H, W)
    else:
        X = imgs

    # --- Build y: neural response windows ---
    print("Building y (neural response windows)...")
    y = np.empty((n_trials, n_neurons, n_timesteps), dtype=np.float32)

    for i in range(n_trials):
        y[i] = neural_data[:, starts[i]:ends[i]]

    if downsample:
        y_bin_size = 50
        y = quantize_to_step(y, y_bin_size)

    # --- Save ---
    kept_times   = grating_times[keep_trial_ids]
    kept_params  = grating_data[keep_trial_ids]   # (n_trials, 3) — orientation, sf, phase

    print("Saving dataset...")
    np.savez_compressed(
        output_path,
        X           = X,
        y           = y,
        timestamps  = kept_times,
        grating_params = kept_params,              # replaces image_indices
        time_axis   = np.linspace(-PRE_STIM, POST_STIM, n_timesteps),
    )

    print(f"\nDataset saved to: {output_path}")
    print(f"Number of samples: {n_trials}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

def main():
    NUM_SAMPLES = 500
    DOWNSAMPLE = True
    IMAGE_SIZE = (229, 293)
    get_full_gratings_dataset(NUM_SAMPLES, DOWNSAMPLE, IMAGE_SIZE)


if __name__ == "__main__":
    main()