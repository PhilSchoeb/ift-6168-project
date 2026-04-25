"""
Download locally, from a .nwb file that was previously downloaded, a dataset (X; y) of the format:
- X represents images (stimuli) presented to mice so X has shape (num_images, image_height, image_width)
- y represents neuronal activity of mice so y has shape (num_images, num_neurons, num_timestamps)

X, y is built so that the neuronal activity matches the timestamps of the image shown to the mice. This way, we can
analyze neuronal activity based on the image shown to a mouse.

"""

import cv2
import h5py
import numpy as np
import os

# To change if you run this file by itself so it matches the .nwb you aim to convert to a dataset
PATH_TO_NWB = "brain_observatory/ophys_experiment_data/501794235.nwb"

current_file_path = os.path.dirname(os.path.abspath(__file__))
path_to_nwb = os.path.join(current_file_path, PATH_TO_NWB)
CHUNK_SIZE   = 100  # process 100 presentations at a time

def quantize_to_step(y_values, step=50):
    return np.round(y_values / step) * step

def get_unique_natural_scenes_analysis():
    with h5py.File(path_to_nwb, "r") as f:
        # --- Stimulus ---
        scene_indices = f["stimulus/presentation/natural_scenes_stimulus/data"][()]
        scene_times = f["stimulus/presentation/natural_scenes_stimulus/timestamps"][()]
        images = f["stimulus/templates/natural_scenes_image_stack/data"]  # keep as h5py dataset, don't load yet

        # --- Neural data ---
        neural_data = f["processing/brain_observatory_pipeline/Fluorescence/imaging_plane_1/data"][
            ()]  # (152, 113850) — small enough
        neural_times = f["processing/brain_observatory_pipeline/Fluorescence/imaging_plane_1/timestamps"][()]

        # --- Filter blanks ---
        valid_mask = scene_indices != -1
        scene_indices = scene_indices[valid_mask]
        scene_times = scene_times[valid_mask]
        n_valid = len(scene_indices)
        print(f"Valid presentations: {n_valid}")

        unique, counts = np.unique(scene_indices, return_counts=True)
        print("Unique images shown:", len(unique))
        print("Total presentations:", len(scene_indices))
        print(f"Presentations per image — min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.1f}")

def get_full_images_dataset(num_samples: int = 500, downsample=True):
    """
    Trial-level dataset with full neural response windows.

    Keeps chunked processing to avoid memory allocation errors.

    Output:
        X : (n_trials, H, W)
        y : (n_trials, n_neurons, n_timesteps)
    """
    if downsample:
        output_path = f"./data/natural_scenes_dataset_{num_samples}_downsampled.npz"
    else:
        output_path = f"./data/natural_scenes_dataset_{num_samples}.npz"


    PRE_STIM = 0.1
    POST_STIM = 5.0

    # Downsampling hyperparameters for images
    DOWNSAMPLE_FACTOR = 4
    N_BINS = 20

    new_h = 918 // DOWNSAMPLE_FACTOR
    new_w = 1174 // DOWNSAMPLE_FACTOR

    # 20 equidistant values from 0 to 255
    bin_values = np.linspace(0, 255, N_BINS, dtype=np.float32)

    with h5py.File(path_to_nwb, "r") as f:
        # --- Stimulus ---
        scene_indices_raw = f["stimulus/presentation/natural_scenes_stimulus/data"][()]
        scene_times_raw = f["stimulus/presentation/natural_scenes_stimulus/timestamps"][()]
        images = f["stimulus/templates/natural_scenes_image_stack/data"]

        # --- Neural data ---
        neural_data = f[
            "processing/brain_observatory_pipeline/Fluorescence/imaging_plane_1/data"
        ][()]  # (n_neurons, T)

        neural_times = f[
            "processing/brain_observatory_pipeline/Fluorescence/imaging_plane_1/timestamps"
        ][()]

        # --- Remove blank stimuli ---
        valid_mask = scene_indices_raw != -1
        scene_indices = scene_indices_raw[valid_mask]
        scene_times = scene_times_raw[valid_mask]

        # --- Window params ---
        dt = neural_times[1] - neural_times[0]
        n_pre = int(PRE_STIM / dt)
        n_post = int(POST_STIM / dt)
        n_timesteps = n_pre + n_post
        n_neurons = neural_data.shape[0]

        print(f"Sampling interval: {dt:.4f}s (~{1/dt:.1f} Hz)")
        print(f"Window: -{PRE_STIM}s to +{POST_STIM}s -> {n_timesteps} timepoints")

        # ---------------------------------------------------------
        # Pass 1: determine valid trials (skip boundary windows)
        # ---------------------------------------------------------
        keep_trial_ids = []
        starts = []
        ends = []

        for i, t in enumerate(scene_times):
            neural_idx = np.searchsorted(neural_times, t)
            start = neural_idx - n_pre
            end = neural_idx + n_post

            if start >= 0 and end <= neural_data.shape[1]:
                keep_trial_ids.append(i)
                starts.append(start)
                ends.append(end)

        keep_trial_ids = np.array(keep_trial_ids, dtype=np.int32)
        starts = np.array(starts, dtype=np.int32)
        ends = np.array(ends, dtype=np.int32)

        # Subsample only num_samples samples
        idx = np.random.choice(len(keep_trial_ids), size=num_samples, replace=False)
        keep_trial_ids = keep_trial_ids[idx]

        n_trials = len(keep_trial_ids)

        print(f"Trials kept: {n_trials}")

        # --- Allocate y as memmap instead of RAM ---
        y_shape = (n_trials, n_neurons, n_timesteps)
        y_memmap = np.memmap(
            "y_temp.dat",
            dtype="float32",
            mode="w+",
            shape=y_shape
        )

        if downsample:
            # --- Allocate X as memmap ---
            img_shape = (new_h, new_w)
            X_shape = (n_trials,) + img_shape

            X_memmap = np.memmap(
                "X_temp.dat",
                dtype="float32",
                mode="w+",
                shape=X_shape
            )
        else:
            # --- Allocate X as memmap ---
            img_shape = images[0].shape
            X_shape = (n_trials,) + img_shape

            X_memmap = np.memmap(
                "X_temp.dat",
                dtype="float32",
                mode="w+",
                shape=X_shape
            )

        kept_indices = np.empty(n_trials, dtype=np.int32)
        kept_times = np.empty(n_trials, dtype=np.float64)

        # ---------------------------------------------------------
        # Pass 2: build dataset in chunks
        # ---------------------------------------------------------
        print("Building dataset...")
        print("Applying downsampling + quantization...")

        for i in range(0, n_trials, CHUNK_SIZE):
            j = min(i + CHUNK_SIZE, n_trials)

            chunk_trials = keep_trial_ids[i:j]

            # --- images ---
            for local_k, trial_id in enumerate(chunk_trials):
                img_idx = scene_indices[trial_id]
                img = images[img_idx]

                if downsample:
                    # ----------------------------
                    # 1. Downsample
                    # ----------------------------
                    img_small = cv2.resize(
                        img,
                        (new_w, new_h),  # OpenCV uses (width, height)
                        interpolation=cv2.INTER_AREA
                    )
                    # ----------------------------
                    # 2. Quantize to 20 bins
                    # ----------------------------
                    # map [0,255] -> indices [0,19]
                    bin_idx = np.round(img_small / 255 * (N_BINS - 1)).astype(np.int32)

                    # map indices -> actual equidistant values
                    img_quant = bin_values[bin_idx]

                    # store
                    X_memmap[i + local_k] = img_quant.astype(np.float32)

                else:
                    X_memmap[i + local_k] = img

            # --- neural windows ---
            for local_k in range(j - i):
                y_memmap[i + local_k] = neural_data[:, starts[i + local_k]:ends[i + local_k]]

            kept_indices[i:j] = scene_indices[chunk_trials]
            kept_times[i:j] = scene_times[chunk_trials]

            print(f"  Processed {j}/{n_trials}", end="\r")

        if downsample:
            # We will quantize the y values to bins to reduce its cardinality
            y_bin_size = 50
            y_memmap = quantize_to_step(y_memmap, y_bin_size)

        print("\nSaving dataset...")

        np.savez_compressed(
            output_path,
            X=X_memmap,
            y=y_memmap,
            timestamps=kept_times,
            image_indices=kept_indices,
            time_axis=np.linspace(-PRE_STIM, POST_STIM, n_timesteps),
        )

        del X_memmap
        del y_memmap

    import os
    os.remove("X_temp.dat")
    os.remove("y_temp.dat")

    print(f"\nDataset saved to: {output_path}")
    print(f"Number of samples: {n_trials}")
    print(f"X shape: {X_shape}")
    print(f"y shape: {y_shape}")


def get_averaged_images_dataset(downsample=True):
    """
    Similar to the 'get_full_images_dataset' function but only keeps each unique image once and averages over all
    corresponding neuronal responses.
    """
    if downsample:
        output_path = "./data/natural_scenes_dataset_averaged_downsampled.npz"
    else:
        output_path = "./data/natural_scenes_dataset_averaged.npz"


    PRE_STIM = 0.1  # seconds before stimulus onset
    POST_STIM = 5.0  # seconds after stimulus onset

    # Downsampling hyperparameters for images
    DOWNSAMPLE_FACTOR = 4
    N_BINS = 20

    new_h = 918 // DOWNSAMPLE_FACTOR
    new_w = 1174 // DOWNSAMPLE_FACTOR

    # 20 equidistant values from 0 to 255
    bin_values = np.linspace(0, 255, N_BINS, dtype=np.float32)

    with h5py.File(path_to_nwb, "r") as f:
        # --- Stimulus ---
        scene_indices_raw = f["stimulus/presentation/natural_scenes_stimulus/data"][()]
        scene_times_raw = f["stimulus/presentation/natural_scenes_stimulus/timestamps"][()]
        images_h5 = f["stimulus/templates/natural_scenes_image_stack/data"]  # keep as h5py, don't load

        # --- Neural data ---
        neural_data = f["processing/brain_observatory_pipeline/Fluorescence/imaging_plane_1/data"][()]  # (152, 113850)
        neural_times = f["processing/brain_observatory_pipeline/Fluorescence/imaging_plane_1/timestamps"][()]

        # --- Filter blanks ---
        valid_mask = scene_indices_raw != -1
        scene_indices = scene_indices_raw[valid_mask]
        scene_times = scene_times_raw[valid_mask]

        # --- Time window setup ---
        dt = neural_times[1] - neural_times[0]
        n_pre = int(PRE_STIM / dt)
        n_post = int(POST_STIM / dt)
        n_timesteps = n_pre + n_post
        n_neurons = neural_data.shape[0]  # 152
        n_images = 118

        print(f"Sampling interval: {dt:.4f}s (~{1 / dt:.1f} Hz)")
        print(f"Window: -{PRE_STIM}s to +{POST_STIM}s → {n_timesteps} timepoints")
        print(f"Final X shape: ({n_images}, 918, 1174)")
        print(f"Final y shape: ({n_images}, {n_neurons}, {n_timesteps})")

        # --- Build X: 118 unique images ---
        print("\nLoading unique images...")
        X = np.array([images_h5[i] for i in range(n_images)], dtype=np.float32)  # (118, 918, 1174)

        if downsample:
            print("Applying downsampling + quantization...")

            # allocate reduced image tensor
            X_processed = np.empty((n_images, new_h, new_w), dtype=np.float32)

            for img_idx in range(n_images):
                img = X[img_idx]

                # ----------------------------
                # 1. Downsample
                # ----------------------------
                img_small = cv2.resize(
                    img,
                    (new_w, new_h),  # OpenCV uses (width, height)
                    interpolation=cv2.INTER_AREA
                )

                # ----------------------------
                # 2. Quantize to N_BINS bins
                # ----------------------------
                # map [0,255] -> indices [0, N_BINS-1]
                bin_idx = np.round(
                    img_small / 255.0 * (N_BINS - 1)
                ).astype(np.int32)

                # map indices -> equidistant values
                img_quant = bin_values[bin_idx]

                X_processed[img_idx] = img_quant.astype(np.float32)

            X = X_processed

        # --- Build y: average over 50 repeats per image ---
        y = np.zeros((n_images, n_neurons, n_timesteps), dtype=np.float32)
        counts = np.zeros(n_images, dtype=np.int32)

        print("Building y (averaged neural responses)...")
        for i, (img_idx, t) in enumerate(zip(scene_indices, scene_times)):
            neural_idx = np.searchsorted(neural_times, t)  # faster than argmin for sorted arrays
            start = neural_idx - n_pre
            end = neural_idx + n_post

            # Skip edge cases
            if start < 0 or end > neural_data.shape[1]:
                continue

            y[img_idx] += neural_data[:, start:end]  # accumulate
            counts[img_idx] += 1
            print(f"  Processing trial {i + 1}/5900...", end="\r")

        # Divide by counts to get mean
        for img_idx in range(n_images):
            if counts[img_idx] > 0:
                y[img_idx] /= counts[img_idx]

        if downsample:
            # We will quantize the y values to bins to reduce its cardinality
            y_bin_size = 50
            # Apply binning
            y = quantize_to_step(y, y_bin_size)

        print(f"\nRepeats per image — min: {counts.min()}, max: {counts.max()}")

    # --- Save ---
    print("Saving dataset...")
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        time_axis=np.linspace(-PRE_STIM, POST_STIM, n_timesteps)  # useful for plotting later
    )

    print(f"\nDataset saved to: {output_path}")
    print(f"Number of samples: {n_images}")
    print(f"X shape: {X.shape}  (n_images, height, width)")
    print(f"y shape: {y.shape}  (n_images, n_neurons, n_timesteps)")


def main():
    DOWNSAMPLE = True

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="full_images_dataset",
        help="Task to run; can be 'full_images_dataset', 'averaged_images_dataset', 'unique_images_analysis'"
    )

    args = parser.parse_args()

    task = args.task

    if task == "full_images_dataset":
        get_full_images_dataset(DOWNSAMPLE)
    elif task == "averaged_images_dataset":
        get_averaged_images_dataset(DOWNSAMPLE)
    elif task == "unique_images_analysis":
        get_unique_natural_scenes_analysis()
    else:
        raise ValueError(f"Unknown --task value: `{task}")


if __name__ == "__main__":
    main()