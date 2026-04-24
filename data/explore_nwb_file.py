"""
Exploration of the structure of the .nwb file with h5py reader.
"""
import h5py
import os

PATH_TO_NWB = "brain_observatory/ophys_experiment_data/501794235.nwb"

current_file_path = os.path.dirname(os.path.abspath(__file__))
path_to_nwb = os.path.join(current_file_path, PATH_TO_NWB)

print(f"\nComplete list of sections in the nwb file downloaded")
with h5py.File(path_to_nwb, "r") as f:
    # See top-level keys
    print(list(f.keys()))

    # Recursively print the full structure
    f.visititems(lambda name, obj: print(name))

# Explore with ROI 0000
roi_path = "processing/brain_observatory_pipeline/ImageSegmentation/imaging_plane_1/roi_0000"

print(f"\nExplore the data in roi_0000S")
with h5py.File(path_to_nwb, "r") as f:
    roi = f[roi_path]

    img_mask        = roi["img_mask"][()]           # 2D array
    pix_mask        = roi["pix_mask"][()]           # (N, 2) array
    pix_mask_weight = roi["pix_mask_weight"][()]    # (N,) array
    description     = roi["roi_description"][()].decode()

    print("img_mask shape:", img_mask.shape)
    print("pix_mask shape:", pix_mask.shape)
    print("weights shape: ", pix_mask_weight.shape)
    print("description:   ", description)

# Look at neuronal activity
print(f"\nLooking at DfOverF/imaging_plane_1 (neuronal activity)")
with h5py.File(path_to_nwb, "r") as f:
    base = "processing/brain_observatory_pipeline/DfOverF/imaging_plane_1"

    data = f[base + "/data"][()]  # (n_timepoints, n_rois)
    timestamps = f[base + "/timestamps"][()]  # (n_timepoints,)
    roi_names = f[base + "/roi_names"][()]  # (n_rois,)

# Activity for a single neuron (e.g. roi_0000, which is index 0)
print(f"\nActivity for neuron at index 0:")
trace_roi0 = data[0, :]
print(trace_roi0.shape)
print(type(trace_roi0))
print(timestamps[:5])   # first few timestamps
print(timestamps[-1])   # total duration in seconds
print(len(timestamps))  # should confirm 152

# Check fluorescence
print(f"\nCheck fluorescence/imaging_plane_1 data")
with h5py.File(path_to_nwb, "r") as f:
    base_f = "processing/brain_observatory_pipeline/Fluorescence/imaging_plane_1"

    data_f = f[base_f + "/data"][()]
    timestamps_f = f[base_f + "/timestamps"][()]

    print("Fluorescence data shape:", data_f.shape)
    print("Fluorescence timestamps shape:", timestamps_f.shape)

# See images (stimuli)
print(f"\nSee stimuli: natural scenes")
with h5py.File(path_to_nwb, "r") as f:
    # Stimulus presentation
    scene_indices = f["stimulus/presentation/natural_scenes_stimulus/data"][()]
    scene_times = f["stimulus/presentation/natural_scenes_stimulus/timestamps"][()]

    # Image templates
    images = f["stimulus/templates/natural_scenes_image_stack/data"][()]

    print("scene_indices shape:", scene_indices.shape)  # (n_presentations,)
    print("scene_times shape:  ", scene_times.shape)  # (n_presentations,)
    print("images shape:       ", images.shape)  # (n_images, H, W)

    print("index range:", scene_indices.min(), scene_indices.max())
    print("time range: ", scene_times[0], scene_times[-1])

# See static gratings (different format, we do not have access to images but rather parameters to generate the static gratings)
print(f"\nSee stimuli: static gratings")
with h5py.File(path_to_nwb, "r") as f:
    grating_data = f["stimulus/presentation/static_gratings_stimulus/data"][()]
    grating_times = f["stimulus/presentation/static_gratings_stimulus/timestamps"][()]
    grating_features = f["stimulus/presentation/static_gratings_stimulus/features"][()]

    print("grating_data shape:    ", grating_data.shape)
    print("grating_times shape:   ", grating_times.shape)
    print("grating_features:      ", grating_features)
    print("grating_data sample:\n ", grating_data[:3])

# See stimuli presentation (static gratings more precisely)
print("\nStimuli presentation:")
with h5py.File(path_to_nwb, "r") as f:
    print(list(f["stimulus/presentation"].keys()))

    print(f["stimulus/presentation/static_gratings_stimulus"].keys())


# See natural movies
print(f"\nSee stimuli: natural movies")
with h5py.File(path_to_nwb, "r") as f:
    movie_indices = f["stimulus/presentation/natural_movie_one_stimulus/data"][()]
    movie_times = f["stimulus/presentation/natural_movie_one_stimulus/timestamps"][()]
    movie_images = f["stimulus/templates/natural_movie_one_image_stack/data"][()]

    print("movie_indices shape:", movie_indices.shape)
    print("movie_times shape:  ", movie_times.shape)
    print("movie_images shape: ", movie_images.shape)
    print("index range:        ", movie_indices.min(), movie_indices.max())
    print("time range:         ", movie_times[0], movie_times[-1])

# Explore spontaneous stimuli
print(f"\nSee stimuli: spontaneous")
with h5py.File(path_to_nwb, "r") as f:
    spont = f["stimulus/presentation/spontaneous_stimulus"]

    # Check all fields
    print("Keys:", list(spont.keys()))

    for key in spont.keys():
        val = spont[key][()]
        print(f"\n{key}:")
        print("  shape:", val.shape if hasattr(val, 'shape') else "scalar")
        print("  sample:", val[:5] if hasattr(val, '__len__') else val)