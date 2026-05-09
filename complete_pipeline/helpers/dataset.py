"""
Fetch dataset.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
FILE_PATH = os.path.dirname(__file__)
SESSION_DATA_PATH = os.path.join(FILE_PATH, "../../session_data/")

from data import StaticGratingsDataset
from density_learning import get_NMF_reduction, get_PCA_reduction, get_TSVD_reduction

import numpy as np


def fetch_dataset(experiment_name, i_dataset, orientations, units, num_bins, num_neurons, neuron_selection):
    if i_dataset == "static_gratings_params":
        if experiment_name == "all":
            raise NotImplementedError("TODO: experiment_name == \"all\" not implemented yet.")
        else:
            if num_bins == 100:
                if num_neurons == "all":
                    experiment_name_int = int(experiment_name)
                    sg_dataset = StaticGratingsDataset(experiment_name_int)
                    selected_orientations = sg_dataset.get_presentation_ids(orientation=orientations)
                    selected_units = sg_dataset.get_unit_ids(units)
                    X_sg, y_sg = sg_dataset.get_data(presentation_ids=selected_orientations, unit_ids=selected_units, stimulus_type="params")
                    i = X_sg
                    # Transpose to have shape (num_samples, num_neurons, num_bins)
                    j = np.transpose(y_sg, (0, 2, 1))
                    return i, j
                else:
                    # Use neuron_selection
                    raise NotImplementedError("TODO: num_neurons != \"all\" not implemented yet.")
            else:
                raise NotImplementedError("TODO: num_bins != 100 not implemented yet.")

    else:
        raise NotImplementedError("TODO: i_dataset != \"static_gratings_params\" not implemented yet.")


def apply_dimensionality_reduction(i, j, experiment_file, i_dataset, dim_reduction, reduced_dimension):
    if i_dataset == "static_gratings_params":
        # Only reduce J
        if dim_reduction == "autoencoder":
            # Fetch data from session_data
            if experiment_file == "all":
                raise NotImplementedError("TODO: experiment_name == \"all\" not implemented yet.")
            else:
                # Only reduced dimension available in our data for now
                if reduced_dimension in [32, 64]:
                    path_to_latent_data = os.path.join(SESSION_DATA_PATH, f"session_{experiment_file}/latent_static_gratings_{reduced_dimension}.npz")
                    data_j = np.load(path_to_latent_data)
                    latent_j = data_j["y"]

                    # Verify X with original data i
                    X = data_j["X"]
                    assert np.allclose(X, i, rtol=1e-3)

                    # Present information that is unused
                    presentation_ids = data_j["presentation_ids"]
                    unit_ids = data_j["unit_ids"]
                    timestamps = data_j["timestamps"]

                    return i, latent_j, None, None
                else:
                    raise NotImplementedError(f"TODO: reduced_dimension == {reduced_dimension} not available in session_data folder.")

        elif dim_reduction == "pca":
            reduced_j, explained_var = get_PCA_reduction(j, reduced_dimension, True)
            return i, reduced_j, None, explained_var

        elif dim_reduction == "truncatedsvd":
            reduced_j, explained_var = get_TSVD_reduction(j ,reduced_dimension, True)
            return i, reduced_j, None, explained_var

        elif dim_reduction == "nmf":
            reduced_j = get_NMF_reduction(j, reduced_dimension)
            return i, reduced_j, None, None

    else:
        # Reduce I and J
        raise NotImplementedError("TODO: i_dataset != static_gratings_params not implemented yet.")
