"""
Test density estimation routines or functions
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)

from data import StaticGratingsDataset
from density_learning import (
    get_nadaraya_watson_density,
    j_dimensionality_reductions,
    standardize_data
)

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def get_kernel_vector(elem, all_elems, kernel_type="gaussian", **kwargs):
    """
    Helper function for `get_n_w_density_by_hand`.
    """
    if kernel_type == "gaussian":
        assert all_elems.ndim == 2
        if len(elem.shape) == 1:
            elem = elem.reshape(1, -1)
        else:
            assert elem.shape[0] == 1, "Only accept single length elem in this function."
        assert elem.ndim == 2
        bandwidth = kwargs.get("bandwidth", 1.0)
        gamma = 1. / bandwidth**2
        kernel_vector = rbf_kernel(all_elems, elem, gamma=gamma)
        kernel_vector = kernel_vector.reshape(-1)

    else:
        raise ValueError(f"Unsupported kernel_type: {kernel_type}.")

    assert kernel_vector.ndim == 1
    return kernel_vector


def get_n_w_density_by_hand(i, j, bandwidth_i=1.0, bandwidth_j=1.0):
    """
    According to equation (1) in https://arxiv.org/pdf/1206.5278

    Helper function (non-efficient to verify the `get_nadaraya_watson_sklearn_pairwise_density` function).
    """
    density_matrix = []  # P(J=j | man(I=i)))
    # first dim: change j
    # second dim: change i

    num_samples = i.shape[0]
    assert num_samples == j.shape[0]

    for index_j in range(num_samples):
        j_elem = j[index_j]
        j_densities = []
        for index_i in range(num_samples):
            i_elem = i[index_i]
            kernel_vector_j = get_kernel_vector(j_elem, j, "gaussian", bandwidth=bandwidth_j)
            kernel_vector_i = get_kernel_vector(i_elem, i, "gaussian", bandwidth=bandwidth_i)
            assert kernel_vector_j.shape == (num_samples,)
            assert kernel_vector_i.shape == (num_samples,)
            mult_kernel = kernel_vector_j * kernel_vector_i
            sum_kernel_i = np.sum(kernel_vector_i)
            assert mult_kernel.shape == (num_samples,)
            density = np.sum(mult_kernel) / sum_kernel_i
            j_densities.append(density)

        density_matrix.append(j_densities)

    return density_matrix


def main():
    print("Getting dataset...")
    sg_dataset = StaticGratingsDataset(750332458)
    h_v_bars = sg_dataset.get_presentation_ids(orientation=[0, 90])
    visp_units = sg_dataset.get_unit_ids("VISp")
    X_sg, y_sg = sg_dataset.get_data(presentation_ids=h_v_bars, unit_ids=visp_units, stimulus_type="params")
    i = X_sg
    j = y_sg
    print(f"i shape: {i.shape}")
    print(f"j shape: {j.shape}")

    print(f"Only keep 50 samples")
    i = i[:50]
    j = j[:50]
    print(f"i shape: {i.shape}")
    print(f"j shape: {j.shape}")

    print("Applying dimensionality reductions to J...")
    j_reduced_svd, j_reduced_nmf, j_reduced_pca = j_dimensionality_reductions(j, 20, True)
    print(f"j_reduced_pca shape: {j_reduced_pca.shape}")

    print(f"Standardizing data...")
    i = standardize_data(i)
    j_reduced_pca = standardize_data(j_reduced_pca)
    print(f"i means: {np.mean(i, axis=0)}")
    print(f"i stds: {np.std(i, axis=0)}")
    print(f"j means: {np.mean(j_reduced_pca, axis=0)}")
    print(f"j stds: {np.std(j_reduced_pca, axis=0)}")

    print("Calculating N-W conditional density...")
    num_features_i = i.shape[1]
    num_features_j = j_reduced_pca.shape[1]
    bandwidth_i = 1.0 / float(num_features_i)
    bandwidth_j = 1.0 / float(num_features_j)

    first_bandwidth = bandwidth_i
    second_bandwidth = bandwidth_j
    density = get_nadaraya_watson_density(i, j_reduced_pca, first_bandwidth, second_bandwidth)

    print("Calculating N-W conditional density by hand...")
    first_bandwidth = bandwidth_i
    second_bandwidth = bandwidth_j
    density_by_hand = get_n_w_density_by_hand(i, j_reduced_pca, first_bandwidth, second_bandwidth)

    print(f"Density comparison...")
    if np.allclose(density, density_by_hand, rtol=1e-3):
        print(f"Density == Density_by_hand")

    elif np.allclose(np.transpose(density), density_by_hand, rtol=1e-3):
        print(f"Density.T == Density_by_hand")

    else:
        print(f" Density and Density.T != Density_by_hand")


if __name__ == "__main__":
    main()