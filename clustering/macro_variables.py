"""
From the cluster assignment, induce the macro effects and causes into probability vectors
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)
DENSITY_PATH = os.path.join(FILE_PATH, "../density_learning/")

from clustering import load_object, get_micro_causes_effects, visualize_samples_from_clusters
from data import StaticGratingsDataset

import argparse
import numpy as np


def load_cluster_info(num_clusters, clustering_method):
    eft_clusters = np.load(os.path.join(FILE_PATH, f"out/eft_clusters_{num_clusters}_{clustering_method}.npy"))
    cs_clusters = np.load(os.path.join(FILE_PATH, f"out/cs_clusters_{num_clusters}_{clustering_method}.npy"))
    eft_means = np.load(os.path.join(FILE_PATH, f"out/eft_means_{num_clusters}_{clustering_method}.npy"))
    cs_means = np.load(os.path.join(FILE_PATH, f"out/cs_means_{num_clusters}_{clustering_method}.npy"))
    return eft_clusters, cs_clusters, eft_means, cs_means


def renormalize_vector(vector):
    assert vector.ndim == 1
    return vector / np.sum(vector)


def get_macro_variables(eft_mic, cs_mic, eft_clusters, cs_clusters):
    """
    The method used to retrieve macro variable is simply to approximate the probability P(E=e | man(C=c))
    by grouping rows of eft_mic by cluster assignment, looking at every single vector within each cs_cluster,
    looking at every single value in the vector, adding that proba of that value at cs_mac[eft_cluster, cs_cluster]
    where eft_cluster is the corresponding cluster for the J associated to the value.

    Finally, renormalization is conducted on the columns of cs_mac afterward since we just kept adding probabilities
    without minding normalization before.

    eft_mac = np.transpose(cs_mac) by definition.
    """
    # Get cs_mac
    num_eft_clusters = len(np.unique(eft_clusters))
    num_cs_clusters = len(np.unique(cs_clusters))
    cs_clustering = {
        cluster: eft_mic[cs_clusters == cluster]
        for cluster in np.unique(cs_clusters)
    }
    # cs_mac initialization
    cs_mac = np.zeros((num_eft_clusters, num_cs_clusters))

    # For every cs_cluster
    for cs_cluster in range(num_cs_clusters):
        # Retrieve vectors assigned to this cs_cluster
        cluster_vectors = cs_clustering[cs_cluster]
        # For every single vector in this cs_cluster
        for cluster_vector in cluster_vectors:
            # This vector should already be normalized
            sum_vector = np.sum(cluster_vector)
            assert np.allclose(sum_vector, np.ones_like(sum_vector))
            for elem_index in range(len(cluster_vector)):
                elem = cluster_vector[elem_index]
                eft_cluster = eft_clusters[elem_index]
                cs_mac[eft_cluster, cs_cluster] += elem

        # Done with this eft_cluster

    # Normalize all columns in cs_mac which is now complete
    for cs_cluster in range(num_cs_clusters):
        cs_mac[:, cs_cluster] = renormalize_vector(cs_mac[:, cs_cluster])

    # Complete cs_mac
    sum_columns_cs_mac = np.sum(cs_mac, axis=0)
    assert np.allclose(sum_columns_cs_mac, np.ones_like(sum_columns_cs_mac), rtol=1e-3)

    # Get eft_mac
    eft_mac = np.copy(np.transpose(cs_mac))
    sum_rows_eft_mac = np.sum(eft_mac, axis=1)
    assert np.allclose(sum_rows_eft_mac, np.ones_like(sum_rows_eft_mac), rtol=1e-3)

    return eft_mac, cs_mac


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_name", type=str, required=False, default="density.pkl")
    args = parser.parse_args()

    name = args.file_name
    print(f"Loading density...")
    density = load_object(name)

    num_clusters = 10
    clustering_method = "dirichlet_process_gaussian_mixture"
    eft_clusters, cs_clusters, eft_means, cs_means = load_cluster_info(num_clusters, clustering_method)
    eft_mic, cs_mic = get_micro_causes_effects(density)

    eft_mac, cs_mac = get_macro_variables(eft_mic, cs_mic, eft_clusters, cs_clusters)

    eft_mac_save_path = os.path.join(FILE_PATH, f"out/eft_mac_{num_clusters}_{clustering_method}")
    cs_mac_save_path = os.path.join(FILE_PATH, f"out/cs_mac_{num_clusters}_{clustering_method}")

    np.save(eft_mac_save_path, eft_mac)
    np.save(cs_mac_save_path, cs_mac)

    # Visualize samples per cluster
    print("Getting dataset...")
    sg_dataset = StaticGratingsDataset(750332458)
    h_v_bars = sg_dataset.get_presentation_ids(orientation=[0, 90])
    visp_units = sg_dataset.get_unit_ids("VISp")
    X_sg, y_sg = sg_dataset.get_data(presentation_ids=h_v_bars, unit_ids=visp_units, stimulus_type="params")
    i = X_sg
    # Transpose to have shape (num_samples, num_neurons, num_bins)
    j = np.transpose(y_sg, (0, 2, 1))

    visualize_samples_from_clusters(i, j, eft_clusters, cs_clusters)


if __name__ == "__main__":
    main()