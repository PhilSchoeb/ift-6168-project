"""
Clustering routine used on a group of vectors to obtain clusters
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)
DENSITY_PATH = os.path.join(FILE_PATH, "../density_learning/")

from clustering import get_micro_causes_effects, load_object
from data import generate_gratings

import argparse
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import BayesianGaussianMixture


def cluster_dirichlet_process_gaussian_mixture(eft_mic, cs_mic, simplify_clustering=False, **kwargs):
    if simplify_clustering:
        max_fit_samples = 3000
        covariance_type = "diag"
        max_iter = 50
    else:
        max_fit_samples = None
        covariance_type = "full"
        max_iter = 100

    def _subsample(data: np.ndarray) -> np.ndarray:
        if max_fit_samples is None or len(data) <= max_fit_samples:
            return data
        idx = np.random.choice(len(data), size=max_fit_samples, replace=False)
        return data[idx]

    d_p_g_m_eft = BayesianGaussianMixture(covariance_type=covariance_type, max_iter=max_iter, **kwargs)
    d_p_g_m_cs = BayesianGaussianMixture(covariance_type=covariance_type, max_iter=max_iter, **kwargs)

    # Cluster micro effects
    eft_mic_train = _subsample(eft_mic)
    d_p_g_m_cs.fit(eft_mic_train)
    cs_clusters = d_p_g_m_cs.predict(eft_mic)
    means_cs = d_p_g_m_cs.means_

    # Cluster micro causes
    cs_mic_train = _subsample(cs_mic)
    d_p_g_m_eft.fit(cs_mic_train)
    eft_clusters = d_p_g_m_eft.predict(cs_mic)
    means_eft = d_p_g_m_eft.means_

    return eft_clusters, cs_clusters, means_eft, means_cs


'''def cluster_dirichlet_process_gaussian_mixture(
    eft_mic,
    cs_mic,
    max_fit_samples: int = 2000,
    **kwargs
):
    """
    Optimized DPGMM clustering with:
    - Subsampled fitting (fits on a representative sample, predicts on all points)
    - Parallel fitting of both models using threads
    - Tighter defaults for large datasets

    Args:
        eft_mic:          Input array for the 'eft' model.
        cs_mic:           Input array for the 'cs' model.
        max_fit_samples:  Max points used for .fit(); set None to disable subsampling.
    """

    def _subsample(data: np.ndarray) -> np.ndarray:
        if max_fit_samples is None or len(data) <= max_fit_samples:
            return data
        idx = np.random.choice(len(data), size=max_fit_samples, replace=False)
        return data[idx]

    def _fit_and_predict(data: np.ndarray):
        # BGM is not thread-safe during fit, so each call gets its own instance
        model = BayesianGaussianMixture(**kwargs)
        sample = _subsample(data)
        model.fit(sample)
        clusters = model.predict(data)  # full data
        return clusters, model.means_

    # Fit both models in parallel (I/O-friendly with threading since sklearn
    # releases the GIL during the core BLAS/LAPACK calls in .fit())
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_cs  = executor.submit(_fit_and_predict, eft_mic)
        future_eft = executor.submit(_fit_and_predict, cs_mic)

        cs_clusters,  means_cs  = future_cs.result()
        eft_clusters, means_eft = future_eft.result()

    return eft_clusters, cs_clusters, means_eft, means_cs'''


def cluster(eft_mic, cs_mic, cluster_method="dirichlet_process_gaussian_mixture", simplify_clustering=False, **kwargs):
    if cluster_method == "dirichlet_process_gaussian_mixture":
        eft_clusters, cs_clusters, means_eft, means_cs = cluster_dirichlet_process_gaussian_mixture(eft_mic, cs_mic, simplify_clustering, **kwargs)
        return eft_clusters, cs_clusters, means_eft, means_cs
    else:
        raise ValueError(f"Unsupported cluster method: {cluster_method}")


def convert_probs_to_logprobs(density):
    return np.log(density)


def renumerate(clusters):
    """
    Helper for merging clusters
    """
    unique_cluster_values = np.unique(clusters)
    # Create mapping
    mapping = {}
    for index in range(len(unique_cluster_values)):
        mapping[unique_cluster_values[index]] = index

    # Renumerate
    for cluster_index in range(len(clusters)):
        clusters[cluster_index] = mapping[clusters[cluster_index]]
    return clusters


def merge_clusters(clusters, merges: list[list[int]]):
    """
    After seeing the macro-variables, a user might want to merge specific clusters together

    clusters: specifically eft_clusters or cs_clusters, for assigning samples to their respective cluster
    (numpy.ndarray of shape [num_samples])

    merges: list of cluster indices to merge together. For example [[0, 2], [1, 5]] means to merge cluster 0 and 2
    together as well as clusters 1 and 5.

    After merge, cluster numbers are renumbered to make sure their value goes from 0 to num_clusters - 1.
    """
    for i in range(len(clusters)):
        for change in merges:
            for value in change[1:]:

                if clusters[i] == value:
                    clusters[i] = change[0]

    clusters = renumerate(clusters)

    return clusters


def visualize_samples_from_clusters(original_i, original_j, eft_clusters, cs_clusters, path_i=None, path_j=None):
    # Cluster i
    unique_cs_clusters = np.unique(cs_clusters)
    clustered_i = [
        original_i[cs_clusters == cluster]
        for cluster in unique_cs_clusters
    ]
    # Cluster j
    unique_eft_clusters = np.unique(eft_clusters)
    clustered_j = [
        original_j[eft_clusters == cluster]
        for cluster in unique_eft_clusters
    ]

    n_samples_to_show = 5

    # --- Visualization for i (cs_clusters) ---
    n_clusters_i = len(unique_cs_clusters)
    fig_i, axes_i = plt.subplots(n_clusters_i, n_samples_to_show,
                                  figsize=(n_samples_to_show * 3, n_clusters_i * 3))
    axes_i = np.atleast_2d(axes_i)

    for cluster_idx, cluster_data in enumerate(clustered_i):
        sample_indices = np.random.choice(len(cluster_data),
                                           size=min(n_samples_to_show, len(cluster_data)),
                                           replace=False)
        for sample_idx, ax in enumerate(axes_i[cluster_idx]):
            if sample_idx < len(sample_indices):
                params = cluster_data[sample_indices[sample_idx]]  # shape [3]
                # Reshape parameters since `generate_gratings` expects batched inputs
                param_0 = np.array([params[0]]).reshape(1, -1)
                param_1 = np.array([params[1]]).reshape(1, -1)
                param_2 = np.array([params[2]]).reshape(1, -1)
                img = generate_gratings(param_0, param_1, param_2)
                img = img.reshape(img.shape[1], img.shape[2])
                ax.imshow(img)
                ax.axis("off")
                if sample_idx == 0:
                    ax.set_title(f"Cluster {unique_cs_clusters[cluster_idx]}", fontsize=9)
            else:
                ax.axis("off")

    fig_i.suptitle("CS Cluster Samples (i)", fontsize=12)
    fig_i.tight_layout(rect=[0, 0, 1, 0.96])

    if path_i is None:
        save_path_i = os.path.join(FILE_PATH, "out/i_cluster_samples.png")
    else:
        save_path_i = path_i

    os.makedirs(os.path.dirname(save_path_i), exist_ok=True)
    fig_i.savefig(save_path_i)
    plt.close(fig_i)

    # --- Visualization for j (eft_clusters) ---
    n_clusters_j = len(unique_eft_clusters)
    fig_j, axes_j = plt.subplots(n_clusters_j, n_samples_to_show,
                                  figsize=(n_samples_to_show * 3, n_clusters_j * 3))
    axes_j = np.atleast_2d(axes_j)

    for cluster_idx, cluster_data in enumerate(clustered_j):
        sample_indices = np.random.choice(len(cluster_data),
                                           size=min(n_samples_to_show, len(cluster_data)),
                                           replace=False)
        for sample_idx, ax in enumerate(axes_j[cluster_idx]):
            if sample_idx < len(sample_indices):
                img = cluster_data[sample_indices[sample_idx]]  # shape [n_neurons, n_bins]
                ax.imshow(img, aspect="auto")
                ax.axis("off")
                if sample_idx == 0:
                    ax.set_title(f"Cluster {unique_eft_clusters[cluster_idx]}", fontsize=9)
            else:
                ax.axis("off")

    fig_j.suptitle("EFT Cluster Samples (j)", fontsize=12)
    fig_j.tight_layout(rect=[0, 0, 1, 0.96])

    if path_j is None:
        save_path_j = os.path.join(FILE_PATH, "out/j_cluster_samples.png")
    else:
        save_path_j = path_j

    fig_j.savefig(save_path_j)
    plt.close(fig_j)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_name", type=str, required=False, default="density.pkl")
    args = parser.parse_args()

    name = args.file_name
    print(f"Loading density...")
    density = load_object(name)
    num_samples = density.shape[0]
    eft_mic, cs_mic = get_micro_causes_effects(density)
    eft_mic = convert_probs_to_logprobs(eft_mic)
    cs_mic = convert_probs_to_logprobs(cs_mic)

    cluster_method = "dirichlet_process_gaussian_mixture"
    max_num_cluster = 10
    print(f"Clustering...")
    eft_clusters, cs_clusters, means_eft, means_cs = cluster(
        eft_mic,
        cs_mic,
        cluster_method=cluster_method,
        n_components=max_num_cluster
    )
    assert eft_clusters.ndim == 1 and cs_clusters.ndim == 1
    assert means_eft.ndim == 2 and means_cs.ndim == 2
    num_eft_clusters = len(np.unique(eft_clusters))
    num_cs_clusters = len(np.unique(cs_clusters))

    print(f"Clustering done;\nnumber of eft clusters: {num_eft_clusters}\nnumber of cs clusters: {num_cs_clusters}")

    file_path_eft_clusters = os.path.join(FILE_PATH, f"out/eft_clusters_{num_eft_clusters}_{cluster_method}")
    np.save(file_path_eft_clusters, eft_clusters)
    file_path_cs_clusters = os.path.join(FILE_PATH, f"out/cs_clusters_{num_cs_clusters}_{cluster_method}")
    np.save(file_path_cs_clusters, cs_clusters)
    file_path_eft_means = os.path.join(FILE_PATH, f"out/eft_means_{num_eft_clusters}_{cluster_method}")
    np.save(file_path_eft_means, means_eft)
    file_path_cs_means = os.path.join(FILE_PATH, f"out/cs_means_{num_cs_clusters}_{cluster_method}")
    np.save(file_path_cs_means, means_cs)


if __name__ == "__main__":
    main()