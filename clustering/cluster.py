"""
Clustering routine used on a group of vectors to obtain clusters
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)
DENSITY_PATH = os.path.join(FILE_PATH, "../density_learning/")

from clustering import get_micro_causes_effects, load_object

import argparse
import numpy as np
from sklearn.mixture import BayesianGaussianMixture


def cluster_dirichlet_process_gaussian_mixture(eft_mic, cs_mic, **kwargs):
    d_p_g_m_eft = BayesianGaussianMixture(**kwargs)
    d_p_g_m_cs = BayesianGaussianMixture(**kwargs)

    # Cluster micro effects
    d_p_g_m_eft.fit(eft_mic)
    eft_clusters = d_p_g_m_eft.predict(eft_mic)
    means_eft = d_p_g_m_eft.means_

    # Cluster micro causes
    d_p_g_m_cs.fit(cs_mic)
    cs_clusters = d_p_g_m_cs.predict(cs_mic)
    means_cs = d_p_g_m_cs.means_

    return eft_clusters, cs_clusters, means_eft, means_cs



def cluster(eft_mic, cs_mic, cluster_method="dirichlet_process_gaussian_mixture", **kwargs):
    if cluster_method == "dirichlet_process_gaussian_mixture":
        eft_clusters, cs_clusters, means_eft, means_cs = cluster_dirichlet_process_gaussian_mixture(eft_mic, cs_mic, **kwargs)
        return eft_clusters, cs_clusters, means_eft, means_cs
    else:
        raise ValueError(f"Unsupported cluster method: {cluster_method}")


def convert_probs_to_logprobs(density):
    return np.log(density)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_name", type=str, required=False, default="density.pkl")
    args = parser.parse_args()

    name = args.file_name
    print(f"Loading density...")
    density = load_object(name)
    num_samples = density.shape[0]
    eft_mic, cs_mic = get_micro_causes_effects(density)
    #eft_mic = convert_probs_to_logprobs(eft_mic)
    #cs_mic = convert_probs_to_logprobs(cs_mic)

    cluster_method = "dirichlet_process_gaussian_mixture"
    print(f"Clustering...")
    eft_clusters, cs_clusters, means_eft, means_cs = cluster(
        eft_mic,
        cs_mic,
        cluster_method=cluster_method,
        n_components=20
    )
    assert eft_clusters.ndim == 1 and cs_clusters.ndim == 1
    assert means_eft.ndim == 2 and means_cs.ndim == 2
    num_eft_clusters = len(np.unique(eft_clusters))
    num_cs_clusters = len(np.unique(cs_clusters))

    print(f"Clustering done;\nnumber of eft clusters: {num_eft_clusters}\nnumber of cs clusters: {num_cs_clusters}")

    file_path_eft_clusters = os.path.join(FILE_PATH, f"out/eft_clusters_{cluster_method}")
    np.save(file_path_eft_clusters, eft_clusters)
    file_path_cs_clusters = os.path.join(FILE_PATH, f"out/cs_clusters_{cluster_method}")
    np.save(file_path_cs_clusters, cs_clusters)
    file_path_eft_means = os.path.join(FILE_PATH, f"out/eft_means_{cluster_method}")
    np.save(file_path_eft_means, means_eft)
    file_path_cs_means = os.path.join(FILE_PATH, f"out/cs_means_{cluster_method}")
    np.save(file_path_cs_means, means_cs)


if __name__ == "__main__":
    main()