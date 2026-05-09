"""
Access clustering routines
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
FILE_PATH = os.path.dirname(__file__)

from clustering import get_micro_causes_effects, convert_probs_to_logprobs, cluster, get_macro_variables

import numpy as np


def get_clustering_and_macro(density, density_to_log, clustering, num_clusters, simplify_clustering):
    eft_mic, cs_mic = get_micro_causes_effects(density)

    if density_to_log:
        eft_mic = convert_probs_to_logprobs(eft_mic)
        cs_mic = convert_probs_to_logprobs(cs_mic)

    eft_clusters, cs_clusters, means_eft, means_cs = cluster(
        eft_mic,
        cs_mic,
        cluster_method=clustering,
        simplify_clustering=simplify_clustering,
        n_components=num_clusters
    )
    assert eft_clusters.ndim == 1 and cs_clusters.ndim == 1
    assert means_eft.ndim == 2 and means_cs.ndim == 2
    num_eft_clusters = len(np.unique(eft_clusters))
    num_cs_clusters = len(np.unique(cs_clusters))

    eft_mac, cs_mac = get_macro_variables(eft_mic, cs_mic, eft_clusters, cs_clusters)

    return eft_clusters, num_eft_clusters, cs_clusters, num_cs_clusters, means_eft, means_cs, eft_mac, cs_mac


def get_macro_only(density, eft_clusters, cs_clusters):
    eft_mic, cs_mic = get_micro_causes_effects(density)
    eft_mac, cs_mac = get_macro_variables(eft_mic, cs_mic, eft_clusters, cs_clusters)
    return eft_mac, cs_mac
