"""
Getting the density estimation
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
FILE_PATH = os.path.dirname(__file__)

from density_learning import get_nadaraya_watson_density

import numpy as np


def get_density_estimation(i, j, density_estimator, bandwidth_i, bandwidth_j):
    if density_estimator == "nadaraya_watson":
        if bandwidth_i == "default":
            num_features_i = i.shape[1]
            bandwidth_i = 1.0 / float(num_features_i)
        if bandwidth_j == "default":
            num_features_j = j.shape[1]
            bandwidth_j = 1.0 / float(num_features_j)
        density = get_nadaraya_watson_density(i, j, bandwidth_i, bandwidth_j)
        return density

    else:
        raise NotImplementedError(f"TODO: density_estimator != \"nadaraya_watson\" not implemented.")


def group_density_by_clusters(density, eft_clusters, cs_clusters):
    # Get row order: sort sample indices by their eft cluster label
    row_order = np.argsort(eft_clusters, kind="stable")

    # Get column order: sort sample indices by their cs cluster label
    col_order = np.argsort(cs_clusters, kind="stable")

    # Rearrange: first reorder rows, then reorder columns
    rearranged = density[np.ix_(row_order, col_order)]

    return rearranged