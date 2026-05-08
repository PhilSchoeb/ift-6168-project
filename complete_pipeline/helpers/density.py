"""
Getting the density estimation
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)

from density_learning import get_nadaraya_watson_density

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
