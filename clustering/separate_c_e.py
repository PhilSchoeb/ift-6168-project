"""
Retrieve vectors of micro-effects and micro-causes from the conditional density matrix
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)
DENSITY_PATH = os.path.join(FILE_PATH, "../density_learning/")

import argparse
import numpy as np
import pickle

def get_micro_causes_effects(density):
    # Verify that dim 0 is for j and dim 1 is for i
    verify_j_dim(density)

    eft_mic = np.transpose(density)
    cs_mic = density

    return eft_mic, cs_mic


def verify_j_dim(density, j_dim=0):
    sum_axis_0 = np.sum(density, axis=0)
    sum_axis_1 = np.sum(density, axis=1)
    if j_dim == 0:
        assert np.allclose(sum_axis_0, np.ones_like(sum_axis_0), rtol=1e-3)
        assert not np.allclose(sum_axis_1, np.ones_like(sum_axis_1), rtol=1e-3)
    else:
        assert not np.allclose(sum_axis_0, np.ones_like(sum_axis_0), rtol=1e-3)
        assert np.allclose(sum_axis_1, np.ones_like(sum_axis_1), rtol=1e-3)
    return density


def load_object(name="density.pkl"):
    save_path = os.path.join(DENSITY_PATH, "out", name)
    try:
        with open(save_path, "rb") as f:
            density = pickle.load(f)
        return density
    except Exception as e:
        raise e


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_name", type=str, required=False, default="density.pkl")
    args = parser.parse_args()

    name = args.file_name
    density = load_object(name)
    eft_mic, cs_mic = get_micro_causes_effects(density)


if __name__ == "__main__":
    main()