"""
Load density object after running density estimation routine
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)

import argparse
from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pickle

from matplotlib import cm


def load_object(name="density.pkl"):
    save_path = os.path.join(FILE_PATH, "out", name)
    try:
        with open(save_path, "rb") as f:
            density = pickle.load(f)
        return density
    except Exception as e:
        raise e


def visualize_density(density, name="density.pkl"):
    density_name = name[:-4]
    assert len(density.shape) == 2
    assert density.shape[0] == density.shape[1]

    num_samples = density.shape[0]

    # Create figure and axis
    fig, ax = plt.subplots()

    # Add mask to highlight values of 0
    masked = np.ma.masked_where(density == 0, density)

    if np.any(density == 0):
        # At least one probability is exactly zero
        cmap = cm.gray_r.copy()
        cmap.set_bad(color="red")
    else:
        # Default when all probabilities are superior to zero
        cmap = cm.viridis.copy()

    # LogNorm mapping from values to colors because all values are really close to zero
    # This is simply to have a better visualization of density estimation
    nonzero = density[density > 0]

    vmin = np.percentile(nonzero, 15)  # clip v_min as the 15 percentile of lower values because true v_min is too small
    vmax = nonzero.max()

    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    im = ax.imshow(masked, cmap=cmap, norm=norm)
    plt.colorbar(im, ax=ax)

    # Optional labels/title
    ax.set_title("Conditional density estimation (P(J=j | man(I=i)))")
    ax.set_xlabel("j")
    ax.set_ylabel("i")

    if np.any(density == 0):
        # Add legend if at least one probability is exactly zero
        zero_patch = mpatches.Patch(color="red", label="Exact 0 values")
        ax.legend(handles=[zero_patch], loc="upper right")

    figure_name = f"{density_name}_visu"
    save_path = os.path.join(FILE_PATH, "out", figure_name)
    plt.savefig(save_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_name", type=str, required=False, default="density.pkl")
    args = parser.parse_args()

    name = args.file_name
    density = load_object(name)
    visualize_density(density, name)
    print(f"density object loaded. Type: {type(density)}")
    print(f"density of shape: {density.shape}")
    sum_axis_0 = np.sum(density, axis=0)
    sum_axis_1 = np.sum(density, axis=1)
    print(f"sum across axis 0 = {sum_axis_0}")
    print(f"sum across axis 1 = {sum_axis_1}")
    print(f"The axis for which the sum across returns only values of 1 is the axis for which j changes for a constant i"
          f" given the conditional probabilities: P(j | man(i)). The other axis is the axis for which j is constant and"
          f" i changes.")
    axis_0_changes_j = False
    axis_1_changes_j = False
    if np.allclose(sum_axis_0, np.ones_like(sum_axis_0), atol=1e-3):
        axis_0_changes_j = True
    if np.allclose(sum_axis_1, np.ones_like(sum_axis_1), atol=1e-3):
        axis_1_changes_j = True

    if not axis_0_changes_j and not axis_1_changes_j:
        print(f"No sum across axis returns only values of 1: density does not represent target conditional "
              f"probabilities.")
    elif axis_0_changes_j and axis_1_changes_j:
        print(f"Both sums across axis return only values of 1: density might represent target conditional probabilities"
              f" but sums across both axis should not all equal to 1.")
    else:
        if axis_0_changes_j:
            print(f"Axis 0 changes j values and density is the target conditional probability distribution.")
        else:
            print(f"Axis 1 changes j values and density is the target conditional probability distribution.")


if __name__ == "__main__":
    main()