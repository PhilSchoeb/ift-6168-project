"""
Load density object after running density estimation routine
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)

import matplotlib.pyplot as plt
import numpy as np


def visualize_macros(eft_mac, cs_mac):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    matrices = [
        (eft_mac, "EFT MAC"),
        (cs_mac, "CS MAC")
    ]

    for ax, (mat, title) in zip(axes, matrices):

        im = ax.imshow(mat, cmap="viridis")

        ax.set_title(title)
        if title == "EFT MAC":
            ax.set_xlabel("E")
            ax.set_ylabel("C")
        elif title == "CS MAC":
            ax.set_xlabel("C")
            ax.set_ylabel("E")

        # ---- GRID LINES ----

        # Put grid lines between cells
        ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)

        # Draw grid
        ax.grid(which="minor", color="white", linestyle='-', linewidth=1)

        # Remove minor tick marks
        ax.tick_params(which="minor", bottom=False, left=False)

        fig.colorbar(im, ax=ax)

    fig.suptitle("Macro Variables")

    plt.tight_layout()

    figure_name = "macro_variables_visu.png"
    save_path = os.path.join(FILE_PATH, "out", figure_name)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    num_clusters = 10
    clustering_method = "dirichlet_process_gaussian_mixture"
    eft_mac = np.load(os.path.join(FILE_PATH, f"out/eft_mac_{num_clusters}_{clustering_method}.npy"))
    cs_mac = np.load(os.path.join(FILE_PATH, f"out/cs_mac_{num_clusters}_{clustering_method}.npy"))
    visualize_macros(eft_mac, cs_mac)


if __name__ == "__main__":
    main()