"""
Load macro variables after calculating them for visualization
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)

import matplotlib.pyplot as plt
import numpy as np


def visualize_macros(eft_mac, cs_mac, path=None):

    if eft_mac.shape[0]>10:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    matrices = [
        (eft_mac, "EFT MAC"),
        (cs_mac, "CS MAC")
    ]

    for ax, (mat, title) in zip(axes, matrices):

        im = ax.imshow(mat, cmap="viridis")

        ax.set_title(title)

        ax.set_xlabel("E" if title == "EFT MAC" else "C")
        ax.set_ylabel("C" if title == "EFT MAC" else "E")

        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_xticklabels(np.arange(mat.shape[1]))
        ax.set_yticklabels(np.arange(mat.shape[0]))

        # ---- GRID LINES ----

        # Put grid lines between cells
        ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)

        # Draw grid
        ax.grid(which="minor", color="white", linestyle='-', linewidth=1)

        # Remove minor tick marks
        ax.tick_params(which="minor", bottom=False, left=False)

        # ---- CELL VALUES ----
        for (row, col), val in np.ndenumerate(mat):
            text_color = "white" if val < 0.5 else "black"
            label = "0" if val <= 0.005 else f"{val:.2f}"
            ax.text(col, row, label, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

        fig.colorbar(im, ax=ax)

    fig.suptitle("Macro Variables")

    plt.tight_layout()

    if path is None:
        figure_name = "macro_variables_visu.png"
        save_path = os.path.join(FILE_PATH, "out", figure_name)
    else:
        save_path = path

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