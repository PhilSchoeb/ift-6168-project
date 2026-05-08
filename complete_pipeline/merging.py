"""
For manual merging of effect and cause clusters
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)

from helpers.cluster import get_macro_only
from clustering import merge_clusters, visualize_macros, visualize_samples_from_clusters

import argparse
import glob
import numpy as np


def parse_merge_groups(merge_list):
    """
    Parse merge arguments of the form 'cluster_a_cluster_b_cluster_c_...' into lists.
    E.g. ['0_2', '1_3_5'] -> [[0, 2], [1, 3, 5]]
    """
    groups = []
    for item in merge_list:
        parts = item.split('_')
        if len(parts) < 2:
            raise ValueError(f"Invalid merge format '{item}'. Expected at least 2 clusters: <cluster_a>_<cluster_b>[_<cluster_c>...]")
        groups.append([int(p) for p in parts])
    return groups


def main():
    parser = argparse.ArgumentParser(
        description="Merge macro variable clusters after the automation pipeline."
    )
    parser.add_argument(
        '--run_folder',
        type=str,
        required=True,
        help="Path to the run folder produced by the pipeline."
    )
    parser.add_argument(
        '--merge',
        type=str,
        nargs='+',
        required=True,
        help=(
            "Pairs of clusters to merge, each in the format <cluster_a>_<cluster_b>. "
            "cluster_b will be merged into cluster_a. "
            "Example: --merge 0_2 1_3"
        )
    )
    parser.add_argument(
        '--cluster_type',
        type=str,
        required=True,
        choices=['eft', 'cs'],
        help="Type of cluster to merge: 'eft' or 'cs'."
    )

    args = parser.parse_args()

    run_folder = args.run_folder
    if not os.path.isdir(run_folder):
        raise ValueError(f"Run folder '{run_folder}' does not exist.")

    merge_groups = parse_merge_groups(args.merge)
    print(f"Merge groups : {merge_groups}")

    cluster_type = args.cluster_type
    print(f"Cluster type: {cluster_type}")

    clusters_path = os.path.join(run_folder, f"{cluster_type}_clusters.npy")
    clusters = np.load(clusters_path)
    merged_clusters = merge_clusters(clusters, merge_groups)

    # Rename old clusters file
    old_clusters_path = os.path.join(
        run_folder,
        f"{cluster_type}_clusters_old.npy"
    )

    os.rename(clusters_path, old_clusters_path)

    # Save merged clusters
    np.save(clusters_path, merged_clusters)

    print(f"Regenerating macro-variables and visualizations...")

    # Reload potentially new clusters
    eft_clusters = np.load(os.path.join(run_folder, "eft_clusters.npy"))
    cs_clusters = np.load(os.path.join(run_folder, "cs_clusters.npy"))

    density_files = glob.glob(os.path.join(run_folder, "density_*.npy"))

    if len(density_files) == 0:
        raise FileNotFoundError(f"No density file found in {run_folder}")
    if len(density_files) > 1:
        raise ValueError(f"Multiple density files found in {run_folder}: {density_files}")

    density_path = density_files[0]
    density = np.load(density_path)

    eft_mac, cs_mac = get_macro_only(density, eft_clusters, cs_clusters)

    # Rename old macro-variables file
    old_macro_path = os.path.join(
        run_folder,
        f"{cluster_type}_mac_old.npy"
    )
    current_macro_path = os.path.join(
        run_folder,
        f"{cluster_type}_mac.npy"
    )

    os.rename(current_macro_path, old_macro_path)

    # Save macro-variables
    np.save(os.path.join(run_folder, "eft_mac.npy"), eft_mac)
    np.save(os.path.join(run_folder, "cs_mac.npy"), cs_mac)

    original_i_path = os.path.join(run_folder, "original_i.npy")
    original_j_path = os.path.join(run_folder, "original_j.npy")
    original_i = np.load(original_i_path)
    original_j = np.load(original_j_path)

    print(f"Generating new visualizations...")
    # Visualizations
    # Visualize samples from each cluster
    i_samples_path = os.path.join(run_folder, "i_merged_cluster_samples.png")
    j_samples_path = os.path.join(run_folder, "j_merged_cluster_sample.png")
    visualize_samples_from_clusters(original_i, original_j, eft_clusters, cs_clusters, path_i=i_samples_path,
                                    path_j=j_samples_path)

    # Visualize macro-variables
    macro_path = os.path.join(run_folder, "merged_macro_variables_visu.png")
    visualize_macros(eft_mac, cs_mac, path=macro_path)

    print("Merging complete.")


if __name__ == '__main__':
    main()