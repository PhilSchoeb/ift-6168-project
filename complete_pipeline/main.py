"""
Main file for complete pipeline runs of Causal Feature Learning (CFL) with real world data; mice neuronal responses to
stimuli images for the Allen Brain Observatory
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)

from complete_pipeline import generate_help_text
from dataset import fetch_dataset, apply_dimensionality_reduction
from density import get_density_estimation
from density_learning import standardize_data

import argparse
from datetime import datetime
import json
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()

    # Remove default --help
    parser = argparse.ArgumentParser(
        add_help=False
    )
    # Override default --help
    parser.add_argument(
        "-h", "--help",
        action="store_true",
    )
    parser.add_argument("--config", type=str, required=False, default="baseline_1.yaml")
    args = parser.parse_args()

    if args.help:
        help_message = generate_help_text()
        print(help_message)
        return

    config = args.config
    if config[-5:] != ".yaml":
        config += ".yaml"
    config_path = os.path.join(FILE_PATH, f"configs/{config}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    experiment_name = config["experiment_name"]
    experiment_file = config["experiment_file"]
    i_dataset = config["i_dataset"]
    orientations = config["orientations"]
    units = config["units"]
    num_bins = config["num_bins"]
    num_neurons = config["num_neurons"]
    neuron_selection = config["neuron_selection"]
    dim_reduction = config["dimensionality_reduction"]
    reduced_dimension = config["reduced_dimension"]
    standardize_post_reduction = config["standardize_post_reduction"]
    density_estimator = config["density_estimator"]
    bandwidth_i = config["bandwidth_i"]
    bandwidth_j = config["bandwidth_j"]
    clustering = config["clustering"]
    num_clusters = config["num_clusters"]

    assert not any(
        isinstance(v, list)
        for v in [
            experiment_name,
            experiment_file,
            i_dataset,
            units,
            num_bins,
            num_neurons,
            neuron_selection,
            dim_reduction,
            reduced_dimension,
            standardize_post_reduction,
            density_estimator,
            bandwidth_i,
            bandwidth_j,
            clustering,
            num_clusters,
        ]
    ), "Config file used must have only one value for each hyperparameter."

    # Different treatment for "orientations" which is a list
    assert not isinstance(orientations[0], list), "Config file used must have only one value for each hyperparameter."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(FILE_PATH, f"out/run_{experiment_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Initiating experiment {experiment_name}...")
    print(f"Saving artefacts at: {out_dir}/...")

    # Save used hyperparameters
    used_config = {
        "experiment_name": experiment_name,
        "experiment_file": experiment_file,
        "i_dataset": i_dataset,
        "orientations": orientations,
        "units": units,
        "num_bins": num_bins,
        "num_neurons": num_neurons,
        "neuron_selection": neuron_selection,
        "dimensionality_reduction": dim_reduction,
        "reduced_dimension": reduced_dimension,
        "standardize_post_reduction": standardize_post_reduction,
        "density_estimator": density_estimator,
        "bandwidth_i": bandwidth_i,
        "bandwidth_j": bandwidth_j,
        "clustering": clustering,
        "num_clusters": num_clusters,
    }

    with open(os.path.join(out_dir, "used_config.json"), "w") as f:
        json.dump(used_config, f, indent=4)

    # Fetch data
    print(f"Fetching dataset...")
    i, j = fetch_dataset(experiment_file, i_dataset, orientations, units, num_bins, num_neurons, neuron_selection)

    print(f"Applying dimensionality reduction...")
    i, j, exp_var_i, exp_var_j = apply_dimensionality_reduction(i, j, experiment_file, i_dataset, dim_reduction, reduced_dimension)
    if exp_var_i is not None:
        path_exp_var_i = os.path.join(out_dir, "i_reduc_explained_variance.txt")
        with open(path_exp_var_i, "w") as f:
            f.write(f"Using {dim_reduction}: " + str(exp_var_i))
        print(f"i reduction explained variance: {exp_var_i} with {dim_reduction}")
    if exp_var_j is not None:
        path_exp_var_j = os.path.join(out_dir, "j_reduc_explained_variance.txt")
        with open(path_exp_var_j, "w") as f:
            f.write(f"Using {dim_reduction}: " + str(exp_var_j))
        print(f"j reduction explained variance: {exp_var_j} with {dim_reduction}")

    # After dimensionality reduction, we expect 2D data
    assert i.ndim == 2, f"Expected 2 dimensions for i but got: {i.ndim}"
    assert i.ndim == 2, f"Expected 2 dimensions for j but got: {j.ndim}"

    if standardize_post_reduction:
        print(f"Applying standardization...")
        i = standardize_data(i)
        j = standardize_data(j)

    # Conditional density estimation
    density = get_density_estimation(i, j, density_estimator, bandwidth_i, bandwidth_j)
    density_path = os.path.join(out_dir, f"density_{density_estimator}.npy")
    np.save(density_path, density)

    # CLustering
    # TODO


if __name__ == "__main__":
    main()