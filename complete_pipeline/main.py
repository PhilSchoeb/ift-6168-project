"""
Main file for complete pipeline runs of Causal Feature Learning (CFL) with real world data; mice neuronal responses to
stimuli images for the Allen Brain Observatory
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)

from complete_pipeline import generate_help_text

import argparse
from datetime import datetime
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
    num_bins = config["num_bins"]
    num_neurons = config["num_neurons"]
    neuron_selection = config["neuron_selection"]
    dim_reduction = config["dimensionality_reduction"]
    density_estimator = config["density_estimator"]
    clustering = config["clustering"]
    num_clusters = config["num_clusters"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(FILE_PATH, f"out/run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Initiating experiment {experiment_name}...")
    print(f"Saving artefacts at: {out_dir}/...")


if __name__ == "__main__":
    main()