"""
Helper for the --help command on `main.py`.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)

import yaml


def format_possible_values(values, indent=8):
    text = ""
    for value in values:
        text += " " * indent + f"- {value}\n"
    return text


def generate_help_text():
    help_with_main_part_1 = """
Usage:
    python main.py [options]

Description:
    Run causal feature learning experiments using a configuration file.

Options:
    -h, --help
        Show this help message and exit.

    --config CONFIG_FILE
        Name of the configuration file to use.

        Example:
            --config baseline_1.yaml

        Default:
            baseline_1.yaml
"""

    config_path = config_path = os.path.join(FILE_PATH, f"configs/all_hyperparameters.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    help_with_main_part_2 = f"""

===========================================================================
Hyperparameters
===========================================================================

experiment_name:
    Name of the experiment configuration.

    Possible values:
        - any string

    Example:
        "baseline"


experiment_file:
    Allen Brain Observatory experiment identifier(s) to use.

    Possible values:
{format_possible_values(config["experiment_file"])}

    Notes:
        "all" runs over all supported experiment files.


i_dataset:
    Dataset used for the i-variable distribution.

    Possible values:
{format_possible_values(config["i_dataset"])}

    Notes:
        TODO


num_bins:
    Number of bins used for the j-dataset representation.

    Possible values:
{format_possible_values(config["num_bins"])}


num_neurons:
    Number of neurons used from the j-dataset.

    Possible values:
{format_possible_values(config["num_neurons"])}

    Notes:
        "all" uses every available neuron.


neuron_selection:
    Method used to select neurons from the dataset.

    Possible values:
{format_possible_values(config["neuron_selection"])}

    Notes:
        "variance" selects neurons according to response variance.


dimensionality_reduction:
    Dimensionality reduction routine applied before density estimation
    and clustering.

    Possible values:
{format_possible_values(config["dimensionality_reduction"])}

    Notes:
        autoencoder:
            Neural-network-based nonlinear dimensionality reduction.

        pca:
            Principal Component Analysis.

        truncatedsvd:
            Truncated Singular Value Decomposition.

        nmf:
            Non-negative Matrix Factorization.


reduced_dimension:
    Number of dimensions the data is reduced to.
    
    Possible values:
{format_possible_values(config["reduced_dimension"])}

    Notes:
        TODO
        
        
density_estimator:
    Density estimation routine used for macro-variable estimation.

    Possible values:
{format_possible_values(config["density_estimator"])}

    Notes:
        TODO


clustering:
    Clustering routine used to identify macro-variable clusters.

    Possible values:
{format_possible_values(config["clustering"])}

    Notes:
        TODO


num_clusters:
    Number of clusters used by the clustering routine.

    Possible values:
{format_possible_values(config["num_clusters"])}

    Notes:
        Some clustering methods may ignore this value depending on
        their internal configuration.
"""
    return help_with_main_part_1 + help_with_main_part_2