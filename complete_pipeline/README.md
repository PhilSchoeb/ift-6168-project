# Data availability

For this complete pipeline to work, you must have downloaded the required data and placed it at the project root as `./session_data/session_{session_number}/...`

# How to run

````
python complete_pipeline/main.py --config {CONFIG_NAME}
````
The default value for `--config` is `baseline_1`.

For any additional help or information on the hyperparameters used:
````
python complete_pipeline/main.py --help
````