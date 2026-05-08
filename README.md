# ift-6168-project
Causal Feature Learning (CFL) application on mice neuronal activity as an effect to visual stimuli (the cause).

The algorithm and project idea come from [this paper](https://arxiv.org/abs/1512.07942) by Chalupka et al.

## How to use

Setup:
```bash
# Create venv
python -m venv .venv

# Activate venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

Before running the algorithm, you have to download the session_data:
https://drive.google.com/drive/folders/1VUATvv57rQbks9GIToRdCQ9uzClQ24K0?usp=sharing

Normally, anyone can access this drive to view the files. You have to download session_data.zip. Unzip it as session_data at the project root and you are good to go.

To run the algorithm, we propose using the complete_pipeline main file:
```bash
# With default config "baseline_1.json"
python complete_pipeline/main.py

# With another config named "example.json"
pyton complete_pipeline/main.py --config example.json
```

For more help around the pipeline and hyperparameter, go check `complete_pipeline/README.md`

You can also use individual python files, but they have not been optimized for this usage.
