# Data availability

For this complete pipeline to work, you must have downloaded the required data and placed it at the project root as `./session_data/session_{session_number}/...`

# How to run

```bash
python complete_pipeline/main.py --config {CONFIG_NAME}
```
The default value for `--config` is `baseline_1`.

For any additional help or information on the hyperparameters used:
```bash
python complete_pipeline/main.py --help
```

## Manual merging

After having run the complete_pipeline, you can manually merge the clusters (macro-variables) you want. To do so you
need three arguments:
1. run_folder: the path to the folder where the results of interest are stored.
2. cluster_type: which cluster type do you want to merge. You have to select "eft" (for effect aka J) or "cs" (for cause aka I).
3. merge: the cluster numbers you wish to merge in groups. For example, if you wish to merge cluster 0 and 1 together as well as a different merge of cluster 2, 3 and 5, the merge variable should take: 0_1 2_3_5 as argument.

```bash
python complete_pipeline/merging.py --run_folder {run_folder} --cluster_type {eft_or_cs} --merge {merge_groups}
```
You usually should merge both the effects and the causes, so we encourage you to merge them iteratively:
```bash
# Merge effects
python complete_pipeline/merging.py --run_folder {run_folder} --cluster_type eft --merge {eft_merges}

# Merge causes
python complete_pipeline/merging.py --run_folder {run_folder} --cluster_type cs --merge {cs_merges}
```
Running these commands will generate new up-to-date figures and keep the old onces. Same for the data.