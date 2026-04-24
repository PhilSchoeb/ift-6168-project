"""
Download locally, from the allensdk API, .nwb files that contain every information relating to the brain observatory
experiments on mouse neuronal activations.
"""

PATH_TO_MANIFEST = "brain_observatory/manifest.json"

import os
import pandas as pd
from tqdm import tqdm
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

current_file_path = os.path.dirname(os.path.abspath(__file__))
path_to_manifest = os.path.join(current_file_path, PATH_TO_MANIFEST)

boc = BrainObservatoryCache(manifest_file=path_to_manifest)
sessions = boc.get_ophys_experiments()
sessions = pd.DataFrame(sessions)
session_ids = sessions["id"]

for session_id in tqdm(session_ids, desc="Downloading for different sessions"):
    print(f"Downloading for session_id: {session_id}:")
    try:
        dataset = boc.get_ophys_experiment_data(session_id)
    except Exception as e:
        print(e)
        print(f"Failed to download {session_id}")