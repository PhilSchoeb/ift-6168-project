import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from allensdk.brain_observatory.ecephys.ecephys_project_api.rma_engine import RmaEngine
from allensdk.brain_observatory.ecephys.ecephys_project_api.utilities import build_and_execute
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache


def retrieve_link(session_id: int, cache_dir: str = "./data") -> str:
    """
    Returns the download link of a session.  
    Also creates the directory for the nwb file. 

    Inputs
        session_id: session to download
        cache_dir: path to cache (where manifest.json is)
    
    Output
        download link
    """
    os.makedirs(cache_dir+f"/session_{session_id}",exist_ok=True)
    
    rma_engine = RmaEngine(scheme="http", host="api.brain-map.org")
    
    well_known_files = build_and_execute(
        (
        "criteria=model::WellKnownFile"
        ",rma::criteria,well_known_file_type[name$eq'EcephysNwb']"
        "[attachable_type$eq'EcephysSession']"
        r"[attachable_id$eq{{session_id}}]"
        ),
        engine=rma_engine.get_rma_tabular, 
        session_id=session_id
    )
    
    return 'http://api.brain-map.org/' + well_known_files['download_link'].iloc[0]


def download_data(session_id: int, cache_dir: str = "./data", n_bins_activation: int = 100):
    """
    Saves the relevant data of a session.  
    The session must have been downloaded before. 

    Inputs
        session_id: session to save
        cache_dir: path to cache (where manifest.json is)
        n_bins_activation: number of bins for activation data (default is 100)
    """
    assert os.path.exists(cache_dir+f"/session_{session_id}/session_{session_id}.nwb"), "session nwb file not found"
    cache = EcephysProjectCache.from_warehouse(manifest=cache_dir+"/manifest.json")
    session = cache.get_session_data(session_id)
    os.makedirs(cache_dir+f"/session_{session_id}/static_gratings",exist_ok=True)
    os.makedirs(cache_dir+f"/session_{session_id}/natural_scenes",exist_ok=True)

    # Metadata and units
    f = open(cache_dir+f"/session_{session_id}/metadata.json","w")
    metadata = session.metadata.copy()
    metadata["session_start_time"] = metadata.get("session_start_time",datetime.datetime(1,1,1)).strftime("%Y-%m-%d %H:%M:%S")
    json.dump(metadata,f)
    f.close()
    VIS_units = session.units.query("ecephys_structure_acronym in ['VISp','VISl','VISal','VISrl','VISpm','VISam']")
    VIS_units[["ecephys_structure_acronym"]].to_csv(cache_dir+f"/session_{session_id}/units.csv")

    # Stimulus
    stimulus = session.get_stimulus_table()

    ## Static gratings
    static_gratings = stimulus.query("stimulus_name=='static_gratings'")
    static_gratings[["orientation","spatial_frequency","phase","contrast","size","duration"]].to_csv(cache_dir+f"/session_{session_id}/static_gratings/stimulus.csv")

    ## Natural scenes
    natural_scenes = stimulus.query("stimulus_name=='natural_scenes'")
    natural_scenes[["frame","duration"]].to_csv(cache_dir+f"/session_{session_id}/natural_scenes/stimulus.csv")

    # Neural activity
    
    ## Static gratings
    activations1 = session.presentationwise_spike_counts(np.linspace(0,static_gratings.duration.max(),n_bins_activation+1),static_gratings.index,VIS_units.index)
    np.savez(cache_dir+f"/session_{session_id}/static_gratings/activation.npz",data=activations1.data,presentation_ids=activations1.stimulus_presentation_id.data,timestamps=activations1.time_relative_to_stimulus_onset.data,unit_ids=activations1.unit_id.data)

    ## Natural scenes
    activations2 = session.presentationwise_spike_counts(np.linspace(0,natural_scenes.duration.max(),n_bins_activation+1),natural_scenes.index,VIS_units.index)
    np.savez(cache_dir+f"/session_{session_id}/natural_scenes/activation.npz",data=activations2.data,presentation_ids=activations2.stimulus_presentation_id.data,timestamps=activations2.time_relative_to_stimulus_onset.data,unit_ids=activations2.unit_id.data)


def get_full_raster(spike_times: pd.DataFrame, duration: float, unit_ids: list) -> np.ndarray:
    """
    Returns an array representing a raster plot (full, not binned). 
    
    Inputs
        spike_times: DataFrame containing the spike times (since stimulus presentation onset) of all units
        duration: duration (in seconds) of the stimulus presentation
        unit_ids: list of all units to use in the raster
    
    Output
        array of shape (n_units, duration_ms) where n_units is the number of units and duration_ms is the duration in microseconds
    """
    assert isinstance(spike_times,pd.DataFrame), f"spike_times must be a DataFrame, not {type(spike_times)}"
    assert "unit_id" in spike_times.columns, "spike_times must have a column 'unit_id'"
    assert "time_since_stimulus_presentation_onset" in spike_times.columns, "spike_times must have a column 'time_since_stimulus_presentation_onset'"
    raster = np.zeros((len(unit_ids),int(duration*1e6)+1))
    for i, unit in enumerate(unit_ids):
        times = np.array((1e6*spike_times[spike_times.unit_id==unit].time_since_stimulus_presentation_onset).astype(int))
        raster[i,times] = 1.0
    return raster


def plot_full_raster(spike_times: pd.DataFrame, unit_ids: list):
    """
    Plots the raster according to spike_times data.  
    Source : ChatGPT

    Inputs
        spike_times: DataFrame containing the spike times (since stimulus presentation onset) of all units
        unit_ids: list of all units to use in the plot
    """
    assert isinstance(spike_times,pd.DataFrame), f"spike_times must be a DataFrame, not {type(spike_times)}"
    assert "unit_id" in spike_times.columns, "spike_times must have a column 'unit_id'"
    assert "time_since_stimulus_presentation_onset" in spike_times.columns, "spike_times must have a column 'time_since_stimulus_presentation_onset'"
    plt.figure(figsize=(10, 6))
    for i, unit in enumerate(unit_ids):
        x = spike_times[spike_times['unit_id'] == unit].time_since_stimulus_presentation_onset.sort_values()
        # Plot vertical lines (raster ticks)
        plt.vlines(x, i + 0.5, i + 1.5)
    plt.yticks(range(1, len(unit_ids) + 1), unit_ids)
    plt.xlabel("Time since stimulus presentation onset")
    plt.ylabel("Unit ID")
    plt.title("Raster Plot")
    plt.ylim(0.5, len(unit_ids) + 0.5)
    plt.show()
