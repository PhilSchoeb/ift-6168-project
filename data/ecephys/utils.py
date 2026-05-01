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
    Fonction qui retourne le lien de téléchargement d'une session. 

    Entrées
        session_id: id de la session à télécharger
        cache_dir: chemin de la cache
    
    Sortie
        lien de téléchargement
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

def download_data(session_id: int, cache_dir: str = "./data"):
    """
    Fonction qui enregistre les données d'une session. 
    La session doit avoir été téléchargée auparavant. 

    Entrées
        session_id: id de la session
        cache_dir: chemin de la cache
    """
    assert os.path.exists(cache_dir+f"/session_{session_id}/session_{session_id}.nwb"), "la session doit avoir été téléchargée"
    cache = EcephysProjectCache.from_warehouse(manifest=cache_dir+"/manifest.json")
    session = cache.get_session_data(session_id)

    if not os.path.exists(cache_dir+f"/session_{session_id}/metadata.json"):
        f = open(cache_dir+f"/session_{session_id}/metadata.json","w")
        try:
            metadata = session.metadata.copy()
            metadata["session_start_time"] = metadata.get("session_start_time",datetime.datetime(1,1,1)).strftime("%Y-%m-%d %H:%M:%S")
            json.dump(metadata,f)
        except Exception:
            print(f"Erreur lors de la récupération des métadonnées de la session {session_id}")
        f.close()

    if not os.path.exists(cache_dir+f"/session_{session_id}/units.csv"):
        try:
            session.units.to_csv(cache_dir+f"/session_{session_id}/units.csv")
        except Exception:
            print(f"Erreur lors de la récupération des unités de la session {session_id}")

    if not os.path.exists(cache_dir+f"/session_{session_id}/stimulus_table.csv"):
        try:
            stimulus_table = session.get_stimulus_table(include_detailed_parameters=True,include_unused_parameters=True)
        except Exception:
            print(f"Erreur lors de la récupération de la table des stimulus de la session {session_id}")
        stimulus_table.to_csv(cache_dir+f"/session_{session_id}/stimulus_table.csv")

    if not os.path.exists(cache_dir+f"/session_{session_id}/spike_times.json"):
        try:
            spike_times = session.presentationwise_spike_times()
        except Exception:
            print(f"Erreur lors de la récupération des temps de pic de la session {session_id}")
        spike_times.to_csv(cache_dir+f"/session_{session_id}/spike_times.csv")

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
