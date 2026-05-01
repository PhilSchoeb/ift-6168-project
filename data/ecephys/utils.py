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


def generate_gratings(orientations,
                      spatial_freqs,
                      phases,
                      size=(250, 250),
                      contrast=0.8,
                      max_pixel_value=255.0,
                      stripes_per_unit=75):  # calibrated: sf=0.32 → 24 stripes
    """
    Generate a batch of sinusoidal gratings.

    Based on parameters presented here: https://observatory.brain-map.org/visualcoding/stimulus/static_gratings

    Parameters
    ----------
    orientations     : array-like (N,) — degrees
    spatial_freqs    : array-like (N,) — cycles per degree
    phases           : array-like (N,) — fraction of cycle [0, 1]
    size             : (H, W) in pixels
    contrast         : float
    max_pixel_value  : float
    stripes_per_unit : float — calibration constant (sf=0.32 → 48 stripes)

    Returns
    -------
    imgs : (N, H, W)
    """
    orientations  = np.asarray(orientations)
    spatial_freqs = np.asarray(spatial_freqs)
    phases        = np.asarray(phases)

    N = len(orientations)

    # x axis: 0 to 1, so sf * stripes_per_unit = number of cycles across width
    x = np.linspace(0, 1, size[1])
    y = np.linspace(0, 1, size[0])
    X, Y = np.meshgrid(x, y)

    imgs = np.empty((N, size[0], size[1]), dtype=np.float32)

    for i in range(N):
        theta = np.deg2rad(orientations[i])
        sf    = spatial_freqs[i] * stripes_per_unit  # cycles across image
        phi   = 2 * np.pi * phases[i]

        # rotate coordinates
        Xr = X * np.cos(theta) + Y * np.sin(theta)

        # sinusoidal grating
        img = contrast * np.sin(2 * np.pi * sf * Xr + phi)

        # map [-1, 1] -> [0, max_pixel_value]
        img = (0.5 + 0.5 * img) * max_pixel_value

        imgs[i] = img.astype(np.float32)

    return imgs


class StaticGratingsDataset:
    def __init__(self, session_id: int, cache_dir: str = "./session_data"):
        """
        Dataset for 1 session. Contains stimulus and neural activation data. 

        Parameters
            session_id: session to use
            cache_dir: path to data directory
        """
        self.session_id = session_id
        self.stimulus_name = "static_gratings"
        self.data_dir = cache_dir+f"/session_{session_id}"

        f = open(self.data_dir+"/metadata.json")
        self.session_metadata = json.load(f)
        f.close()
        self.units = pd.read_csv(self.data_dir+"/units.csv",index_col=0)

        self.stimulus_table = pd.read_csv(self.data_dir+"/static_gratings/stimulus.csv",index_col=0)
        self.stimulus_table.dropna(inplace=True)
        act_data = np.load(self.data_dir+"/static_gratings/activation.npz")
        self.activation_data = act_data["data"]
        self.presentation_ids = act_data["presentation_ids"]
        self.timestamps = act_data["timestamps"]
        self.unit_ids = act_data["unit_ids"]
    
    def list_possible_values(self, parameter_name: str) -> list:
        """
        Returns a list of all possible values of a parameter. 

        Input
            parameter_name: name of the parameter ('orientation', 'spatial_frequency', 'phase' or 'ecephys_structure_acronym')
        
        Output
            list of possible values
        """
        if parameter_name=="ecephys_structure_acronym":
            values = self.units.ecephys_structure_acronym.unique().tolist()
        else:
            values = self.stimulus_table[parameter_name].unique().tolist()
        values.sort()
        return values

    def get_presentation_ids(self, orientation: int | list = None, spatial_frequency: float | list = None, phase: float | list = None) -> list:
        """
        Returns the presentation ids of a specific set of stimulus. 

        Inputs
            orientation: orientation of the gratings (if a list, returns all presentations where orientation is in the list)
            spatial_frequency: spatial_frequency of the gratings (if a list, returns all presentations where spatial_frequency is in the list)
            phase: phase of the gratings (if a list, returns all presentations where phase is in the list)

        Outputs
            list of presentation ids
        """
        selection = self.stimulus_table
        if isinstance(orientation,(int,float)):
            selection = selection[selection.orientation==orientation]
        elif isinstance(orientation,(list,tuple,np.ndarray)):
            selection = selection[selection.orientation.isin(orientation)]
        if isinstance(spatial_frequency,(int,float)):
            selection = selection[selection.spatial_frequency==spatial_frequency]
        elif isinstance(spatial_frequency,(list,tuple,np.ndarray)):
            selection = selection[selection.spatial_frequency.isin(spatial_frequency)]
        if isinstance(phase,(int,float)):
            selection = selection[selection.phase==phase]
        elif isinstance(phase,(list,tuple,np.ndarray)):
            selection = selection[selection.phase.isin(phase)]
        return selection.index.to_list()

    def get_unit_ids(self, ecephys_structure_acronym: str | list = None) -> list:
        """
        Returns the unit ids of a specific structure. 

        Inputs
            ecephys_structure_acronym: ecephys_structure_acronym of the units (if a list, returns all units where ecephys_structure_acronym is in the list)

        Outputs
            list of unit ids
        """
        selection = self.units
        if isinstance(ecephys_structure_acronym,str):
            selection = selection[selection.ecephys_structure_acronym==ecephys_structure_acronym]
        elif isinstance(ecephys_structure_acronym,(list,tuple,np.ndarray)):
            selection = selection[selection.ecephys_structure_acronym.isin(ecephys_structure_acronym)]
        return selection.index.to_list()

    def get_data(self, presentation_ids: list = None, unit_ids: list = None, stimulus_type: str = "images") -> tuple[np.ndarray,np.ndarray]:
        """
        Returns data (stimulus and neural activations) for specific presentations and units. 

        Inputs
            presentation_ids: list of presentation ids
            unit_ids: list of unit ids
            stimulus_type: type of stimulus to output ('images' or 'params')
        
        Outputs
            stimulus array (len(presentation_ids), 250, 250) if stimulus_type='images' else (len(presentation_ids), 3)
            neural activations array (len(presentation_ids), 100, len(unit_ids))
        """
        # Get index
        if presentation_ids is None:
            presentations = self.stimulus_table.index
        else:
            presentations = presentation_ids
        presentation_map = {v: i for i,v in enumerate(self.presentation_ids)}
        presentation_idx = [presentation_map[v] for v in presentations]
        if unit_ids is None:
            units = self.units.index
        else:
            units = unit_ids
        unit_map = {v: i for i,v in enumerate(self.unit_ids)}
        unit_idx = [unit_map[v] for v in units]

        # Stimulus
        stimulus_params = self.stimulus_table.loc[presentations]
        if stimulus_type=="images":
            stimulus = generate_gratings(stimulus_params.orientation,stimulus_params.spatial_frequency,stimulus_params.phase)
        elif stimulus_type=="params":
            stimulus = stimulus_params[["orientation","spatial_frequency","phase"]].to_numpy()
        else:
            raise ValueError("stimulus_type must be 'images' or 'params'")

        # Neural activations
        activations = self.activation_data[presentation_idx,:,:][:,:,unit_idx]

        return stimulus, activations

    def __repr__(self):
        return self.__class__.__name__+f"(session_id={self.session_id})"


class NaturalScenesDataset:
    def __init__(self, session_id: int, cache_dir: str = "./session_data"):
        """
        Dataset for 1 session. Contains stimulus and neural activation data. 

        Parameters
            session_id: session to use
            cache_dir: path to data directory
        """
        self.session_id = session_id
        self.stimulus_name = "natural_scenes"
        self.data_dir = cache_dir+f"/session_{session_id}"

        f = open(self.data_dir+"/metadata.json")
        self.session_metadata = json.load(f)
        f.close()
        self.units = pd.read_csv(self.data_dir+"/units.csv",index_col=0)

        self.stimulus_table = pd.read_csv(self.data_dir+"/natural_scenes/stimulus.csv",index_col=0)
        self.stimulus_table.dropna(inplace=True)
        self.stimulus_table["frame"] = self.stimulus_table.frame.astype(int)
        self.natural_scenes = np.load(cache_dir+"/natural_scenes.npy")
        act_data = np.load(self.data_dir+"/natural_scenes/activation.npz")
        self.activation_data = act_data["data"]
        self.presentation_ids = act_data["presentation_ids"]
        self.timestamps = act_data["timestamps"]
        self.unit_ids = act_data["unit_ids"]
    
    def list_possible_values(self, parameter_name: str) -> list:
        """
        Returns a list of all possible values of a parameter. 

        Input
            parameter_name: name of the parameter ('frame' or 'ecephys_structure_acronym')
        
        Output
            list of possible values
        """
        if parameter_name=="ecephys_structure_acronym":
            values = self.units.ecephys_structure_acronym.unique().tolist()
        else:
            values = self.stimulus_table[parameter_name].unique().tolist()
        values.sort()
        return values

    def get_presentation_ids(self, frame: int | list = None) -> list:
        """
        Returns the presentation ids of a specific set of stimulus. 

        Inputs
            frame: index of an image (if a list, returns all presentations where frame is in the list)

        Outputs
            list of presentation ids
        """
        selection = self.stimulus_table
        if isinstance(frame,(int,float)):
            selection = selection[selection.frame==frame]
        elif isinstance(frame,(list,tuple,np.ndarray)):
            selection = selection[selection.frame.isin(frame)]
        return selection.index.to_list()

    def get_unit_ids(self, ecephys_structure_acronym: str | list = None) -> list:
        """
        Returns the unit ids of a specific structure. 

        Inputs
            ecephys_structure_acronym: ecephys_structure_acronym of the units (if a list, returns all units where ecephys_structure_acronym is in the list)

        Outputs
            list of unit ids
        """
        selection = self.units
        if isinstance(ecephys_structure_acronym,str):
            selection = selection[selection.ecephys_structure_acronym==ecephys_structure_acronym]
        elif isinstance(ecephys_structure_acronym,(list,tuple,np.ndarray)):
            selection = selection[selection.ecephys_structure_acronym.isin(ecephys_structure_acronym)]
        return selection.index.to_list()

    def get_data(self, presentation_ids: list = None, unit_ids: list = None) -> tuple[np.ndarray,np.ndarray]:
        """
        Returns data (stimulus and neural activations) for specific presentations and units. 

        Inputs
            presentation_ids: list of presentation ids
            unit_ids: list of unit ids
        
        Outputs
            stimulus array (len(presentation_ids), 918, 1174)
            neural activations array (len(presentation_ids), 100, len(unit_ids))
        """
        # Get index
        if presentation_ids is None:
            presentations = self.stimulus_table.index
        else:
            presentations = presentation_ids
        presentation_map = {v: i for i,v in enumerate(self.presentation_ids)}
        presentation_idx = [presentation_map[v] for v in presentations]
        if unit_ids is None:
            units = self.units.index
        else:
            units = unit_ids
        unit_map = {v: i for i,v in enumerate(self.unit_ids)}
        unit_idx = [unit_map[v] for v in units]

        # Stimulus
        frames = self.stimulus_table.loc[presentations,"frame"].values
        stimulus = self.natural_scenes[frames,:,:]

        # Neural activations
        activations = self.activation_data[presentation_idx,:,:][:,:,unit_idx]

        return stimulus, activations

    def __repr__(self):
        return self.__class__.__name__+f"(session_id={self.session_id})"


def plot_data_samples(stimulus: np.ndarray, activation: np.ndarray, n_samples: int, random_sate: int = None):
    """
    Plots samples from a dataset (stimulus and neural activations). 

    Inputs
        stimulus: stimulus data
        activation: neural activations data
        n_samples: number of samples to show
        random_sate: random seed
    """
    if random_sate!=None:
        np.random.seed(random_sate)
    samples = np.random.choice(range(stimulus.shape[0]),size=n_samples,replace=False)
    if n_samples>1:
        fig, ax = plt.subplots(nrows=n_samples,ncols=2,figsize=(10,10))
        for i, x in enumerate(samples):
            if stimulus.ndim==3:
                ax[i,0].imshow(stimulus[i],cmap="gray")
            else:
                stim = generate_gratings([stimulus[i,0]],[stimulus[i,1]],[stimulus[i,2]])
                ax[i,0].imshow(stim[0],cmap="gray")
                ax[i,0].text(1.05, 0.8, f"orientation={stimulus[i,0]}", transform=ax[i,0].transAxes, fontsize=8, va='top')
                ax[i,0].text(1.05, 0.6, f"spatial_frequency={stimulus[i,1]}", transform=ax[i,0].transAxes, fontsize=8, va='top')
                ax[i,0].text(1.05, 0.4, f"phase={stimulus[i,2]}", transform=ax[i,0].transAxes, fontsize=8, va='top')
            ax[i,0].axis("off")
            ax[i,1].imshow(activation[i].T,aspect="auto")
            ax[i,1].set_xlabel("Timestep")
            ax[i,1].set_ylabel("Unit")
    else:
        fig, ax = plt.subplots(ncols=2,figsize=(8,6))
        if stimulus.ndim==3:
            ax[0].imshow(stimulus[samples[0]],cmap="gray")
        else:
            stim = generate_gratings([stimulus[samples[0],0]],[stimulus[samples[0],1]],[stimulus[samples[0],2]])
            ax[0].imshow(stim[0],cmap="gray")
            ax[0].text(1.05, 0.8, f"orientation={stimulus[samples[0],0]}", transform=ax[0].transAxes, fontsize=10, va='top')
            ax[0].text(1.05, 0.6, f"spatial_frequency={stimulus[samples[0],1]}", transform=ax[0].transAxes, fontsize=10, va='top')
            ax[0].text(1.05, 0.4, f"phase={stimulus[samples[0],2]}", transform=ax[0].transAxes, fontsize=10, va='top')
        ax[0].axis("off")
        ax[1].imshow(activation[samples[0]].T,aspect="auto")
        ax[1].set_xlabel("Timestep")
        ax[1].set_ylabel("Unit")
    plt.tight_layout()
    plt.show()
