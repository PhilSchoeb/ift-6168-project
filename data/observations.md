# Observations about the data from the brain observatory

## Specifications about the causal multi-level system

1. Input I causes output J
2. I and J are discrete

> But how do we define I and J in our context with data from the observatory ?

## The data

There are a lot of informations.

### What could be in I:

#### Metadata (with examples)
- sex: "male",
- targeted_structure: "VISp",
- excitation_lambda: "910 nanometers",
- indicator: "GCaMP6f",
- fov: '400x400 microns (512 x 512 pixels)',
- genotype: 'Cux2-CreERT2/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/Ai93(TITL-GCaMP6f)',
- session_start_time: datetime.datetime(2016, 2, 4, 10, 25, 24),
- session_type: 'three_session_B',
- specimen_name: 'Cux2-CreERT2;Camk2a-tTA;Ai93-222426',
- cre_line: 'Cux2-CreERT2/wt',
- imaging_depth_um: 175,
- age_days: 104,
- device: 'Nikon A1R-MP multiphoton microscope',
- device_name: 'CAM2P.2',
- pipeline_version: '3.0',
- stimulus: "static_gratings", "natural_scenes", "spontaneous", "natural_movie_one", etc.

#### Visual data
- Images of shape \[1174 x 918\] for \[x_axis x y_axis\] (example)

### What could be in J:

#### Time series
- dff traces (seems to represent neurotic activity) shape \[174 x 113888\] for \[num_neurons x time_unit\]
- running speed
- events (not sure what this represents)
- pupil diameter
- lick times (not sure)
- reward times (not sure once again, this might go in I instead)

### What will be used to build our dataset
- Time (to build time windows)

## Different avenues

1. Consider data for one mouse only to find I and J for that single mouse across wide variety of data
2. Consider many mice (but will have to reduce dataset features considered)