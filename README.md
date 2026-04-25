# ift-6168-project
Causal Feature Learning (CFL) application

## Dataset creation

The first step to this project is to build the dataset we will use to apply Algo 1. onto. The first version of this 
dataset is now obtainable (with many hyperparameter to figure out). Here are the steps to acquire the dataset:

Note that you can use the notebook at `data/explore_datasets.ipynb` once the steps 1 to 3 were completed to visualize the
different dataset versions you have built.

1. Download ophys_experiment_data (not all of it)

    We only need a few experiments as they contain a lot of data and downloading the entirety of the data available would
    take around 800 GB of memory.
    
    ```bash
    python data/download_ophys_experiment_data.py
    ```

2. Build natural scenes dataset (different versions)

    You can modify the DOWNSAMPLE variable in the `main()` function and choose which task to run with the --task 
    argument.

    Default --task = "full_images_dataset":

    ```bash
    python data/build_natural_scenes_dataset.py
    ```
   
   Other tasks:

    ```bash
   # Build dataset with neuronal activations averaged across every single image
    python data/build_natural_scenes_dataset.py --task averaged_images_dataset
   
   # or
   
   # Check statistics across unique images
   python data/build_natural_scenes_dataset.py --task unique_images_analysis
    ```
   
3. Build static gratings dataset (different versions)

    Modify hyperparameter in `main()` function.

    ```bash
    python data/build_static_gratings_dataset.py
    ```
   
4. TODO Choose which dataset versions to use and build official dataset built upon natural scenes and static gratings
images and their correspond neuronal signals across time.



## Details

I tried using Python3.12 but ran into trouble when downloading the allensdk library. I recommend using Python3.11 or 
less.
