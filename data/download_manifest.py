"""
Once the virtual environment is set up and allensdk is installed, run this file from the data folder to download the
manifest at `repo_root/data/brain_observatory/manifest.json`
"""

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

boc = BrainObservatoryCache()