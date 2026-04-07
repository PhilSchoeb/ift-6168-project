PATH_TO_MANIFEST = "brain_observatory/manifest.json"

import os
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

path_to_manifest = os.path.join(os.curdir, PATH_TO_MANIFEST)

boc = BrainObservatoryCache(manifest_file=path_to_manifest)

