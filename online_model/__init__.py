import os
import h5py
import numpy as np
from online_model.util import fix_units, build_image_pvs

# set keras backend to tensorflow to prevent theano import errors
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["EPICS_CA_MAX_ARRAY_BYTES"] = "1000000"
# Set up pvdbs
DEFAULT_PRECISION = 8
DEFAULT_COLOR_MODE = 0

ARRAY_PVS = ["x:y", "image"]
