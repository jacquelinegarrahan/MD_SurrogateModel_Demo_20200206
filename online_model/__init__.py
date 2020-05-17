import os
import h5py
import numpy as np

# set keras backend to tensorflow to prevent theano import errors
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["EPICS_CA_MAX_ARRAY_BYTES"] = "1000000"

from online_model.util import fix_units

# SCALAR_MODEL_FILE = "online_model/files/Scalar_NN_SurrogateModel.h5"
# YAG_MODEL_FILE = "online_model/files/YAG_NN_SurrogateModel.h5"
MODEL_FILE = "online_model/files/CNN_051620_SurrogateModel.h5"
STOCK_LASER_IMAGE = "online_model/files/example_input_image.npy"
STOCK_INPUT_SCALARS = "online_model/files/example_input_scalars.npy"

# pva prefix
PREFIX = "smvm"

# load info for both models
# SCALAR_MODEL_INFO = {}
# with h5py.File(SCALAR_MODEL_FILE, "r") as h5:
#     SCALAR_MODEL_INFO = dict(h5.attrs)

# YAG_MODEL_INFO = {}
# with h5py.File(YAG_MODEL_FILE, "r") as h5:
#     YAG_MODEL_INFO = dict(h5.attrs)

MODEL_INFO = {}
with h5py.File(MODEL_FILE, "r") as h5:
    MODEL_INFO = dict(h5.attrs)

# Start with the nice example from Lipi

DEFAULT_LASER_IMAGE = np.load(STOCK_LASER_IMAGE)
DEFAULT_INPUTS_SCALARS = [ 3.47986980e-01,  4.02751972e-02, -7.99101687e+00,  1.41576322e+02,
 -3.53964583e-04,  3.44330666e-04, -3.47874295e-04,  3.45778376e-04]

DEFAULT_INPUTS = dict(zip(MODEL_INFO["input_ordering"], DEFAULT_INPUTS_SCALARS))
DEFAULT_INPUTS["image"] = DEFAULT_LASER_IMAGE


# Set up pvdbs
CMD_PVDB = {}
for ii, input_name in enumerate(MODEL_INFO["input_names"]):
    CMD_PVDB[input_name] = {
        "type": "float",
        "prec": 8,
        "value": DEFAULT_INPUTS[input_name],
        "units": fix_units(MODEL_INFO["input_units"][ii]),
        "range": list(MODEL_INFO["input_ranges"][ii]),
    }

SIM_PVDB = {}
for ii, output_name in enumerate(MODEL_INFO["output_names"]):
    SIM_PVDB[output_name] = {
        "type": "float",
        "prec": 8,
        # "value": default_output[output_name],
        "units": fix_units(MODEL_INFO["output_units"][ii]),
    }

# sim_pvdb['z:pz']={'type': 'float', 'prec': 8, 'count':len(default_output['z:pz']),'units':'mm:delta','value':list(default_output['z:pz'])}
SIM_PVDB["x:y"] = {
    "type": "float",
    "prec": 8,
    # "count": len(default_output["x:y"]),
    "units": "mm:mm",
    # "value": list(default_output["x:y"]),
}

ARRAY_PVS = ["x:y"]
# Add in noise
# sim_pvdb["x_95coremit"]["scan"] = 0.2
# noise_params = {"x_95coremit": {"sigma": 0.5e-7, "dist": "uniform"}}

# dims
DIMS = [50, 50]
