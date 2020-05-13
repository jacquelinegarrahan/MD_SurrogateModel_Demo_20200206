import os
import h5py

# set keras backend to tensorflow to prevent theano import errors
os.environ["KERAS_BACKEND"] = "tensorflow"

from online_model.util import fix_units

SCALAR_MODEL_FILE = "online_model/files/Scalar_NN_SurrogateModel.h5"
YAG_MODEL_FILE = "online_model/files/YAG_NN_SurrogateModel.h5"

# pva prefix
PREFIX = "smvm"

# load info for both models
SCALAR_MODEL_INFO = {}
with h5py.File(SCALAR_MODEL_FILE, "r") as h5:
    SCALAR_MODEL_INFO = dict(h5.attrs)

YAG_MODEL_INFO = {}
with h5py.File(YAG_MODEL_FILE, "r") as h5:
    YAG_MODEL_INFO = dict(h5.attrs)

# Start with the nice example from Lipi
DEFAULT_INPUTS = {
    "maxb(2)": 0.06125866317542922,
    "phi(1)": 8.351877669807294,
    "q_total": 0.020414630732101164,
    "sig_x": 0.4065596830730608,
}


# Set up pvdbs
CMD_PVDB = {}
for ii, input_name in enumerate(SCALAR_MODEL_INFO["input_names"]):
    CMD_PVDB[input_name] = {
        "type": "float",
        "prec": 8,
        "value": DEFAULT_INPUTS[input_name],
        "units": fix_units(SCALAR_MODEL_INFO["input_units"][ii]),
        "range": list(SCALAR_MODEL_INFO["input_ranges"][ii]),
    }

SIM_PVDB = {}
for ii, output_name in enumerate(SCALAR_MODEL_INFO["output_names"]):
    SIM_PVDB[output_name] = {
        "type": "float",
        "prec": 8,
        # "value": default_output[output_name],
        "units": fix_units(SCALAR_MODEL_INFO["output_units"][ii]),
    }

# sim_pvdb['z:pz']={'type': 'float', 'prec': 8, 'count':len(default_output['z:pz']),'units':'mm:delta','value':list(default_output['z:pz'])}
SIM_PVDB["x:y"] = {
    "type": "float",
    "prec": 8,
    # "count": len(default_output["x:y"]),
    "units": "mm:mm",
    # "value": list(default_output["x:y"]),
}


# Add in noise
# sim_pvdb["x_95coremit"]["scan"] = 0.2
# noise_params = {"x_95coremit": {"sigma": 0.5e-7, "dist": "uniform"}}
