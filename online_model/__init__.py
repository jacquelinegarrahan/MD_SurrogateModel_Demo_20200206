import os
import h5py
import numpy as np
from online_model.util import fix_units, build_image_pvs

# set keras backend to tensorflow to prevent theano import errors
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["EPICS_CA_MAX_ARRAY_BYTES"] = "1000000"

# set protocol
PROTOCOL = os.environ.get("PROTOCOL")
if not PROTOCOL:
    raise ValueError("Protocol is not defined")

# pva prefix
PREFIX = "smvm"

MODEL_FILE = "online_model/files/CNN_060420_SurrogateModel.h5"
STOCK_LASER_IMAGE = "online_model/files/example_input_image.npy"

# pva prefix
PREFIX = "smvm"

# Build model info
MODEL_INFO = {}
with h5py.File(MODEL_FILE, "r") as h5:
    MODEL_INFO = dict(h5.attrs)

DEFAULT_LASER_IMAGE = np.load(STOCK_LASER_IMAGE)
DEFAULT_INPUTS_SCALARS = [
    3.47986980e-01,
    4.02751972e-02,
    -7.99101687e00,
    1.41576322e02,
    -3.53964583e-04,
    3.44330666e-04,
    -3.47874295e-04,
    3.45778376e-04,
]

DEFAULT_INPUTS = dict(zip(MODEL_INFO["input_ordering"], DEFAULT_INPUTS_SCALARS))
DEFAULT_INPUTS["image"] = DEFAULT_LASER_IMAGE


# TEMPORARY FIX FOR SAME NAME INPUT/OUTPUT VARS
REDUNDANT_INPUT_OUTPUT = ["xmin", "xmax", "ymin", "ymax"]

# sliders to exclude
EXCLUDE_SLIDERS = ["in_" + input_name for input_name in REDUNDANT_INPUT_OUTPUT]

# Set up pvdbs
DEFAULT_PRECISION = 8
DEFAULT_COLOR_MODE = 0


CMD_PVDB = {}
for ii, input_name in enumerate(MODEL_INFO["input_names"]):
    label = input_name
    if input_name in REDUNDANT_INPUT_OUTPUT:
        label = f"in_{input_name}"

    CMD_PVDB[label] = {
        "type": "float",
        "prec": DEFAULT_PRECISION,
        "value": DEFAULT_INPUTS[input_name],
        "units": fix_units(MODEL_INFO["input_units"][ii]),
        "range": list(MODEL_INFO["input_ranges"][ii]),
    }

SIM_PVDB = {}
for ii, output_name in enumerate(MODEL_INFO["output_names"]):
    label = output_name
    if output_name in REDUNDANT_INPUT_OUTPUT:
        label = f"out_{output_name}"

    SIM_PVDB[label] = {
        "type": "float",
        "prec": DEFAULT_PRECISION,
        # "value": default_output[output_name],
        "units": fix_units(MODEL_INFO["output_units"][ii]),
    }


# sim_pvdb['z:pz']={'type': 'float', 'prec': 8, 'count':len(default_output['z:pz']),'units':'mm:delta','value':list(default_output['z:pz'])}

IMAGE_SHAPE = np.array([50, 50])
IMAGE_UNITS = "mm:mm"

# TODO: ASSIGN START FOR PVA
if PROTOCOL == "pva":
    SIM_PVDB["x:y"] = {
        "type": "float",
        "prec": 8,
        # "count": len(default_output["x:y"]),
        "units": IMAGE_UNITS,
    }
elif PROTOCOL == "ca":
    image_pvs = build_image_pvs(
        "x:y",  # pvame
        IMAGE_SHAPE,
        IMAGE_UNITS,  # get units
        DEFAULT_PRECISION,
        DEFAULT_COLOR_MODE,
    )
    SIM_PVDB.update(image_pvs)

ARRAY_PVS = ["x:y"]

MODEL_KWARGS = {"model_file": MODEL_FILE, "stock_image_input": DEFAULT_LASER_IMAGE}
