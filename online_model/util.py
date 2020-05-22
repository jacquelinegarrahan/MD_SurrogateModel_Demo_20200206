import numpy as np


def fix_units(unit_str):

    unit_str = unit_str.strip()
    if len(unit_str.split(" ")) > 1:
        unit_str = unit_str.split(" ")[-1]
    unit_str = unit_str.replace("(", "")
    unit_str = unit_str.replace(")", "")

    return unit_str


def build_image_pvs(pvname, image_shape, image_units, precision, color_mode):
    ndim = len(image_shape)

    # confirm dimensions make sense
    assert ndim > 0

    # assign default PVS
    pvdb = {
        f"{pvname}.NDimensions_RBV": {
            "type": "float",
            "prec": precision,
            "value": ndim,
        },
        f"{pvname}.Dimensions_RBV": {
            "type": "int",
            "prec": precision,
            "count": ndim,
            "value": image_shape,
        },
        f"{pvname}.ArraySizeX_RBV": {"type": "int", "value": image_shape[0]},
        f"{pvname}.ArraySize_RBV": {"type": "int", "value": int(np.prod(image_shape))},
        f"{pvname}.ArrayData_RBV": {
            "type": "float",
            "prec": precision,
            "count": int(np.prod(image_shape)),
            "units": image_units,
        },
        f"{pvname}.ColorMode_RBV": {"type": "int", "value": color_mode},
        f"{pvname}.AttributeList": {
            "type": "string",
            "value": ["dw", "dh"],
            "count": 2,
        },
        f"{pvname}.dw": {"type": "float", "prec": precision},
        f"{pvname}.dh": {"type": "float", "prec": precision},
    }

    # assign dimension specific pvs
    if ndim > 1:
        pvdb[f"{pvname}.ArraySizeY_RBV"] = {"type": "int", "value": image_shape[1]}

    if ndim > 2:
        pvdb[f"{pvname}.ArraySizeZ_RBV"] = {"type": "int", "value": image_shape[2]}

    return pvdb
