import numpy as np


def fix_units(unit_str):

    unit_str = unit_str.strip()
    if len(unit_str.split(" ")) > 1:
        unit_str = unit_str.split(" ")[-1]
    unit_str = unit_str.replace("(", "")
    unit_str = unit_str.replace(")", "")

    return unit_str


def build_image_pvs(pvname, image, image_units, dw, dh, precision, color_mode):
    flattened_image = image.flatten()

    # confirm dimensions make sense
    assert image.ndim > 0

    # assign default PVS
    pvdb = {
        f"{pvname}.NDimensions_RBV": {
            "type": "float",
            "prec": precision,
            "value": image.ndim,
        },
        f"{pvname}.Dimensions_RBV": {
            "type": "float",
            "prec": precision,
            "count": image.ndim,
            "value": np.array(image.shape),
        },
        f"{pvname}.ArraySizeX_RBV": {"type": "int", "value": image.shape[0]},
        f"{pvname}.ArraySize_RBV": {"type": "int", "value": flattened_image.shape[0]},
        f"{pvname}.ArrayData_RBV": {
            "type": "float",
            "prec": precision,
            "count": len(flattened_image),
            "value": flattened_image,
            "units": image_units,
        },
        f"{pvname}.ColorMode_RBV": {"type": "int", "value": color_mode},
        f"{pvname}.AttributeList": {
            "type": "char",
            "value": np.array(["dw", "dh"]),
            "count": 2,
        },
        f"{pvname}.dw": {"type": "float", "prec": precision, "value": dw},
        f"{pvname}.dh": {"type": "float", "prec": precision, "value": dh},
    }

    # assign dimension specific pvs
    if image.ndim > 1:
        pvdb[f"{pvname}.ArraySizeY_RBV"]: {"type": "int", "value": image.shape[1]}

    if image.ndim > 2:
        pvdb[f"{pvname}.ArraySizeZ_RBV"] = {"type": "int", "value": image.shape[2]}

    return pvdb
