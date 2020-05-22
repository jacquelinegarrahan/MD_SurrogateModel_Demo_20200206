import copy
import functools
from p4p.nt.ndarray import ntndarray as NTNDArrayData
from online_model import REDUNDANT_INPUT_OUTPUT, PROTOCOL, IMAGE_PVS

# Some input/output variables have the same name and must be unique.
# Below are utility functions to fix this:


def apply_temporary_ordering_patch(ordering, prefix):
    # TEMPORARY PATCH FOR INPUT/OUTPUT REDUNDANT VARS
    rebuilt_order = copy.copy(ordering)
    for i, val in enumerate(ordering):
        if val in REDUNDANT_INPUT_OUTPUT:
            rebuilt_order[i] = f"{prefix}_{val}"

    return rebuilt_order


def apply_temporary_output_patch(output):
    # TEMPORARY PATCH FOR OUTPUT ORDERING
    rebuilt_output = {}
    for item in output:
        if item in REDUNDANT_INPUT_OUTPUT:
            rebuilt_output[f"out_{item}"] = output[item]

        else:
            rebuilt_output[item] = output[item]
    return rebuilt_output


def format_outputs_by_protocol(f):
    def format_wrapper(*args, **kwargs):
        output_state = f(*args, **kwargs)

        rebuilt_output = {}
        if PROTOCOL == "ca":
            for pv, value in output_state.items():
                if pv in IMAGE_PVS:
                    rebuilt_output[f"{pv}.ArrayData_RBV"] = value
                else:
                    rebuilt_output[pv] = value

        elif PROTOCOL == "pva":
            for pv, value in output_state.items():
                if pv in IMAGE_PVS:
                    # populate image data
                    array_data = value.view(NTNDArrayData)

                    # get dw and dh from model output
                    array_data.attrib = {
                        # "ColorMode": DEFAULT_COLOR_MODE,
                        "dw": output_state[f"{pv}.dw"],
                        "dh": output_state[f"{pv}.dh"],
                    }
                    rebuilt_output[pv] = array_data

                # do not build attribute pvs
                elif not ".dw" in pv and not ".dh" in pv:
                    rebuilt_output[pv] = value

        return rebuilt_output

    return format_wrapper
