import copy
import functools

# TEMPORARY FIX FOR SAME NAME INPUT/OUTPUT VARS
REDUNDANT_INPUT_OUTPUT = ["xmin", "xmax", "ymin", "ymax"]


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
