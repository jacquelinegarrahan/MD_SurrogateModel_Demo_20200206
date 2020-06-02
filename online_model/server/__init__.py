from online_model.server import ca, pva
import pickle
import xarray as xr
from typing import Union


def pvdb_from_xarray(dset):
    input_pvdb = {}
    output_pvdb = {}

    for variable in dset.keys():

        entry = {
            "prec": dset[variable].attrs["precision"],
            "units": dset[variable].attrs["units"],
            "range": dset[variable].attrs["range"],
            "type": "float",  # For channel access
        }

        if dset[variable].attrs["is_input"]:

            # set values for the inputs
            if dset[variable].attrs["pv_type"] == "scalar":
                entry["value"] = dset[variable].values[
                    0
                ]  # Have to extract our scalar value from the xarray

            else:
                entry["value"] = dset[variable]

            input_pvdb[variable] = entry

        else:
            output_pvdb[variable] = entry

    return input_pvdb, output_pvdb


def pvdb_from_classes(variables):
    input_pvdb = {}
    output_pvdb = {}

    for variable in variables:
        # no manual formatting needed and have control over what is included/excluded
        entry = variable.dict(exclude_unset=True, exclude={"io_type"})

        if variable.io_type == "input":
            input_pvdb[variable.name] = entry

        elif variable.io_type == "output":
            output_pvdb[variable.name] = entry

        else:
            # pydantic enum validation will prohibit any other assignment
            pass

    return input_pvdb, output_pvdb


def get_server(
    prefix: str,
    model_class,
    model_kwargs: dict,
    protocol: str,
    data_file: str,
    from_xarray: bool = False,
) -> Union[ca.CAServer, pva.PVAServer]:
    # process data
    pickled_data = open(data_file, "rb")
    data = pickle.load(pickled_data)
    if from_xarray:
        input_pvdb, output_pvdb = pvdb_from_xarray(data)

    else:
        input_pvdb, output_pvdb = pvdb_from_classes(data)

    if protocol == "ca":
        server = ca.CAServer(model_class, model_kwargs, input_pvdb, output_pvdb, prefix)

    elif protocol == "pva":
        server = pva.PVAServer(
            model_class, model_kwargs, input_pvdb, output_pvdb, prefix
        )

    else:
        raise Exception("Must use ca or pva for protocol.")

    return server
