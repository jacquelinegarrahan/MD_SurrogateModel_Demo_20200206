from argparse import ArgumentParser

# These should go away when we stop using pvbds entirely
from online_model.server import pvdb_from_classes
from online_model.server import pvdb_from_xarray

import pickle


def load_data(data_file, from_xarray, protocol):
    pickled_data = open(data_file, "rb")
    data = pickle.load(pickled_data)

    if from_xarray:
        input_pvdb, output_pvdb = pvdb_from_xarray(data, protocol)

    else:
        input_pvdb, output_pvdb = pvdb_from_classes(data, protocol)

    return input_pvdb, output_pvdb


def parse_args():
    # Parse arguments passed through bokeh serve
    # requires protocol to be set
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--protocol",
        metavar="PROTOCOL",
        nargs=1,
        type=str,
        choices=["pva", "ca"],
        help="Protocol to use (ca, pva)",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--file",
        nargs=1,
        type=str,
        help="Data file to load variables.",
        required=True,
    )
    parser.add_argument("--from-xarray", default=False, action="store_true")
    parser.add_argument("--not-xarray", dest="from-xarray", action="store_false")
    args = parser.parse_args()

    protocol = args.protocol[0]
    data_file = args.file[0]
    from_xarray = args.from_xarray
    input_pvdb, output_pvdb = load_data(data_file, from_xarray, protocol)

    return protocol, input_pvdb, output_pvdb
