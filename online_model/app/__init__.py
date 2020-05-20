import copy
import time

from argparse import ArgumentParser
from functools import partial
import numpy as np

from bokeh.plotting import figure
from bokeh.models import Slider
from bokeh.models import ColumnDataSource

from epics import caget, caput
from p4p.client.thread import Context

from online_model import PREFIX, ARRAY_PVS


def get_protocol():
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
    args = parser.parse_args()

    # get protocol
    return args.protocol[0]
