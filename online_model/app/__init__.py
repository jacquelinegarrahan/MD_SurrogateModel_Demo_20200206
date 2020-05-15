from argparse import ArgumentParser
from functools import partial
from bokeh.models import Slider
import numpy as np

from epics import caget, caput
from p4p.client.thread import Context

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
PROTOCOL = args.protocol[0]

# initialize context for pva
CONTEXT = None
if PROTOCOL == "pva":
    CONTEXT = Context("pva")


def set_pv_from_slider(attr, old, new, pvname: str, scale) -> None:
    """
    Callback function for slider change.

    Parameters
    ----------
    attr

    old

    new

    pvname: str
        Process variable name

    scale:

    """
    if PROTOCOL == "pva":
        CONTEXT.put(pvname, new * scale)

    elif PROTOCOL == "ca":
        caput(self.pvname, new * scale)


def build_slider(title: str, pvname, scale, start, end, step) -> Slider:
    """
    Utility function for building a slider.

    Parameters
    ----------
    title:str
        Slider title

    pvname:str
        Process variable name

    scale:float/int
        Scale of the slider

    start:float
        Lower range of the slider

    end:float
        Upper range of the slider

    step:np.float64
        The step between consecutive values

    Returns
    -------
    bokeh.models.widgets.sliders.Slider

    """

    # initialize value
    start_val = None
    if PROTOCOL == "pva":
        start_val = CONTEXT.get(pvname)
    elif PROTOCOL == "ca":
        start_val = caget(pvname)

    slider = Slider(
        title=title, value=scale * start_val, start=start, end=end, step=step
    )

    slider.on_change("value", partial(set_pv_from_slider, pvname=pvname, scale=scale))

    return slider
