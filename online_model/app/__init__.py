from argparse import ArgumentParser
from functools import partial
from bokeh.models import Slider

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


def set_pv_from_slider(attrname, old, new, pvname, scale):
    if PROTOCOL == "pva":
        CONTEXT.put(pvname, new * scale)

    elif PROTOCOL == "ca":
        caput(self.pvname, new * scale)


def build_slider(title, pvname, scale, start, end, step):

    # initialize value
    start_val = None
    if PROTOCOL == "pva":
        start_val = CONTEXT.get(pvname)
    elif PROTOCOL == "ca":
        start_val = caget(pvname)

    slider = Slider(
        title=title, value=scale * start_val, start=start, end=end, step=step
    )

    slider.on_change("value", partial(set_pv_from_slider, pvname, scale))

    return slider
