import numpy as np
import math
from argparse import ArgumentParser

import json

from epics import caget, caput
from p4p.client.thread import Context

from bokeh.driving import count
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models.glyphs import MultiLine
from bokeh.models.glyphs import VArea

from online_model import CMD_PVDB, PREFIX

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


class pv_slider:
    def __init__(self, title, pvname, scale, start, end, step):

        self.pvname = pvname
        self.scale = scale

        # initialize value
        if PROTOCOL == "pva":
            start_val = CONTEXT.get(pvname)
        elif PROTOCOL == "ca":
            start_val = caget(pvname)

        # TODO : Catch exception if no value returned

        self.slider = Slider(
            title=title, value=scale * start_val, start=start, end=end, step=step
        )

        self.slider.on_change("value", self.set_pv_from_slider)

    def set_pv_from_slider(self, attrname, old, new):
        if PROTOCOL == "pva":
            CONTEXT.put(self.pvname, new * self.scale)

        elif PROTOCOL == "ca":
            caput(self.pvname, new * self.scale)


sliders = []

for ii, pv in enumerate(CMD_PVDB):

    title = pv + " (" + CMD_PVDB[pv]["units"] + ")"
    pvname = PREFIX + ":" + pv
    step = (CMD_PVDB[pv]["range"][1] - CMD_PVDB[pv]["range"][0]) / 100.0

    pvs = pv_slider(
        title=title,
        pvname=pvname,
        scale=1,
        start=CMD_PVDB[pv]["range"][0],
        end=CMD_PVDB[pv]["range"][1],
        step=step,
    )
    sliders.append(pvs.slider)

scol = column(sliders, width=350)
curdoc().add_root(row(scol))
curdoc().title = "Online Surrogate Model Virtual Machine"
