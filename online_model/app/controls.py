import numpy as np
import math

import json

from epics import caget, caput

from bokeh.driving import count
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models.glyphs import MultiLine
from bokeh.models.glyphs import VArea


class pv_slider:
    def __init__(self, title, pvname, scale, start, end, step):

        self.pvname = pvname
        self.scale = scale

        self.slider = Slider(
            title=title, value=scale * caget(pvname), start=start, end=end, step=step
        )
        self.slider.on_change("value", self.set_pv_from_slider)

    def set_pv_from_slider(self, attrname, old, new):
        caput(self.pvname, new * self.scale)


# Controls looks for a pvdef file written by the online model server
with open("pvdef.json", "r") as fp:
    pvdefs = json.load(fp)

prefix = pvdefs["prefix"]
sliders = []
for pv in pvdefs["input"]:

    pvdef = pvdefs["input"][pv]
    title = pv + " (" + pvdef["units"] + ")"
    pvname = prefix + pv

    pvrange = pvdef["range"]
    step = (pvrange[-1] - pvrange[0]) / 100.0

    pvs = pv_slider(
        title=title, pvname=pvname, scale=1, start=pvrange[0], end=pvrange[1], step=step
    )
    sliders.append(pvs.slider)

scol = column(sliders, width=350)
curdoc().add_root(row(scol))

# curdoc().add_periodic_callback(update, 250)
curdoc().title = "Online Surrogate Model Virtual Machine"
