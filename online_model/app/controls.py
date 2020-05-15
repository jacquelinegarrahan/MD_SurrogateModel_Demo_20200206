import numpy as np
from argparse import ArgumentParser

from epics import caget, caput
from p4p.client.thread import Context

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.layouts import column, row

from online_model import CMD_PVDB, PREFIX
from online_model.app import build_slider


sliders = []

for ii, pv in enumerate(CMD_PVDB):
    title = pv + " (" + CMD_PVDB[pv]["units"] + ")"
    pvname = PREFIX + ":" + pv
    step = (CMD_PVDB[pv]["range"][1] - CMD_PVDB[pv]["range"][0]) / 100.0
    scale = 1

    slider = build_slider(
        title, pvname, scale, CMD_PVDB[pv]["range"][0], CMD_PVDB[pv]["range"][1], step
    )
    sliders.append(slider)

scol = column(sliders, width=350)
curdoc().add_root(row(scol))
curdoc().title = "Online Surrogate Model Virtual Machine"
