import numpy as np
import copy
import sys
import os
from functools import partial

from epics import caget, PV
from p4p.client.thread import Context

from bokeh.models.widgets import Select
from bokeh.io import curdoc
from bokeh.layouts import column, row

from bokeh import palettes, colors

# fix for bokeh path error, maybe theres a better way to do this
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

from online_model import PREFIX, SIM_PVDB, CMD_PVDB
from online_model.app import PlotController, build_sliders

pal = palettes.Viridis[256]
white = colors.named.white

# set up plot controller
plot_controller = PlotController(SIM_PVDB)
plot_controller.build_plot(pal)

# set current_pv globally
current_pv = plot_controller.current_pv

# set up selection toggle
select = Select(
    title="Image PV", value=current_pv, options=list(plot_controller.pvimages.keys())
)


def on_selection(attrname, old, new):
    """
    Callback function for dropdown selection that updates the global current variable.
    """
    global current_pv
    current_pv = new


select.on_change("value", on_selection)

# build sliders for the command process variable database
sliders = build_sliders(CMD_PVDB)
scol = column(sliders, width=350)

curdoc().title = "Online Surrogate Model Virtual Machine"

curdoc().add_root(column(row(scol), row(select), row(plot_controller.p), width=300))
curdoc().add_periodic_callback(partial(plot_controller.update, current_pv), 250)
