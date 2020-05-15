import numpy as np
import time
import sys
import os
from functools import partial
from argparse import ArgumentParser

from epics import caget, caput, PV
from p4p.client.thread import Context

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models.widgets import Select

# fix for bokeh path error, maybe theres a better way to do this
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

from online_model import PREFIX, SIM_PVDB, ARRAY_PVS
from online_model.app import PlotController


plot_controller = PlotController(SIM_PVDB)
plot_controller.build_plot()
current_pv = plot_controller.current_pv

# set up selection
def pv_select_callback(attr, old, new):
    global current_pv
    current_pv = new


select = Select(
    title="PV to Plot:",
    value=current_pv,
    options=list(plot_controller.pvmonitors.keys()),
)
select.on_change("value", pv_select_callback)

scol = column(row(select), row(plot_controller.p), width=350)
curdoc().add_root(scol)
curdoc().add_periodic_callback(partial(plot_controller.update, current_pv), 250)
curdoc().title = "Online Surrogate Model Strip Tool"
