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

from online_model import SIM_PVDB
from online_model.app import PlotController

# Set up the controller for the plot
plot_controller = PlotController(SIM_PVDB)
plot_controller.build_plot()
current_pv = plot_controller.current_pv

# Set up select option
select = Select(
    title="PV to Plot:",
    value=current_pv,
    options=list(plot_controller.pvmonitors.keys()),
)

# Set up selection callback
def pv_select_callback(attr, old, new):
    """
    Update global current process variable.
    """
    global current_pv
    current_pv = new


select.on_change("value", pv_select_callback)

# Set up periodic plot callback
def plot_callback():
    """
    Calls plot controller update with the current global process variable
    """
    global current_pv
    plot_controller.update(current_pv)


# Set up page
curdoc().title = "Online Surrogate Model Strip Tool"
curdoc().add_root(column(row(select), row(plot_controller.p), width=300))
curdoc().add_periodic_callback(plot_callback, 250)
