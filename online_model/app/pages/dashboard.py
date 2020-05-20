import numpy as np
import copy
import sys
import os
from functools import partial

from epics import caget, PV
from p4p.client.thread import Context
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import Select
from bokeh.io import curdoc
from bokeh.layouts import column, row, Spacer

from bokeh import palettes, colors

# fix for bokeh path error, maybe theres a better way to do this
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

from online_model import PREFIX, SIM_PVDB, CMD_PVDB
from online_model.app.controllers import Controller
from online_model.app.widgets.sliders import build_sliders
from online_model.app.widgets.plots import ImagePlot
from online_model.app.widgets.plots import Striptool
from online_model.app.widgets.tables import ValueTable
from online_model.app import get_protocol

# need surrogate model image processing methods
from online_model.model.MySurrogateModel import MySurrogateModel

# get protocol
PROTOCOL = get_protocol()

# create controller
controller = Controller(PROTOCOL)

# Create custom palette with low values set to white
pal = list(palettes.viridis(244))  # 256 - 12 (set lowest 5% to white)
pal = ["#FFFFFF"] * 12 + pal
pal = tuple(pal)

# set up plot
image_plot = ImagePlot(SIM_PVDB, controller, MySurrogateModel)
image_plot.build_plot(pal)

# set current_pv globally
current_image_pv = image_plot.current_pv

# set up image toggle
image_select = Select(
    title="Image PV",
    value=current_image_pv,
    options=list(image_plot.pv_monitors.keys()),
)


def on_image_selection(attrname, old, new):
    """
    Callback function for dropdown selection that updates the global current variable.
    """
    global current_image_pv
    current_image_pv = new


image_select.on_change("value", on_image_selection)

# Set up image update callback
def image_callback():
    """
    Calls plot controller update with the current global process variable
    """
    global current_image_pv
    image_plot.update(current_image_pv)


# build sliders for the command process variable database
# TEMPORARILY EXCLUDE THE EXTENTS
sliders_to_render = {}
for var, value in CMD_PVDB.items():
    if "in_" not in var:
        sliders_to_render[var] = value

sliders = build_sliders(CMD_PVDB, controller)
slider_col = column(sliders, width=350)

# Set up the controller for the plot
striptool = Striptool(SIM_PVDB, controller)
striptool.build_plot()

# set up global pv
current_striptool_pv = striptool.current_pv

# set up selection
def striptool_select_callback(attr, old, new):
    global current_striptool_pv
    current_striptool_pv = new


striptool_select = Select(
    title="PV to Plot:",
    value=current_striptool_pv,
    options=list(striptool.pv_monitors.keys()),
)
striptool_select.on_change("value", striptool_select_callback)

# add table
value_table = ValueTable(SIM_PVDB, controller)

# Set up plot and table update callback
def data_callback():
    """
    Calls plot controller update with the current global process variable
    and updates the value table.
    """
    global current_striptool_pv

    striptool.update(current_striptool_pv)
    value_table.update()


# Set up the document
curdoc().title = "Online Surrogate Model Virtual Machine"

curdoc().add_root(
    column(
        row(slider_col, Spacer(width=50), value_table.table),  # add sliders
        row(
            column(image_select, image_plot.p),  # add image controls
            column(striptool_select, striptool.p),  # add plot controls
        ),
        width=300,
    )
)

curdoc().add_periodic_callback(image_callback, 250)
curdoc().add_periodic_callback(data_callback, 250)
