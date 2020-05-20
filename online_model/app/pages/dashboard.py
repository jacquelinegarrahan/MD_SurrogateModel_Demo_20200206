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
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

from online_model import PREFIX, SIM_PVDB, CMD_PVDB
from online_model.app.widgets.sliders import build_sliders
from online_model.app.widgets.controllers import (
    ImageController,
    PlotController,
    TableController,
)

# Create custom palette with low values set to white
pal = list(palettes.viridis(244))  # 256 - 12 (set lowest 5% to white)
pal = ["#FFFFFF"] * 12 + pal
pal = tuple(pal)

# set up plot controller
image_controller = ImageController(SIM_PVDB)
image_controller.build_plot(pal)

# set current_pv globally
current_image_pv = image_controller.current_pv

# set up image toggle
image_select = Select(
    title="Image PV",
    value=current_image_pv,
    options=list(image_controller.pv_monitors.keys()),
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
    image_controller.update(current_image_pv)


# build sliders for the command process variable database
# TEMPORARILY EXCLUDE THE EXTENTS
sliders_to_render = {}
for var, value in CMD_PVDB.items():
    if "in_" not in var:
        sliders_to_render[var] = value

sliders = build_sliders(sliders_to_render)
slider_col = column(sliders, width=350)

# set up striptool
plot_controller = PlotController(SIM_PVDB)
plot_controller.build_plot()
current_striptool_pv = plot_controller.current_pv

# set up selection
def striptool_select_callback(attr, old, new):
    global current_striptool_pv
    current_striptool_pv = new


striptool_select = Select(
    title="PV to Plot:",
    value=current_striptool_pv,
    options=list(plot_controller.pv_monitors.keys()),
)
striptool_select.on_change("value", striptool_select_callback)

# add table
table_controller = TableController(SIM_PVDB)

# Set up plot and table update callback
def data_callback():
    """
    Calls plot controller update with the current global process variable
    """
    global current_striptool_pv
    plot_controller.update(current_striptool_pv)

    table_controller.update()


# Set up the document
curdoc().title = "Online Surrogate Model Virtual Machine"

curdoc().add_root(
    column(
        row(slider_col, Spacer(width=50), table_controller.table),  # add sliders
        row(
            column(image_select, image_controller.p),  # add image controls
            column(striptool_select, plot_controller.p),  # add plot controls
        ),
        width=300,
    )
)

curdoc().add_periodic_callback(image_callback, 250)
curdoc().add_periodic_callback(data_callback, 250)
