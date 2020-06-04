import sys
import os

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models.widgets import Select

# fix for bokeh path error, maybe theres a better way to do this
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

import epics

# fix for bug that made vars unreachable...
# need to find origin
epics.ca.initialize_libca()

from online_model.app.controllers import Controller
from online_model.app.widgets.plots import Striptool
from online_model.app import parse_args
from online_model import ARRAY_PVS

PROTOCOL, CMD_PVDB, SIM_PVDB = parse_args()

# server prefix
PREFIX = "smvm"

# exclude channel access data items from plots
PLOT_PVDB = {
    item: value for item, value in SIM_PVDB.items() if "units" in SIM_PVDB[item]
}

# create controller
controller = Controller(PROTOCOL)

# Set up the controller for the plot
striptool = Striptool(PLOT_PVDB, controller, ARRAY_PVS, PREFIX)
striptool.build_plot()
current_pv = striptool.current_pv

# Set up select option
select = Select(
    title="PV to Plot:", value=current_pv, options=list(striptool.pv_monitors.keys())
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
    striptool.update(current_pv)


# Set up page
curdoc().title = "Online Surrogate Model Strip Tool"
curdoc().add_root(column(row(select), row(striptool.p), width=300))
curdoc().add_periodic_callback(plot_callback, 250)
