import sys
import os

from bokeh.models.widgets import Select
from bokeh.io import curdoc
from bokeh.layouts import column, row

from bokeh import palettes

# fix for bokeh path error, maybe theres a better way to do this
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

from online_model import PREFIX, SIM_PVDB
from online_model.app.controllers import Controller
from online_model.app.widgets.plots import ImagePlot
from online_model.app import get_protocol

# need surrogate model image processing methods
from online_model.model.MySurrogateModel import MySurrogateModel

# get protocol
PROTOCOL = get_protocol()

# create controller
controller = Controller(PROTOCOL)

# Create custom palette with low values set to white
white = colors.named.white
pal = list(palettes.viridis(244))  # 256 - 12 (set lowest 5% to white)
pal = ["#FFFFFF"] * 12 + pal
pal = tuple(pal)

# set up plot
image_plot = ImagePlot(SIM_PVDB, controller, MySurrogateModel)
image_plot.build_plot(pal)

# set current_pv globally
current_pv = image_plot.current_pv

# set up selection toggle
select = Select(
    title="Image PV", value=current_pv, options=list(image_plot.pv_monitors.keys())
)

# Set up selection callback
def on_selection(attrname, old, new):
    """
    Callback function for dropdown selection that updates the global current variable.
    """
    global current_pv
    current_pv = new


select.on_change("value", on_selection)

# Set up image update callback
def image_callback():
    """
    Calls plot controller update with the current global process variable
    """
    global current_pv
    image_plot.update(current_pv)


curdoc().title = "Online Surrogate Model Image Viewer"
curdoc().add_root(column(row(select), row(image_plot.p), width=300))
curdoc().add_periodic_callback(image_callback, 250)
