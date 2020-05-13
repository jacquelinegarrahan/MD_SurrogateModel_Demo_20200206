import numpy as np
import copy
from argparse import ArgumentParser

from epics import caget, PV
from p4p.client.thread import Context

from bokeh.plotting import figure
from bokeh.models.widgets import Select
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource

from bokeh import palettes, colors

from online_model import PREFIX, SIM_PVDB


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


pal = palettes.Viridis[256]
white = colors.named.white
# pal[0] = white # replace 0 with white


class PVImageMonitor:
    def __init__(self, pvname, units):
        self.pvname = pvname
        self.units = units

    def poll(self):
        if PROTOCOL == "ca":
            # self.pv = PV(pv, auto_monitor=True)
            data = caget(self.pvname)

        elif PROTOCOL == "pva":
            # context returns np array with WRITEABLE=False
            # copy to manipulate array below
            data = copy.copy(CONTEXT.get(self.pvname))

        nx = data[0]
        ny = data[1]
        ext = data[2:6]

        image = data[6:]
        image[np.where(image <= 0)] = 0  # Set negative to zero
        image = image.reshape(int(nx), int(ny))
        return (image, ext)

    def get_units(self):
        return self.units.split(":")

    def variables(self):
        return self.pvname.split(":")


pvimages = {}
for opv in SIM_PVDB:
    if len(SIM_PVDB[opv]["units"].split(":")) == 2:
        pvimages[opv] = PVImageMonitor(f"{PREFIX}:{opv}", SIM_PVDB[opv]["units"])

current_pv = list(pvimages.keys())[0]

img, ext = pvimages[current_pv].poll()

source = ColumnDataSource(
    {
        "image": [img],
        "x": [ext[0]],
        "y": [ext[2]],
        "dw": [ext[1] - ext[0]],
        "dh": [ext[3] - ext[2]],
    }
)

select = Select(title="Image PV", value=current_pv, options=list(pvimages.keys()))


def on_selection(attrname, old, new):
    global current_pv
    current_pv = new


select.on_change("value", on_selection)

p = figure(
    tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")], height=400, width=400
)
p.x_range.range_padding = p.y_range.range_padding = 0

img_obj = p.image(
    name="img",
    image="image",
    x="x",
    y="y",
    dw="dw",
    dh="dh",
    source=source,
    palette=pal,
)

variables = pvimages[current_pv].variables()
units = pvimages[current_pv].get_units()

p.xaxis.axis_label = variables[-2] + " (" + units[0] + ")"
p.yaxis.axis_label = variables[-1] + " (" + units[1] + ")"


def update():

    global current_pv

    variables = pvimages[current_pv].variables()
    units = pvimages[current_pv].get_units()

    p.xaxis.axis_label = variables[-2] + " (" + units[0] + ")"
    p.yaxis.axis_label = variables[-1] + " (" + units[1] + ")"

    img, ext = pvimages[current_pv].poll()

    img_obj.data_source.data.update(
        {
            "image": [img],
            "x": [ext[0]],
            "y": [ext[2]],
            "dw": [ext[1] - ext[0]],
            "dh": [ext[3] - ext[2]],
        }
    )


curdoc().add_root(column(row(select), row(p), width=300))
curdoc().add_periodic_callback(update, 250)
