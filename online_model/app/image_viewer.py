import numpy as np
import json

from epics import caget_many, PV

from bokeh.plotting import figure
from bokeh.models.widgets import Select
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource

# from bokeh.palettes import magma

from bokeh import palettes, colors

pal = palettes.Viridis[256]
white = colors.named.white
# pal[0] = white # replace 0 with white


class PVImageMonitor:
    def __init__(self, pvname, units):
        self.pvname = pvname
        self.pv = PV(pvname, auto_monitor=True)
        self.units = units

    def poll(self):
        data = self.pv.value
        nx = data[0]
        ny = data[1]
        ext = data[2:6]

        print(ext)
        image = data[6:]
        image[np.where(image <= 0)] = 0  # Set negative to zero
        image = image.reshape(int(nx), int(ny))
        return (image, ext)

    def get_units(self):
        return self.units.split(":")

    def variables(self):
        return self.pvname.split(":")


with open("pvdef.json") as json_file:
    pvdefs = json.load(json_file)
prefix = pvdefs["prefix"]

pvimages = {}
for opv in pvdefs["output"].keys():

    if len(pvdefs["output"][opv]["units"].split(":")) == 2:
        pvimages[opv] = PVImageMonitor(prefix + opv, pvdefs["output"][opv]["units"])

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

    # current_index = get_screen_index(

    # global current_index
    # avgxs = avgx_monitor.get_pv_array(1000)
    # avgys = avgy_monitor.get_pv_array(1000)
    # stdxs = stdx_monitor.get_pv_array(1000)
    # stdys = stdy_monitor.get_pv_array(1000)

    # if(current_index is not None):

    #    avgx = avgxs[current_index]
    #    avgy = avgys[current_index]
    #    stdx = stdxs[current_index]
    #    stdy = stdys[current_index]

    # else:

    #    avgx = float("NaN")
    #    avgy = float("NaN")
    #    stdx = float("NaN")
    #    stdy = float("NaN")

    # img = gaus2d(x,y,avgx,avgy,stdx,stdy)
    # img_obj.data_source.data.update({'image': [img]})


curdoc().add_root(column(row(select), row(p), width=300))
curdoc().add_periodic_callback(update, 250)
