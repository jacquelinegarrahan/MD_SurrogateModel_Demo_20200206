import copy
import time

from argparse import ArgumentParser
from functools import partial
import numpy as np

from bokeh.plotting import figure
from bokeh.models import Slider
from bokeh.models import ColumnDataSource

from epics import caget, caput
from p4p.client.thread import Context

from online_model import PREFIX, ARRAY_PVS


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


class PVImageMonitor:
    """
    Monitor for updating and formatting image data.

    Attributes
    ----------
    pvname: str
        Process variable name

    units: str
        Units for process variable

    """

    def __init__(self, pvname, units):
        self.pvname = pvname
        self.units = units

    def poll(self):

        image_data = {"image": None}
        if PROTOCOL == "ca":
            # self.pv = PV(pv, auto_monitor=True)
            data = caget(self.pvname)
            nx = data[0]
            ny = data[1]
            ext = data[2:6]
            dw = ext[1] - ext[0]
            dh = ext[3] - ext[2]
            image = data[6:]
            image[np.where(image <= 0)] = 0  # Set negative to zero
            image = image.reshape(int(nx), int(ny))

        elif PROTOCOL == "pva":
            # context returns np array with WRITEABLE=False
            # copy to manipulate array below
            pv = CONTEXT.get(self.pvname)
            attrib = pv.attrib
            dw = attrib["dw"]
            dh = attrib["dh"]
            nx, ny = pv.shape
            image = copy.copy(pv)

        return {"image": [image], "x": [nx], "y": [ny], "dw": [dw], "dh": [dh]}

    def get_units(self):
        return self.units.split(":")

    def variables(self):
        return self.pvname.split(":")


class ImageController:
    def __init__(self, SIM_PVDB):
        self.pvimages = {}

        for opv in SIM_PVDB:
            if len(SIM_PVDB[opv]["units"].split(":")) == 2:
                self.pvimages[opv] = PVImageMonitor(
                    f"{PREFIX}:{opv}", SIM_PVDB[opv]["units"]
                )

        self.current_pv = list(self.pvimages.keys())[0]
        self.image_data = self.pvimages[self.current_pv].poll()
        self.source = ColumnDataSource(self.image_data)

    def build_plot(self, palette):
        # create plot
        self.p = figure(
            tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
            height=400,
            width=400,
        )
        self.p.x_range.range_padding = self.p.y_range.range_padding = 0

        self.img_obj = self.p.image(
            name="img",
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            source=self.source,
            palette=palette,
        )

        variables = self.pvimages[self.current_pv].variables()
        units = self.pvimages[self.current_pv].get_units()

        self.p.xaxis.axis_label = variables[-2] + " (" + units[0] + ")"
        self.p.yaxis.axis_label = variables[-1] + " (" + units[1] + ")"

    def update(self, current_pv):
        # update internal pv trackinng
        self.current_pv = current_pv

        # Update x and y axes
        variables = self.pvimages[current_pv].variables()
        units = self.pvimages[current_pv].get_units()

        self.p.xaxis.axis_label = variables[-2] + " (" + units[0] + ")"
        self.p.yaxis.axis_label = variables[-1] + " (" + units[1] + ")"

        # get image data
        self.image_data = self.pvimages[current_pv].poll()

        # update data source
        self.img_obj.data_source.data.update(self.image_data)


class PVMonitor:
    def __init__(self, pvname, units):
        self.pvname = pvname
        self.units = units
        self.tstart = time.time()
        self.time = np.array([])
        self.data = np.array([])

    def poll(self):
        t = time.time()

        if PROTOCOL == "ca":
            v = caget(self.pvname)

        elif PROTOCOL == "pva":
            v = CONTEXT.get(self.pvname)

        self.time = np.append(self.time, t)
        self.data = np.append(self.data, v)

        return self.time - self.tstart, self.data

    def get_units(self):
        return self.units.split(":")


class PlotController:
    def __init__(self, SIM_PVDB):
        self.pvmonitors = {}

        # only creating pvs for non-image pvs
        for opv in SIM_PVDB:
            if opv not in ARRAY_PVS:
                self.pvmonitors[opv] = PVMonitor(
                    f"{PREFIX}:{opv}", SIM_PVDB[opv]["units"]
                )

        self.current_pv = list(self.pvmonitors.keys())[0]
        ts, ys = self.pvmonitors[self.current_pv].poll()
        self.source = ColumnDataSource(dict(x=ts, y=ys))

    def build_plot(self):
        self.p = figure(plot_width=400, plot_height=400)
        self.p.line(x="x", y="y", line_width=2, source=self.source)
        self.p.yaxis.axis_label = (
            self.current_pv
            + " ("
            + self.pvmonitors[self.current_pv].get_units()[0]
            + ")"
        )
        self.p.xaxis.axis_label = "time (sec)"

    def update(self, current_pv):
        self.current_pv = current_pv
        ts, ys = self.pvmonitors[current_pv].poll()
        units = self.pvmonitors[current_pv].get_units()[0]
        self.source.data = dict(x=ts, y=ys * 1e6)
        self.p.yaxis.axis_label = f"{current_pv} ({units})"
