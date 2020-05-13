import numpy as np
import time
from argparse import ArgumentParser

from epics import caget, caput, PV
from p4p.client.thread import Context

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models.widgets import Select
from online_model import PREFIX, SIM_PVDB, ARRAY_PVS

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


class pv_buffer:
    def __init__(self, pv, buffer_size):

        self.pvname = pv
        print(self.pvname)
        # initialize data and time depending on protocol
        # TODO: Check monitors for pva as an alternative to raw polling
        if PROTOCOL == "ca":
            # self.pv = PV(pv, auto_monitor=True)
            self.data = np.array([caget(self.pvname)])

        elif PROTOCOL == "pva":
            self.data = np.array([CONTEXT.get(self.pvname)])

        self.tstart = time.time()
        self.time = np.array([self.tstart])

        self.buffer_size = buffer_size

    def poll(self):
        t = time.time()

        if PROTOCOL == "ca":
            v = caget(self.pvname)

            if len(self.data) < self.buffer_size:
                self.time = np.append(self.time, t)
                self.data = np.append(self.data, v)

            else:
                self.time[:-1] = self.time[1:]
                self.time[-1] = t
                self.data[:-1] = self.data[1:]
                self.data[-1] = v

        elif PROTOCOL == "pva":
            v = CONTEXT.get(self.pvname)

            self.time = np.append(self.time, t)
            self.data = np.append(self.data, v)

        return self.time - self.tstart, self.data


pvbuffers = {}

# only creating pvs for non-image pvs
for opv in SIM_PVDB:
    if opv not in ARRAY_PVS:
        pvbuffers[opv] = pv_buffer(f"{PREFIX}:{opv}", 100)

plot_pvs = list(pvbuffers.keys())
current_pv = plot_pvs[0]


def pv_select_callback(attr, old, new):
    global current_pv
    current_pv = new


select = Select(title="PV to Plot:", value=current_pv, options=plot_pvs)
select.on_change("value", pv_select_callback)

ts, ys = pvbuffers[current_pv].poll()
source = ColumnDataSource(dict(x=ts, y=ys))
p = figure(plot_width=400, plot_height=400)
p.line(x="x", y="y", line_width=2, source=source)
p.yaxis.axis_label = current_pv + " (" + SIM_PVDB[current_pv]["units"] + ")"
p.xaxis.axis_label = "time (sec)"


def update():
    global current_pv

    ts, ys = pvbuffers[current_pv].poll()
    ys = ys * 1e6
    source.data = dict(x=ts, y=ys)
    p.yaxis.axis_label = current_pv + " (" + SIM_PVDB[current_pv]["units"] + ")"


scol = column(row(select), row(p), width=350)
curdoc().add_root(scol)
curdoc().add_periodic_callback(update, 250)
curdoc().title = "Online Surrogate Model Strip Tool"
