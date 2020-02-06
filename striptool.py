import numpy as np
import math
import time

from epics import caget, caput, PV

from bokeh.driving import count
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models.glyphs import MultiLine
from bokeh.models.widgets import Select


import json

class pv_buffer():

    def __init__(self, pv, buffer_size):

        self.pvname=pv
        self.pv = PV(pv,auto_monitor=True)
        self.data = np.array([self.pv.get()])

        self.tstart = time.time()

        self.time = np.array([self.tstart])
        self.buffer_size=buffer_size

    def poll(self):

        t=time.time()
        v=caget(self.pvname)#self.pv.get()

        #print(t,v)

        if(len(self.data)<self.buffer_size):
            self.time = np.append(self.time,t)
            self.data = np.append(self.data,v)

        else:
            self.time[:-1]=self.time[1:]
            self.time[-1]=t
            self.data[:-1]=self.data[1:]
            self.data[-1]=v

        return self.time-self.tstart, self.data

with open('pvdef.json', 'r') as fp:
    pvdefs = json.load(fp)
prefix = pvdefs['prefix']

pvs = pvdefs['output']

pvbuffers = {}
for opv in pvs:

    pvdef = pvs[opv]
    if(pvdef['type']=='float' and isinstance(pvdef['value'],float)):
        pvbuffers[opv] = pv_buffer(prefix+opv,100)

plot_pvs = list(pvbuffers.keys())
current_pv = plot_pvs[0]

def pv_select_callback(attr, old, new):
    global current_pv
    current_pv = new

select = Select(title="PV to Plot:", value=current_pv, options=plot_pvs)
select.on_change("value", pv_select_callback)

ts,ys = pvbuffers[current_pv].poll()
source = ColumnDataSource(dict(x=ts, y=ys))
p = figure(plot_width=400, plot_height=400)
p.line(x='x', y='y', line_width=2, source=source)
p.yaxis.axis_label = current_pv + ' ('+pvs[current_pv]['units']+')'
p.xaxis.axis_label = 'time (sec)'

def update():
    global current_pv
    ts,ys = pvbuffers[current_pv].poll()
    source.data = dict(x=ts, y=ys)
    p.yaxis.axis_label = current_pv + ' ('+pvs[current_pv]['units']+')'

scol = column(row(select),row(p),width=350)
curdoc().add_root( scol ) 
curdoc().add_periodic_callback(update, 250)
curdoc().title = "Online Surrogate Model Strip Tool"



