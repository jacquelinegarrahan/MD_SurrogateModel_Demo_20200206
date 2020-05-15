import sys
import os
from argparse import ArgumentParser

from bokeh.io import curdoc
from bokeh.layouts import column, row

# fix for bokeh path error, maybe theres a better way to do this
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

from online_model import CMD_PVDB
from online_model.app.widgets.sliders import build_sliders

# build sliders for the command process variable database
sliders = build_sliders(CMD_PVDB)
scol = column(sliders, width=350)

curdoc().add_root(row(scol))
curdoc().title = "Online Surrogate Model Virtual Machine"
