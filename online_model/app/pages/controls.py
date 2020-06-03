import sys
import os
from argparse import ArgumentParser

from bokeh.io import curdoc
from bokeh.layouts import column, row

# fix for bokeh path error, maybe theres a better way to do this
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

from online_model.app.widgets.sliders import build_sliders
from online_model.app.controllers import Controller

# server prefix
PREFIX = "smvm"

PROTOCOL, CMD_PVDB, _ = parse_args()


# TEMPORARILY EXCLUDE THE EXTENTS
sliders_to_render = {}
for var, value in CMD_PVDB.items():
    if "in_" not in var:
        sliders_to_render[var] = value

# create controller
controller = Controller(PROTOCOL)

# build sliders for the command process variable database
sliders = build_sliders(sliders_to_render, controller, PREFIX)
scol = column(sliders, width=350)

curdoc().add_root(row(scol))
curdoc().title = "Online Surrogate Model Virtual Machine"
