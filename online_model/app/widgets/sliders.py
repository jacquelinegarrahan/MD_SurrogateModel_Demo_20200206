import copy
import time

from argparse import ArgumentParser
from functools import partial
import numpy as np
from typing import Union, List

from bokeh.plotting import figure
from bokeh.models import Slider
from bokeh.models import ColumnDataSource

from epics import caget, caput
from p4p.client.thread import Context

from online_model import PREFIX, ARRAY_PVS
from online_model.app import PROTOCOL, CONTEXT


def set_pv_from_slider(
    attr: str, old: float, new: float, pvname: str, scale: Union[float, int]
) -> None:
    """
    Callback function for slider change.

    Parameters
    ----------
    attr:str
        Attribute to update

    old:float
        Old value

    new:float
        New value

    pvname: str
        Process variable name

    scale:float/int
        Scale of the slider

    """
    if PROTOCOL == "pva":
        CONTEXT.put(pvname, new * scale)

    elif PROTOCOL == "ca":
        caput(pvname, new * scale)


def build_slider(title: str, pvname, scale, start, end, step) -> Slider:
    """
    Utility function for building a slider.

    Parameters
    ----------
    title:str
        Slider title

    pvname:str
        Process variable name

    scale:float/int
        Scale of the slider

    start:float
        Lower range of the slider

    end:float
        Upper range of the slider

    step:np.float64
        The step between consecutive values

    Returns
    -------
    bokeh.models.widgets.sliders.Slider

    """

    # initialize value
    try:
        if PROTOCOL == "pva":
            start_val = CONTEXT.get(pvname)
        elif PROTOCOL == "ca":
            start_val = caget(pvname)
    except TimeoutError:
        print(f"No process variable found for {pvname}")
        start_val = 0

    slider = Slider(
        title=title, value=scale * start_val, start=start, end=end, step=step
    )

    slider.on_change("value", partial(set_pv_from_slider, pvname=pvname, scale=scale))

    return slider


def build_sliders(CMD_PVDB: dict) -> List[Slider]:
    """
    Build sliders from the CMD_PVDB.

    Parameters
    ----------
    CMD_PVDB: dict

    Return
    ------
    list
        List of slider objects

    """
    sliders = []

    for ii, pv in enumerate(CMD_PVDB):
        title = pv + " (" + CMD_PVDB[pv]["units"] + ")"
        pvname = PREFIX + ":" + pv
        step = (CMD_PVDB[pv]["range"][1] - CMD_PVDB[pv]["range"][0]) / 100.0
        scale = 1

        slider = build_slider(
            title,
            pvname,
            scale,
            CMD_PVDB[pv]["range"][0],
            CMD_PVDB[pv]["range"][1],
            step,
        )
        sliders.append(slider)

    return sliders
