import copy
import time

from argparse import ArgumentParser
from functools import partial
import numpy as np
from typing import Union, List, Dict, Tuple
from abc import ABC, abstractmethod

from bokeh.plotting import figure
from bokeh.models import Slider
from bokeh.models import ColumnDataSource

from epics import caget, caput
from p4p.client.thread import Context

from online_model import PREFIX, ARRAY_PVS
from online_model.app import PROTOCOL, CONTEXT


class Monitor(ABC):
    """
    Base class for the monitor types.

    Attributes
    ----------
    pvname: str
        Name of the process variable

    units: str
        Unit label of the variable

    """

    def __init__(self, pvname: str, units: str) -> None:
        """
        Initialize monitor with name and units.

        Parameters
        ----------
        pvname: str
            Name of the process variable

        units: str
            Unit label of the variable
        """
        self.pvname = pvname
        self.units = units

    @abstractmethod
    def poll(self):
        """
        Must be defined in inheriting class.
        """
        pass

    def get_units(self) -> List[str]:
        """
        Returns units to be used in labels.
        """
        return self.units.split(":")


class PVImageMonitor(Monitor):
    """
    Monitor for updating and formatting image data.

    Attributes
    ----------
    pvname: str
        Process variable name

    units: str
        Units for process variable

    """

    def __init__(self, pvname: str, units: str) -> None:
        """
        Initializes monitor attributes.

        Parameters
        ----------
        pvname: str
            Process variable name

        units: str
            Units for process variable
        """
        super(PVImageMonitor, self).__init__(pvname, units)

    def poll(self) -> Dict[str, list]:
        """
        Collects image data via appropriate protocol and builds image data dictionary.

        Returns
        -------
        dict
            Dictionary mapping image components to values.
        """
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

    def variables(self) -> List[str]:
        """
        Returns variables to be plotted. 'x:y' -> ['x', 'y']
        """
        return self.pvname.split(":")


class ImageController:
    """
    Controller for image display.

    Attributes
    ----------
    current_pv: str
        Current process variable to be displayed

    source: bokeh.models.sources.ColumnDataSource
        Data source for the viewer.

    pv_monitors: PVImageMonitor
        Monitors for the process variables.

    p: bokeh.plotting.figure.Figure
        Plot object

    img_obj: bokeh.models.renderers.GlyphRenderer
        Image renderer

    """

    def __init__(self, SIM_PVDB: dict) -> None:
        """
        Initialize monitors, current process variable, and data source.

        Parameters
        ----------
        SIM_PVDB: dict
            Dictionary of process variable values
        """
        self.pv_monitors = {}

        for opv in SIM_PVDB:
            if len(SIM_PVDB[opv]["units"].split(":")) == 2:
                self.pv_monitors[opv] = PVImageMonitor(
                    f"{PREFIX}:{opv}", SIM_PVDB[opv]["units"]
                )

        self.current_pv = list(self.pv_monitors.keys())[0]
        image_data = self.pv_monitors[self.current_pv].poll()
        self.source = ColumnDataSource(image_data)

    def build_plot(self, palette: tuple) -> Nonoe:
        """
        Creates the plot object.

        Parameters
        ----------
        palette: tuple
            Color palette to use for plot.
        """
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

        variables = self.pv_monitors[self.current_pv].variables()
        units = self.pv_monitors[self.current_pv].get_units()

        self.p.xaxis.axis_label = variables[-2] + " (" + units[0] + ")"
        self.p.yaxis.axis_label = variables[-1] + " (" + units[1] + ")"

    def update(self, current_pv: str) -> None:
        """
        Update the plot to reflect current process variable.

        Parameters
        ----------
        current_pv: str
            Current process variable
        """
        # update internal pv trackinng
        self.current_pv = current_pv

        # Update x and y axes
        variables = self.pv_monitors[current_pv].variables()
        units = self.pv_monitors[current_pv].get_units()

        self.p.xaxis.axis_label = variables[-2] + " (" + units[0] + ")"
        self.p.yaxis.axis_label = variables[-1] + " (" + units[1] + ")"

        # get image data
        image_data = self.pv_monitors[current_pv].poll()

        # update data source
        self.img_obj.data_source.data.update(image_data)


class PVScalarMonitor(Monitor):
    """
    Monitor for scalar process variables.

    Attributes
    ----------
    tstart:

    time: np.ndarray
        Array of sample times

    data: np.ndarray
        Array of data samples
    """

    def __init__(self, pvname, units):
        """
        Initializes monitor attributes.

        Parameters
        ----------
        pvname: str
            Process variable name

        units: str
            Units for process variable

        """
        super(PVScalarMonitor, self).__init__(pvname, units)
        self.tstart = time.time()
        self.time = np.array([])
        self.data = np.array([])

    def poll(self) -> Tuple[np.ndarray]:
        """
        Collects image data via appropriate protocol and returns time and data.
        """
        t = time.time()

        if PROTOCOL == "ca":
            v = caget(self.pvname)

        elif PROTOCOL == "pva":
            v = CONTEXT.get(self.pvname)

        self.time = np.append(self.time, t)
        self.data = np.append(self.data, v)

        return self.time - self.tstart, self.data


class PlotController:
    """
    Controller for plot display.

    Attributes
    ----------
    current_pv: str
        Current process variable to be displayed

    source: bokeh.models.sources.ColumnDataSource
        Data source for the viewer.

    pv_monitors: PVScalarMonitor
        Monitors for the scalar variables.

    p: bokeh.plotting.figure.Figure
        Plot object

    """

    def __init__(self, SIM_PVDB) -> None:
        """
        Initialize monitors, current process variable, and data source.

        Parameters
        ----------
        SIM_PVDB: dict
            Dictionary of process variable values
        """
        self.pv_monitors = {}

        # only creating pvs for non-image pvs
        for opv in SIM_PVDB:
            if opv not in ARRAY_PVS:
                self.pv_monitors[opv] = PVScalarMonitor(
                    f"{PREFIX}:{opv}", SIM_PVDB[opv]["units"]
                )

        self.current_pv = list(self.pv_monitors.keys())[0]
        ts, ys = self.pv_monitors[self.current_pv].poll()
        self.source = ColumnDataSource(dict(x=ts, y=ys))

    def build_plot(self) -> None:
        """
        Creates the plot object.
        """
        self.p = figure(plot_width=400, plot_height=400)
        self.p.line(x="x", y="y", line_width=2, source=self.source)
        self.p.yaxis.axis_label = (
            self.current_pv
            + " ("
            + self.pv_monitors[self.current_pv].get_units()[0]
            + ")"
        )
        self.p.xaxis.axis_label = "time (sec)"

    def update(self, current_pv: str) -> None:
        """
        Update the plot to reflect current process variable.

        Parameters
        ----------
        current_pv: str
            Current process variable
        """
        self.current_pv = current_pv
        ts, ys = self.pv_monitors[current_pv].poll()
        units = self.pv_monitors[current_pv].get_units()[0]
        self.source.data = dict(x=ts, y=ys * 1e6)
        self.p.yaxis.axis_label = f"{current_pv} ({units})"
