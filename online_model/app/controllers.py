from typing import Union
import numpy as np
from epics import caget, caput
from p4p.client.thread import Context


class Controller:
    """
    Controller class used to get and put process variables.

    Attributes
    ----------
    protocol: str
        Protocol to use ("pva", "ca")

    context: p4p.client.thread.Context
        p4p threaded context instance

    """

    def __init__(self, protocol: str):
        """
        Store protocol and initialize context if using PVAccess.
        """
        self.protocol = protocol

        # initalize context for pva
        self.context = None
        if protocol == "pva":
            self.context = Context("pva")

    def get(self, pvname: str):
        """
        Get the value of a process variable.

        Parameters
        ----------
        pvname: str
            Name of the process variable

        Returns
        -------
        np.ndarray
            Returns numpy array containing value.

        """
        if self.protocol == "ca":
            return caget(pvname)

        elif self.protocol == "pva":
            return self.context.get(pvname)

    def put(self, pvname, value: Union[np.ndarray, float]) -> None:
        """
        Assign the value of a process variable.

        Parameters
        ----------
        pvname: str
            Name of the process variable

        value
            Value to put. Either float or numpy array

        """
        if self.protocol == "ca":
            caput(pvname, value)

        elif self.protocol == "pva":
            self.context.put(pvname, value)
