from epics import caget, caput
from p4p.client.thread import Context


class Controller:
    def __init__(self, protocol):
        self.protocol = protocol

        # initalize context for pva
        self.context = None
        if protocol == "pva":
            self.context = Context("pva")

    def get(self, pvname):
        if self.protocol == "ca":
            return caget(self.pvname)

        elif self.protocol == "pva":
            return self.context.get(self.pvname)

    def put(self, pvname, value):
        if self.protocol == "ca":
            caput(pvname, value)

        elif self.protocol == "pva":
            self.context.put(pvname, value)
