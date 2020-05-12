# from matrix_tracker import lattice
import copy
import multiprocessing
from pcaspy import Driver, SimpleServer
import time
from epics import caget, PV
import numpy as np
import random
import json


class SimDriver(Driver):
    def __init__(self, input_pv_state, output_pv_state, noise_params=None):

        super(SimDriver, self).__init__()

        self.input_pv_state = input_pv_state
        self.output_pv_state = output_pv_state

        if noise_params:
            self.noise_params = noise_params
        else:
            self.noise_params = {}

    def read(self, reason):

        if reason in self.output_pv_state:
            value = self.get_noisy_pv(reason)
        else:
            value = self.getParam(reason)

        return value

    def write(self, reason, value):

        if reason in self.output_pv_state:
            print(reason + " is a read-only pv")
            return False
        else:

            if reason in self.input_pv_state:
                self.input_pv_state[reason] = value
            self.setParam(reason, value)
            self.updatePVs()
            return True

    def set_output_pvs(self, outpvs):
        post_updates = False
        for opv in outpvs:
            if opv in self.output_pv_state:
                self.output_pv_state[opv] = outpvs[opv]
                self.setParam(opv, self.get_noisy_pv(opv))
                post_updates = True

        if post_updates:
            self.updatePVs()

    def set_pvs(self, pvs):

        post_updates = False

        for pv in pvs:
            self.setParam(pv, pvs[pv])
            post_updates = True

        if post_updates:
            self.updatePVs()

    def get_noisy_pv(self, pv):

        noise = 0
        if pv in self.noise_params:

            dist = self.getParam(pv + ":dist")  # self.noise_params[pv]['dist']
            sigma = self.getParam(pv + ":sigma")  # self.noise_params[pv]['sigma']

            if dist == "uniform":
                full_width = np.sqrt(12) * sigma
                noise = random.uniform(-full_width / 2.0, full_width / 2.0)

            elif dist == "normal":
                noise = random.uniform(0, sigma)

        return self.output_pv_state[pv] + noise


class SyncedSimPVServer:

    """Defines basic PV server that continuously syncs the input model to the input (command) EPICS PV values
    and publishes updated model data to output EPICS PVs.  Assumes fast model execution, as the model executes
    in the main CAS server thread.  CAS for the input and ouput PVs is handled by the SimDriver object"""

    def __init__(
        self, name, input_pvdb, output_pvdb, noise_params, model, sim_params=None
    ):

        self.name = name

        self.pvdb = {}
        self.input_pv_state = {}
        self.output_pv_state = {}

        self.model = model

        for pv in input_pvdb:
            # print(pv)
            self.pvdb[pv] = input_pvdb[pv]
            self.input_pv_state[pv] = input_pvdb[pv]["value"]

        output_pv_state = {}
        for pv in output_pvdb:
            self.pvdb[pv] = output_pvdb[pv]
            output_pv_state[pv] = output_pvdb[pv]["value"]

        for pv in output_pvdb:
            if pv in noise_params:
                self.pvdb[pv + ":sigma"] = {
                    "type": "float",
                    "value": noise_params[pv]["sigma"],
                }
                self.pvdb[pv + ":dist"] = {
                    "type": "char",
                    "count": 100,
                    "value": noise_params[pv]["dist"],
                }

        prefix = self.name + ":"
        self.server = SimpleServer()
        self.server.createPV(prefix, self.pvdb)

        self.driver = SimDriver(self.input_pv_state, output_pv_state, noise_params)

        self.serve_data = False
        self.sim_params = sim_params

    def set_sim_params(**params):
        self.sim_params = params

    def start_server(self):

        self.serve_data = True

        sim_pv_state = copy.deepcopy(self.input_pv_state)

        # Do initial simulation
        print("Initializing sim...")
        output_pv_state = self.model.run(self.input_pv_state, verbose=True)
        self.driver.set_output_pvs(output_pv_state)
        print("...done.")

        while self.serve_data:
            # process CA transactions
            self.server.process(0.1)

            while sim_pv_state != self.input_pv_state:

                sim_pv_state = copy.deepcopy(self.input_pv_state)
                output_pv_state = self.model.run(self.input_pv_state, verbose=True)
                self.driver.set_output_pvs(output_pv_state)

    def stop_server(self):
        self.serve_data = False
