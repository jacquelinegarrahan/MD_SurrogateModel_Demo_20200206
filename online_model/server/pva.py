import random
import threading
import numpy as np
from typing import Dict, Tuple, Mapping, Union

from p4p.nt import NTScalar, NTNDArray
from p4p.server.thread import SharedPV
from p4p.server import Server

from online_model.model.surrogate_model import OnlineSurrogateModel
from online_model import PREFIX

# global mapping of process variable string to SharedPV instance
providers = {}

# global mapping of input process variable string to value
input_pvs = {}


class ModelLoader(threading.local):
    """
    Subclass of threading.local that will initialize the OnlineSurrogateModel in each thread.

    Attributes
    ----------
    model: online_model.model.surrogate_model.OnlineSurrogateModel
        OnlineSurrogateModel instance used for getting predictions

    Note
    ----
    Keras models are not thread safe so the model must be loaded in each thread and referenced locally.
    """

    def __init__(self):
        """
        Initializes OnlineSurrogateModel.
        """
        self.model = OnlineSurrogateModel()


# initialize loader for model
model_loader = ModelLoader()


class InputHandler:
    """
    Handler object that defines the callbacks to execute on put operations to input process variables.
    """

    def put(self, pv, op) -> None:
        """
        Updates the global input process variable state, posts the input process variable value change, runs the thread local OnlineSurrogateModel instance using the updated global input process variable states, and posts the model output values to the output process variables.

        Parameters
        ----------
        pv: p4p.server.thread.SharedPV
            Input process variable on which the put is operating

        op: p4p.server.raw.ServOpWrap
            Server operation initiated by the put call

        """
        global providers
        global input_pvs

        # update input values and global input process variable state
        pv.post(op.value())
        input_pvs[op.name().replace(f"{PREFIX}:", "")] = op.value()

        # run model using global input process variable state
        output_pv_state = model_loader.model.run(input_pvs)

        # now update output variables
        for pv_item, value in output_pv_state.items():
            output_provider = providers[f"{PREFIX}:{pv_item}"]
            output_provider.post(value)

        # mark server operation as complete
        op.done()


class PVAServer:
    """
    Server object for PVA process variables.

    Attributes
    ----------
    in_pvdb: dict
        Dictionary that maps the input process variable string to type (str), prec (precision), value (float), units (str), range (List[float])

    out_pvdb: dict
        Dictionary that maps the output process variable string to type (str), prec (precision), value (float), units (str), range (List[float])

    """

    def __init__(self, in_pvdb: Dict[str, dict], out_pvdb: Dict[str, dict]) -> None:
        """
        Initialize the global process variable list, populate the initial values for the global input variable state, generate starting output from the main thread OnlineSurrogateModel model instance, and initialize input and output process variables.

        Parameters
        ----------
        in_pvdb: dict
            Dictionary that maps the input process variable string to type (str), prec (precision), value (float), units (str), range (List[float])

        out_pvdb: dict
            Dictionary that maps the output process variable string to type (str), prec (precision), value (float), units (str), range (List[float])
        """

        global providers
        global input_pvs

        self.in_pvdb = in_pvdb
        self.out_pvdb = out_pvdb

        # initialize model and state
        for in_pv in in_pvdb:
            input_pvs[in_pv] = in_pvdb[in_pv]["value"]

        # use main thread loaded model to do initial model run
        starting_output = model_loader.model.run(input_pvs)

        # create PVs for model inputs
        for in_pv in in_pvdb:
            pvname = f"{PREFIX}:{in_pv}"
            pv = SharedPV(
                handler=InputHandler(),
                nt=NTScalar("d"),
                initial=in_pvdb[in_pv]["value"],
            )
            providers[pvname] = pv

        # create PVs for model outputs
        for out_pv in out_pvdb:
            pvname = f"{PREFIX}:{out_pv}"

            # use default handler for the output process variables
            # updates to output pvs are handled from post calls within the input update processing
            if isinstance(starting_output[out_pv], (float,)):
                pv = SharedPV(nt=NTScalar("d"), initial=starting_output[out_pv])
            else:
                pv = SharedPV(nt=NTNDArray(), initial=starting_output[out_pv])

            # update global provider list
            providers[pvname] = pv

    def start_server(self) -> None:
        """
        Starts PVAServer and runs until KeyboardInterrupt.
        """
        Server.forever(providers=[providers])
