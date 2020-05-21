import random
import threading
import numpy as np
import time
from typing import Dict, Mapping, Union

from p4p.nt.ndarray import ntndarray as NTNDArrayData
from p4p.nt import NTScalar, NTNDArray
from p4p.server.thread import SharedPV
from p4p.server import Server

from online_model.model.surrogate_model import OnlineSurrogateModel
from online_model.model.MySurrogateModel import MySurrogateModel
from online_model import PREFIX, MODEL_INFO, MODEL_FILE, DEFAULT_LASER_IMAGE


class ModelLoader(threading.local):
    """
    Subclass of threading.local that will initialize the OnlineSurrogateModel in each \\
    thread.

    Attributes
    ----------
    model: online_model.model.surrogate_model.OnlineSurrogateModel
        OnlineSurrogateModel instance used for getting predictions

    Note
    ----
    Keras models are not thread safe so the model must be loaded in each thread and \\
    referenced locally.
    """

    def __init__(self, model_class, model_kwargs: dict) -> None:
        """
        Initializes OnlineSurrogateModel.

        Parameters
        ----------
        model_class
            Model class to be instantiated

        model_kwargs: dict
            kwargs for initialization
        """

        surrogate_model = model_class(**model_kwargs)
        self.model = OnlineSurrogateModel([surrogate_model])


class InputHandler:
    """
    Handler object that defines the callbacks to execute on put operations to input \\
    process variables.
    """

    def put(self, pv, op) -> None:
        """
        Updates the global input process variable state, posts the input process \\
        variable value change, runs the thread local OnlineSurrogateModel instance \\
        using the updated global input process variable states, and posts the model \\
        output values to the output process variables.

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

            if isinstance(value, (np.ndarray,)):
                image_array = value[6:].reshape((50, 50))

                # populate image data
                array_data = image_array.view(NTNDArrayData)

                # get dw and dh from model output
                array_data.attrib = {
                    "ColorMode": 0,
                    "dw": value[3] - value[2],
                    "dh": value[5] - value[4],
                }

                output_provider.post(array_data)

            else:

                output_provider.post(value)

        # mark server operation as complete
        op.done()


class PVAServer:
    """
    Server object for PVA process variables.

    Attributes
    ----------
    in_pvdb: dict
        Dictionary that maps the input process variable string to type (str), prec \\
        (precision), value (float), units (str), range (List[float])

    out_pvdb: dict
        Dictionary that maps the output process variable string to type (str), prec \\
        (precision), value (float), units (str), range (List[float])

    """

    def __init__(
        self,
        model_class,
        model_kwargs: dict,
        in_pvdb: Dict[str, dict],
        out_pvdb: Dict[str, dict],
    ) -> None:
        """
        Initialize the global process variable list, populate the initial values for \\
        the global input variable state, generate starting output from the main thread \\
        OnlineSurrogateModel model instance, and initialize input and output process \\
        variables.

        Parameters
        ----------
        model_class: class
            Model class to be instantiated

        model_kwargs: dict
            kwargs for initialization

        in_pvdb: dict
            Dictionary that maps the input process variable string to type (str), prec \\
            (precision), value (float), units (str), range (List[float])

        out_pvdb: dict
            Dictionary that maps the output process variable string to type (str), \\
            prec (precision), value (float), units (str), range (List[float])
        """
        # need these to be global to access from threads
        global providers
        global input_pvs
        global model_loader
        providers = {}
        input_pvs = {}

        # initialize loader for model
        model_loader = ModelLoader(model_class, model_kwargs)

        # these aren't currently used; but, probably not a bad idea to have around
        # for introspection
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
                handler=InputHandler(),  # Use InputHandler class to handle callbacks
                nt=NTScalar("d"),
                initial=in_pvdb[in_pv]["value"],
            )
            providers[pvname] = pv

        # create PVs for model outputs
        for out_pv in out_pvdb:
            pvname = f"{PREFIX}:{out_pv}"

            # use default handler for the output process variables
            # updates to output pvs are handled from post calls within the input update
            # processing
            if isinstance(starting_output[out_pv], (float,)):
                pv = SharedPV(nt=NTScalar("d"), initial=starting_output[out_pv])

            elif isinstance(starting_output[out_pv], (np.ndarray,)):
                # reshape output
                # should probably be moved into the surrogate model code rather than
                # being done here
                image_array = starting_output[out_pv][6:].reshape((50, 50))

                # populate image data
                array_data = image_array.view(NTNDArrayData)

                # get dw and dh from model output
                array_data.attrib = {
                    "ColorMode": 0,
                    "dw": starting_output[out_pv][3] - starting_output[out_pv][2],
                    "dh": starting_output[out_pv][5] - starting_output[out_pv][4],
                }

                pv = SharedPV(nt=NTNDArray(), initial=array_data)

            else:
                pass  # throw exception for incorrect data type

            # update global provider list
            providers[pvname] = pv

    def start_server(self) -> None:
        """
        Starts the server and runs until KeyboardInterrupt.
        """
        print("Starting Server...")
        Server.forever(providers=[providers])
