import threading
import numpy as np
from typing import Dict

from p4p.nt import NTScalar, NTNDArray
from p4p.server.thread import SharedPV
from p4p.server import Server

from online_model.model.surrogate_model import OnlineSurrogateModel
from online_model import ARRAY_PVS, DEFAULT_COLOR_MODE


def format_model_output(model_output):
    """
    Reformat model for pva server compatibility.

    Parameters
    ----------
    model_ouptut: dict
        Output from the surrogate model.

    Returns
    -------
    dict
        Output with metadata assigned.
    """
    rebuilt_output = {}
    for pv, value in model_output.items():
        if pv in ARRAY_PVS:
            # populate image data
            array_data = value.view(NTNDArrayData)

            # get dw and dh from model output
            array_data.attrib = {
                # "ColorMode": DEFAULT_COLOR_MODE,
                "dw": output_state[f"{pv}:dw"],
                "dh": output_state[f"{pv}:dh"],
            }
            rebuilt_output[pv] = array_data

        # do not build attribute pvs
        elif not ".dw" in pv and not ".dh" in pv:
            rebuilt_output[pv] = value

    return rebuilt_output


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

    def __init__(self, prefix):
        self.prefix = prefix

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
        input_pvs[op.name().replace(f"{self.prefix}:", "")] = op.value()

        # run model using global input process variable state
        output_pv_state = model_loader.model.run(input_pvs)
        output_pv_state = format_model_output(output_pv_state)

        # now update output variables
        for pv, value in output_pv_state.items():
            output_provider = providers[f"{self.prefix}:{pv}"]
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
        prefix: str,
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

        prefix: str
            Prefix to use when serving
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
        starting_output = format_model_output(starting_output)

        # create PVs for model inputs
        for in_pv in in_pvdb:
            pvname = f"{prefix}:{in_pv}"

            if in_pv not in ARRAY_PVS:
                pv = SharedPV(
                    handler=InputHandler(
                        prefix
                    ),  # Use InputHandler class to handle callbacks
                    nt=NTScalar("d"),
                    initial=in_pvdb[in_pv]["value"],
                )
            else:
                pv = SharedPV(
                    handler=InputHandler(
                        prefix
                    ),  # Use InputHandler class to handle callbacks
                    nt=NTNDArray(),
                    initial=in_pvdb[in_pv]["value"],
                )
            providers[pvname] = pv

        # use default handler for the output process variables
        # updates to output pvs are handled from post calls within the input update
        for out_pv, value in starting_output.items():
            pvname = f"{prefix}:{out_pv}"
            if out_pv not in ARRAY_PVS:
                pv = SharedPV(nt=NTScalar(), initial=value)

            elif out_pv in ARRAY_PVS:
                pv = SharedPV(nt=NTNDArray(), initial=value)

            providers[pvname] = pv

        else:
            pass  # throw exception for incorrect data type

    def start_server(self) -> None:
        """
        Starts the server and runs until KeyboardInterrupt.
        """
        print("Starting Server...")
        self.server = Server.forever(providers=[providers])

    def stop_server(self) -> None:
        """
        Stops the server manually.
        """
        print("Stopping Server...")
        self.server.stop()
