import copy
import time
import numpy as np
import random
from typing import Dict, Mapping, Union

from epics import caget, PV
from pcaspy import Driver, SimpleServer

from online_model.model.surrogate_model import OnlineSurrogateModel
from online_model.model.custom_model import MySurrogateModel, MyScaler
from online_model import ARRAY_PVS, PREFIX, MODEL_INFO, MODEL_FILE


class SimDriver(Driver):
    """
    Class that reacts to read an write requests to process variables.

    Attributes
    ----------
    input_pv_state: dict
        Dictionary mapping initial input process variables to values.

    output_pv_state: dict
        Dictionary mapping initial output process variables to values (np.ndarray in \\
        the case of image x:y)

    noise_params:
        Dictionary mapping noisy process variables to a dictionary containing their \\
        distribution and standard deviation.

    """

    def __init__(
        self,
        input_pv_state: Dict[str, float],
        output_pv_state: Mapping[str, Union[float, np.ndarray]],
        noise_params: Dict[str, dict] = {},
    ) -> None:
        """
        Initialize the driver. Store input state, output state, and noise parameters.

        Parameters
        ----------
        input_pv_state: dict
            Dictionary that maps the input process variables to their inital values

        output_pv_state:
            Dictionary that maps the output process variables to their inital values

        noise_params: dict
            Dictionary mapping noisy process variables to a dictionary containing their \\
            distribution and standard deviation.
        """

        super(SimDriver, self).__init__()

        # track input state, output state, and noise parameters
        self.input_pv_state = input_pv_state
        self.output_pv_state = output_pv_state
        self.noise_params = noise_params

    def read(self, pv: str) -> Union[float, np.ndarray]:
        """
        Method used by server when clients read a process variable.

        Parameters
        ----------
        pv: str
            Process variable name

        Returns
        -------
        float/np.ndarray
            Returns the value of the process variable

        Notes
        -----
        In the pcaspy documentation, 'reason' is used instead of pv.

        """
        if pv in self.output_pv_state:
            value = self.getParam(pv)
            if self.noise_params:
                value += self.get_noise(pv)
        else:
            value = self.getParam(pv)

        return value

    def write(self, pv: str, value: Union[float, np.ndarray]) -> bool:
        """
        Method used by server when clients write a process variable.


        Parameters
        ----------
        pv: str
            Process variable name

        value: float/np.ndarray
            Value to assign to the process variable.

        Returns
        -------
        bool
            Returns True if the value is accepted, False if rejected

        Notes
        -----
        In the pcaspy documentation, 'reason' is used instead of pv.
        """

        if pv in self.output_pv_state:
            print(pv + " is a read-only pv")
            return False

        else:

            if pv in self.input_pv_state:
                self.input_pv_state[pv] = value

            self.setParam(pv, value)
            self.updatePVs()

            return True

    def set_output_pvs(
        self, output_pvs: Mapping[str, Union[float, np.ndarray]]
    ) -> None:
        """
        Set output process variables.

        Parameters
        ----------
        output_pvs: dict
            Dictionary that maps ouput process variable name to values
        """
        # update output process variable state
        self.output_pv_state.update(output_pvs)

        # update the output process variables that have been changed
        for pv in output_pvs:
            value = self.output_pv_state[pv]

            # add noise
            if pv in self.noise_params:
                value += self.get_noise(pv)

            # set parameter with value
            self.setParam(pv, value)

    def get_noise(self, pv: str) -> float:
        """
        Add noise to a process variable.

        Parameters
        ----------
        pv: str
            Process variable name

        Returns
        -------
        float
            Return noise value to add to model output value
        """
        noise = 0

        # check pv included in noise pv attribute
        if pv in self.noise_params:
            dist = self.getParam(pv + ":dist")
            sigma = self.getParam(pv + ":sigma")

            # sample uniform distribution
            if dist == "uniform":
                full_width = np.sqrt(12) * sigma
                noise = random.uniform(-full_width / 2.0, full_width / 2.0)

            # sample normal distribution
            elif dist == "normal":
                noise = random.uniform(0, sigma)

        return noise


class CAServer:
    """
    Server object for channel access process variables that updates and reads process \\
    values in a single thread.

    Attributes
    ----------
    model: online_model.model.surrogate_model.OnlineSurrogateModel
        OnlineSurrogateModel instance used for getting predictions

    pvdb: dict
        Dictionary that maps the process variable string to type (str), prec \\
        (precision), value (float), units (str), range (List[float])

    input_pv_state: dict
        Dictionary that maps the input process variables to their current values

    output_pv_state:
        Dictionary that maps the output process variables to their current values

    server: pcaspy.driver.SimpleServer
        Server class that interfaces between the channel access client and the driver. \\
        Forwards the read/write requests to the driver

    driver: online_model.server.ca.SimDriver
        Class that reacts to process variable read/write requests

    """

    def __init__(
        self,
        input_pvdb: Dict[str, dict],
        output_pvdb: Dict[str, dict],
        noise_params: Dict[str, dict] = {},
        prefix: str = PREFIX,
    ) -> None:
        """
        Create OnlineSurrogateModel instance and initialize output variables by running \\
        with the input process variable state, set up the proces variable database and \\
        input/output variable tracking, start the server, create the process variables, \\
        and start the driver.

        Parameters
        ----------
        in_pvdb: dict
            Dictionary that maps the input process variable string to type (str), prec \\
             (precision), value (float), units (str), range (List[float])

        out_pvdb: dict
            Dictionary that maps the output process variable string to type (str), prec \\
            (precision), value (float), units (str), range (List[float])

        noise_params: dict
            Dictionary that maps noisy output process variables to their distribution \\
            and standard deviation

        """
        # prepare info necessary to initialize
        image_input_scales = MODEL_INFO["input_scales"][-1]
        image_output_scales = MODEL_INFO["output_scales"][-1]
        image_offset = MODEL_INFO["output_offsets"][-1]
        output_scales = MODEL_INFO["output_scales"][:-1]
        output_offsets = MODEL_INFO["output_offsets"][:-1]
        n_scalar_vars = len(MODEL_INFO["input_ordering"])
        n_scalar_outputs = len(MODEL_INFO["output_ordering"])
        input_scales = MODEL_INFO["input_scales"][:n_scalar_vars]
        input_offsets = MODEL_INFO["input_offsets"][:n_scalar_vars]
        model_value_min = MODEL_INFO["lower"]
        model_value_max = MODEL_INFO["upper"]
        image_shape = (MODEL_INFO["bins"][0], MODEL_INFO["bins"][1])

        # Create instance of scaler object
        my_scaler_obj = MyScaler(
            input_scales,
            input_offsets,
            output_scales,
            output_offsets,
            model_value_min,
            model_value_max,
            image_input_scales,
            image_output_scales,
            n_scalar_vars,
            image_shape,
        )

        surrogate_model = MySurrogateModel(MODEL_FILE, my_scaler_obj)
        self.model = OnlineSurrogateModel([surrogate_model])

        # set up db for initializing process variables
        self.pvdb = {}

        # set up input process variables
        self.pvdb.update(input_pvdb)
        self.input_pv_state = {pv: input_pvdb[pv]["value"] for pv in input_pvdb}

        # get starting output from the model and set up output process variables
        self.output_pv_state = self.model.run(self.input_pv_state)
        self.pvdb.update(output_pvdb)

        # for array pv values, add count to the db entry
        for pv in ARRAY_PVS:
            self.pvdb[pv]["count"] = len(self.output_pv_state[pv])

        # set up noise process variables
        for pv in noise_params:
            self.pvdb[pv + ":sigma"] = {
                "type": "float",
                "value": noise_params[pv]["sigma"],
            }
            self.pvdb[pv + ":dist"] = {
                "type": "char",
                "count": 100,
                "value": noise_params[pv]["dist"],
            }

        # initialize channel access server
        self.server = SimpleServer()

        # create all process variables using the process variables stored in self.pvdb
        # with the given prefix
        self.server.createPV(prefix + ":", self.pvdb)

        # set up driver for handing read and write requests to process variables
        self.driver = SimDriver(self.input_pv_state, self.output_pv_state, noise_params)

    def start_server(self) -> None:
        """
        Start the channel access server and continually update.
        """
        sim_pv_state = copy.deepcopy(self.input_pv_state)

        # Initialize output variables
        print("Initializing sim...")
        output_pv_state = self.model.run(self.input_pv_state)
        self.driver.set_output_pvs(output_pv_state)
        print("...finished initializing.")

        while True:
            # process channel access transactions
            self.server.process(0.1)

            # check if the input process variable state has been updated as
            # an indicator of new input values
            while not all(
                np.array_equal(sim_pv_state[key], self.input_pv_state[key])
                for key in self.input_pv_state
            ):

                sim_pv_state = copy.deepcopy(self.input_pv_state)
                output_pv_state = self.model.run(self.input_pv_state)
                self.driver.set_output_pvs(output_pv_state)
