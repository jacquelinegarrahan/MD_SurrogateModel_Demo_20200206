import numpy as np
import time
from typing import Dict, Tuple, Mapping, Union
from abc import ABC, abstractmethod

import tensorflow as tf
import h5py


# TODO: What are bins? What is ext?

# model info files are loaded as dicts that map strings to numpy arrays, ints,
# bytes, and strings
ModelInfo = Mapping[str, Union[bytes, np.ndarray, str, np.int64]]


class SurrogateModel(ABC):
    """
    Base class for the surrogate models that includes abstract predict method, which \\
    must be initialized by children.

    """

    @abstractmethod
    def predict(self):
        """
        Abstract prediction method that must be overwritten by inheriting classes.
        """
        pass


def load_model_info(model_file: str) -> ModelInfo:
    """
    Utility function for loading model info.

    Parameters
    ----------
    model_file: str
        Filename of the image model

    Returns
    -------
    dict
        Dictionary containing info relevant to the model build.

    """
    model_info = {}
    with h5py.File(model_file, "r") as h5:
        model_info = dict(h5.attrs)

    return model_info


class OnlineSurrogateModel:
    """
    Class for running the executing both the scalar and image model.

    Attributes
    ----------
    scalar_model: online_model.model.surrogate_model.ScalarSurrogateModel
        Model instance used for predicting scalar outputs.

    image_model: online_model.model.surrogate_model.ImageSurrogateModel
        Model instance used for predicting image outputs.

    NOTES
    -----
    TODO:
    Understand the preprocessing here
    """

    def __init__(self, models) -> None:
        """
        Initialize OnlineSurrogateModel instance using given scalar and image model \\
        files.

        Parameters
        ----------
        models: list
            list of model objects

        """
        self.models = models

    def run(self, pv_state: Dict[str, float]) -> Mapping[str, Union[float, np.ndarray]]:
        """
        Executes both scalar and image model given process variable value inputs.

        Parameters
        ----------
        pv_state: dict
            State of input process variables.

        Returns
        -------
        dict
            Mapping of process variables to model output values.

        """
        t1 = time.time()

        output = {}

        for model in self.models:
            predicted_output = model.predict(pv_state)
            output.update(predicted_output)

        t2 = time.time()
        print("Running model...", end="")
        print("Ellapsed time: " + str(t2 - t1))

        return output
