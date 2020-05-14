import numpy as np
import sys, os
import time
from typing import Dict, Tuple, Mapping, Union
from abc import ABC, abstractmethod

import tensorflow as tf
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import h5py
import random
import pickle

from online_model import YAG_MODEL_FILE, SCALAR_MODEL_FILE


scalerfile = "online_model/files/transformer_frontend_y_imgs.sav"
transformer_y = pickle.load(open(scalerfile, "rb"))

# TODO: What are bins? What is ext?

# model info files are loaded as dicts that map strings to numpy arrays, ints,
# bytes, and strings
ModelInfo = Mapping[str, Union[bytes, np.ndarray, str, np.int64]]


class ReconstructedScaler:
    """
    Object for performing appropriate scaling on inputs and outputs.

    Attributes
    ----------
    input_scales: numpy.ndarray

    input_offsets: numpy.ndarray

    output_scales: numpy.ndarray

    output_offsets: numpy.ndarray

    min_value: numpy.int64

    max_value: numpy.int64

    Notes
    -----
    This object aims to reconstruct the original scaling method during model building. It uses the same method names as the scikit-learn transformation methods in order to be consistent with the provided pickled image scaler.
    """

    def __init__(
        self,
        input_scales,
        input_offsets,
        output_scales,
        output_offsets,
        min_value,
        max_value,
    ):
        self.input_scales = input_scales
        self.input_offsets = input_offsets
        self.output_scales = output_scales
        self.output_offsets = output_offsets
        self.min_value = min_value
        self.max_value = max_value

    def transform(self, values):
        """
        Transforms input values to the model range.

        Parameters
        ----------
        values: numpy.ndarray
            Values to scale, the shape of the array should be (1, number of input pvs)
        """
        return (
            self.min_value
            + (values - self.input_offsets)
            * (self.max_value - self.min_value)
            / self.input_scales
        )

    def inverse_transform(self, values):
        """
        Transforms outputed model values to the appropriate range.

        Parameters
        ----------
        values: numpy.ndarray
            Values to scale, the shape of the array should be (1, number of output pvs - 1). The output process variable that is not scaled by the inverse transform is handled by the image scaler object.
        """
        return (
            (values - self.min_value)
            * self.output_scales
            / (self.max_value - self.min_value)
        ) + self.output_offsets


class SurrogateModel(ABC):
    """
    Base class for the surrogate models that includes abstract predict method, which must be initialized by children.

    Attributes
    ----------
    model_file: str
        Path to the model h5 file

    input_ordering: numpy.ndarray
        Array of process variable names that indicates order in other related attributes such as scales and offsets and indicates order in model input composition.

    thread_graph: tensorflow.python.framework.ops.Graph

    thread_session: tensorflow.python.client.session.Session

    model: keras.engine.training.Model

    """

    def __init__(self, model_file):
        self.model_file = model_file

    def configure(self, model_info) -> None:
        """
        Stores fundamental input_ordering attribute from provided model info and loads and initializes the keras model in a threadsafe session.

        Parameters
        ----------
        model_info: dict
            Contains the model info dictionary loaded from the provided h5 model files.

        """
        # Open the File
        self.input_ordering = model_info["input_ordering"]

        # load model in thread safe manner
        self.thread_graph = tf.Graph()
        with self.thread_graph.as_default():
            self.thread_session = tf.Session()
            with self.thread_session.as_default():
                self.model = model_from_json(model_info["JSON"].decode("utf-8"))
                self.model.load_weights(self.model_file)

    @abstractmethod
    def predict(self):
        """
        Abstract prediction method that must be overwritten by inheriting classes.
        """
        pass


class ScalarSurrogateModel(SurrogateModel):
    """
    Class with specific attributes for scalar surrogate model.

    Attributes
    ----------
    output_ordering:

    """

    def __init__(self, model_file: str) -> None:
        """
        Initialize ScalarSurrogateModel instance, create scaler, and configure the threadsafe model session.

        Parameters
        ----------
        model_file: str
            Filename of the scalar model

        """
        super(ScalarSurrogateModel, self).__init__(model_file)

        # load model info and save scalar specific attributes
        model_info = load_model_info(model_file)
        self.output_ordering = model_info["output_ordering"]

        # reconstruct scaler from model info
        self.scaler = ReconstructedScaler(
            model_info["input_scales"],
            model_info["input_offsets"],
            model_info["output_scales"],
            model_info["output_offsets"],
            model_info["lower"],
            model_info["upper"],
        )

        # Configure model attributes and setup model session
        self.configure(model_info)

    def predict(self, settings: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        Given input process variable values, executes scalar model prediction.

        Parameters
        ----------
        settings: dict
            Dictionary mapping input process variable names to values

        Returns
        -------
        dict
            Mapping of output process variables to single element numpy.ndarray containing the model output.

        """
        input_values = np.array([[settings[key] for key in self.input_ordering]])
        inputs_scaled = self.scaler.transform(input_values)

        # call thread-safe predictions
        with self.thread_graph.as_default():
            with self.thread_session.as_default():
                predicted_outputs = self.model.predict(inputs_scaled)

        predicted_outputs_unscaled = self.scaler.inverse_transform(predicted_outputs)

        return dict(zip(self.output_ordering, predicted_outputs_unscaled.T))


class ImageSurrogateModel(SurrogateModel):
    """
    Class with specific attributes for image surrogate model.

    Attributes
    ----------
    ndim: numpy.int64
        Number of non-image outputs.

    bins: numpy.ndarray
        Maybe shape of the image (x, y) ?

    image_scaler: sklearn.preprocessing.data.MinMaxScaler
        Scaler used for generating final image from image model outputs.
    """

    def __init__(
        self, model_file: str, image_scaler: MinMaxScaler = transformer_y
    ) -> None:
        """
        Initialize ImageSurrogateModel instance, create scaler, and configure the threadsafe model session.

        Parameters
        ----------
        model_file: str
            Filename of the image model

        image_scaler: sklearn.preprocessing.data.MinMaxScaler
            Pre-fit scaler for image outputs

        """
        super(ImageSurrogateModel, self).__init__(model_file)

        # load model info
        model_info = load_model_info(model_file)

        # store image specific keys from model_info
        self.ndim = model_info["ndim"][0]
        self.bins = model_info["bins"]
        self.image_scaler = transformer_y

        # reconstruct scaler from model info
        self.scaler = ReconstructedScaler(
            model_info["input_scales"],
            model_info["input_offsets"],
            model_info["output_scales"][:-1],  # exclude array from scale
            model_info["output_offsets"][:-1],  # exclude array from offset
            model_info["lower"],
            model_info["upper"],
        )

        # Configure model attributes and setup model session
        self.configure(model_info)

    def predict(self, settings: Dict[str, int]) -> np.ndarray:
        """
        Given input process variable values, executes scalar model prediction.

        Parameters
        ----------
        settings: dict
            Dictionary mapping input process variable names to values

        Returns
        -------
        numpy.ndarray
            Array of image elements

        Notes
        -----
        TODO: ask about the other image output vars
        """
        input_values = np.array([[settings[key] for key in self.input_ordering]])
        inputs_scaled = self.scaler.transform(input_values)

        # call thread-safe predictions
        with self.thread_graph.as_default():
            with self.thread_session.as_default():
                predicted_outputs = self.model.predict(inputs_scaled)

        predicted_outputs_limits = self.scaler.inverse_transform(
            predicted_outputs[:, : self.ndim]
        )

        predicted_outputs_image = self.image_scaler.inverse_transform(
            predicted_outputs[:, self.ndim :]
        )

        # format image data for viewing
        image_output = self.format_image_data(
            self.bins, predicted_outputs_image, predicted_outputs_limits
        )

        return image_output

    @staticmethod
    def format_image_data(
        bins: np.ndarray, image_array: np.ndarray, ext: list
    ) -> np.ndarray:
        """
        Formats data

        Parameters
        ----------
        bins: np.ndarray
            Maybe shape of the image (x, y) ?

        image_array: np.ndarray
            Array of unformatted image data

        ext: list
            Non-image outputs ?

        Returns
        -------
        np.ndarray
            Array of formatted image data
        """

        # Note from Lipi: # At the moment there is some scaling done by hand, this can be changed!

        ext = [ext[0, 0], ext[0, 1], ext[0, 2], ext[0, 3]]

        image_values = np.zeros((2 + len(ext) + image_array.shape[1],))
        image_values[0] = bins[0]
        image_values[1] = bins[1]
        image_values[2:6] = ext
        image_values[6:] = image_array

        return image_values


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

    def __init__(
        self,
        scalar_model_file: str = SCALAR_MODEL_FILE,
        image_model_file: str = YAG_MODEL_FILE,
    ) -> None:
        """
        Initialize OnlineSurrogateModel instance using given scalar and image model files.

        Parameters
        ----------
        scalar_model_file: str
            File path to scalar model h5, defaults to SCALAR_MODEL_FILE loaded online_model.__init___

        image_model_file: str
            File path to image model h5, defaults to YAG_MODEL_FILE loaded in online_model.__init__
        """
        self.scalar_model = ScalarSurrogateModel(scalar_model_file)
        self.image_model = ImageSurrogateModel(image_model_file)

    def run(self, pv_state: Dict[str, float]) -> Mapping[str, Union[float, np.ndarray]]:
        """
        Executes both scalar and image model given process variable value inputs.

        Parameters
        ----------
        pv_state:

        Returns
        -------
        dict
            Mapping of process variables to model output values.

        """
        t1 = time.time()
        scalar_data = self.scalar_model.predict(pv_state)

        output = {}
        for scalar in scalar_data:
            output[scalar] = scalar_data[scalar][0]

        # format image data for viewing
        output["x:y"] = self.image_model.predict(pv_state)

        t2 = time.time()
        print("Running model...", end="")
        print("Ellapsed time: " + str(t2 - t1))

        return output


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
