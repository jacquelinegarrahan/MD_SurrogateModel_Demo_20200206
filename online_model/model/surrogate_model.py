import numpy as np
import sys, os
import time
from abc import ABC, abstractmethod

import tensorflow as tf
from keras.models import model_from_json
import h5py
import random
import pickle

from online_model import YAG_MODEL_FILE, SCALAR_MODEL_FILE


scalerfile = "online_model/files/transformer_frontend_y_imgs.sav"
transformer_y = pickle.load(open(scalerfile, "rb"))


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
    Base class for the surrogate models that includes abstract predict method which must be initialized by children.

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
    def __init__(self, model_file):
        super().__init__(model_file)

        # load model info
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

    # dunder method conserved from initial implementation
    def __str__(self):
        return f"The inputs are: {', '.join(self.input_names)} and the outputs: {', '.join(self.output_names)}"

    def predict(self, settings) -> dict:
        input_values = np.array([[settings[key] for key in self.input_ordering]])
        inputs_scaled = self.scaler.transform(input_values)

        # call thread-safe predictions
        with self.thread_graph.as_default():
            with self.thread_session.as_default():
                predicted_outputs = self.model.predict(inputs_scaled)

        predicted_outputs_unscaled = self.scaler.inverse_transform(predicted_outputs)
        return dict(zip(self.output_ordering, predicted_outputs_unscaled.T))


class ImageSurrogateModel(SurrogateModel):
    def __init__(self, model_file):
        super().__init__(model_file)

        # load model info
        model_info = load_model_info(model_file)

        # store image specific keys from model_info
        self.ndim = model_info["ndim"]
        self.bins = model_info["bins"]

        # reconstruct scaler from model info
        self.scaler = ReconstructedScaler(
            model_info["input_scales"],
            model_info["input_offsets"],
            model_info["output_scales"][:-1],  # exclude array from scale
            model_info["output_offsets"][:-1],  # exclude array from offset
            model_info["lower"],
            model_info["upper"],
        )

        # set image scaler
        self.image_scaler = transformer_y

        # Configure model attributes and setup model session
        self.configure(model_info)

    # dunder method conserved from initial implementation
    def __str__(self):
        return f"The inputs are: {', '.join(self.input_names)} and the output is an image of LPS."

    def predict(self, settings):
        input_values = np.array([[settings[key] for key in self.input_ordering]])
        inputs_scaled = self.scaler.transform(input_values)

        # call thread-safe predictions
        with self.thread_graph.as_default():
            with self.thread_session.as_default():
                predicted_outputs = self.model.predict(inputs_scaled)

        predicted_outputs_limits = self.scaler.inverse_transform(
            predicted_outputs[:, : self.ndim[0]]
        )

        predicted_outputs_image = self.image_scaler.inverse_transform(
            predicted_outputs[:, self.ndim[0] :]
        )

        return predicted_outputs_image, predicted_outputs_limits


class OnlineSurrogateModel:
    """

    TODO:
    Include note about defaults
    Refactor run and create utility function for manual scaling


    """

    def __init__(
        self, scalar_model_file=SCALAR_MODEL_FILE, image_model_file=YAG_MODEL_FILE
    ):

        self.scalar_model = ScalarSurrogateModel(scalar_model_file)
        self.image_model = ImageSurrogateModel(image_model_file)

    def run(self, pv_state):

        t1 = time.time()
        print("Running model...", end="")
        scalar_data = self.scalar_model.predict(pv_state)

        output = {}
        for scalar in scalar_data:
            output[scalar] = scalar_data[scalar][0]

        image_array, ext = self.image_model.predict(pv_state)
        print(ext)
        ext = [
            ext[0, 0],
            ext[0, 1],
            ext[0, 2],
            ext[0, 3],
        ]  # From Lipi: # At the moment there is some scaling done by hand, this can be changed!

        image_values = np.zeros((2 + len(ext) + image_array.shape[1],))
        image_values[0] = self.image_model.bins[0]
        image_values[1] = self.image_model.bins[1]
        image_values[2:6] = ext
        image_values[6:] = image_array

        # output['z:pz']=image_values
        output["x:y"] = image_values
        t2 = time.time()
        print("Ellapsed time: " + str(t2 - t1))

        return output


def load_model_info(model_file):
    model_info = {}
    with h5py.File(model_file, "r") as h5:
        model_info = dict(h5.attrs)

    return model_info
