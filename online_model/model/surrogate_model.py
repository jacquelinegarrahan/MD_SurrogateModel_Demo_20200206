import numpy as np
import sys, os
import time
from abc import ABC, abstractmethod

import keras
import tensorflow as tf
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Activation
import h5py
import random
import pprint
import pickle
import sklearn

from online_model import YAG_MODEL_FILE, SCALAR_MODEL_FILE


scalerfile = "online_model/files/transformer_frontend_y_imgs.sav"
transformer_y = pickle.load(open(scalerfile, "rb"))

# reconstruct missing scaler object
# using same method names as the MinMaxScaler sci-kit learn method
# to be consistent with the pickled image scaler
class ReconstructedScaler:
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
        return (
            self.min_value
            + (values - self.input_offsets)
            * (self.max_value - self.min_value)
            / self.input_scales
        )

    def inverse_transform(self, values):
        return (
            (values - self.min_value)
            * self.output_scales
            / (self.max_value - self.min_value)
        ) + self.output_offsets


class SurrogateModel(ABC):
    """

    Use of the abstract base class SurrogateMode requires implementation of predict, scale, and
    """

    def __init__(self, model_file):
        self.model_file = model_file

    def configure(self, model_info) -> None:
        # Open the File
        self.json_string = model_info["JSON"]
        self.input_ordering = model_info["input_ordering"]

        # load model in thread safe manner
        self.thread_graph = tf.Graph()
        with self.thread_graph.as_default():
            self.thread_session = tf.Session()
            with self.thread_session.as_default():
                self.model = model_from_json(self.json_string.decode("utf-8"))
                self.model.load_weights(self.model_file)

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
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

    def evaluate(self, settings) -> dict:
        vec = np.array([[settings[key] for key in self.input_ordering]])
        model_output = self.predict(vec)
        return dict(zip(self.output_ordering, model_output.T))

    def predict(self, input_values):
        inputs_scaled = self.scaler.transform(input_values)

        # call thread-safe predictions
        with self.thread_graph.as_default():
            with self.thread_session.as_default():
                predicted_outputs = self.model.predict(inputs_scaled)

        predicted_outputs_unscaled = self.scaler.inverse_transform(predicted_outputs)
        return predicted_outputs_unscaled


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

    def evaluate(self, settings):
        vec = np.array([[settings[key] for key in self.input_ordering]])
        output, extent = self.predict(vec)
        return output, extent

    def predict(self, input_values, plotting=True):
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

    def run(self, pv_state, verbose=True):

        t1 = time.time()
        print("Running model...", end="")
        scalar_data = self.scalar_model.evaluate(pv_state)

        output = {}
        for scalar in scalar_data:
            output[scalar] = scalar_data[scalar][0]

        image_array, ext = self.image_model.evaluate(pv_state)
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
