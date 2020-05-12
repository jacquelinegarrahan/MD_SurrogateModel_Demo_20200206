import numpy as np
import matplotlib.pyplot as plt
import sys, os
import keras
import tensorflow
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Activation
import h5py
import random
import pprint
import pickle
import sklearn


scalerfile = "transformer_frontend_y_imgs.sav"
transformer_y = pickle.load(open(scalerfile, "rb"))


class SurrogateModel:
    """
Example Usage:
    Load model and use a dictionary of inputs to evaluate the NN.
    """

    def __init__(self, model_file=None):
        # Save init
        self.model_file = model_file

        # Run control
        self.configure()

    def __str__(self):
        if self.type == "scalar":
            s = f"""The inputs are: {', '.join(self.input_names)} and the outputs: {', '.join(self.output_names)}"""
        elif self.type == "image":
            s = f"""The inputs are: {', '.join(self.input_names)} and the output is an image of LPS."""
        return s

    def configure(self):

        ## Open the File
        with h5py.File(self.model_file, "r") as h5:
            attrs = dict(h5.attrs)
        self.__dict__.update(attrs)
        self.json_string = self.JSON
        self.model = model_from_json(self.json_string.decode("utf-8"))
        self.model.load_weights(self.model_file)
        ## Set basic values needed for input and output scaling
        self.model_value_max = attrs["upper"]
        self.model_value_min = attrs["lower"]
        # print(self.output_scales)
        # print(self.output_offsets)
        if self.type == "image":
            self.image_scale = self.output_scales[-1]
            self.image_offset = self.output_offsets[-1]
            self.output_scales = self.output_scales[:-1]
            self.output_offsets = self.output_offsets[:-1]

    def scale_inputs(self, input_values):
        data_scaled = self.model_value_min + (
            (input_values - self.input_offsets)
            * (self.model_value_max - self.model_value_min)
            / self.input_scales
        )
        return data_scaled

    def scale_outputs(self, output_values):
        data_scaled = self.model_value_min + (
            (output_values - self.output_offsets)
            * (self.model_value_max - self.model_value_min)
            / self.output_scales
        )
        return data_scaled

    def scale_image(self, image_values):
        # data_scaled = 2*((image_values/self.image_scale)-self.image_offset)
        data_scaled = transformer_y.transform(image_values)
        return data_scaled

    def unscale_image(self, image_values):
        # data_scaled = (((image_values/2)+self.image_offset)*self.image_scale)
        data_scaled = transformer_y.inverse_transform(image_values)
        return data_scaled

    def predict(self, input_values):
        inputs_scaled = self.scale_inputs(input_values)
        predicted_outputs = self.model.predict(inputs_scaled)
        predicted_outputs_unscaled = self.unscale_outputs(predicted_outputs)
        return predicted_outputs_unscaled

    def predict_image(self, input_values, plotting=True):
        inputs_scaled = self.scale_inputs(input_values)
        predicted_outputs = self.model.predict(inputs_scaled)
        predicted_outputs_limits = self.unscale_outputs(
            predicted_outputs[:, : self.ndim[0]]
        )
        predicted_outputs_image = self.unscale_image(
            predicted_outputs[:, self.ndim[0] :]
        )
        # predicted_outputs_unscaled = np.concatenate((predicted_outputs_limits, predicted_outputs_image), axis = 1)
        # predicted_outputs_unscaled = predicted_outputs
        return predicted_outputs_image, predicted_outputs_limits

    def unscale_inputs(self, input_values):
        data_unscaled = (
            (input_values - self.model_value_min)
            * (self.input_scales)
            / (self.model_value_max - self.model_value_min)
        ) + self.input_offsets
        return data_unscaled

    def unscale_outputs(self, output_values):
        data_unscaled = (
            (output_values - self.model_value_min)
            * (self.output_scales)
            / (self.model_value_max - self.model_value_min)
        ) + self.output_offsets
        return data_unscaled

    def evaluate(self, settings):
        if self.type == "image":
            print(
                "To evaluate an image NN, please use the method .evaluateImage(settings)."
            )
            output = 0
        else:
            vec = np.array([[settings[key] for key in self.input_ordering]])
            model_output = self.predict(vec)
            output = dict(zip(self.output_ordering, model_output.T))
        return output

    def evaluate_image(self, settings, position_scale=10e6):

        vec = np.array([[settings[key] for key in self.input_ordering]])
        model_output, extent = self.predict_image(vec)
        output = model_output.reshape((int(self.bins[0]), int(self.bins[1])))
        return output, extent

    def evaluate_image_array(self, settings, position_scale=10e6):
        vec = np.array([[settings[key] for key in self.input_ordering]])
        output, extent = self.predict_image(vec)
        return output, extent

    def generate_random_input(self):
        values = np.zeros(len(self.input_ordering))
        for i in range(len(self.input_ordering)):
            values[i] = random.uniform(self.input_ranges[i][0], self.input_ranges[i][1])
        return dict(zip(self.input_ordering, values.T))

    def random_evaluate(self):
        individual = self.generate_random_input()
        if self.type == "scalar":
            random_eval_output = self.evaluate(individual)
        else:
            random_eval_output, extent = self.evaluate_image(individual)
            print("Output Generated")
            print(extent)
        return random_eval_output
