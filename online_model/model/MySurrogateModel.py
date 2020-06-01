import random
import h5py
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras

from online_model.model.surrogate_model import SurrogateModel
from online_model.model import apply_temporary_ordering_patch


class MySurrogateModel(SurrogateModel):
    """
Example Usage:
    Load model and use a dictionary of inputs to evaluate the NN.
    """

    def __init__(self, model_file=None, stock_image_input=None):
        # Save init
        self.model_file = model_file
        self.stock_image_input = stock_image_input
        # Run control
        self.configure()

    def __str__(self):
        if self.type == "scalar":
            s = f"""The inputs are: {', '.join(self.input_names)} and the outputs: {', '.join(self.output_names)}"""
        elif self.type == "image":
            s = f"""The inputs are: {', '.join(self.input_names)} and the output: {', '.join(self.output_names)}"""
        elif self.type == "both":
            s = f"""The inputs are: {', '.join(self.input_names)} and the output: {', '.join(self.output_names)}. Requires image input and output as well"""
        return s

    def configure(self):

        ## Open the File
        with h5py.File(self.model_file, "r") as h5:
            attrs = dict(h5.attrs)
            print("Loaded Attributes successfully")
        self.__dict__.update(attrs)
        self.json_string = self.JSON
        print("Loaded Architecture successfully")

        # load model in thread safe manner
        self.thread_graph = tf.Graph()
        with self.thread_graph.as_default():
            self.model = tf.keras.models.model_from_json(
                self.json_string.decode("utf-8")
            )
            self.model.load_weights(self.model_file)

        # TEMPORARY PATCH FOR INPUT/OUTPUT REDUNDANT VARS
        self.input_ordering = apply_temporary_ordering_patch(self.input_ordering, "in")
        self.output_ordering = apply_temporary_ordering_patch(
            self.output_ordering, "out"
        )

        print("Loaded Weights successfully")
        ## Set basic values needed for input and output scaling
        self.model_value_max = 1  # attrs['upper']
        self.model_value_min = 0  # attrs['lower']

        if self.type == "image":
            self.image_scale = self.output_scales[-1]
            self.image_offset = self.output_offsets[-1]
            self.output_scales = self.output_scales[:-1]
            self.output_offsets = self.output_offsets[:-1]
        elif self.type == "both":
            self.image_scale = self.output_scales[-1]
            self.image_offset = self.output_offsets[-1]
            self.output_scales = self.output_scales[:-1]
            self.output_offsets = self.output_offsets[:-1]
            self.scalar_variables = len(self.input_ordering)
            self.scalar_outputs = len(self.output_ordering)

    def scale_inputs(self, input_values):
        data_scaled = self.model_value_min + (
            (input_values - self.input_offsets[0 : self.scalar_variables])
            * (self.model_value_max - self.model_value_min)
            / self.input_scales[0 : self.scalar_variables]
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
        data_scaled = 2 * ((image_values / self.image_scale) - self.image_offset)
        return data_scaled

    def unscale_image(self, image_values):
        data_scaled = ((image_values / 2) + self.image_offset) * self.image_scale
        return data_scaled

    def unscale_inputs(self, input_values):
        data_unscaled = (
            (input_values - self.model_value_min)
            * (self.input_scales[0 : self.scalar_variables])
            / (self.model_value_max - self.model_value_min)
        ) + self.input_offsets[0 : self.scalar_variables]
        return data_unscaled

    def unscale_outputs(self, output_values):
        data_unscaled = (
            (output_values - self.model_value_min)
            * (self.output_scales)
            / (self.model_value_max - self.model_value_min)
        ) + self.output_offsets
        return data_unscaled

    def evaluate_scalar(self, settings):
        if self.type == "image":
            print(
                "To evaluate an image NN, please use the method .evaluateImage(settings)."
            )
            output = 0
        else:
            vec = np.array([[settings[key] for key in self.input_ordering]])
            inputs_scaled = self.scale_inputs(vec)
            model_output = self.model.predict(inputs_scaled)
            model_output = self.unscale_outputs(predicted_outputs)
            output = dict(zip(self.output_ordering, model_output.T))

        return output

    def predict(self, settings):
        if not "image" in settings:
            settings["image"] = self.stock_image_input

        vec = np.array([settings[key] for key in self.input_ordering])
        image = np.array([settings["image"]])

        inputs_scalar_scaled = self.scale_inputs([vec])
        inputs_image_scaled = self.scale_image(image)

        # call prediction in threadsafe manner
        with self.thread_graph.as_default():
            predicted_output = self.model.predict(
                [inputs_image_scaled, inputs_scalar_scaled]
            )

        predicted_image_scaled = np.array(predicted_output[0])
        predicted_scalars_scaled = predicted_output[1]
        predicted_scalars_unscaled = self.unscale_outputs(predicted_scalars_scaled)
        predicted_extents = predicted_scalars_unscaled[
            :, int(self.scalar_outputs - self.ndim[0]) :
        ]
        predicted_image_unscaled = self.unscale_image(
            predicted_image_scaled.reshape(
                predicted_image_scaled.shape[0], int(self.bins[0] * self.bins[1])
            )
        )

        predicted_output = dict(zip(self.output_ordering, predicted_scalars_unscaled.T))
        predicted_image = predicted_image_unscaled.reshape(
            (int(self.bins[0]), int(self.bins[1]))
        )
        predicted_output["extents"] = predicted_extents
        predicted_output["image"] = predicted_image_unscaled

        return self.prepare_outputs(predicted_output)

    def evaluate_image(self, settings, position_scale=10e6):
        vec = np.array([[settings[key] for key in self.input_ordering]])

        inputs_scaled = self.scale_inputs(vec)
        predicted_outputs = self.model.predict(inputs_scaled)
        predicted_outputs_limits = self.unscale_outputs(
            predicted_outputs[:, : self.ndim[0]]
        )
        predicted_outputs_image = self.unscale_image(
            predicted_outputs[:, int(scalar_outputs - 4) :]
        )

        output = predicted_outputs_image.reshape((int(self.bins[0]), int(self.bins[1])))
        return output, extent

    def use_stock_input_image(self):
        data = np.load(self.stock_image_input)
        return data

    def generate_random_input(self):
        if self.type == "both":
            values = np.zeros(len(self.input_ordering))
            for i in range(len(self.input_ordering)):
                values[i] = random.uniform(
                    self.input_ranges[i][0], self.input_ranges[i][1]
                )
            individual = dict(zip(self.input_ordering, values.T))
            individual["image"] = self.use_stock_input_image()
        else:
            values = np.zeros(len(self.input_ordering))
            for i in range(len(self.input_ordering)):
                values[i] = random.uniform(
                    self.input_ranges[i][0], self.input_ranges[i][1]
                )
            individual = dict(zip(self.input_ordering, values.T))

        return individual

    def random_evaluate(self):
        individual = self.generate_random_input()
        if self.type == "scalar":
            random_eval_output = self.evaluate(individual)
            print("Output Generated")
        elif self.type == "image":
            random_eval_output, extent = self.evaluate_image(individual)
            print("Output Generated")
        else:
            random_eval_output = self.evaluate(individual)
            print("Output Generated")
        return random_eval_output

    def prepare_outputs(self, predicted_output):
        """
        Prepares the model outputs to be served so no additional
        manipulation happens in the OnlineSurrogateModel class

        Parameters
        ----------
        model_outputs: dict
            Dictionary of output variables to np.ndarrays of outputs

        Returns
        -------
        dict
            Dictionary of output variables to respective scalars
            (reduced dimensionality of the numpy arrays)

        Note
        ----
        This could also be accomplished by reshaping/sampling the arrays
        in scaling.
        """
        output = {}
        for scalar in self.output_ordering:
            output[scalar] = predicted_output[scalar][0]

        extents = list(predicted_output["extents"][0])

        output["x:y"] = predicted_output["image"].reshape((self.bins[0], self.bins[1]))
        output["x:y:dw"] = extents[1] - extents[0]
        output["x:y:dh"] = extents[3] - extents[2]

        return output
