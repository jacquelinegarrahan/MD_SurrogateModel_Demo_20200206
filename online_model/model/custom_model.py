import numpy as np
import copy
from online_model.model.surrogate_model import (
    SurrogateModel,
    Scaler,
    apply_temporary_ordering_patch,
    apply_temporary_output_patch,
    load_model_info,
)
from online_model import DEFAULT_LASER_IMAGE


class MyScaler(Scaler):
    def __init__(
        self,
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
    ):

        self.input_scales = input_scales
        self.input_offsets = input_offsets
        self.output_scales = output_scales
        self.output_offsets = output_offsets
        self.model_value_min = model_value_min
        self.model_value_max = model_value_max
        self.n_scalar_vars = n_scalar_vars
        self.image_input_scales = image_input_scales
        self.image_output_scales = image_output_scales
        self.image_shape = image_shape

    # MUST OVERWRITE
    def transform(self, values, scale_type):
        if scale_type == "image":
            data_scaled = np.array([values / self.image_input_scales])

        elif scale_type == "scalar":
            data_scaled = self.model_value_min + (
                (values - self.input_offsets[0 : self.n_scalar_vars])
                * (self.model_value_max - self.model_value_min)
                / self.input_scales[0 : self.n_scalar_vars]
            )

            data_scaled = np.array([data_scaled])

        return data_scaled

    # MUST OVERWRITE
    def inverse_transform(self, values, scale_type):
        if scale_type == "image":
            data_unscaled = values * self.image_output_scales
            data_unscaled = data_unscaled.reshape(
                (1, int(self.image_shape[0] * self.image_shape[1]))
            )

        elif scale_type == "scalar":
            data_unscaled = (
                (values - self.model_value_min)
                * (self.output_scales)
                / (self.model_value_max - self.model_value_min)
            ) + self.output_offsets

        # data_unscaled = data_unscaled.reshape(self.image_shape)
        return data_unscaled


class MySurrogateModel(SurrogateModel):
    def __init__(self, model_file, scaler):
        # Set up base class
        super(MySurrogateModel, self).__init__(model_file)
        self.model_file = model_file
        self.scaler = scaler

        # can use the utility function to load the model info
        # and populate needed info
        model_info = load_model_info(model_file)
        self.ndim = model_info["ndim"]

        input_ordering = model_info["input_ordering"]
        output_ordering = model_info["output_ordering"]

        # TEMPORARY PATCH FOR INPUT/OUTPUT REDUNDANT VARS
        self.input_ordering = apply_temporary_ordering_patch(input_ordering, "in")
        self.output_ordering = apply_temporary_ordering_patch(output_ordering, "out")

        # Configure model attributes and setup model session
        self.configure(model_info)

    # no need for an evaluate method,
    # can handle everything in predict
    def predict(self, input_values):
        input_values["image"] = DEFAULT_LASER_IMAGE
        image_input = self.scaler.transform(np.array(input_values["image"]), "image")

        scalar_inputs = np.array([input_values[key] for key in self.input_ordering])
        scalar_inputs = self.scaler.transform(scalar_inputs, "scalar")

        # now that this is scaled, we access the model attribute loaded by the
        # SurrogateModel base class
        with self.thread_graph.as_default():
            predicted_output = self.model.predict([image_input, scalar_inputs])

        # process the model output
        image_output = np.array(predicted_output[0])
        scalar_output = predicted_output[1]

        # unscale
        image_output = self.scaler.inverse_transform(image_output, "image")
        scalar_output = self.scaler.inverse_transform(scalar_output, "scalar")

        # select extents
        extent_output = scalar_output[
            :, int(len(self.output_ordering) - self.ndim[0]) :
        ]

        # Now, format for server use
        formatted_output = dict(zip(self.output_ordering, scalar_output.T))

        # convert from arrays
        for scalar in self.output_ordering:
            formatted_output[scalar] = formatted_output[scalar][0]

        image_values = np.zeros((2 + len(extent_output[0]) + image_output.shape[1],))

        image_values[0] = self.scaler.image_shape[0]
        image_values[1] = self.scaler.image_shape[1]
        image_values[2:6] = extent_output
        image_values[6:] = image_output

        # format image data for viewing
        formatted_output["x:y"] = image_values

        return apply_temporary_output_patch(formatted_output)
