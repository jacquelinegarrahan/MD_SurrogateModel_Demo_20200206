import numpy as np

# image input
# reshape because isn't in the correct format
default_image_array = np.load("online_model/files/example_input_image.npy")
# default_image_array = default_image_array.reshape((50, 50))


VARIABLES = {
    "laser_radius": {
        "pv_type": "scalar",
        "value": 3.47986980e-01,
        "default": 3.47986980e-01,
        "units": "mm",
        "range": np.array([1.000000e-01, 5.000000e-01]),
        "xarray_dim": ("length",),
        "is_input": 1,
    },
    "maxb(2)": {
        "pv_type": "scalar",
        "value": 4.02751972e-02,
        "default": 4.02751972e-02,
        "units": "T",
        "range": np.array([0.000000e00, 1.000000e-01]),
        "xarray_dim": ("induction",),
        "is_input": 1,
    },
    "phi(1)": {
        "pv_type": "scalar",
        "value": -7.99101687e00,
        "default": -7.99101687e00,
        "units": "degrees",
        "range": np.array([-1.000000e01, 1.000000e01]),
        "xarray_dim": ("phi",),
        "is_input": 1,
    },
    "total_charge:value": {
        "pv_type": "scalar",
        "value": 1.41576322e02,
        "default": 1.41576322e02,
        "units": "pC",
        "range": np.array([0.000000e00, 3.000000e02]),
        "xarray_dim": ("charge",),
        "is_input": 1,
    },
    "in_xmin": {
        "pv_type": "scalar",
        "value": -3.53964583e-04,
        "default": -3.53964583e-04,
        "units": "m",
        "range": np.array([-4.216000e-04, 3.977000e-04]),
        "xarray_dim": ("length",),
        "is_input": 1,
    },
    "in_xmax": {
        "pv_type": "scalar",
        "value": 3.45778376e-04,
        "default": 3.45778376e-04,
        "units": "m",
        "range": np.array([-1.117627e-01, 1.120053e-01]),
        "xarray_dim": ("length",),
        "is_input": 1,
    },
    "in_ymin": {
        "pv_type": "scalar",
        "value": -3.47874295e-04,
        "default": -3.47874295e-04,
        "units": "m",
        "range": np.array([-1.117627e-01, 1.120053e-01]),
        "xarray_dim": ("length",),
        "is_input": 1,
    },
    "in_ymax": {
        "pv_type": "scalar",
        "value": 3.45778376e-04,
        "default": 3.45778376e-04,
        "units": "m",
        "range": np.array([-1.117627e-01, 1.120053e-01]),
        "xarray_dim": ("length"),
        "is_input": 1,
    },
    #    "image": {
    #        "pv_type": "image",
    #        "value": default_image_array,
    #        "default": default_image_array,
    #        "units": "W",
    #        "value_range": np.array([0.0, 9.0]),
    #        "xarray_dim": ("length_x", "length_y", "length_z"),
    #        "is_input": 1,
    #    },
    "end_core_emit_95percent_x": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "mm-mrad",
        "xarray_dim": ("mm-mrad",),
        "is_input": 0,
    },
    "end_core_emit_95percent_y": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "mm-mrad",
        "xarray_dim": ("mm-mrad",),
        "is_input": 0,
    },
    "end_core_emit_95percent_z": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "mm-mrad",
        "xarray_dim": ("mm-rad",),
        "is_input": 0,
    },
    "end_mean_kinetic_energy": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "eV",
        "xarray_dim": ("eV",),
        "is_input": 0,
    },
    "end_mean_x": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "mm",
        "xarray_dim": ("mm",),
        "is_input": 0,
    },
    "end_mean_y": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "mm",
        "xarray_dim": ("mm",),
        "is_input": 0,
    },
    "end_n_particle_loss": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "number",
        "xarray_dim": ("number",),
        "is_input": 0,
    },
    "end_norm_emit_x": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "mm-mrad",
        "xarray_dim": ("mm-rad",),
        "is_input": 0,
    },
    "end_norm_emit_y": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "mm-mrad",
        "xarray_dim": ("mm-rad",),
        "is_input": 0,
    },
    "end_norm_emit_z": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "mm-mrad",
        "xarray_dim": ("mm-rad",),
        "is_input": 0,
    },
    "end_sigma_x": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "mm",
        "xarray_dim": ("mm",),
        "is_input": 0,
    },
    "end_sigma_xp": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "mrad",
        "xarray_dim": ("mrad",),
        "is_input": 0,
    },
    "end_sigma_y": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "mm",
        "xarray_dim": ("mm",),
        "is_input": 0,
    },
    "end_sigma_yp": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "mrad",
        "xarray_dim": ("mrad",),
        "is_input": 0,
    },
    "end_sigma_z": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "mm",
        "xarray_dim": ("mm",),
        "is_input": 0,
    },
    "end_total_charge": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "C",
        "xarray_dim": ("C",),
        "is_input": 0,
    },
    "out_xmin": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "m",
        "xarray_dim": ("m",),
        "is_input": 0,
    },
    "out_xmax": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "m",
        "xarray_dim": ("m",),
        "is_input": 0,
    },
    "out_ymin": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "m",
        "xarray_dim": ("m",),
        "is_input": 0,
    },
    "out_ymax": {
        "pv_type": "scalar",
        "default": 0.0,
        "units": "m",
        "xarray_dim": ("m",),
        "is_input": 0,
    },
    "x:y": {"pv_type": "image", "xarray_dim": (), "is_input": 0, "units": "mm:mm"},
}
