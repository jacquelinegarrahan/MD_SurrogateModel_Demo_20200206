import numpy as np
import xarray as xr
from enum import Enum
from typing import Any, List, Union, Optional
from pydantic import BaseModel, Field

# custom validator for ndarrays
class NumpyNDArray(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> np.ndarray:
        # validate data...
        if not isinstance(v, np.ndarray):
            raise TypeError("Numpy array required")
        return v


class IOEnum(str, Enum):
    pv_input = "input"
    pv_output = "output"


class ProcessVariable(BaseModel):
    name: str
    io_type: IOEnum  # requires selection of input or output for creation
    # defaults for pvdb
    value_type: str = Field("float", alias="type")
    precision: int = 8

    class Config:
        use_enum_values = True


class ScalarProcessVariable(ProcessVariable):
    value: Optional[float]
    default: Optional[float]
    value_range: list = Field(alias="range")
    units: Optional[str]


class NDProcessVariable(ProcessVariable):
    value: Optional[NumpyNDArray]
    default: Optional[NumpyNDArray]
    value_range: list = Field(alias="range")
    units: str


class ImageProcessVariable(NDProcessVariable):
    color_mode: int = 0
    shape: tuple  # need for channel access AreaDetector
