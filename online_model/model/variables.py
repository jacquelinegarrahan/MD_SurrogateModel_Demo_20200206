import numpy as np
import xarray as xr
from enum import Enum
from typing import Any, List, Union, Optional
from pydantic import BaseModel
from online_model import DEFAULT_PRECISION

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


# custom validator for xarray DataArrays
class XarrayDataArray(xr.DataArray):
    __slots__ = []  # xarrray requires explicit definition of slots on subclasses

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> xr.DataArray:
        # validate data...
        if not isinstance(v, xr.DataArray):
            raise TypeError("Xarray DataArray required")
        return v


class IOEnum(str, Enum):
    input = "input"
    output = "output"


class ProcessVariable(BaseModel):
    name: str
    io_type: IOEnum  # requires selection of input or output for creation
    units: Optional[str]
    # defaults for pvdb
    type: str = "float"
    precision: int = DEFAULT_PRECISION

    class Config:
        use_enum_values = True


class ScalarProcessVariable(ProcessVariable):
    value: Optional[float]
    default: Optional[float]
    range: Optional[Union[NumpyNDArray, XarrayDataArray]]


class NDProcessVariable(ProcessVariable):
    value: Optional[NumpyNDArray]
    default: Optional[NumpyNDArray]
    range: Optional[Union[NumpyNDArray, XarrayDataArray]]
