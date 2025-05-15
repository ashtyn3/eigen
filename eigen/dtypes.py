from enum import Enum, IntEnum
from typing import Literal, Optional, Union
from dataclasses import dataclass, fields
import ctypes
import typing


FmtStr = Literal["?", "b", "B", "h", "H", "i", "I", "q", "Q", "e", "f", "d"]


class FastEnum(IntEnum):
    def __str__(self):
        return Enum.__str__(self)

    @staticmethod
    def _generate_next_value_(_, __, ___, last_values):
        return 1 + max([
            0,
            *last_values,
            *[max(c) for c in FastEnum.__subclasses__()],
        ])


Consts = Union[float, int, bool]


def py(t):
    if isinstance(t, float):
        return dtypes.float32
    if isinstance(t, int):
        return dtypes.int32


@dataclass
class Eigen_Dtype:
    name: str
    size: int
    py_str: str

    def is_int(self):
        if self in dtypes.all_ints:
            return True
        return False

    def is_float(self):
        if self in dtypes.floats:
            return True
        return False

    def is_bool(self):
        if self.py_str == "b":
            return True
        return False

    def const(self):
        return (
            int() if self.is_int() else float() if self.is_float() else bool()
        )

    def ctype(self):
        assert self.py_str in typing.get_args(FmtStr)
        return {
            "f": ctypes.c_float,
            "d": ctypes.c_double,
            "i": ctypes.c_int32,
            "q": ctypes.c_long,
        }[self.py_str]


class dtypes:
    float32: Eigen_Dtype = Eigen_Dtype("float", 4, "f")
    float64: Eigen_Dtype = Eigen_Dtype("double", 8, "d")

    int32: Eigen_Dtype = Eigen_Dtype("int", 4, "i")
    int64: Eigen_Dtype = Eigen_Dtype("long", 8, "q")

    uint32: Eigen_Dtype = Eigen_Dtype("unsigned int", 4, "I")
    uint64: Eigen_Dtype = Eigen_Dtype("unsigned long", 8, "Q")

    bool: Eigen_Dtype = Eigen_Dtype("bool", 1, "b")
    sig_ints = (int32, int64)
    unsig_ints = (uint32, uint64)
    all_ints = sig_ints + unsig_ints
    floats = (float32, float64)
