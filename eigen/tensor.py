import array
import ctypes

from eigen.dtypes import Consts, Eigen_Dtype, py


class Tensor:
    _buffer: array.ArrayType
    dtype: Eigen_Dtype
    gpu: bool
    shape: tuple[int, int]
    flat_len: int

    @property
    def flat_len(self):
        return self.shape[0] * self.shape[1]

    def __init__(
        self,
        shape: tuple[int, int],
        data: list | None = None,
        fill: Consts | None = None,
        dtype: Eigen_Dtype | None = None,
    ):
        self.shape = shape
        if dtype is None:
            if data is not None:
                self.dtype = py(data[0])
            elif fill is not None:
                self.dtype = py(fill)
            else:
                raise TypeError("Cannot detect dtype")
        else:
            self.dtype = dtype

        self._buffer = array.array(self.dtype.py_str)

        if fill is not None:
            assert data is None
            self._buffer = [fill] * self.flat_len
        else:
            if data is not None:
                if self.flat_len != len(data):
                    raise ValueError("Data doesn't match shape")
                self._buffer = data
            else:
                self._buffer = [dtype.const()] * self.flat_len

    def to_ctype(self) -> ctypes.Array:
        view = (self.dtype.ctype() * self.flat_len)()
        view[:] = self._buffer
        return view

    @property
    def nbytes(self):
        return ctypes.sizeof(self.to_ctype()) * self.flat_len

    def __str__(self):
        rows, cols = self.shape
        s = "[\n"
        for r in range(rows):
            # s += "  [ "
            for c in range(cols):
                idx = r * cols + c
                s += f"{self._buffer[idx]} "
            s += "\n"
        s += "]"
        return f"<dtype={self.dtype.name} shape={self.shape} nbytes={self.nbytes}>\n{s}"
