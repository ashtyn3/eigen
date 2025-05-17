import array
import ctypes

from eigen.dtypes import Consts, Eigen_Dtype, py
from eigen.node import Node
from eigen.device import Device
import numpy as np
import math
import eigen.lazy
import functools


class Tensor:
    _buffer: array.ArrayType
    dtype: Eigen_Dtype
    gpu: bool
    shape: tuple[int, int]
    flat_len: int
    _node: Node
    data: None | Consts
    realized: bool

    @property
    def flat_len(self):
        return math.prod(self.shape)

    def __hash__(self):
        return hash(str(self._buffer) + str(self.shape))

    def __init__(
        self,
        shape: tuple[int, int],
        data: list | None = None,
        fill: Consts | None = None,
        dtype: Eigen_Dtype | None = None,
        node: Node | None = None,
    ):
        self._node = node
        self.shape = shape
        self.realized = False
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

    @functools.cached_property
    def node(self):
        if self._node is None:
            key = hash(str(self._buffer) + str(self.shape))
            self._node = Node.make_const(self, key)
        return self._node

    @property
    def nbytes(self):
        return ctypes.sizeof(self.to_ctype()) * self.flat_len

    def __str__(self):
        real = "realized" if self.realized else "unrealized"
        return f"<dtype={self.dtype.name} shape={self.shape} nbytes={
            self.nbytes
        } {real}>"

    def realize(self):
        self.realized = True
        self.data = self.node.forward()
        return self.data

    def to_numpy(self):
        return np.array(self._buffer).reshape(self.shape)

    def numpy(self):
        if self.realized is False:
            self.data = self.node.forward()
            self.realized = True
        if self.data is None:
            return np.array(self._buffer).reshape(self.shape)

        return np.array(self.data._buffer).reshape(self.data.shape)

    def __add__(self, x):
        from eigen.ops import Ops

        def add_kernel(a_data, b_data):
            return Device().Runtime().add(a_data, b_data)

        out = Tensor(
            self.shape,
            dtype=self.dtype,
            node=Node(add_kernel, Ops.ADD, inputs=[self, x]),
        )

        return out

    def __sub__(self, x):
        from eigen.ops import Ops

        def add_kernel(a_data, b_data):
            return Device().Runtime().sub(a_data, b_data)

        out = Tensor(
            self.shape,
            dtype=self.dtype,
            node=Node(add_kernel, Ops.SUB, inputs=[self, x]),
        )

        return out

    def __mul__(self, x):
        from eigen.ops import Ops

        def add_kernel(a_data, b_data):
            return Device().Runtime().mul(a_data, b_data)

        out = Tensor(
            self.shape,
            dtype=self.dtype,
            node=Node(add_kernel, Ops.MUL, inputs=[self, x]),
        )

        return out

    def __truediv__(self, x):
        from eigen.ops import Ops

        def div_kernel(a_data, b_data):
            return Device().Runtime().div(a_data, b_data)

        out = Tensor(
            self.shape,
            dtype=self.dtype,
            node=Node(div_kernel, Ops.DIV, inputs=[self, x]),
        )

        return out

    def sum(self, axis=0):
        from eigen.ops import Ops

        def div_kernel(host, a):
            return Device().Runtime().sum(host, a)

        out = Tensor(
            self.shape,
            dtype=self.dtype,
            node=Node(div_kernel, Ops.SUM, inputs=[self, axis]),
        )

        return out

    def mean(self, axis=0):
        from eigen.ops import Ops

        def div_kernel(host, a):
            return Device().Runtime().mean(host, a)

        out = Tensor(
            self.shape,
            dtype=self.dtype,
            node=Node(div_kernel, Ops.MEAN, inputs=[self, axis]),
        )

        return out
