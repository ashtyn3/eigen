from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Self, Union
import hashlib

from eigen.lazy import LazyOp, tensor_map
from eigen.ops import Ops

if TYPE_CHECKING:
    from eigen.tensor import Tensor

    Consts = Union[Tensor, int, float]


GenericKernel = Callable[..., object]


class Node:
    kernel: GenericKernel
    gpu: bool

    def __init__(self, kernel: GenericKernel, op, inputs=[]):
        self.kernel = kernel
        self.gpu = False
        self.inputs: list[Consts] = inputs
        self.op = op

    def _walk(self):
        from eigen.tensor import Tensor

        input_ops = []
        for inp in self.inputs:
            if isinstance(inp, Tensor):
                op = inp.node._walk()
                input_ops.append(op)
            else:
                input_ops.append(inp)

        op = LazyOp(self.op, input_ops)

        return op

    @classmethod
    def make_const(cls, tensor, key):
        tensor_map.set(key, tensor)

        def kernel():
            return LazyOp(Ops.CONST, srcs=(key,))

        # no need to include tensor
        return cls(op=Ops.CONST, kernel=kernel, inputs=(key,))

    def GPU(self) -> Self:
        self.gpu = True
        return self
