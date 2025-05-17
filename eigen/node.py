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

    def _walk(self) -> LazyOp:
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
        op = LazyOp(Ops.CONST, srcs=(key,))
        tensor_map.set(op, tensor)

        def kernel(*args):
            return tensor

        # no need to include tensor
        return cls(op=Ops.CONST, kernel=kernel, inputs=(key,))

    def forward(self, cache=None):
        tree = self._walk()
        exec_items = tree.toposort()
        results = []
        for item in exec_items:
            inputs = []
            for src in item.srcs:
                if (d := tensor_map.get(src)) is not None:
                    inputs.append(d)
            res = self.kernel(*inputs)
            tensor_map.set(item, res)
            results.append(res)
        return results[-1]

    def GPU(self) -> Self:
        self.gpu = True
        return self
