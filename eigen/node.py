from __future__ import annotations
import os

from typing import TYPE_CHECKING, Callable, Self, Union
import hashlib

from eigen.lazy import LazyOp, tensor_map, node_map
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
        if node_map.get(op) is None:
            node_map.set(op, self)

        return op

    @classmethod
    def make_const(cls, tensor, key):
        op = LazyOp(Ops.CONST, srcs=(key,))
        tensor_map.set(op, tensor)

        def kernel(*args):
            return None

        return cls(op=Ops.CONST, kernel=kernel, inputs=(key,))

    def debug(self):
        tree = self._walk().toposort(debug=True)
        print(f"{'step':<4} {'name':<6} {'op':<10} args")

        op_ids = {}
        items = []
        counter = 0

        for i, op in enumerate(tree):
            if op not in op_ids:
                op_ids[op] = f"op{counter}"
                counter += 1

            args = []
            for s in op.srcs:
                if isinstance(s, LazyOp):
                    if s not in op_ids:
                        op_ids[s] = f"op{counter}"
                        counter += 1
                    args.append(op_ids[s])
                else:
                    if op.op == Ops.CONST:
                        if s in items:
                            args.append(str(items.index(s)))
                        else:
                            items.append(s)
                            args.append(str(len(items) - 1))
                    else:
                        args.append(str(s))

            print(f"{i:<4} {op_ids[op]:<6} {str(op.op):<10} {', '.join(args)}")

    def forward(self, cache=None):
        tree = self._walk()
        exec_items = tree.toposort()
        results = []

        for item in exec_items:
            cached = tensor_map.get(item)
            if cached is not None:
                results.append(cached)
                continue

            inputs = []
            for src in item.srcs:
                d = tensor_map.get(src)
                inputs.append(d if d is not None else src)

            res = node_map.get(item).kernel(*inputs)
            tensor_map.set(item, res)
            results.append(res)

        if not results:
            return None
        return results[-1]

    def GPU(self) -> Self:
        self.gpu = True
        return self
