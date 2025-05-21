from __future__ import annotations
import os

from typing import TYPE_CHECKING, Callable, Self, Union
from functools import cached_property
import inspect
import sys
import base64
import json


from eigen.lazy import LazyOp, tensor_map, node_map
import eigen.lazy
from eigen.ops import Ops

if TYPE_CHECKING:
    from eigen.tensor import Tensor

    Consts = Union[Tensor, int, float]


GenericKernel = Callable[..., object]


class Node:
    kernel: GenericKernel
    gpu: bool
    computed: bool

    @cached_property
    def _source(self):
        return inspect.getsource(self.kernel)

    def __init__(self, kernel: GenericKernel, op, inputs=[]):
        self.kernel = kernel
        self.gpu = False
        self.inputs: list[Consts] = inputs
        self.op = op
        self.computed = False

    def _walk(self) -> LazyOp:
        from eigen.tensor import Tensor

        input_ops = []
        dtype = None
        for inp in self.inputs:
            if isinstance(inp, Tensor):
                op = inp.node._walk()
                input_ops.append(op)
                dtype = inp.dtype
            else:
                input_ops.append(inp)

        op = LazyOp(self.op, input_ops, dtype=dtype)
        if node_map.get(op) is None:
            node_map.set(op, self)

        return op

    @classmethod
    def make_const(cls, tensor, key):
        op = LazyOp(Ops.CONST, srcs=(key,), dtype=tensor.dtype)
        tensor_map.set(op, tensor)

        def kernel(*args):
            return None

        return cls(op=Ops.CONST, kernel=kernel, inputs=(key,))

    def edge_list(self):
        tree = self._walk().toposort(debug=True)
        ops: dict[LazyOp, str] = {}
        edges = []
        counter = 0
        for op in tree:
            if op not in ops:
                ops[op] = f"op{counter}"
                counter += 1
            current_op = ops[op]
            for s in op.srcs:
                if isinstance(s, LazyOp):
                    if s not in ops:
                        ops[s] = f"op{counter}"
                        counter += 1
                    s_id = ops[s]
                    edges.append((s_id, current_op))
        jsonable = {}

        for k, v in ops.items():
            jsonable[v] = k.to_json()

        return {"edges": edges, "nodes": jsonable}

    def debug(self):
        from tabulate import tabulate

        tree = self._walk().toposort(debug=True)
        headers = ["step", "name", "op", "args"]
        table = []

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
                            args.append("#" + str(items.index(s)))
                        else:
                            items.append(s)
                            args.append("#" + str(len(items) - 1))
                    else:
                        args.append(str(s))

            table.append([i, op_ids[op], str(op.op), ", ".join(args)])

        print(tabulate(table, headers=headers, tablefmt="rounded_grid"))

    def forward(self, cache=None):
        self.computed = True
        tree = self._walk()
        exec_items = tree.toposort()
        eigen.lazy.root_node = node_map.get(exec_items[-1])
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
            node_map.get(item).computed = True
            tensor_map.set(item, res)
            results.append(res)

            if item == exec_items[-1]:
                if int(os.getenv("G", "0")):
                    grump_path = (
                        os.path.join(os.path.dirname(__file__))
                        + "/grumpy/serve.py"
                    )
                    os.environ["G_DATA"] = str(
                        base64.b64encode(
                            json.dumps(
                                eigen.lazy.root_node.edge_list()
                            ).encode()
                        ).decode()
                    )
                    os.execv(sys.executable, [sys.executable, grump_path])

        if not results:
            return None
        return results[-1]

    def GPU(self) -> Self:
        self.gpu = True
        return self
