from __future__ import annotations

from typing import TYPE_CHECKING
import os

from eigen.edge import Edge
from eigen.node import GenericKernel, Node
from eigen.scheduler import Scheduler
from eigen.tensor import Tensor
from eigen.ops import OpsTrait

if TYPE_CHECKING:
    from eigen.node import Node


class Graph:
    ontology: dict[str, Node] = {}
    op_counter: dict[str, int] = {}
    scheduler: Scheduler
    runtime: OpsTrait

    def __init__(self):
        self.scheduler = Scheduler()
        if int(os.getenv("METAL", "0")):
            raise NotImplementedError
        else:
            from eigen.cpu_runtime import CPU_Ops

            self.runtime = CPU_Ops

    def add_node(self, n: str, src: GenericKernel) -> Node:
        self.ontology[n] = Node(n, src, self.scheduler)
        if int(os.getenv("METAL", "0")):
            self.ontology[n].GPU()
        return self.ontology[n]

    def connect(self, f: str, t: str):
        e = Edge(self.ontology[f], self.ontology[t], self.ontology[f].name)
        self.ontology[f].outputs.append(e)

    def constant(self, v, name: str | None = None) -> Node:
        if self.op_counter.get("constant") is None:
            self.op_counter["constant"] = 0

        self.op_counter["constant"] = self.op_counter["constant"] + 1

        node_name = f"constant_{self.op_counter['constant']}"
        if name is not None:
            node_name = name

        if isinstance(v, list):
            v = Tensor((1, 1), v)
        elif isinstance(v, Tensor):
            v = v
        else:
            v = Tensor((1, 1), [v])

        def const():
            return v

        return self.add_node(node_name, const).immediate(self.scheduler)
