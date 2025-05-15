from __future__ import annotations

from typing import TYPE_CHECKING

from eigen.edge import Edge
from eigen.node import GenericKernel, Node
from eigen.scheduler import Scheduler
from eigen.tensor import Tensor

if TYPE_CHECKING:
    from eigen.node import Node


class Graph:
    ontology: dict[str, Node] = {}
    op_counter: dict[str, int] = {}
    scheduler: Scheduler

    def __init__(self):
        self.scheduler = Scheduler()

    def add_node(self, n: str, src: GenericKernel) -> Node:
        self.ontology[n] = Node(n, src, self.scheduler)
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
        else:
            v = Tensor((1, 1), [v])

        def const():
            return v

        return self.add_node(node_name, const).immediate(self.scheduler)
