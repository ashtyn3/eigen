from __future__ import annotations

from typing import TYPE_CHECKING

from eigen.edge import Edge
from eigen.node import GenericKernel, Node
from eigen.scheduler import Scheduler

if TYPE_CHECKING:
    from eigen.node import Node


class Graph:
    ontology: dict[str, Node] = {}
    scheduler: Scheduler

    def __init__(self):
        self.scheduler = Scheduler()

    def add_node(self, n: str, src: GenericKernel) -> Node:
        self.ontology[n] = Node(n, src, self.scheduler)
        return self.ontology[n]

    def connect(self, f: str, t: str):
        e = Edge(self.ontology[f], self.ontology[t], self.ontology[f].name)
        self.ontology[f].outputs.append(e)
