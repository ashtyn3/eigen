from __future__ import annotations

from typing import TYPE_CHECKING

from eigen.edge import Edge
from eigen.node import GenericKernel
from eigen.node import Node

if TYPE_CHECKING:
    from eigen.node import Node


class Graph:
    ontology: dict[str, Node] = {}

    def add_node(self, n: str, src: GenericKernel) -> Node:
        self.ontology[n] = Node(n, src)
        return self.ontology[n]

    def connect(self, f: str, t: str):
        e = Edge(self.ontology[f], self.ontology[t], self.ontology[f].name)
        self.ontology[f].outputs.append(e)
