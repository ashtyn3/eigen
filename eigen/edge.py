from __future__ import annotations
from typing import TYPE_CHECKING
from eigen.scheduler import Scheduler

if TYPE_CHECKING:
    from eigen.node import Node


class Edge:
    name: str
    f: Node
    t: Node

    def __init__(self, f: Node, t: Node, n: str):
        self.f = f
        self.t = t
        self.name = n

    def forward(self, res: object, s: Scheduler):
        assert isinstance(s, Scheduler)
        self.t.put(res, s)
