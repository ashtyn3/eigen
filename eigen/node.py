from __future__ import annotations
import inspect
import os
import threading
from typing import Callable, Self
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eigen.scheduler import Scheduler
    from eigen.edge import Edge


GenericKernel = Callable[..., object]


class Node:
    name: str
    kernel: GenericKernel
    inputs: list[object]
    outputs: list["Edge"]
    gpu: bool
    lock: threading.Lock
    named: dict[int, str]
    scheduler: Scheduler

    def __init__(self, name: str, kernel: GenericKernel, sched: Scheduler):
        self.name = name
        self.kernel = kernel
        self.gpu = False
        self.named = {}
        self.scheduler = sched
        kernel_data = inspect.signature(kernel)
        for [i, v] in enumerate(kernel_data.parameters.values()):
            self.named[v] = i

        self.inputs: list[object] = [None] * len(kernel_data.parameters)

        self.outputs = []
        self.lock = threading.Lock()

    def GPU(self) -> Self:
        self.gpu = True
        return self

    def ready(self) -> bool:
        return all(x is not None for x in self.inputs)

    def reset(self):
        for [i, _] in enumerate(self.inputs):
            self.inputs[i] = None

    def put(self, v: object, k: str | None = None):
        with self.lock:
            if k is not None:
                self.inputs[self.named[k]] = v
            else:
                self.inputs[self.inputs.index(None)] = v
            # self.inputs.append(v)
            if self.ready():
                self.scheduler.work.put(self)

    def immediate(self, s: Scheduler) -> Node:
        with self.lock:
            if self.ready():
                s.work.put(self)
        return self

    def fire(self):
        if not self.ready():
            if int(os.getenv("V", "0")):
                print(f"info: skipping name={self.name}")
            return
        if int(os.getenv("V", "0")):
            print(f"info: running name={self.name}")

        with self.lock:
            out = self.kernel(*self.inputs)
            self.reset()

        return out
