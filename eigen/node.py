from __future__ import annotations
import inspect
import os
import threading
from typing import Callable, Self
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eigen.tensor import Tensor


GenericKernel = Callable[..., object]


class Node:
    # name: str
    kernel: GenericKernel
    inputs: list[object]
    gpu: bool
    lock: threading.Lock
    named: dict[int, str]

    def __init__(self, kernel: GenericKernel, inputs=[]):
        # self.name = name
        self.kernel = kernel
        self.gpu = False
        self.named = {}
        kernel_data = inspect.signature(kernel)
        for [i, v] in enumerate(kernel_data.parameters.values()):
            self.named[v] = i

        self.inputs: list[Tensor] = inputs

        self.outputs = []
        self.lock = threading.Lock()

    def forward(self):
        in_data = [t.realize() for t in self.inputs]
        return self.kernel(*in_data)

    def GPU(self) -> Self:
        self.gpu = True
        return self

    def ready(self) -> bool:
        return all(x is not None for x in self.inputs)

    def reset(self):
        for [i, _] in enumerate(self.inputs):
            self.inputs[i] = None
