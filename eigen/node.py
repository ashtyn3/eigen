from __future__ import annotations
import inspect
import os
import threading
from typing import Callable, Self, Union
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eigen.tensor import Tensor

    Consts = Union[Tensor, int, float]


GenericKernel = Callable[..., object]


class Node:
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

        self.inputs: list[Consts] = inputs

        self.outputs = []
        self.lock = threading.Lock()

    def forward(self):
        from eigen.tensor import Tensor

        # in_data = [t.realize() for t in self.inputs]
        in_data = []
        for t in self.inputs:
            if isinstance(t, Tensor):
                in_data.append(t.realize())
                continue
            in_data.append(t)
        return self.kernel(*in_data)

    def GPU(self) -> Self:
        self.gpu = True
        return self

    def ready(self) -> bool:
        return all(x is not None for x in self.inputs)

    def reset(self):
        for [i, _] in enumerate(self.inputs):
            self.inputs[i] = None
