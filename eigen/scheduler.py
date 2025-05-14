from __future__ import annotations
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eigen.node import Node


class Scheduler:
    work: Queue[Node | None]
    running: bool
    pool: list[Thread]

    def __init__(self, size: int = 4):
        self.work = Queue()
        self.pool = [Thread(target=self.worker_loop) for _ in range(size)]
        self.running = True
        for w in self.pool:
            w.start()

    def worker_loop(self):
        while True:
            node = self.work.get()
            if node is None:
                self.work.task_done()
                break
            result = node.fire()
            for edge in node.outputs:
                edge.forward(result, self)
            if len(node.outputs) == 0:
                node.outputs.append(result)
            self.work.task_done()

    def shutdown(self):
        for _ in self.pool:
            self.work.put(None)
        self.work.join()
        for w in self.pool:
            w.join()
