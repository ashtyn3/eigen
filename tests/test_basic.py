import eigen
from eigen import Graph
from eigen import Scheduler
# from eigen import Runtime


def test_basic():
    def add(a: int, b: int):
        return a + b

    def n1():
        return 2

    def n2():
        return 2

    g = Graph()

    b = g.add_node("bob", add)

    x = g.add_node("a", n1)
    y = g.add_node("b", n2)

    t = g.add_node("steve", add)

    g.connect("a", "bob")
    g.connect("b", "bob")

    g.connect("bob", "steve")
    g.connect("bob", "steve")

    x.immediate(g.scheduler)
    y.immediate(g.scheduler)

    g.scheduler.work.join()
    g.scheduler.shutdown()

    # _ = Runtime()
    assert t.outputs[0] == 8


h = eigen.Tensor((2, 3), [0, 1, 2, 3, 4, 5])
# b = eigen.Tensor((3, 1), [0, 1, 2])

n = eigen.cpu_runtime.CPU_Ops(h).sum(0)
n2 = eigen.cpu_runtime.CPU_Ops(h).sum(1)
print(n)
print(n2)
