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
