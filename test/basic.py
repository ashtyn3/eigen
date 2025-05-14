import threading

from eigen import Graph
from eigen import Scheduler
# from eigen import Runtime


def add(a: int, b: int):
    return a + b


def n1():
    return 2


def n2():
    return 2


g = Graph()
s = Scheduler()

b = g.add_node("bob", add)

x = g.add_node("a", n1).GPU()
y = g.add_node("b", n2)

t = g.add_node("steve", add)


g.connect("a", "bob")
g.connect("b", "bob")

g.connect("bob", "steve")
g.connect("bob", "steve")

x.immediate(s)
y.immediate(s)

s.work.join()
s.shutdown()

# _ = Runtime()
print(t.outputs[0])
