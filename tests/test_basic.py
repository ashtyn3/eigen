import eigen
from eigen import Graph
from eigen import Scheduler
from eigen import Tensor
# from eigen import Runtime


def test_basic():
    def add(a: Tensor, b: Tensor):
        return b

    g = Graph()

    x, y = (
        g.constant(9),
        g.constant(Tensor((3, 2), dtype=eigen.dtypes.dtypes.float32)),
    )

    t = g.add_node("steve", add)

    g.connect(x.name, t.name)
    g.connect(y.name, t.name)
    #
    # g.connect("bob", "steve")
    # g.connect("bob", "steve")

    g.scheduler.work.join()
    g.scheduler.shutdown()
    #
    print(t.outputs[0])
    # assert t.outputs[0] == 8


test_basic()
h = eigen.Tensor((2, 3), [0, 1, 2, 3, 4, 5])
# b = eigen.Tensor((3, 1), [0, 1, 2])

n = eigen.cpu_runtime.CPU_Ops(h).add(h)
print(n)
