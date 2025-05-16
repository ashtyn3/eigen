from eigen import Tensor


h = Tensor((3, 2), fill=1)
b = Tensor((3, 2), fill=1)
a = b + h
# n = a.sum(0)

for n in a.node._walk().toposort():
    print(n)
