from eigen import Tensor


h = Tensor((3, 2), fill=1)
b = Tensor((3, 2), fill=1)
a = b + h
# n = a.sum(0)

print(a.node._walk())
print(a.node._walk().srcs[0] == a.node._walk().srcs[1])
