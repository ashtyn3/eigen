from eigen import Tensor


h = Tensor((3, 2), fill=1)
b = Tensor((3, 2), fill=1)
a = b + h

print(a.numpy())
# a.node.forward()
# n = a.sum(0)
