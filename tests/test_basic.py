from eigen import Tensor


a = Tensor((4, 4), fill=1)
z = a + a


b = z.matmul(z)
print(b.numpy())
print(b._node._walk().toposort())
# print(b.numpy())
