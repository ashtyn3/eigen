from eigen import Tensor


n = Tensor.rand((4, 2))
a = Tensor.rand((2, 3))

print(n.matmul(a).numpy())
