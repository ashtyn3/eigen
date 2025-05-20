from eigen import Tensor


n = Tensor.rand((2, 4, 4))

print(n.matmul(n).numpy())
