from eigen import Tensor


a = Tensor.arange(8).reshape((4, 2))
n = Tensor.fill((2, 2), 2)

print(a.matmul(n).numpy())
