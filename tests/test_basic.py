from eigen import Tensor


a = Tensor.arange(8)
a = Tensor.arange(4)

# n = Tensor.fill((2, 4), 2)

print(a.matmul(a).numpy())
