from eigen import Tensor


n = Tensor.fill((2, 2), 2)

print(n.matmul(n).numpy())
