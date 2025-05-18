from eigen import Tensor


b = Tensor.ones((2, 3, 3))


print(b.matmul(b).numpy())
