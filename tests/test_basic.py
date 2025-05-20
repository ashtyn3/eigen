from eigen import Tensor


a = Tensor.arange(8).reshape((4, 2))
n = Tensor.zeros((4, 1))

print((a**2).numpy())
