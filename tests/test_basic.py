from eigen import Tensor


a = Tensor.arrange(8).reshape((4, 2))


b = (a * Tensor.fill((4, 2), 0.5)).reshape((2, 4))

print(a, b)
