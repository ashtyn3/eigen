from eigen import Tensor


a = Tensor.arrange(8).reshape((4, 2))


b = a * 0.5 + 2

print(b.numpy())
