from eigen import Tensor


a = Tensor.arange(8).reshape((4, 2))

print((a / 2).realize())

print(a)
