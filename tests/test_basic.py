from eigen import Tensor


a = Tensor.arange(8).reshape((4, 2))
b = Tensor.arange(start=1, stop=9).reshape((4, 2))

a.realize()
b.realize()
print(a)
