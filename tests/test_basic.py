from eigen import Tensor


b = Tensor((3, 2), fill=1)


print(b.numpy().sum(axis=0))
print(b.sum(0).realize()._buffer)
