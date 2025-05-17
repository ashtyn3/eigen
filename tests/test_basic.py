from eigen import Tensor


b = Tensor((3, 2), fill=1)


print(b.numpy())
print(b.sum(1).numpy())
