from eigen import Tensor


b = Tensor((3, 2), fill=1)


print(b.numpy().mean(axis=1))
print(b.mean(1).numpy())
