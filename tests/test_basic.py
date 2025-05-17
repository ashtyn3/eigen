from eigen import Tensor


b = Tensor.ones((3, 2))


print(b.numpy().mean(axis=1))
print(b.mean(1).numpy())
