from eigen import Tensor


a = Tensor((4, 4), fill=1)
b = Tensor((4, 4), fill=2)
z = a + b


b = z.sum(0)
print(b.numpy())
