from eigen import Tensor


a = Tensor((4, 4), fill=1)
b = Tensor((4, 4), fill=2)
z = a + b
# z = z * z


b = z.sum(0)
print(b.numpy())
print(b.reshape((2, 2)).numpy())
