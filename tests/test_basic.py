from eigen import Tensor


b = Tensor((3, 2), fill=1)
twos = Tensor((3, 2), fill=2)

res = b / twos
a = res + twos

print(res.numpy())

print(a.numpy())
