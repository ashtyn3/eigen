from eigen import Tensor


h = Tensor((3, 3), fill=1)
b = Tensor((3, 3), fill=1)
n = h + b
a = n - h * Tensor((3, 3), fill=5)
s = (a / a).sum(0)

print(s.numpy())
