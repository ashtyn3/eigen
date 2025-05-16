from eigen import Tensor


h = Tensor((3, 2), fill=1)
b = Tensor((3, 2), fill=1)
n = h + b

print(n.realize())
print(n)
