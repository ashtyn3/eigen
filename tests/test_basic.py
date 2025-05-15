# import eigen
from eigen import Tensor
# from eigen import Runtime


h = Tensor((3, 3), fill=1)
b = Tensor((3, 3), fill=1)
n = h + b
a = n - h * Tensor((3, 3), fill=5)

print((a / a).numpy())
