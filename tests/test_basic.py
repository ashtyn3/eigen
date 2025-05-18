from eigen import Tensor


a = Tensor((4, 4), fill=2)


print(a.matmul(a).numpy())
