from eigen import Tensor
import eigen.lazy
import json


n = Tensor.rand((2, 100, 10))
a = Tensor.rand((2, 10, 100))

res = n.matmul(a).sum(1) + 2
res.realize()
