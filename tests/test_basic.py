from eigen import Tensor
import eigen.lazy
import json


n = Tensor.rand((2, 100, 10))
a = Tensor.rand((2, 10, 100))


b = Tensor.rand((100, 100))

(n.linear(a, b) + 69).realize()
