from eigen import Tensor
import json


n = Tensor.rand((2, 100, 10))
a = Tensor.rand((2, 10, 100))

res = n.matmul(a).sum(1)
# res.realize()

# print(res.sum(1).sum(1).sum(0).numpy())
print(json.dumps(res.node.edge_list()))
