from eigen.tensor import Tensor


class Layer:
    def backward() -> Tensor:
        raise NotImplementedError

    def forward() -> Tensor:
        raise NotImplementedError
