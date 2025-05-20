from eigen.tensor import Tensor
import math


class BroadcastView:
    from_t: Tensor
    other: Tensor

    def __init__(self, f, t: tuple[int]):
        self.to_t = t
        if isinstance(f, Tensor):
            self.from_t = f
        else:
            self.from_t = Tensor(self.to_t, fill=f)

    @property
    def to_shape(self):
        return self._to_shape()

    @property
    def from_strides(self):
        return self.compute_strides(self.from_t.shape)

    def _to_shape(self):
        result = []
        for t, o in zip(reversed(self.from_t.shape), reversed(self.to_t)):
            if t == 1:
                result.append(o)
            elif o == 1:
                result.append(t)
            elif t == o:
                result.append(t)
            else:
                raise ValueError("Cannot broadcast with shapes")

        longer = (
            self.from_t.shape
            if len(self.from_t.shape) > len(self.to_t)
            else self.to_t
        )
        result.extend(
            reversed(longer[: abs(len(self.from_t.shape) - len(self.to_t))])
        )
        return tuple(reversed(result))

    def _map_index(self, idx: int) -> int:
        out_idx = []
        tmp = idx
        for dim in reversed(self.to_shape):
            out_idx.insert(0, tmp % dim)
            tmp //= dim

        in_idx = 0
        for i, out_dim_idx in enumerate(out_idx[-len(self.from_t.shape) :]):
            in_dim = self.from_t.shape[i]
            stride = self.from_strides[i]
            dim_idx = out_dim_idx if in_dim != 1 else 0
            in_idx += dim_idx * stride

        return in_idx

    def __len__(self):
        return math.prod(self.to_shape)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int):
        return self.from_t._buffer[self._map_index(idx)]

    def __setitem__(self, idx: int, data):
        self.from_t._buffer[self._map_index(idx)] = data

    def compute_strides(self, shape):
        strides = []
        acc = 1
        for dim in reversed(shape):
            strides.insert(0, acc)
            acc *= dim
        return strides
