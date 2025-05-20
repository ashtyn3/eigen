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
        from_shape = self.from_t.shape
        to_shape = self.to_t

        # Pad the shorter shape with leading 1s for proper alignment
        len_diff = len(to_shape) - len(from_shape)
        if len_diff > 0:
            from_shape = (1,) * len_diff + from_shape
        elif len_diff < 0:
            to_shape = (1,) * (-len_diff) + to_shape

        result = []
        for t, o in zip(from_shape, to_shape):
            if t == o or t == 1 or o == 1:
                result.append(max(t, o))
            else:
                raise ValueError(f"Cannot broadcast dimensions: {t} and {o}")

        return tuple(result)

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
