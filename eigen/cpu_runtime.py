import itertools
from concurrent.futures import ThreadPoolExecutor

import eigen.ops as ops
from eigen.dtypes import Eigen_Dtype
from eigen.broadcast import BroadcastView
from typing import Union

from typing import TYPE_CHECKING

from eigen.tensor import Tensor
from eigen.lazy import LazyOp
import operator
from functools import reduce


def compute_outer_axis_inner(shape, axis):
    def prod(seq):
        return reduce(operator.mul, seq, 1)

    if not shape:
        raise ValueError("Shape must be non-empty")

    if not (0 <= axis < len(shape)):
        raise ValueError(f"Axis {axis} is out of bounds for shape {shape}")

    outer = prod(shape[:axis])
    axis_dim = shape[axis]
    inner = prod(shape[axis + 1 :])

    return outer, axis_dim, inner


def broadcast(host: Tensor, x: Union[int, float, Tensor]):
    if not isinstance(x, Tensor):
        return Tensor.fill(host.shape, x)._buffer
    return x


class Runtime(ops.OpsTrait):
    dtype: Eigen_Dtype

    def add_op(self, host: Tensor, other: Tensor):
        to = BroadcastView(other, host.shape).to_shape
        # if host.shape != other.shape:
        #     raise ValueError("op needs matching shapes")
        return Tensor(
            to,
            [x + y for x, y in zip(BroadcastView(other, to), host._buffer)],
        )

    def sub_op(self, host: Tensor, other: Tensor):
        to = BroadcastView(other, host.shape).to_shape
        # if host.shape != other.shape:
        #     raise ValueError("op needs matching shapes")
        return Tensor(
            to,
            [x - y for x, y in zip(host._buffer, BroadcastView(other, to))],
        )

    def mul_op(self, host: Tensor, other: Tensor):
        to = BroadcastView(other, host.shape).to_shape
        # if host.shape != other.shape:
        #     raise ValueError("op needs matching shapes")
        return Tensor(
            to,
            [x * y for x, y in zip(host._buffer, BroadcastView(other, to))],
        )

    def div_op(self, host: Tensor, other: Tensor):
        to = BroadcastView(other, host.shape).to_shape
        # if host.shape != other.shape:
        #     raise ValueError("op needs matching shapes")
        return Tensor(
            to,
            [x / y for x, y in zip(host._buffer, BroadcastView(other, to))],
        )

    def pow_op(self, host: Tensor, other: Tensor):
        to = BroadcastView(other, host.shape).to_shape
        # if host.shape != other.shape:
        #     raise ValueError("op needs matching shapes")
        return Tensor(
            to,
            [x**y for x, y in zip(host._buffer, BroadcastView(other, to))],
        )

    def neg_op(self, host: Tensor):
        return Tensor(
            host.shape,
            [-x for x in host._buffer],
        )

    def abs_op(self, host: Tensor):
        return Tensor(
            host.shape,
            [abs(x) for x in host._buffer],
        )

    def sum_op(self, host: Tensor, axis: int = 0):
        shape = host.shape
        buffer = host._buffer

        outer, axis_dim, inner = compute_outer_axis_inner(shape, axis)

        result = []
        for o in range(outer):
            for i in range(inner):
                acc = 0
                for j in range(axis_dim):
                    idx = o * axis_dim * inner + j * inner + i
                    acc += buffer[idx]
                result.append(acc)

        new_shape = shape[:axis] + shape[axis + 1 :]
        if not new_shape:
            new_shape = (1,)
        return Tensor(new_shape, result)

    def mean_op(self, host: Tensor, axis: int):
        outer, axis_dim, inner = compute_outer_axis_inner(host.shape, axis)
        summed = self.sum_op(host, axis)
        return self.div_op(summed, Tensor(summed.shape, fill=axis_dim))

    def matmul_op(self, a: Tensor, b: Tensor):
        import math
        from eigen.broadcast import BroadcastView

        a_shape = a.shape
        b_shape = b.shape

        assert len(a_shape) >= 2 and len(b_shape) >= 2, (
            "Both tensors must be at least 2D"
        )
        M, K = a_shape[-2], a_shape[-1]
        K2, N = b_shape[-2], b_shape[-1]
        assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

        # Broadcast batch shape
        batch_shape = []
        for x, y in itertools.zip_longest(
            a_shape[:-2], b_shape[:-2], fillvalue=1
        ):
            if x == y or x == 1 or y == 1:
                batch_shape.append(max(x, y))
            else:
                raise ValueError(
                    f"Incompatible batch shapes: {a_shape} vs {b_shape}"
                )

        out_shape = tuple(batch_shape) + (M, N)
        a_view = BroadcastView(a, out_shape[:-2] + (M, K))
        b_view = BroadcastView(b, out_shape[:-2] + (K, N))

        result = []
        total_batches = int(math.prod(batch_shape)) if batch_shape else 1

        def compute_batch(b):
            for i in range(M):
                for j in range(N):
                    acc = 0
                    for k in range(K):
                        a_idx = b * M * K + i * K + k
                        b_idx = b * K * N + k * N + j
                        acc += a_view[a_idx] * b_view[b_idx]
                    result.append(acc)

        with ThreadPoolExecutor() as executor:
            executor.map(compute_batch, range(total_batches))

        return Tensor(out_shape, result)

    def prod_op(self, host: Tensor, axis: int = 0):
        shape = host.shape

        outer, axis_dim, inner = compute_outer_axis_inner(shape, axis)

        result = []
        for o in range(outer):
            for i in range(inner):
                acc = 1
                for j in range(axis_dim):
                    idx = o * axis_dim * inner + j * inner + i
                    acc *= host._buffer[idx]
                result.append(acc)

        new_shape = shape[:axis] + shape[axis + 1 :]
        if not new_shape:
            new_shape = (1,)
        return Tensor(new_shape, result)

    def reshape_op(self, host: Tensor, shape: tuple):
        old_len = host.flat_len
        new_len = reduce(operator.mul, shape, 1)
        assert new_len == old_len
        host.shape = shape
        return host

    def op(
        self,
        op: ops.Ops,
        host: ops.other_consts,
        other: ops.other_consts | None = None,
    ):
        # Ensure host is a Tensor, not a LazyOp
        from eigen.lazy import tensor_map, LazyOp

        if isinstance(host, LazyOp):
            # Try to get the tensor from the tensor_map
            tensor = tensor_map.get(host)
            if tensor is not None:
                host = tensor
            else:
                # If not found, raise a clear error
                raise TypeError("Cannot resolve LazyOp to Tensor in Runtime.op")
        self.dtype = host.dtype
        fn = {
            ops.Ops.ADD: self.add_op,
            ops.Ops.SUB: self.sub_op,
            ops.Ops.MUL: self.mul_op,
            ops.Ops.DIV: self.div_op,
            ops.Ops.POW: self.pow_op,
            ops.Ops.NEG: self.neg_op,
            ops.Ops.ABS: self.abs_op,
            # reductions
            ops.Ops.SUM: self.sum_op,
            ops.Ops.MEAN: self.mean_op,
            ops.Ops.PROD: self.prod_op,
            # other
            ops.Ops.CONST: self.const,
            ops.Ops.RESHAPE: self.reshape_op,
            # linear algebra
            ops.Ops.MATMUL: self.matmul_op,
        }[op]

        if other is not None:
            return fn(host, other)
        return fn(host)
