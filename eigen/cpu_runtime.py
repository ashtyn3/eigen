import os
import eigen.ops as ops
from eigen.dtypes import Eigen_Dtype
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


class Runtime(ops.OpsTrait):
    dtype: Eigen_Dtype

    def add_op(self, host: Tensor, other: Tensor):
        if host.shape != other.shape:
            raise ValueError("op needs matching shapes")

        return Tensor(
            host.shape,
            [x + y for x, y in zip(other._buffer, host._buffer)],
        )

    def sub_op(self, host: Tensor, other: Tensor):
        if host.shape != other.shape:
            raise ValueError("op needs matching shapes")

        return Tensor(
            host.shape,
            [x - y for x, y in zip(host._buffer, other._buffer)],
        )

    def mul_op(self, host: Tensor, other: Tensor):
        if host.shape != other.shape:
            raise ValueError("op needs matching shapes")

        return Tensor(
            host.shape,
            [x * y for x, y in zip(host._buffer, other._buffer)],
        )

    def div_op(self, host: Tensor, other: Tensor):
        if host.shape != other.shape:
            raise ValueError("op needs matching shapes")

        return Tensor(
            host.shape,
            [x / y for x, y in zip(host._buffer, other._buffer)],
        )

    def pow_op(self, host: Tensor, other: ops.other_consts):
        if isinstance(other, Tensor):
            if host.shape != other.shape:
                raise ValueError("op needs matching shapes")

            return Tensor(
                host.shape,
                [(x**y) for x, y in zip(host._buffer, other._buffer)],
            )
        else:
            return Tensor(
                host.shape,
                [x**other for x in host._buffer],
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

    def matmul_op(self, host: Tensor, other: Tensor):
        B, M, N = host.shape
        _, N2, P = other.shape
        assert N == N2, "Inner dimensions must match"

        result = []

        for b in range(B):
            for i in range(M):
                for j in range(P):
                    sum = 0
                    for k in range(N):
                        a = host._get(b, i, k)
                        b_val = other._get(b, k, j)
                        sum += a * b_val
                    result.append(sum)

        return Tensor((B, M, P), result)

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
            # other
            ops.Ops.CONST: self.const,
            ops.Ops.MATMUL: self.matmul_op,
        }[op]

        if other is not None:
            return fn(host, other)
        return fn(host)
