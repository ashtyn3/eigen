import os
import eigen.ops as ops
from eigen.dtypes import Eigen_Dtype
from typing import Union

from typing import TYPE_CHECKING

from eigen.tensor import Tensor
from eigen.lazy import LazyOp


class Runtime(ops.OpsTrait):
    dtype: Eigen_Dtype

    def add_op(host: Tensor, other: Tensor):
        if host.shape != other.shape:
            raise ValueError("Add op needs matching shapes")

        return Tensor(
            host.shape,
            [x + y for x, y in zip(other._buffer, host._buffer)],
        )

    def sub_op(self, other: Tensor):
        if self.host.shape != other.shape:
            raise ValueError("Add op needs matching shapes")

        return Tensor(
            self.host.shape,
            [x - y for x, y in zip(self.host._buffer, other._buffer)],
        )

    def mul_op(self, other: Tensor):
        if self.host.shape != other.shape:
            raise ValueError("Add op needs matching shapes")

        return Tensor(
            self.host.shape,
            [x * y for x, y in zip(self.host._buffer, other._buffer)],
        )

    def div_op(self, other: Tensor):
        if self.host.shape != other.shape:
            raise ValueError("Add op needs matching shapes")

        return Tensor(
            self.host.shape,
            [x / y for x, y in zip(self.host._buffer, other._buffer)],
        )

    def pow_op(self, other: ops.other_consts):
        if isinstance(other, Tensor):
            if self.host.shape != other.shape:
                raise ValueError("Add op needs matching shapes")

            return Tensor(
                self.host.shape,
                [(x**y) for x, y in zip(self.host._buffer, other._buffer)],
            )
        else:
            return Tensor(
                self.host.shape,
                [x**other for x in self.host._buffer],
            )

    def neg_op(self):
        return Tensor(
            self.host.shape,
            [-x for x in self.host._buffer],
        )

    def abs_op(self):
        return Tensor(
            self.host.shape,
            [abs(x) for x in self.host._buffer],
        )

    def sum_op(self, axis: ops.other_consts = 0):
        from functools import reduce

        import operator

        # Compute strides for row-major layout
        strides = []
        acc = 1
        for dim in reversed(self.host.shape):
            strides.insert(0, acc)
            acc *= dim

        # Compute outer, axis, and inner sizes
        outer = reduce(operator.mul, self.host.shape[:axis], 1)
        axis_dim = self.host.shape[axis]
        inner = reduce(operator.mul, self.host.shape[axis + 1 :], 1)

        result = []
        for o in range(outer):
            for i in range(inner):
                base = o * axis_dim * inner + i
                acc = 0
                for j in range(axis_dim):
                    idx = base + j * inner
                    acc += self.host._buffer[idx]
                result.append(acc)

        # Return summed buffer and new shape
        new_shape = self.host.shape[:axis] + self.host.shape[axis + 1 :]
        if len(new_shape) == 1:
            new_shape = (1, new_shape[0])
        return Tensor(new_shape, result)

    def op(
        self,
        op: ops.Ops,
        host: ops.other_consts,
        other: ops.other_consts | None = None,
    ):
        self.dtype = self.host.dtype
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
            ops.Ops.CONST: self.const,
        }[op]

        if other is not None:
            return fn(host, other)
        return fn(host)
