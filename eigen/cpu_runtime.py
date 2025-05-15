from eigen.tensor import Tensor
import eigen.ops as ops
from eigen.dtypes import Eigen_Dtype
from typing import Union


class CPU_Ops(ops.OpsTrait):
    dtype: Eigen_Dtype

    def add_op(self, other: Tensor):
        if self.host.shape != other.shape:
            raise ValueError("Add op needs matching shapes")

        return Tensor(
            self.host.shape,
            [x + y for x, y in zip(other._buffer, self.host._buffer)],
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
        rows, cols = self.host.shape
        if axis == 0:
            # Sum down columns → result has `cols` elements
            result = [0] * cols
            for r in range(rows):
                for c in range(cols):
                    result[c] += self.host._buffer[(r * cols) + c]
            return Tensor((1, len(result)), result)

        elif axis == 1:
            # Sum across rows → result has `rows` elements
            result = [0] * rows
            for r in range(rows):
                for c in range(cols):
                    result[r] += self.host._buffer[(r * cols) + c]
            return Tensor((1, len(result)), result)

        else:
            raise ValueError("Invalid axis for 2D tensor")

    def op(self, op: ops.Ops, other: ops.other_consts | None = None):
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
        }[op]

        if other is not None:
            return fn(other)
        return fn()
