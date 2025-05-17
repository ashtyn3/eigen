from eigen.dtypes import FastEnum, Eigen_Dtype
from enum import auto
from typing import Union
import os


other_consts = Union[int, float]


class Ops(FastEnum):
    # Basic MathOps
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()
    NEG = auto()
    ABS = auto()

    # Redu
    SUM = auto()
    MEAN = auto()
    PROD = auto()
    MAX = auto()
    MIN = auto()
    ARGMAX = auto()
    ARGMIN = auto()

    # Linear Algebra Ops
    MATMUL = auto()
    DOT = auto()
    TRANSPOSE = auto()
    RESHAPE = auto()
    FLATTEN = auto()
    DIAG = auto()
    INVERSE = auto()
    DET = auto()

    # Shape and Indexing Ops
    SHAPE = auto()
    SIZE = auto()
    SLICE = auto()
    GATHER = auto()
    SCATTER = auto()
    EXPAND_DIMS = auto()
    SQUEEZE = auto()
    CONCATENATE = auto()
    STACK = auto()
    SPLIT = auto()

    # Broadcasting Ops
    BROADCAST_TO = auto()
    BROADCAST_SHAPES = auto()

    # Control Flow Ops
    COND = auto()
    WHILE_LOOP = auto()
    STOP_GRADIENT = auto()

    # Autodiff Ops
    GRAD = auto()
    BACKWARD = auto()

    # Graph Utilities
    CONST = auto()

    # High-Level NN Ops (Optional)
    RELU = auto()
    SIGMOID = auto()
    SOFTMAX = auto()
    CROSS_ENTROPY = auto()
    CONV2D = auto()
    MAX_POOL = auto()
    DROPOUT = auto()


class OpsTrait:
    dtype: Eigen_Dtype

    def op(
        self,
        op: Ops,
        host: other_consts | None,
        other: other_consts | None = None,
    ):
        raise NotImplementedError(
            f"Operation {op} must be implemented in subclass"
        )

    def const(self, host: other_consts):
        return self.op(Ops.CONST, host)

    # Basic Math Ops

    def add(self, host: other_consts, other: other_consts):
        return self.op(Ops.ADD, host, other)

    def sub(self, host: other_consts, other: other_consts):
        return self.op(Ops.SUB, host, other)

    def mul(self, host: other_consts, other: other_consts):
        return self.op(Ops.MUL, host, other)

    def div(self, host: other_consts, other: other_consts):
        return self.op(Ops.DIV, host, other)

    def pow(self, host: other_consts, other: other_consts):
        return self.op(Ops.POW, host, other)

    def neg(self, host: other_consts):
        return self.op(Ops.NEG, host)

    def abs(self, host: other_consts):
        return self.op(Ops.ABS, host)

    # Reductions
    def sum(self, host: other_consts, axis: int):
        return self.op(Ops.SUM, host, axis)

    def mean(self, host: other_consts, axis: int):
        return self.op(Ops.MEAN, host, axis)

    def prod(self, host: other_consts):
        return self.op(Ops.PROD, host)

    def max(self, host: other_consts):
        return self.op(Ops.MAX, host)

    def min(self, host: other_consts):
        return self.op(Ops.MIN, host)

    def argmax(self, host: other_consts):
        return self.op(Ops.ARGMAX, host)

    def argmin(self, host: other_consts):
        return self.op(Ops.ARGMIN, host)

    # Linear Algebra
    def matmul(self, host: other_consts, other: other_consts):
        return self.op(Ops.MATMUL, host, other)

    def dot(self, host: other_consts, other: other_consts):
        return self.op(Ops.DOT, host, other)

    def transpose(self, host: other_consts):
        return self.op(Ops.TRANSPOSE, host)

    def reshape(self, host: other_consts, shape):
        return self.op(Ops.RESHAPE, host, shape)

    def flatten(self, host: other_consts):
        return self.op(Ops.FLATTEN, host)

    def diag(self, host: other_consts):
        return self.op(Ops.DIAG, host)

    def inverse(self, host: other_consts):
        return self.op(Ops.INVERSE, host)

    def det(self, host: other_consts):
        return self.op(Ops.DET, host)

    # Shape & Indexing
    def shape(self, host: other_consts):
        return self.op(Ops.SHAPE, host)

    def size(self, host: other_consts):
        return self.op(Ops.SIZE, host)

    def slice(self, host: other_consts, start, end):
        return self.op(Ops.SLICE, host, (start, end))

    def gather(self, host: other_consts, indices):
        return self.op(Ops.GATHER, host, indices)

    def scatter(self, host: other_consts, indices, updates):
        return self.op(Ops.SCATTER, host, (indices, updates))

    def expand_dims(self, host: other_consts, axis):
        return self.op(Ops.EXPAND_DIMS, host, axis)

    def squeeze(self, host: other_consts, axis):
        return self.op(Ops.SQUEEZE, host, axis)

    def concatenate(self, host: other_consts, other, axis):
        return self.op(Ops.CONCATENATE, host, (other, axis))

    def stack(self, host: other_consts, other, axis):
        return self.op(Ops.STACK, host, (other, axis))

    def split(self, host: other_consts, num_or_size_splits, axis):
        return self.op(Ops.SPLIT, host, (num_or_size_splits, axis))

    # Broadcasting
    def broadcast_to(self, host: other_consts, shape):
        return self.op(Ops.BROADCAST_TO, host, shape)

    def broadcast_shapes(self, host: other_consts, other_shape):
        return self.op(Ops.BROADCAST_SHAPES, host, other_shape)

    # Control Flow
    def cond(self, host: other_consts, predicate, true_fn, false_fn):
        return self.op(Ops.COND, host, (predicate, true_fn, false_fn))

    def while_loop(self, host: other_consts, cond_fn, body_fn, loop_vars):
        return self.op(Ops.WHILE_LOOP, host, (cond_fn, body_fn, loop_vars))

    def stop_gradient(self, host: other_consts):
        return self.op(Ops.STOP_GRADIENT, host)

    # Autodiff
    def grad(self, host: other_consts, wrt):
        return self.op(Ops.GRAD, host, wrt)

    def backward(self, host: other_consts):
        return self.op(Ops.BACKWARD, host)

    # Graph utilities
    def create_op(self, host: other_consts, name, inputs, fn):
        return self.op(Ops.CREATE_OP, host, (name, inputs, fn))

    def connect(self, host: other_consts, other):
        return self.op(Ops.CONNECT, host, other)

    def execute(self, host: other_consts):
        return self.op(Ops.EXECUTE, host)

    def forward(self, host: other_consts):
        return self.op(Ops.FORWARD, host)

    # NN Ops
    def relu(self, host: other_consts):
        return self.op(Ops.RELU, host)

    def sigmoid(self, host: other_consts):
        return self.op(Ops.SIGMOID, host)

    def softmax(self, host: other_consts):
        return self.op(Ops.SOFTMAX, host)

    def cross_entropy(self, host: other_consts, label):
        return self.op(Ops.CROSS_ENTROPY, host, label)

    def conv2d(self, host: other_consts, kernel):
        return self.op(Ops.CONV2D, host, kernel)

    def max_pool(self, host: other_consts, kernel_size):
        return self.op(Ops.MAX_POOL, host, kernel_size)

    def dropout(self, host: other_consts, rate):
        return self.op(Ops.DROPOUT, host, rate)
