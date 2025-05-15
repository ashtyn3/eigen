from eigen.dtypes import FastEnum, Eigen_Dtype
from enum import auto
from typing import Union

from eigen.tensor import Tensor

other_consts = Union[int, float, Tensor]


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
    CREATE_OP = auto()
    CONNECT = auto()
    EXECUTE = auto()
    FORWARD = auto()

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

    def __init__(self, host: Tensor):
        self.host = host
        self.dtype = host.dtype

    def op(self, op: Ops, other: other_consts | None = None):
        raise NotImplementedError(
            f"Operation {op} must be implemented in subclass"
        )

    # Basic Math Ops
    def add(self, other: other_consts):
        return self.op(Ops.ADD, other)

    def sub(self, other: other_consts):
        return self.op(Ops.SUB, other)

    def mul(self, other: other_consts):
        return self.op(Ops.MUL, other)

    def div(self, other: other_consts):
        return self.op(Ops.DIV, other)

    def pow(self, other: other_consts):
        return self.op(Ops.POW, other)

    def neg(self):
        return self.op(Ops.NEG)

    def abs(self):
        return self.op(Ops.ABS)

    # Reductions
    def sum(self, axis: int):
        return self.op(Ops.SUM, axis)

    def mean(self):
        return self.op(Ops.MEAN)

    def prod(self):
        return self.op(Ops.PROD)

    def max(self):
        return self.op(Ops.MAX)

    def min(self):
        return self.op(Ops.MIN)

    def argmax(self):
        return self.op(Ops.ARGMAX)

    def argmin(self):
        return self.op(Ops.ARGMIN)

    # Linear Algebra
    def matmul(self, other: other_consts):
        return self.op(Ops.MATMUL, other)

    def dot(self, other: other_consts):
        return self.op(Ops.DOT, other)

    def transpose(self):
        return self.op(Ops.TRANSPOSE)

    def reshape(self, shape):
        return self.op(Ops.RESHAPE, shape)

    def flatten(self):
        return self.op(Ops.FLATTEN)

    def diag(self):
        return self.op(Ops.DIAG)

    def inverse(self):
        return self.op(Ops.INVERSE)

    def det(self):
        return self.op(Ops.DET)

    # Shape & Indexing
    def shape(self):
        return self.op(Ops.SHAPE)

    def size(self):
        return self.op(Ops.SIZE)

    def slice(self, start, end):
        return self.op(Ops.SLICE, (start, end))

    def gather(self, indices):
        return self.op(Ops.GATHER, indices)

    def scatter(self, indices, updates):
        return self.op(Ops.SCATTER, (indices, updates))

    def expand_dims(self, axis):
        return self.op(Ops.EXPAND_DIMS, axis)

    def squeeze(self, axis):
        return self.op(Ops.SQUEEZE, axis)

    def concatenate(self, other: Tensor, axis):
        return self.op(Ops.CONCATENATE, (other, axis))

    def stack(self, other: Tensor, axis):
        return self.op(Ops.STACK, (other, axis))

    def split(self, num_or_size_splits, axis):
        return self.op(Ops.SPLIT, (num_or_size_splits, axis))

    # Broadcasting
    def broadcast_to(self, shape):
        return self.op(Ops.BROADCAST_TO, shape)

    def broadcast_shapes(self, other_shape):
        return self.op(Ops.BROADCAST_SHAPES, other_shape)

    # Control Flow
    def cond(self, predicate, true_fn, false_fn):
        return self.op(Ops.COND, (predicate, true_fn, false_fn))

    def while_loop(self, cond_fn, body_fn, loop_vars):
        return self.op(Ops.WHILE_LOOP, (cond_fn, body_fn, loop_vars))

    def stop_gradient(self):
        return self.op(Ops.STOP_GRADIENT)

    # Autodiff
    def grad(self, wrt):
        return self.op(Ops.GRAD, wrt)

    def backward(self):
        return self.op(Ops.BACKWARD)

    # Graph utilities
    def create_op(self, name, inputs, fn):
        return self.op(Ops.CREATE_OP, (name, inputs, fn))

    def connect(self, other):
        return self.op(Ops.CONNECT, other)

    def execute(self):
        return self.op(Ops.EXECUTE)

    def forward(self):
        return self.op(Ops.FORWARD)

    # NN Ops
    def relu(self):
        return self.op(Ops.RELU)

    def sigmoid(self):
        return self.op(Ops.SIGMOID)

    def softmax(self):
        return self.op(Ops.SOFTMAX)

    def cross_entropy(self, label):
        return self.op(Ops.CROSS_ENTROPY, label)

    def conv2d(self, kernel):
        return self.op(Ops.CONV2D, kernel)

    def max_pool(self, kernel_size):
        return self.op(Ops.MAX_POOL, kernel_size)

    def dropout(self, rate):
        return self.op(Ops.DROPOUT, rate)
