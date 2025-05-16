from __future__ import annotations
import eigen.ops
import weakref
from dataclasses import dataclass
import functools
import hashlib


class GlobalTensorMap:
    _map = weakref.WeakValueDictionary()

    @classmethod
    def set(cls, key, tensor):
        cls._map[key] = tensor

    @classmethod
    def get(cls, key):
        return cls._map.get(key)


tensor_map = GlobalTensorMap()


class LazyOpMeta(type):
    cache: dict[tuple, weakref.ReferenceType[LazyOp]] = {}

    def __call__(cls, op, srcs=tuple()):
        if not isinstance(srcs, (list, tuple)):
            srcs = (srcs,)
        srcs = tuple(srcs)
        key = (op, srcs)
        if (ref := LazyOpMeta.cache.get(key)) is not None and (
            cached := ref()
        ) is not None:
            return cached

        obj = super().__call__(op, srcs)
        LazyOpMeta.cache[key] = weakref.ref(obj)
        return obj


@dataclass(eq=False)
class LazyOp(metaclass=LazyOpMeta):
    op: eigen.ops.Ops
    srcs: tuple

    def __str__(self):
        return self._str_recursive(0)

    @functools.cached_property
    def key(self) -> bytes:
        return hashlib.sha256(
            str((self.op, self.srcs)).encode()
            + b"".join([s.key for s in self.arg])
        ).digest()

    def toposort(self, visited=set(), out=[]):
        if self in visited:
            return out

        for inp in self.srcs:
            if isinstance(inp, LazyOp):
                inp.toposort(visited, out)

        out.append(self)
        return out

    def _str_recursive(self, indent):
        pad = "  " * indent
        if not self.srcs:
            return f"{pad}{self.op.name}\n"
        else:
            s = f"{pad}{self.op.name}(\n"
            for src in self.srcs:
                if isinstance(src, LazyOp):
                    s += src._str_recursive(indent + 1)
                else:
                    s += f"{'  ' * (indent + 1)}{str(src)}\n"
            s += f"{pad})\n"
            return s
