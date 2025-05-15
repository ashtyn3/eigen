import os

if int(os.getenv("TYPED", "0")):
    from typeguard import install_import_hook

    _ = install_import_hook(__name__)

from eigen.edge import Edge  # noqa: F401
from eigen.scheduler import Scheduler  # noqa: F401
from eigen.node import Node  # noqa: F401
from eigen.graph import Graph  # noqa: F401
import eigen.dtypes as dtypes  # noqa: F401
from eigen.tensor import Tensor
import eigen.ops as ops
import eigen.cpu_runtime as cpu_runtime  # noqa: F401
