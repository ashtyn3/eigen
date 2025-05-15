import os

if int(os.getenv("TYPED", "0")):
    from typeguard import install_import_hook

    _ = install_import_hook(__name__)

from eigen.tensor import Tensor
