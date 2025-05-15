import os


class Device:
    def __init__(self):
        self.backend = self._select_backend()

    def _select_backend(self):
        if int(os.getenv("METAL", "0")):
            import eigen.metal_runtime as metal_runtime

            return metal_runtime
        else:
            import eigen.cpu_runtime as cpu_runtime

            return cpu_runtime

    def __getattr__(self, name):
        # Automatically delegate missing attributes to the backend
        return getattr(self.backend, name)
