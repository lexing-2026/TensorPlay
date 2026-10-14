"""Backend configuration for the executorch backend.

The native quantized kernels are the only backend available in this project,
so the executorch configuration reuses the native operator patterns.
"""

from .native import get_native_backend_config


def get_executorch_backend_config():
    """Return the backend config for executorch (native patterns)."""
    return get_native_backend_config()


def get_executorch_backend_config_dict():
    """Return the backend config for executorch as a dictionary."""
    return get_native_backend_config().to_dict()
