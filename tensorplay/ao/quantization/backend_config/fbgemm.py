"""Backend configuration for the fbgemm backend.

The native quantized kernels are the only backend available in this project,
so the fbgemm configuration reuses the native operator patterns.
"""

from .native import get_native_backend_config


def get_fbgemm_backend_config():
    """Return the backend config for fbgemm (native patterns)."""
    return get_native_backend_config()


def get_fbgemm_backend_config_dict():
    """Return the backend config for fbgemm as a dictionary."""
    return get_native_backend_config().to_dict()
