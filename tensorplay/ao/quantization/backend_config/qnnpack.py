"""Backend configuration for the qnnpack backend.

The native quantized kernels are the only backend available in this project,
so the qnnpack configuration reuses the native operator patterns.
"""

from .native import get_native_backend_config


def get_qnnpack_backend_config():
    """Return the backend config for qnnpack (native patterns)."""
    return get_native_backend_config()


def get_qnnpack_backend_config_dict():
    """Return the backend config for qnnpack as a dictionary."""
    return get_native_backend_config().to_dict()
