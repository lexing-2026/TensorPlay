"""Backend configuration for the onednn backend.

The native quantized kernels are the only backend available in this project,
so the onednn configuration reuses the native operator patterns.
"""

from .native import get_native_backend_config


def get_onednn_backend_config():
    """Return the backend config for onednn (native patterns)."""
    return get_native_backend_config()


def get_onednn_backend_config_dict():
    """Return the backend config for onednn as a dictionary."""
    return get_native_backend_config().to_dict()
