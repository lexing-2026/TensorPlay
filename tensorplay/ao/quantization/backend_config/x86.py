"""Backend configuration for the x86 backend (native patterns)."""

from .native import get_native_backend_config


def get_x86_backend_config():
    return get_native_backend_config()
