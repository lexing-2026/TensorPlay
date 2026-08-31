"""NNPACK execution controls."""

from contextlib import contextmanager

import tensorplay
from tensorplay.backends import __allow_nonbracketed_mutation

__all__ = ["is_available", "flags", "set_flags"]


def is_available() -> bool:
    """Return whether NNPACK kernels were included in this build."""
    settings = tensorplay._C._get_build_info().get("BUILD_SETTINGS", "")
    return "USE_NNPACK=ON" in settings


def set_flags(_enabled=None):
    """Set the process-wide NNPACK enable flag."""
    original = (tensorplay._C._get_nnpack_enabled(),)
    if _enabled is not None:
        tensorplay._C._set_nnpack_enabled(bool(_enabled))
    return original


@contextmanager
def flags(enabled=False):
    """Temporarily set the process-wide NNPACK enable flag."""
    with __allow_nonbracketed_mutation():
        original = set_flags(enabled)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(*original)
