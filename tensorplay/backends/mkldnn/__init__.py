from contextlib import contextmanager

import sys

import tensorplay
from tensorplay.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

__all__ = ["is_available", "flags", "set_flags", "enabled"]


def is_available() -> bool:
    """Return whether oneDNN kernels were included in this build."""
    return bool(tensorplay._C.has_mkldnn())


def set_flags(_enabled=None):
    """Set the process-wide oneDNN enable flag."""
    original = (tensorplay._C.is_mkldnn_enabled(),)
    if _enabled is not None:
        tensorplay._C.set_mkldnn_enabled(bool(_enabled))
    return original


@contextmanager
def flags(enabled=False):
    """Temporarily set the process-wide oneDNN enable flag."""
    with __allow_nonbracketed_mutation():
        original = set_flags(enabled)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(*original)


class MkldnnModule(PropModule):
    enabled = ContextProp(
        tensorplay._C.is_mkldnn_enabled,
        tensorplay._C.set_mkldnn_enabled,
    )


sys.modules[__name__] = MkldnnModule(sys.modules[__name__], __name__)
