"""Availability shims for the functional layer (see ``audio._extension``).

gated behind flags so importing the functional layer never fails.
"""
from functools import wraps

from tensorplay.audio import _extension as _root_extension

_IS_AUDIO_EXT_AVAILABLE = _root_extension._IS_AUDIO_EXT_AVAILABLE

__all__ = ["_IS_AUDIO_EXT_AVAILABLE", "fail_if_no_align"]


def fail_if_no_align(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _IS_AUDIO_EXT_AVAILABLE:
            raise RuntimeError(
                "Forced alignment requires the tensorplay.audio C++ extension, "
                "which is not available in this build."
            )
        return fn(*args, **kwargs)

    return wrapper
