# mypy: allow-untyped-defs
"""cuDNN backend property helpers.

Exposes ``tensorplay.backends.cudnn.allow_tf32`` (defaults to True, like
which times candidate convolution algorithms per shape and caches the
fastest.
"""

import re
import sys
from contextlib import contextmanager

import tensorplay
from tensorplay.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

__all__ = [
    "allow_tf32",
    "benchmark",
    "flags",
    "is_available",
    "set_flags",
    "version",
]


def is_available() -> bool:
    r"""Returns a bool indicating if CUDNN is currently available."""
    info = tensorplay._C._get_build_info().get("CUDNN_INFO", "")
    return info.lower().endswith("enabled")


def version() -> int | None:
    r"""Returns the build-reported cuDNN version when available."""
    if not is_available():
        return None
    info = tensorplay._C._get_build_info().get("CUDNN_INFO", "")
    match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", info)
    if match is None:
        return None
    major, minor, patch = (int(part or 0) for part in match.groups())
    return major * 10000 + minor * 100 + patch


def set_flags(_benchmark=None, _allow_tf32=None):
    """Set the native cuDNN execution flags and return their old values."""
    original = (
        tensorplay._C._get_cudnn_benchmark(),
        tensorplay._C._get_cudnn_allow_tf32(),
    )
    if _benchmark is not None:
        tensorplay._C._set_cudnn_benchmark(bool(_benchmark))
    if _allow_tf32 is not None:
        tensorplay._C._set_cudnn_allow_tf32(bool(_allow_tf32))
    return original


@contextmanager
def flags(benchmark=False, allow_tf32=True):
    """Temporarily set the native cuDNN execution flags."""
    with __allow_nonbracketed_mutation():
        original = set_flags(benchmark, allow_tf32)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(*original)


class _CuDNNModule(PropModule):
    allow_tf32 = ContextProp(
        tensorplay._C._get_cudnn_allow_tf32,
        tensorplay._C._set_cudnn_allow_tf32,
    )
    benchmark = ContextProp(
        tensorplay._C._get_cudnn_benchmark,
        tensorplay._C._set_cudnn_benchmark,
    )


sys.modules[__name__] = _CuDNNModule(sys.modules[__name__], __name__)
