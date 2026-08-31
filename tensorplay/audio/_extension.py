"""Availability helpers for optional native audio features.

tensorplay.audio gates optional native features behind
``tensorplay.audio._extension``. Audio kernels are provided by the native
backend, while optional extension-only features report their documented
unavailable state instead of failing at import time.  The flag is kept at
module scope so callers can make one inexpensive capability check before
selecting an optional code path.
"""
_IS_AUDIO_EXT_AVAILABLE = False


def fail_if_no_align():
    from tensorplay._C import TP_THROW  # noqa: F401  (unreachable guard below)
    raise RuntimeError("forced alignment requires the tensorplay.audio C++ extension")
