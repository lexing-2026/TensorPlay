"""Availability shims standing in for tensorplay.audio's C++ extension.

tensorplay.audio gates optional native features behind ``tensorplay.audio._extension``.
TensorPlay implements the audio ops natively in p10 (see
p10/src/backend/cpu/SpectralKernels.cpp), so the extension-dependent
features degrade to their documented unavailable state instead of failing
at import time (mirrors tensorplay.audio/_extension/loader.py semantics).
"""
_IS_AUDIO_EXT_AVAILABLE = False


def fail_if_no_align():
    from tensorplay._C import TP_THROW  # noqa: F401  (unreachable guard below)
    raise RuntimeError("forced alignment requires the tensorplay.audio C++ extension")
