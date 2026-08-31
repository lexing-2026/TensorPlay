# mypy: allow-untyped-defs
"""CUDA backend property helpers.

Exposes ``tensorplay.backends.cuda.matmul.allow_tf32``, backed by the same
global state as :func:`tensorplay.set_float32_matmul_precision`.
"""

import sys

import tensorplay
from tensorplay.backends import PropModule

__all__ = [
    "get_name",
    "is_available",
    "is_built",
    "matmul",
]


def is_built() -> bool:
    r"""Returns whether TensorPlay is built with CUDA support. Note that this
    doesn't mean CUDA is available; just that if TensorPlay is built for the machine."""
    return bool(tensorplay._C.is_cuda_available)


def is_available() -> bool:
    r"""Returns a bool indicating if CUDA is currently available."""
    return bool(tensorplay.cuda.is_available())


def get_name() -> str:
    r"""Returns the CUDA device name."""
    return tensorplay.cuda.get_device_name()


class cuBLASModule:
    def __getattr__(self, name):
        if name == "allow_tf32":
            return tensorplay._C._get_cublas_allow_tf32()
        raise AttributeError("Unknown attribute " + name)

    def __setattr__(self, name, value):
        if name == "allow_tf32":
            return tensorplay._C._set_cublas_allow_tf32(value)
        raise AttributeError("Unknown attribute " + name)


class _CUDAModule(PropModule):
    matmul = cuBLASModule()


sys.modules[__name__] = _CUDAModule(sys.modules[__name__], __name__)
