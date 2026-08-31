from __future__ import annotations

import os
from typing import Any

from ._shmem_triton_utils import ShmemKernelRegistry

__all__ = ["NvshmemLibFinder", "enable_triton", "NvshmemKernelRegistry", "_nvshmem_init_hook", "requires_nvshmem"]


class NvshmemLibFinder:
    found_device_lib_path: str | None = None

    @classmethod
    def find_device_library(cls) -> str:
        path = cls.found_device_lib_path or os.environ.get("TP_NVSHMEM_DEVICE_LIBRARY")
        if not path or not os.path.isfile(path):
            raise RuntimeError("NVSHMEM device library is unavailable")
        cls.found_device_lib_path = path
        return path


def enable_triton(lib_dir: str | None = None) -> dict[str, str]:
    del lib_dir
    raise NotImplementedError("use requires_nvshmem on a Triton kernel")


class NvshmemKernelRegistry(ShmemKernelRegistry):
    _to_init: dict[str, Any] = {}


def _nvshmem_init_hook(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    return None


def requires_nvshmem(jit_func: Any) -> Any:
    if not callable(jit_func):
        raise TypeError("requires_nvshmem expects a callable kernel")
    NvshmemLibFinder.find_device_library()
    NvshmemKernelRegistry.register(getattr(jit_func, "__name__", type(jit_func).__name__))
    return jit_func
