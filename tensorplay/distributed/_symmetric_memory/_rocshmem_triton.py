from __future__ import annotations

import os
from typing import Any

from ._shmem_triton_utils import ShmemKernelRegistry

__all__ = ["RocshmemLibFinder", "RocshmemKernelRegistry", "_rocshmem_init_hook", "requires_rocshmem"]


class RocshmemLibFinder:
    found_device_lib_path: str | None = None

    @classmethod
    def find_device_library(cls) -> str:
        path = cls.found_device_lib_path or os.environ.get("TP_ROCSHMEM_DEVICE_LIBRARY")
        if not path or not os.path.isfile(path):
            raise RuntimeError("rocSHMEM device library is unavailable")
        cls.found_device_lib_path = path
        return path


class RocshmemKernelRegistry(ShmemKernelRegistry):
    _to_init: dict[str, Any] = {}


def _rocshmem_init_hook(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    return None


def requires_rocshmem(jit_func: Any) -> Any:
    if not callable(jit_func):
        raise TypeError("requires_rocshmem expects a callable kernel")
    RocshmemLibFinder.find_device_library()
    RocshmemKernelRegistry.register(getattr(jit_func, "__name__", type(jit_func).__name__))
    return jit_func
