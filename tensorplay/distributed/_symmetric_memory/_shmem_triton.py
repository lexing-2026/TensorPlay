from __future__ import annotations

from typing import Any

__all__ = ["get_shmem_backend_module", "requires_shmem"]


def get_shmem_backend_module() -> Any:
    from . import _nvshmem_triton

    return _nvshmem_triton


def requires_shmem(jit_func: Any) -> Any:
    backend = get_shmem_backend_module()
    return backend.requires_nvshmem(jit_func)
