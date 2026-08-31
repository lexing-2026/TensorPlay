from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["ShmemKernelRegistry", "run_shmem_init_hook"]


class ShmemKernelRegistry:
    """Registry for compiled kernels that need one-time module setup."""

    _to_init: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> None:
        cls._to_init.setdefault(name, None)

    @classmethod
    def deregister(cls, name: str) -> None:
        cls._to_init.pop(name, None)

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls._to_init


def run_shmem_init_hook(*, kwargs: dict[str, Any], registry: type[ShmemKernelRegistry], module_init: Callable[[Any], None], logger: Any) -> None:
    function = kwargs.get("fn")
    name = getattr(getattr(function, "jit_function", function), "__name__", None)
    if name is None or not registry.has(name):
        return
    kernel = kwargs.get("kernel")
    module = getattr(kernel, "module", None)
    if module is None:
        logger.warning("compiled kernel %s has no loadable module", name)
        return
    module_init(module)
