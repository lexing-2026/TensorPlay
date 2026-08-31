"""Private Stax backend host: compilation orchestration and backend registry."""

from .api import compile, reset
from .aot import AOTError, build_aot
from .codecache import CodeCache, default_cache
from .cudagraphs import CudaGraphError, CudaGraphManager
from .guards import Guard, GuardChain, format_recompile_reasons
from .registry import (
    get_default_backend,
    list_backends,
    lookup_backend,
    register_backend,
    register_debug_backend,
    register_experimental_backend,
    set_default_backend,
    unregister_backend,
)

__all__ = [
    "AOTError",
    "CodeCache",
    "CudaGraphError",
    "CudaGraphManager",
    "Guard",
    "GuardChain",
    "build_aot",
    "compile",
    "default_cache",
    "format_recompile_reasons",
    "get_default_backend",
    "list_backends",
    "lookup_backend",
    "register_backend",
    "register_debug_backend",
    "register_experimental_backend",
    "reset",
    "set_default_backend",
    "unregister_backend",
]
