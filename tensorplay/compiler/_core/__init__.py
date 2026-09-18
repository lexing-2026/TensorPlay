"""Compiler orchestration layer: capture, specialization caching, backend
registry, guards and ahead-of-time autograd partitioning.

This package is the implementation behind the :mod:`tensorplay.compiler`
facade; backends live in :mod:`tensorplay.compiler.backends`.
"""

from .api import compile, reset
from .aot import AOTError, build_aot
from .guards import Guard, GuardChain, format_recompile_reasons
from .registry import (
    BackendCapabilities,
    CORE_BACKEND_CONTRACT_VERSION,
    InvalidBackend,
    declares_capabilities,
    get_backend_capabilities,
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
    "BackendCapabilities",
    "CORE_BACKEND_CONTRACT_VERSION",
    "Guard",
    "GuardChain",
    "InvalidBackend",
    "build_aot",
    "compile",
    "declares_capabilities",
    "format_recompile_reasons",
    "get_backend_capabilities",
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
