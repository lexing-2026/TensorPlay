"""Compatibility shim: the compiler stack moved out of the stax namespace.

The orchestration layer (compile orchestration, backend registry, guards,
ahead-of-time autograd) now lives in :mod:`tensorplay.compiler._core` and the
backends (stax among them, no longer hosting the stack) in
:mod:`tensorplay.compiler.backends`.  Historical ``tensorplay._stax.*`` import
paths are served by lightweight proxy modules seeded into ``sys.modules``:
each proxy resolves to the moved module on first attribute access and then
replaces itself with the real module, so existing code keeps working during
the migration window.

New code must import from the new homes.  This shim is removed once all
in-flight work has switched over and no ``tensorplay._stax`` references
remain.  See devdoc/NOTICE_compiler_layout_migration.md.
"""

import importlib
import sys
import types
from typing import Any

_PREFIX = "tensorplay._stax."

_ALIASES: dict[str, str] = {
    # orchestration
    "api": "tensorplay.compiler._core.api",
    "registry": "tensorplay.compiler._core.registry",
    "guards": "tensorplay.compiler._core.guards",
    "region_cache": "tensorplay.compiler._core.region_cache",
    "aot": "tensorplay.compiler._core.aot",
    "aot_autograd": "tensorplay.compiler._core.aot_autograd",
    "common": "tensorplay.compiler._core.common",
    # backends
    "builtins": "tensorplay.compiler.backends.builtins",
    "debugging": "tensorplay.compiler.backends.debugging",
    "tvm": "tensorplay.compiler.backends.tvm",
    "cudagraphs": "tensorplay.compiler.backends.cudagraphs",
    "onnxrt": "tensorplay.compiler.backends.onnxrt",
    # stax backend internals
    "stax": "tensorplay.compiler.backends.stax.backend",
    "scheduler": "tensorplay.compiler.backends.stax.scheduler",
    "codecache": "tensorplay.compiler.backends.stax.codecache",
    "cpu_vec_isa": "tensorplay.compiler.backends.stax.cpu_vec_isa",
    "cpp_builder": "tensorplay.compiler.backends.stax.cpp_builder",
    "codegen": "tensorplay.compiler.backends.stax.codegen",
    "codegen.triton": "tensorplay.compiler.backends.stax.codegen.triton",
    "codegen.cpp": "tensorplay.compiler.backends.stax.codegen.cpp",
    "codegen.cpp_reduction": "tensorplay.compiler.backends.stax.codegen.cpp_reduction",
    "codegen.cpp_rowfusion": "tensorplay.compiler.backends.stax.codegen.cpp_rowfusion",
    "codegen.triton_rowfusion": "tensorplay.compiler.backends.stax.codegen.triton_rowfusion",
    "codegen.index_expr": "tensorplay.compiler.backends.stax.codegen.index_expr",
    "runtime": "tensorplay.compiler.backends.stax.runtime",
    "runtime.fastlaunch": "tensorplay.compiler.backends.stax.runtime.fastlaunch",
    "runtime.stax_autotune": "tensorplay.compiler.backends.stax.runtime.stax_autotune",
}

_HISTORICAL_SURFACE = {
    "compile": ("tensorplay.compiler._core.api", "compile"),
    "reset": ("tensorplay.compiler._core.api", "reset"),
    "AOTError": ("tensorplay.compiler._core.aot", "AOTError"),
    "build_aot": ("tensorplay.compiler._core.aot", "build_aot"),
    "CodeCache": ("tensorplay.compiler.backends.stax.codecache", "CodeCache"),
    "default_cache": ("tensorplay.compiler.backends.stax.codecache", "default_cache"),
    "CudaGraphError": ("tensorplay.compiler.backends.cudagraphs", "CudaGraphError"),
    "CudaGraphInputDrift": (
        "tensorplay.compiler.backends.cudagraphs",
        "CudaGraphInputDrift",
    ),
    "CudaGraphManager": ("tensorplay.compiler.backends.cudagraphs", "CudaGraphManager"),
    "Guard": ("tensorplay.compiler._core.guards", "Guard"),
    "GuardChain": ("tensorplay.compiler._core.guards", "GuardChain"),
    "format_recompile_reasons": (
        "tensorplay.compiler._core.guards",
        "format_recompile_reasons",
    ),
    "InvalidBackend": ("tensorplay.compiler._core.registry", "InvalidBackend"),
    "get_default_backend": ("tensorplay.compiler._core.registry", "get_default_backend"),
    "list_backends": ("tensorplay.compiler._core.registry", "list_backends"),
    "lookup_backend": ("tensorplay.compiler._core.registry", "lookup_backend"),
    "register_backend": ("tensorplay.compiler._core.registry", "register_backend"),
    "register_debug_backend": (
        "tensorplay.compiler._core.registry",
        "register_debug_backend",
    ),
    "register_experimental_backend": (
        "tensorplay.compiler._core.registry",
        "register_experimental_backend",
    ),
    "set_default_backend": ("tensorplay.compiler._core.registry", "set_default_backend"),
    "unregister_backend": ("tensorplay.compiler._core.registry", "unregister_backend"),
}


class _AliasModule(types.ModuleType):
    """Proxy module that resolves to the moved module on first attribute read.

    Attribute writes (monkeypatching in tests, late configuration) forward to
    the real module as well, so patching through the historical path reaches
    the code that actually runs.
    """

    _new_name: str = ""
    _old_name: str = ""

    #: Names owned by the proxy itself; everything else forwards.
    _PROXY_OWNED = frozenset({"_new_name", "_old_name", "__path__", "__file__"})

    def __getattr__(self, attr: str) -> Any:
        real = importlib.import_module(self._new_name)
        # Self-heal: the real module takes over the historical slot so every
        # later import (and identity check) sees one object.
        sys.modules[f"{_PREFIX}{self._old_name}"] = real
        setattr(sys.modules[__name__], self._old_name, real)
        return getattr(real, attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr in _AliasModule._PROXY_OWNED or attr.startswith("__"):
            super().__setattr__(attr, value)
            return
        try:
            real = sys.modules.get(self._new_name) or importlib.import_module(
                self._new_name
            )
        except ImportError:
            real = None
        if real is not None:
            setattr(real, attr, value)
        self.__dict__[attr] = value


def _proxy_for(old_name: str, new_name: str) -> types.ModuleType:
    fullname = f"{_PREFIX}{old_name}"
    proxy = _AliasModule(fullname)
    proxy._new_name = new_name
    proxy._old_name = old_name
    proxy.__path__ = []  # alias packages stay recognizable as such
    proxy.__file__ = None
    return proxy


def _install() -> None:
    package = sys.modules[__name__]
    for old_name, new_name in _ALIASES.items():
        fullname = f"{_PREFIX}{old_name}"
        if fullname in sys.modules:
            continue
        proxy = _proxy_for(old_name, new_name)
        sys.modules[fullname] = proxy
        if "." not in old_name:
            setattr(package, old_name, proxy)


_install()


def __getattr__(name: str) -> Any:
    target = _HISTORICAL_SURFACE.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_ALIASES) | set(_HISTORICAL_SURFACE))


__all__ = [
    "AOTError",
    "CodeCache",
    "CudaGraphError",
    "CudaGraphInputDrift",
    "CudaGraphManager",
    "Guard",
    "GuardChain",
    "InvalidBackend",
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
