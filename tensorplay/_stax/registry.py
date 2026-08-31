"""Backend registry used by :func:`tensorplay.compile`.

``backend(graph_module, example_inputs, **options) -> callable``.
Backends do not capture Python and do not own graph-break policy.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable, Sequence
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Protocol

from tensorplay.graph import GraphModule


class CompiledFn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


CompilerFn = Callable[..., CompiledFn]

_lock = threading.RLock()
_backends: dict[str, EntryPoint | None] = {}
_compiler_fns: dict[str, CompilerFn] = {}
_backend_tags: dict[str, tuple[str, ...]] = {}
_default_backend: str | CompilerFn = "stax"
_entrypoints_loaded = False
_builtins_loaded = False


def register_backend(
    compiler_fn: CompilerFn | None = None,
    *,
    name: str | None = None,
    tags: Sequence[str] = (),
) -> Callable[[CompilerFn], CompilerFn] | CompilerFn:
    """Register a backend by name.

    A backend may be passed directly to ``tensorplay.compile`` without being
    registered. Registration is only required for string lookup.
    """

    if compiler_fn is None:
        return functools.partial(register_backend, name=name, tags=tags)
    if not callable(compiler_fn):
        raise TypeError(f"compiler_fn must be callable, got {type(compiler_fn)!r}")

    backend_name = name or getattr(compiler_fn, "__name__", None)
    if not backend_name:
        raise ValueError("a backend name is required for unnamed callables")

    with _lock:
        if backend_name in _compiler_fns:
            raise RuntimeError(f"backend {backend_name!r} is already registered")
        _backends.setdefault(backend_name, None)
        _compiler_fns[backend_name] = compiler_fn
        _backend_tags[backend_name] = tuple(tags)
    return compiler_fn


register_debug_backend = functools.partial(register_backend, tags=("debug",))
register_experimental_backend = functools.partial(
    register_backend, tags=("experimental",)
)


def unregister_backend(name: str) -> None:
    """Remove a previously registered backend (tests and tooling)."""

    with _lock:
        _backends.pop(name, None)
        _compiler_fns.pop(name, None)
        _backend_tags.pop(name, None)


def _load_builtins() -> None:
    global _builtins_loaded
    with _lock:
        if _builtins_loaded:
            return
        _builtins_loaded = True

    # Imports are lazy so importing tensorplay does not import Triton or a
    # backend's optional compiler toolchain.
    from . import builtins as _builtins

    _builtins.register()


def _load_entrypoints() -> None:
    global _entrypoints_loaded
    with _lock:
        if _entrypoints_loaded:
            return
        _entrypoints_loaded = True

    try:
        discovered = entry_points(group="tensorplay_compiler_backends")
    except TypeError:  # Python versions with the pre-3.10 API
        discovered = entry_points().get("tensorplay_compiler_backends", ())
    with _lock:
        for item in discovered:
            _backends.setdefault(item.name, item)


def lookup_backend(backend: str | CompilerFn) -> CompilerFn:
    """Resolve a backend name or validate a backend callable."""

    if not isinstance(backend, str):
        if not callable(backend):
            raise TypeError(f"backend must be a string or callable, got {type(backend)!r}")
        return backend

    _load_builtins()
    _load_entrypoints()
    with _lock:
        if backend not in _backends:
            available = ", ".join(list_backends(exclude_tags=None)) or "<none>"
            raise ValueError(f"unknown TensorPlay compiler backend {backend!r}; available: {available}")
        compiler_fn = _compiler_fns.get(backend)
        entrypoint = _backends.get(backend)

    if compiler_fn is None and entrypoint is not None:
        loaded = entrypoint.load()
        register_backend(loaded, name=backend)
        compiler_fn = loaded

    if compiler_fn is None:
        raise RuntimeError(f"backend {backend!r} was discovered but could not be loaded")
    return compiler_fn


def list_backends(
    *, exclude_tags: Sequence[str] | None = ("debug", "experimental")
) -> list[str]:
    """Return names accepted by ``tensorplay.compile(backend=...)``."""

    _load_builtins()
    _load_entrypoints()
    excluded = set(exclude_tags or ())
    with _lock:
        return sorted(
            name
            for name in _backends
            if not excluded.intersection(_backend_tags.get(name, ()))
        )


def set_default_backend(backend: str | CompilerFn | None) -> None:
    """Set the default compiler backend; ``None`` restores ``stax``."""

    global _default_backend
    if backend is None:
        _default_backend = "stax"
        return
    if isinstance(backend, str):
        lookup_backend(backend)
    elif not callable(backend):
        raise TypeError(f"backend must be a string or callable, got {type(backend)!r}")
    _default_backend = backend


def get_default_backend() -> str | CompilerFn:
    return _default_backend
