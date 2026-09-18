"""Backend registry used by :func:`tensorplay.compile`.

``backend(graph_module, example_inputs, **options) -> callable``.
Backends do not capture Python and do not own graph-break policy.

Backends come from three sources, resolved lazily on first lookup:

* explicit :func:`register_backend` calls (decorator or plain call);
* the built-ins in :mod:`tensorplay._stax.builtins`;
* installed third-party packages that declare an entry point in the
  ``tensorplay_compiler_backends`` group, for example::

      [project.entry-points.tensorplay_compiler_backends]
      my_compiler = "my_backend.compiler:my_compiler_function"

  The entry point is only imported when its name is first looked up.

Tags categorize backends; :func:`list_backends` hides ``debug`` and
``experimental`` ones by default.
"""

from __future__ import annotations

import functools
import importlib.util
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Protocol

from tensorplay.graph import GraphModule


class CompiledFn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


CompilerFn = Callable[..., CompiledFn]

#: Revision of the backend contract this core implements.  The contract fixes
#: what a backend receives (graph module + example inputs + keyword options)
#: and what it must return (a callable executing the region).  A backend
#: declaring a higher revision needs a newer core; lower revisions stay
#: callable because the contract is additive.
CORE_BACKEND_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend supports, declared at registration or on the callable.

    ``inference_only``: the backend lowers inference regions only.  The
    frontend wraps such a backend with ahead-of-time autograd for training
    regions (forward through the backend, backward through the eager graph).

    ``handles_training``: the backend accepts regions whose inputs require
    grad on its own.  A backend that is neither training-capable nor
    inference-only is rejected for training regions.

    ``optional_deps``: import names the backend needs at call time.  A
    backend whose dependencies are missing is hidden from
    :func:`list_backends`; selecting it by name still works and explains
    what to install.

    ``contract_version``: the backend contract revision the callable speaks.
    ``min_core_version`` / ``max_core_version``: inclusive range of core
    releases the backend was validated against (subpackages declare this so
    a core IR change is caught at lookup instead of at run time).
    """

    inference_only: bool = False
    handles_training: bool = True
    optional_deps: tuple[str, ...] = field(default=())
    contract_version: int = CORE_BACKEND_CONTRACT_VERSION
    min_core_version: str | None = None
    max_core_version: str | None = None


DEFAULT_CAPABILITIES = BackendCapabilities()

_lock = threading.RLock()
_backends: dict[str, EntryPoint | None] = {}
_compiler_fns: dict[str, CompilerFn] = {}
_backend_tags: dict[str, tuple[str, ...]] = {}
_backend_capabilities: dict[str, BackendCapabilities] = {}
_default_backend: str | CompilerFn = "stax"
_entrypoints_loaded = False
_builtins_loaded = False
_missing_dep_cache: dict[str, bool] = {}

_ENTRY_POINT_GROUP = "tensorplay_compiler_backends"

_CORE_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


class InvalidBackend(ValueError, RuntimeError):
    """A backend name that no registered, built-in or entry-point backend owns.

    Derives from ``ValueError`` (what unknown names raised historically) and
    ``RuntimeError`` (the compiler error family), so either handler works.
    """

    def __init__(self, name: str, suggestions: Sequence[str] | None = None) -> None:
        self.name = name
        self.suggestions = list(suggestions or ())
        message = f"Invalid backend: {name!r}"
        if self.suggestions:
            message += f", did you mean: {', '.join(map(repr, self.suggestions))}?"
        else:
            message += "."
        message += (
            " See `tensorplay.compiler.list_backends()` for available backends."
        )
        super().__init__(message)


def register_backend(
    compiler_fn: CompilerFn | None = None,
    *,
    name: str | None = None,
    tags: Sequence[str] = (),
    capabilities: BackendCapabilities | None = None,
) -> Callable[[CompilerFn], CompilerFn] | CompilerFn:
    """Register a backend by name.

    A backend may be passed directly to ``tensorplay.compile`` without being
    registered. Registration is only required for string lookup.
    ``capabilities`` records what the backend supports; when omitted it is
    read from the callable's ``_tensorplay_capabilities`` attribute if set.
    """

    if compiler_fn is None:
        return functools.partial(
            register_backend, name=name, tags=tags, capabilities=capabilities
        )
    if not callable(compiler_fn):
        raise TypeError(f"compiler_fn must be callable, got {type(compiler_fn)!r}")

    backend_name = name or getattr(compiler_fn, "__name__", None)
    if not backend_name:
        raise ValueError("a backend name is required for unnamed callables")

    resolved = capabilities or getattr(
        compiler_fn, "_tensorplay_capabilities", None
    )
    if resolved is None:
        resolved = DEFAULT_CAPABILITIES
    if not isinstance(resolved, BackendCapabilities):
        raise TypeError(
            f"capabilities must be a BackendCapabilities, got {type(resolved)!r}"
        )

    with _lock:
        if backend_name in _compiler_fns:
            raise RuntimeError(f"backend {backend_name!r} is already registered")
        _backends.setdefault(backend_name, None)
        _compiler_fns[backend_name] = compiler_fn
        _backend_tags[backend_name] = tuple(tags)
        _backend_capabilities[backend_name] = resolved
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
        _backend_capabilities.pop(name, None)


def _load_builtins() -> None:
    global _builtins_loaded
    with _lock:
        if _builtins_loaded:
            return
        _builtins_loaded = True

    # Imports are lazy so importing tensorplay does not import Triton or a
    # backend's optional compiler toolchain.
    from ..backends import builtins as _builtins

    _builtins.register()


def _load_entrypoints() -> None:
    global _entrypoints_loaded
    with _lock:
        if _entrypoints_loaded:
            return
        _entrypoints_loaded = True

    try:
        discovered = entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:  # Python versions with the pre-3.10 API
        discovered = entry_points().get(_ENTRY_POINT_GROUP, ())
    with _lock:
        for item in discovered:
            # A registered or built-in backend keeps its name; the entry point
            # only claims names nobody owns yet.
            _backends.setdefault(item.name, item)


def _is_missing_dep(name: str) -> bool:
    known = _missing_dep_cache.get(name)
    if known is None:
        known = importlib.util.find_spec(name) is None
        _missing_dep_cache[name] = known
    return known


def missing_optional_deps(capabilities: BackendCapabilities) -> tuple[str, ...]:
    """Return the backend's optional dependencies that are not importable."""

    return tuple(
        dep for dep in capabilities.optional_deps if _is_missing_dep(dep)
    )


def _core_version() -> tuple[int, int, int] | None:
    import tensorplay

    match = _CORE_VERSION_PATTERN.match(tensorplay.__version__)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _check_contract(capabilities: BackendCapabilities, backend: str) -> None:
    import tensorplay

    if capabilities.contract_version > CORE_BACKEND_CONTRACT_VERSION:
        raise RuntimeError(
            f"backend {backend!r} implements backend-contract revision "
            f"{capabilities.contract_version}, but this TensorPlay core "
            f"implements up to revision {CORE_BACKEND_CONTRACT_VERSION}; "
            "upgrade TensorPlay to use it"
        )
    version = _core_version()
    if version is None:
        return
    if capabilities.min_core_version is not None:
        floor = _CORE_VERSION_PATTERN.match(capabilities.min_core_version)
        if floor and version < tuple(int(p) for p in floor.groups()):  # type: ignore[arg-type]
            raise RuntimeError(
                f"backend {backend!r} requires TensorPlay >= "
                f"{capabilities.min_core_version}, found {tensorplay.__version__}"
            )
    if capabilities.max_core_version is not None:
        ceiling = _CORE_VERSION_PATTERN.match(capabilities.max_core_version)
        if ceiling and version > tuple(int(p) for p in ceiling.groups()):  # type: ignore[arg-type]
            raise RuntimeError(
                f"backend {backend!r} supports TensorPlay <= "
                f"{capabilities.max_core_version}, found {tensorplay.__version__}; "
                "check the backend package for an update"
            )


def _install_guidance(deps: Sequence[str], backend: str) -> str:
    return (
        f"backend {backend!r} is missing optional dependencies: "
        f"{', '.join(deps)}. Install them with `pip install {' '.join(deps)}` "
        "and retry."
    )


def lookup_backend(backend: str | CompilerFn) -> CompilerFn:
    """Resolve a backend name or validate a backend callable."""

    if not isinstance(backend, str):
        if not callable(backend):
            raise TypeError(f"backend must be a string or callable, got {type(backend)!r}")
        return backend

    _load_builtins()
    _load_entrypoints()
    with _lock:
        known = backend in _backends
        compiler_fn = _compiler_fns.get(backend)
        entrypoint = _backends.get(backend)
    if not known:
        import difflib

        suggestions = difflib.get_close_matches(
            backend, list_backends(exclude_tags=None, include_unavailable=True), n=2
        )
        raise InvalidBackend(backend, suggestions)

    if compiler_fn is None and entrypoint is not None:
        try:
            loaded = entrypoint.load()
        except Exception as exc:
            raise RuntimeError(
                f"failed to load compiler backend {backend!r} from entry point "
                f"{entrypoint.value!r} (group {_ENTRY_POINT_GROUP!r})"
            ) from exc
        if not callable(loaded):
            raise TypeError(
                f"entry point {entrypoint.value!r} for backend {backend!r} "
                f"resolved to {type(loaded)!r}; expected a callable"
            )
        with _lock:
            # Another thread may have finished the same load first.
            compiler_fn = _compiler_fns.setdefault(backend, loaded)
            _backend_tags.setdefault(backend, ())
            _backend_capabilities.setdefault(
                backend,
                getattr(loaded, "_tensorplay_capabilities", None)
                or DEFAULT_CAPABILITIES,
            )

    if compiler_fn is None:
        raise RuntimeError(f"backend {backend!r} was discovered but could not be loaded")

    capabilities = _backend_capabilities.get(backend, DEFAULT_CAPABILITIES)
    missing = missing_optional_deps(capabilities)
    if missing:
        raise RuntimeError(_install_guidance(missing, backend))
    _check_contract(capabilities, backend)
    return compiler_fn


def list_backends(
    *,
    exclude_tags: Sequence[str] | None = ("debug", "experimental"),
    include_unavailable: bool = False,
) -> list[str]:
    """Return names accepted by ``tensorplay.compile(backend=...)``.

    Backends whose optional dependencies are missing are hidden unless
    ``include_unavailable`` is set; selecting one by name still produces an
    error explaining what to install.
    """

    _load_builtins()
    _load_entrypoints()
    excluded = set(exclude_tags or ())
    with _lock:
        snapshot = list(_backends.items())
        tags = dict(_backend_tags)
        caps = dict(_backend_capabilities)
    names = []
    for name, _ in snapshot:
        if excluded.intersection(tags.get(name, ())):
            continue
        if not include_unavailable:
            missing = missing_optional_deps(caps.get(name, DEFAULT_CAPABILITIES))
            if missing:
                continue
        names.append(name)
    return sorted(names)


def get_backend_capabilities(backend: str | CompilerFn) -> BackendCapabilities:
    """Capabilities of a registered backend (defaults for bare callables)."""

    if isinstance(backend, str):
        _load_builtins()
        _load_entrypoints()
        with _lock:
            caps = _backend_capabilities.get(backend)
            compiler_fn = _compiler_fns.get(backend)
    else:
        compiler_fn = backend
        with _lock:
            caps = next(
                (
                    _backend_capabilities.get(name)
                    for name, fn in _compiler_fns.items()
                    if fn is backend
                ),
                None,
            )
    if caps is None and compiler_fn is not None:
        caps = getattr(compiler_fn, "_tensorplay_capabilities", None)
    return caps if caps is not None else DEFAULT_CAPABILITIES


def declares_capabilities(
    capabilities: BackendCapabilities,
) -> Callable[[CompilerFn], CompilerFn]:
    """Attach capabilities to a backend callable (entry-point backends)."""

    def attach(compiler_fn: CompilerFn) -> CompilerFn:
        compiler_fn._tensorplay_capabilities = capabilities  # type: ignore[attr-defined]
        return compiler_fn

    return attach


def _is_registered_backend(compiler_fn: CompilerFn) -> bool:
    """Whether ``compiler_fn`` is a loaded registry backend (not a bare callable)."""

    _load_builtins()
    _load_entrypoints()
    with _lock:
        return any(compiler_fn is fn for fn in _compiler_fns.values())


def reset_backends() -> None:
    """Invoke ``reset()`` on every loaded backend that defines one.

    Backends holding process-wide state (captured CUDA graphs, compiled
    kernel pools) expose ``reset`` so :func:`tensorplay.compiler.reset`
    can release it together with the frontend caches.
    """

    with _lock:
        loaded = list(_compiler_fns.values())
    for compiler_fn in loaded:
        reset = getattr(compiler_fn, "reset", None)
        if callable(reset):
            reset()


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
