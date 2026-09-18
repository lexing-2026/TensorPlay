"""Persistent capture cache for compiled regions.

Capturing a region runs the program eagerly and then applies the
canonicalization pass pipeline.  Both steps are deterministic functions of
(the program source, the traced module's state, the input metadata), so the
resulting graph module can be stored on disk and reloaded by later
processes, leaving only backend lowering per process.

The cache key covers:

* the region schema version -- bump when capture recording, the pass
  pipeline, or this storage format changes in a way that a stored graph
  can no longer reproduce;
* the running package version;
* a source stamp of the traced program (a text hash, as a build system
  would use).  Programs whose source cannot be read are never cached;
* a value fingerprint of the traced module's parameters and buffers, when
  the program belongs to a module.  Capture bakes those values into the
  graph as constants, so a stored entry is only valid while they are
  unchanged.  Modules whose values cannot be read back (unsupported
  dtypes, non-host tensors that cannot be copied) are never cached;
* the caller-visible specialization signature (input metadata, guard and
  gate components), so distinct specializations keep distinct entries.

Stored entries keep the example tensors recorded in node metadata, so a
loaded region looks the same to guard promotion and backends as a freshly
captured one.  Everything here is best-effort: any load or store failure
degrades to an ordinary capture, never to a wrong result.
"""

from __future__ import annotations

import hashlib
import inspect
import pickle
from typing import Any, Callable, Optional, Sequence

from tensorplay.graph import GraphModule

from .codecache import default_cache

# Bump when capture recording, the pass pipeline, or this serialization
# format changes such that a stored graph is no longer faithful.
_SCHEMA_VERSION = "1"

_CACHE_BACKEND = "capture-region"
_ARTIFACT_EXT = "gmp"


def _program_stamp(program: Callable[..., Any]) -> Optional[bytes]:
    """Content hash of the traced program's source, or ``None`` if unreadable."""

    try:
        source = inspect.getsource(program)
    except (OSError, TypeError):
        return None
    qualname = getattr(
        program, "__qualname__", getattr(program, "__name__", "region")
    )
    h = hashlib.sha256()
    h.update(qualname.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(source.encode("utf-8", "replace"))
    return h.digest()


def _host_view(tensor: Any) -> Optional[Any]:
    """A numpy view of ``tensor``'s value, or ``None`` if unavailable."""

    try:
        if getattr(tensor, "device", None) is not None and tensor.device.type != "cpu":
            tensor = tensor.detach().to("cpu")
        else:
            tensor = tensor.detach()
        return tensor.contiguous().numpy()
    except Exception:  # noqa: BLE001 - an unreadable value disables caching
        return None


def _state_fingerprint(module: Any) -> Optional[bytes]:
    """Hash of the module's parameter and buffer values.

    Values are hashed through host views without copying when the storage
    is already contiguous host memory.  ``None`` means the state cannot be
    proven (unsupported dtype or unreadable tensor); callers must then
    treat the region as uncachable rather than skipping the fingerprint.
    """

    entries = [*module.named_parameters(), *module.named_buffers()]
    h = hashlib.sha256()
    for name, tensor in entries:
        view = _host_view(tensor)
        if view is None:
            return None
        h.update(name.encode("utf-8", "replace"))
        h.update(b"\x00")
        h.update(repr((tuple(tensor.shape), str(tensor.dtype))).encode())
        h.update(b"\x00")
        h.update(memoryview(view).cast("B"))
    return h.digest()


def region_key(
    program: Callable[..., Any],
    module: Any,
    signature_parts: Sequence[Any],
) -> Optional[str]:
    """The persistent cache key for one region, or ``None`` if uncachable."""

    program_stamp = _program_stamp(program)
    if program_stamp is None:
        return None
    state_stamp = None
    if module is not None:
        state_stamp = _state_fingerprint(module)
        if state_stamp is None:
            return None
    import tensorplay

    h = hashlib.sha256()
    h.update(b"tensorplay-region\x00")
    h.update(_SCHEMA_VERSION.encode())
    h.update(b"\x00")
    h.update(tensorplay.__version__.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(program_stamp)
    if state_stamp is not None:
        h.update(state_stamp)
    for part in signature_parts:
        h.update(repr(part).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def load_region(key: Optional[str]) -> Optional[GraphModule]:
    """The stored graph for ``key``, or ``None`` on any miss."""

    if not key:
        return None
    payload = default_cache(_CACHE_BACKEND).load(key, ext=_ARTIFACT_EXT)
    if payload is None:
        return None
    try:
        graph_module = pickle.loads(payload)
    except Exception:  # noqa: BLE001 - corrupt or unreadable entry
        return None
    return graph_module if isinstance(graph_module, GraphModule) else None


def store_region(key: Optional[str], graph_module: GraphModule) -> None:
    """Store ``graph_module`` under ``key``; silently skip on failure."""

    if not key:
        return
    try:
        payload = pickle.dumps(graph_module, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # noqa: BLE001 - entries carrying unpicklable state
        return
    try:
        default_cache(_CACHE_BACKEND).store(key, payload, ext=_ARTIFACT_EXT)
    except Exception:  # noqa: BLE001 - cache is best-effort
        return
