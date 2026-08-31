"""CUDA graphs orchestration (L5-M3).

(capture once, replay against static buffers), driven entirely by the native
:class:`tensorplay._C.CUDAGraph` class:

* ``capture_begin/capture_end`` own the dedicated per-device side stream,
  route allocations into a graph-private allocator pool and register
  graph-safe RNG state; instantiation happens eagerly at ``capture_end``.
* ``stage_and_launch`` is the low-overhead replay path: every input is
  copied onto its static buffer with a raw async device-to-device copy and
  the cached executable is launched - one Python-to-native crossing per
  replay instead of one dispatcher round trip per input plus launch.

Tests may inject a stand-in via ``CudaGraphManager(native=...)``; the
stand-in must expose a ``CUDAGraph`` class with ``capture_begin``,
``capture_end``, ``replay``, ``reset`` and optionally
``stage_and_launch``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class CudaGraphError(RuntimeError):
    """Raised for capture/replay contract violations."""


def _default_native() -> Any:
    try:
        from .. import _C  # type: ignore
    except Exception as exc:  # pragma: no cover - import failure diagnostics
        raise NotImplementedError(
            "CUDA graph support requires tensorplay._C; import failed: "
            f"{exc!r}. Was TensorPlay built with CUDA support?"
        ) from exc
    if not hasattr(_C, "CUDAGraph"):
        raise NotImplementedError(
            "CUDA graphs are not supported by this TensorPlay build "
            "(tensorplay._C exposes no CUDAGraph class). Was it built "
            "with CUDA support?"
        )
    return _C


def _shape_signature(args: Sequence[Any]) -> Tuple:
    return tuple(
        (tuple(getattr(a, "shape", ())), str(getattr(a, "dtype", "")))
        for a in args
    )


class _GraphEntry:
    __slots__ = ("key", "signature", "graph", "static_inputs",
                 "static_outputs", "replays", "bulk")

    def __init__(self, key: str, signature: Tuple, graph: Any,
                 static_inputs: List[Any], static_outputs: List[Any]) -> None:
        self.key = key
        self.signature = signature
        self.graph = graph
        self.static_inputs = static_inputs
        self.static_outputs = static_outputs
        self.replays = 0
        # Bulk staging keeps the whole replay inside one native call;
        # stand-in natives without stage_and_launch fall back to per-tensor
        # copies plus replay().
        self.bulk = hasattr(graph, "stage_and_launch")


class CudaGraphManager:
    """Capture functions once, replay them against static buffers."""

    def __init__(self, native: Optional[Any] = None, max_entries: int = 8) -> None:
        self._native_module = native
        self._owns_lookup = native is None
        self._entries: Dict[str, _GraphEntry] = {}
        self.max_entries = max_entries
        self.capturing: Optional[str] = None

    # -- native plumbing ------------------------------------------------------

    @property
    def native(self) -> Any:
        if self._native_module is None:
            self._native_module = _default_native()
        return self._native_module

    def _new_graph(self) -> Any:
        return self.native.CUDAGraph()

    # -- API ------------------------------------------------------------------

    def capture(self, key: str, fn: Callable[..., Any], *sample_args: Any) -> _GraphEntry:
        if self.capturing is not None:
            raise CudaGraphError(
                f"nested capture attempted ({self.capturing!r} already active)"
            )
        existing = self._entries.get(key)
        signature = _shape_signature(sample_args)
        if existing is not None:
            if existing.signature != signature:
                raise CudaGraphError(
                    f"entry {key!r} was captured for {existing.signature}, "
                    f"refusing re-capture for {signature}; use a new key"
                )
            return existing
        if len(self._entries) >= self.max_entries:
            raise CudaGraphError(
                f"graph cache full ({self.max_entries}); clear stale entries"
            )

        graph = self._new_graph()
        try:
            # Warmup executes lazy initialisations outside capture (cuBLAS
            # workspaces etc.); the native capture stream matches the stream
            # capture will run on.
            fn(*sample_args)
            # Static input buffers must be allocated AND filled before the
            # capture window opens: a clone issued inside capture becomes a
            # captured node that would overwrite the staged replay inputs
            # with the sample values on every replay.  Allocating outside
            # also keeps them out of the graph-private pool, so their
            # lifetime is independent of graph reset.  No ordering fence is
            # needed here: nothing executes during capture, and at replay
            # time the staging copies are enqueued on the launch stream ahead
            # of the graph.
            static_inputs = [
                a.clone() if hasattr(a, "clone") else a for a in sample_args
            ]
            self.capturing = key
            graph.capture_begin()
            outputs = fn(*static_inputs)
            graph.capture_end()
        finally:
            self.capturing = None
        out_list = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
        entry = _GraphEntry(key, signature, graph, static_inputs, out_list)
        self._entries[key] = entry
        return entry

    def replay(self, key: str, *args: Any) -> List[Any]:
        entry = self._entries.get(key)
        if entry is None:
            raise CudaGraphError(f"no captured graph under key {key!r}")
        if len(args) != len(entry.static_inputs):
            raise CudaGraphError(
                f"entry {key!r} expects {len(entry.static_inputs)} inputs, got {len(args)}"
            )
        signature = _shape_signature(args)
        if signature != entry.signature:
            raise CudaGraphError(
                f"entry {key!r} captured for {entry.signature}, replay args are {signature}"
            )
        if entry.bulk:
            entry.graph.stage_and_launch(entry.static_inputs, list(args))
        else:
            for dst, src in zip(entry.static_inputs, args):
                dst.copy_(src)
            entry.graph.replay()
        entry.replays += 1
        return list(entry.static_outputs)

    def clear(self, key: Optional[str] = None) -> None:
        if key is None:
            entries = list(self._entries.values())
            self._entries.clear()
        else:
            entry = self._entries.pop(key, None)
            entries = [] if entry is None else [entry]
        for entry in entries:
            reset = getattr(entry.graph, "reset", None)
            if reset is not None:
                reset()
