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

import logging
import weakref
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


# ---------------------------------------------------------------------------
# ``cudagraphs`` compiler backend
# ---------------------------------------------------------------------------
#
# ``tensorplay.compile(fn, backend="cudagraphs")`` runs the captured graph on
# its Python executor once per input layout, records that run into a CUDA
# graph and afterwards only replays it: staging copies into the static input
# buffers, one launch, and copies of the static outputs.  Regions that cannot
# be replayed safely are left uncaptured and run on the executor; the reason
# is logged.

log = logging.getLogger(__name__)

# Frontend keywords every backend receives; they carry no cudagraphs option.
_FRONTEND_KWARGS = frozenset({"name", "dynamic"})

# Host-synchronizing or data-dependent-shape operations: a capture would
# either fail or freeze one run's result/shape into every replay.
_UNSAFE_OPS = frozenset(
    {
        "item",
        "tolist",
        "numpy",
        "nonzero",
        "argwhere",
        "unique",
        "unique_consecutive",
        "masked_select",
        "_local_scalar_dense",
        "eigh",
        "_assert_scalar",
        "_assert_async",
    }
)

_OUT_OF_PLACE_EXCEPTIONS = frozenset({"requires_grad_", "retain_grad_"})


def format_default_skip_message(reason: str) -> str:
    return f"skipping cudagraphs due to {reason}"


def _is_tensor(value: Any) -> bool:
    import tensorplay

    return isinstance(value, tensorplay.Tensor)


def _storage_key(value: Any) -> Any:
    try:
        return value.untyped_storage().data_ptr()
    except Exception:  # noqa: BLE001 - storage-less tensors never alias
        return None


def _target_name(node: Any) -> str:
    if node.op == "call_method":
        return str(node.target)
    return str(getattr(node.target, "__name__", node.target))


def _mutated_argument(node: Any) -> Any:
    """The graph value an in-place node writes to, if any."""

    if node.op not in ("call_function", "call_method"):
        return None
    out = node.kwargs.get("out")
    if out is not None:
        return out
    name = _target_name(node)
    if (
        name.endswith("_")
        and not name.startswith("__")
        and name not in _OUT_OF_PLACE_EXCEPTIONS
        and node.args
    ):
        return node.args[0]
    return None


def find_input_mutations(gm: Any) -> set[int]:
    """Placeholder indices whose storage an in-place node writes."""

    from ..graph import Node

    storages: dict[Any, set[int]] = {}
    mutated: set[int] = set()
    index = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            value = node.meta.get("val")
            if _is_tensor(value):
                key = _storage_key(value)
                if key is not None:
                    storages.setdefault(key, set()).add(index)
            index += 1
            continue
        target = _mutated_argument(node)
        written = target if isinstance(target, (list, tuple)) else (target,)
        for item in written:
            if isinstance(item, Node) and _is_tensor(item.meta.get("val")):
                mutated |= storages.get(_storage_key(item.meta["val"]), set())
    return mutated


def get_device_node_mapping(gm: Any) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for node in gm.graph.nodes:
        value = node.meta.get("val")
        if _is_tensor(value):
            mapping.setdefault(str(value.device), node)
    return mapping


def check_multiple_devices_or_any_cpu_nodes(mapping: dict[str, Any]) -> str | None:
    mapping = {key: node for key, node in mapping.items() if key != "meta"}
    cpu_node = mapping.get("cpu")
    if cpu_node is not None:
        return format_default_skip_message(f"cpu device ({cpu_node.name})")
    if len(mapping) == 1 and next(iter(mapping)).startswith("cuda"):
        return None
    if not mapping:
        return format_default_skip_message("no CUDA tensors")
    return format_default_skip_message(f"multiple devices: {', '.join(mapping)}")


def get_first_incompatible_cudagraph_node(gm: Any) -> Any:
    from ..graph import Node

    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method"):
            continue
        if _target_name(node).rsplit(".", 1)[-1] in _UNSAFE_OPS:
            return node
        # Boolean-mask indexing computes the result size from the data.
        if _target_name(node) in ("getitem", "__getitem__", "index_put", "index_put_"):
            for item in _iter_index_values(node.args[1:] if len(node.args) > 1 else ()):
                if isinstance(item, Node):
                    value = item.meta.get("val")
                    if _is_tensor(value) and str(value.dtype).rsplit(".", 1)[-1] in (
                        "bool",
                        "uint8",
                    ):
                        return node
    return None


def _iter_index_values(value: Any):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_index_values(item)
    else:
        yield value


def check_for_skip(gm: Any) -> str | None:
    mutated = find_input_mutations(gm)
    if mutated:
        placeholders = gm.graph.placeholders
        names = ", ".join(placeholders[i].name for i in sorted(mutated))
        return format_default_skip_message(f"mutated inputs ({names})")
    if skip := check_multiple_devices_or_any_cpu_nodes(get_device_node_mapping(gm)):
        return skip
    if (node := get_first_incompatible_cudagraph_node(gm)) is not None:
        return format_default_skip_message(f"incompatible op ({node.name})")
    return None


def _graph_requires_grad(gm: Any) -> bool:
    for node in gm.graph.nodes:
        value = node.meta.get("val")
        if _is_tensor(value) and value.requires_grad:
            return True
    return False


class _OutputSlot:
    __slots__ = ("index",)

    def __init__(self, index: int) -> None:
        self.index = index


def _flatten_outputs(value: Any, flat: List[Any]) -> Any:
    if _is_tensor(value):
        flat.append(value)
        return _OutputSlot(len(flat) - 1)
    if isinstance(value, tuple):
        return tuple(_flatten_outputs(item, flat) for item in value)
    if isinstance(value, list):
        return [_flatten_outputs(item, flat) for item in value]
    if isinstance(value, dict):
        return {key: _flatten_outputs(item, flat) for key, item in value.items()}
    return value


def _fill_outputs(template: Any, flat: Sequence[Any]) -> Any:
    if isinstance(template, _OutputSlot):
        return flat[template.index]
    if isinstance(template, tuple):
        return tuple(_fill_outputs(item, flat) for item in template)
    if isinstance(template, list):
        return [_fill_outputs(item, flat) for item in template]
    if isinstance(template, dict):
        return {key: _fill_outputs(item, flat) for key, item in template.items()}
    return template


class _CudagraphRunner:
    """Per-specialization replay state for one captured region."""

    def __init__(self, gm: Any, manager: CudaGraphManager) -> None:
        self.gm = gm
        self.executor = gm.forward
        self.manager = manager
        self.placeholders = gm.graph.placeholders
        self.requires_grad = _graph_requires_grad(gm)
        self.templates: Dict[str, Any] = {}
        self._limit_logged = False

    def _bind(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> List[Any]:
        if not kwargs and len(args) == len(self.placeholders):
            return list(args)
        bound = self.gm.signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        values = []
        for node in self.placeholders:
            key = node.target if isinstance(node.target, str) else node.name
            values.append(bound.arguments[key] if key in bound.arguments else bound.arguments[node.name])
        return values

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import tensorplay

        # Replayed outputs carry no autograd history; a region that would
        # record one runs on the executor instead.
        if self.requires_grad and tensorplay.is_grad_enabled():
            return self.executor(*args, **kwargs)
        values = self._bind(args, kwargs)
        positions = [i for i, value in enumerate(values) if _is_tensor(value)]
        tensors = [values[i] for i in positions]
        key = repr(_shape_signature(tensors))
        template = self.templates.get(key)
        if template is None:
            if len(self.templates) >= self.manager.max_entries:
                if not self._limit_logged:
                    self._limit_logged = True
                    log.warning(
                        format_default_skip_message(
                            f"more than {self.manager.max_entries} input layouts"
                        )
                    )
                return self.executor(*args, **kwargs)
            captured_template: List[Any] = []

            def region(*static: Any) -> List[Any]:
                full = list(values)
                for position, tensor in zip(positions, static):
                    full[position] = tensor
                flat: List[Any] = []
                captured_template[:] = [_flatten_outputs(self.executor(*full), flat)]
                return flat

            self.manager.capture(key, region, *tensors)
            template = self.templates[key] = captured_template[0]
        outputs = self.manager.replay(key, *tensors)
        return _fill_outputs(template, [output.clone() for output in outputs])


class CudagraphsBackend:
    """``backend="cudagraphs"``: capture the region once, then replay it."""

    compiler_name = "cudagraphs"
    _managers: "weakref.WeakSet[CudaGraphManager]" = weakref.WeakSet()

    @staticmethod
    def reset() -> None:
        for manager in list(CudagraphsBackend._managers):
            manager.clear()

    @staticmethod
    def __call__(gm: Any, example_inputs: Sequence[Any], **kwargs: Any) -> Any:
        strict_native = bool(kwargs.pop("strict_native", False))
        extra = {key: value for key, value in kwargs.items() if key not in _FRONTEND_KWARGS}
        if extra:
            log.warning("cudagraphs backend ignoring extra kwargs %s", extra)
        return cudagraphs(gm, example_inputs, strict_native=strict_native)


def cudagraphs(gm: Any, example_inputs: Sequence[Any], *, strict_native: bool = False) -> Any:
    skip = check_for_skip(gm)
    if skip is not None:
        if strict_native:
            raise CudaGraphError(f"strict_native=True: {skip}")
        log.warning(skip)
        return gm.forward
    manager = CudaGraphManager()
    CudagraphsBackend._managers.add(manager)
    runner = _CudagraphRunner(gm, manager)
    runner._tensorplay_codegen = "cudagraphs"  # type: ignore[attr-defined]
    return runner
