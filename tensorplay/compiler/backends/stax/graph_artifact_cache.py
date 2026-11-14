"""A cache of finished regions, keyed by what the region is made of.

Scheduling a region and printing its wrapper depends on nothing but the region
itself: the captured operations, the constants it carries, the shapes it was
specialized to, and the toolchain that builds the result.  The finished wrapper
and the metadata the built module reads at call time can therefore be stored
under a key made of exactly those things, and a region seen again is loaded
rather than re-scheduled and re-printed.

The payload holds only plain Python data, so it survives a round trip through
the disk unchanged.  A cache that cannot be read or written must never break a
compile: every failure here leaves the caller on the uncached path, and a
directory that belongs to another user is simply never written.
"""

from __future__ import annotations

import dataclasses
import os
import pickle

from .cache_key import FX_GRAPH_CACHE_KEY_STRATEGY
from .codegen.cpp_wrapper import CppWrapperCode
from .runtime.cache_dir_utils import cache_dir
from .utils import ValueWithLineMap

#: Bumped whenever anything that shapes the printed wrapper changes in a way
#: the key below cannot see.  A key that misses such a change would hand back
#: an artifact built by the old rules, so the version is the escape hatch:
#: raise it, and every stored entry becomes unreadable at once.
SCHEMA_VERSION = "1"

#: How deep a nested argument is rendered before the walk stops.  Captured
#: arguments are shallow: nodes, scalars, and short sequences of both.
_MAX_ARG_DEPTH = 8


def _arg_digest(value: object, depth: int = 0) -> object:
    """A stable, JSON-able stand-in for one captured argument.

    Nodes stand in by name -- two captures of the same program name their
    nodes the same way -- and sequences and maps are walked so that a list of
    nodes does not collapse into one opaque repr.  Anything else is rendered
    with ``repr``, which for scalars and the other values a captured graph
    carries is both complete and stable.
    """

    if depth >= _MAX_ARG_DEPTH:
        return repr(value)
    if hasattr(value, "op") and hasattr(value, "name"):
        return {"kind": "node", "name": value.name}
    if isinstance(value, (list, tuple)):
        return {"kind": "seq", "items": [_arg_digest(v, depth + 1) for v in value]}
    if isinstance(value, dict):
        return {
            "kind": "map",
            "items": sorted(
                (str(k), _arg_digest(v, depth + 1)) for k, v in value.items()
            ),
        }
    return {"kind": "value", "text": repr(value)}


def _node_digest(node: object) -> dict:
    args = [_arg_digest(v) for v in getattr(node, "args", ()) or ()]
    kwargs = {k: _arg_digest(v) for k, v in (getattr(node, "kwargs", None) or {}).items()}
    return {
        "op": node.op,
        "target": repr(node.target),
        "name": node.name,
        "args": args,
        "kwargs": kwargs,
    }


def _tensor_spec(tensor: object) -> list:
    return [
        list(tensor.shape),
        str(tensor.dtype),
        list(tensor.stride()),
        f"{tensor.device.type}{tensor.device.index or 0}",
    ]


def _config_digest() -> dict:
    """Every plain knob of this module's configuration, as one mapping.

    The scheduler and the code generators read dozens of scalar knobs, and a
    knob that changes has to change the key even though nothing else about the
    region did.  Enumerating them by name would fall behind, so everything
    whose value is a number, a string, a boolean, none, or a flat sequence of
    those is taken as-is and the rest is left out: knobs that carry callables
    or objects describe behaviour the version above has to answer for.
    """

    from . import config

    knobs: dict[str, object] = {}
    for name in dir(config):
        if name.startswith("_"):
            continue
        value = getattr(config, name)
        if value is None or isinstance(value, (bool, int, float, str)):
            knobs[name] = value
        elif isinstance(value, (list, tuple)) and all(
            v is None or isinstance(v, (bool, int, float, str)) for v in value
        ):
            knobs[name] = list(value)
    return knobs


def graph_key(graph: object) -> str:
    """The name under which this region's finished wrapper is stored.

    Everything the printed wrapper depends on goes in: the captured program
    node by node, the spec of every input the region is specialized to, a
    description of every constant that carries its content hash, the knobs the
    scheduler and code generators read, the toolchain that will build the
    result, and the version above.  Two regions that agree here agree on every
    line the printer would have written.
    """

    from .codecache import CacheBase
    from .cpp_builder import get_compiler_version_info, get_cpp_compiler

    inputs = [
        _tensor_spec(tensor)
        for tensor in getattr(graph, "example_inputs", ())
        if hasattr(tensor, "shape")
    ]
    components = {
        "schema": SCHEMA_VERSION,
        "graph": [_node_digest(node) for node in graph.graph_module.graph.nodes],
        "inputs": inputs,
        "constants": sorted(
            (name, text) for name, text in graph.constant_reprs.items()
        ),
        "flags": {
            "is_inference": graph.is_inference,
            "is_backward": graph.is_backward,
            "aot_mode": repr(graph.aot_mode),
            "layout_opt": graph.layout_opt,
        },
        "config": _config_digest(),
        "compiler": get_compiler_version_info(get_cpp_compiler()),
        "system": CacheBase.get_system()["hash"],
    }
    return FX_GRAPH_CACHE_KEY_STRATEGY.key_from_json(components)


def _payload(wrapper_code: object) -> dict:
    """The part of a finished wrapper the builder downstream reads.

    A C++ wrapper carries the source plus the metadata the built module reads
    at call time -- names, indices, shapes, strides, all plain values.  A
    Python wrapper is nothing but its text; the line map is the bookkeeping of
    the printing itself (it points errors back at the region that produced
    each line) and is not part of the artifact, so a loaded wrapper reports
    from the wrapper's own lines instead.
    """

    if isinstance(wrapper_code, ValueWithLineMap):
        return {"kind": "py", "value": wrapper_code.value}
    return {
        "kind": "cpp",
        "value": wrapper_code.value,
        "buffer_meta": wrapper_code.buffer_meta,
        "output_names": wrapper_code.output_names,
        "tensor_index": wrapper_code.tensor_index,
        "num_tensors": wrapper_code.num_tensors,
        "num_sizevars": wrapper_code.num_sizevars,
        "graph_input_names": wrapper_code.graph_input_names,
    }


def _artifact_path(key: str) -> str:
    from .codecache import CacheBase

    return os.path.join(
        cache_dir(), "cache", CacheBase.get_system()["hash"], "graph", f"{key}.pkl"
    )


def lookup(graph: object) -> CppWrapperCode | ValueWithLineMap | None:
    """The finished wrapper of this region, if one is already stored.

    A region whose wrapper is on disk comes back as the same dataclass the
    code generators would have produced, with the printing bookkeeping left
    empty.  Anything unreadable -- a missing file, a payload from an older
    schema, a damaged entry -- is a miss, not an error.
    """

    if not _enabled():
        return None
    try:
        key = graph_key(graph)
        path = _artifact_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fp:
            payload = pickle.load(fp)
        if not isinstance(payload, dict) or "value" not in payload:
            return None
        if payload.get("kind") == "py":
            return ValueWithLineMap(value=payload["value"], line_map=[])
        return CppWrapperCode(
            value=payload["value"],
            line_map=[],
            buffer_meta=payload["buffer_meta"],
            output_names=payload["output_names"],
            tensor_index=payload["tensor_index"],
            sizevar_index={},
            num_tensors=payload["num_tensors"],
            num_sizevars=payload["num_sizevars"],
            graph_input_names=payload["graph_input_names"],
        )
    except Exception:
        if os.environ.get("TP_STAX_CACHE_DEBUG"):
            import traceback

            traceback.print_exc()
        return None


def publish(graph: object, wrapper_code: object) -> None:
    """Store the wrapper this region just printed, under its own key.

    Failures are silent: a cache that cannot write costs compiles nothing but
    the work of trying, and the next run simply walks the uncached path again.
    """

    if not _enabled() or not isinstance(
        wrapper_code, (CppWrapperCode, ValueWithLineMap)
    ):
        return
    try:
        key = graph_key(graph)
        path = _artifact_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "wb") as fp:
            pickle.dump(_payload(wrapper_code), fp, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(path + ".tmp", path)
    except Exception:
        if os.environ.get("TP_STAX_CACHE_DEBUG"):
            import traceback

            traceback.print_exc()


def _enabled() -> bool:
    from . import config

    return config.cache_graph_artifacts is not False and (
        not config.force_disable_caches
    )
