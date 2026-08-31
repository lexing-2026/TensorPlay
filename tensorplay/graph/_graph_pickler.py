"""Pickle support for graph modules and their cyclic graph objects."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import pickle
import weakref
from collections.abc import Callable, Generator
from typing import Any

from .graph_module import GraphModule


def _ops_filter_safe(name: str) -> bool:
    """Return whether a qualified operation name is safe for portable data."""

    return name.startswith(("builtins.", "operator.", "math.", "tensorplay."))


def _node_metadata_key_filter_safe(key: str) -> bool:
    """Filter metadata that commonly contains process-local objects."""

    return key not in {"source_fn_stack", "nn_module_stack", "stack_trace"}


@dataclasses.dataclass
class Options:
    ops_filter: Callable[[str], bool] | None = _ops_filter_safe
    node_metadata_key_filter: Callable[[str], bool] | None = _node_metadata_key_filter_safe
    ignore_raw_node: bool = False


def _unpickle_as_none() -> None:
    return None


def _unpickle_as_weakref(referent: object) -> weakref.ref[object]:
    return weakref.ref(referent)


def _unpickle_as_dead_weakref() -> Callable[[], None]:
    return lambda: None


@contextlib.contextmanager
def patch_pytree_map_over_slice() -> Generator[None, None, None]:
    """Provide a scoped hook for callers that register slice tree nodes."""

    yield


def _rebuild_graph_module(
    root: Any, graph: Any, signature: Any, metadata: dict[str, Any]
) -> GraphModule:
    result = GraphModule(root, graph, signature)
    result.meta.update(metadata)
    return result


class GraphPickler(pickle.Pickler):
    """Pickler with an explicit reducer for executable graph modules."""

    def __init__(self, file: io.BufferedIOBase, options: Options | None = None) -> None:
        super().__init__(file)
        self.options = options or Options()

    def reducer_override(self, obj: object):
        if isinstance(obj, GraphModule):
            return (
                _rebuild_graph_module,
                (obj.root, obj.graph, obj.signature, dict(obj.meta)),
            )
        if isinstance(obj, weakref.ReferenceType):
            referent = obj()
            return (
                (_unpickle_as_weakref, (referent,))
                if referent is not None
                else (_unpickle_as_dead_weakref, ())
            )
        return NotImplemented

    @classmethod
    def dumps(cls, obj: object, options: Options | None = None) -> bytes:
        with io.BytesIO() as stream:
            cls(stream, options).dump(obj)
            return stream.getvalue()

    @staticmethod
    def loads(data: bytes, **kwargs: Any) -> object:
        del kwargs
        return pickle.loads(data)

    @classmethod
    def debug_dumps(
        cls,
        obj: object,
        options: Options | None = None,
        *,
        max_depth: int = 80,
        max_iter_items: int = 50,
        verbose: bool = True,
    ) -> str | None:
        del max_depth, max_iter_items, verbose
        try:
            cls.dumps(obj, options)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        return None


class _UnpickleState:
    def __init__(self, value: Any = None) -> None:
        self.value = value


class _GraphUnpickler(pickle.Unpickler):
    def __init__(self, file: io.BufferedIOBase, state: _UnpickleState | None = None) -> None:
        super().__init__(file)
        self.state = state or _UnpickleState()


__all__ = [
    "GraphPickler",
    "Options",
    "_GraphUnpickler",
    "_UnpickleState",
    "_node_metadata_key_filter_safe",
    "_ops_filter_safe",
    "patch_pytree_map_over_slice",
]
