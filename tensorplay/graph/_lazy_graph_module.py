"""Deferred code generation for graph modules that are only inspected later."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ._compatibility import compatibility
from .graph_module import GraphModule

_use_lazy_graph_module_flag = False
_force_skip_lazy_graph_module_flag = False


@compatibility(is_backward_compatible=False)
@contextmanager
def _force_skip_lazy_graph_module() -> Iterator[None]:
    global _force_skip_lazy_graph_module_flag
    previous = _force_skip_lazy_graph_module_flag
    _force_skip_lazy_graph_module_flag = True
    try:
        yield
    finally:
        _force_skip_lazy_graph_module_flag = previous


@compatibility(is_backward_compatible=False)
@contextmanager
def _use_lazy_graph_module(should_use: bool) -> Iterator[None]:
    global _use_lazy_graph_module_flag
    previous = _use_lazy_graph_module_flag
    _use_lazy_graph_module_flag = bool(should_use) and not _force_skip_lazy_graph_module_flag
    try:
        yield
    finally:
        _use_lazy_graph_module_flag = previous


@compatibility(is_backward_compatible=False)
def _get_graph_module_cls() -> type[GraphModule]:
    return _LazyGraphModule if _use_lazy_graph_module_flag else GraphModule


def _make_graph_module(
    *args: Any, graph_module_cls: type[GraphModule] | None = None, **kwargs: Any
) -> GraphModule:
    cls = graph_module_cls or _get_graph_module_cls()
    return cls(*args, **kwargs)


@compatibility(is_backward_compatible=False)
def _unwrap_lazy_graph_module(gm: GraphModule) -> GraphModule:
    if not isinstance(gm, _LazyGraphModule):
        return gm
    gm.real_recompile()
    result = GraphModule(gm.root, gm.graph, gm.signature)
    result.meta.update(gm.meta)
    return result


@compatibility(is_backward_compatible=False)
class _LazyGraphModule(GraphModule):
    """Delay explicit executor generation until code or execution is needed."""

    @classmethod
    def from_graphmodule(cls, gm: GraphModule) -> GraphModule:
        if isinstance(gm, cls):
            return gm
        return cls(gm, gm.graph, gm.signature)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._lazy_needs_recompile = False
        super().__init__(*args, **kwargs)
        self._lazy_needs_recompile = True

    def recompile(self):
        self._lazy_needs_recompile = True
        self._compiled_forward = None
        self._compiled_impl = None
        return self._lazy_forward

    def real_recompile(self) -> Any:
        if self._lazy_needs_recompile or self._compiled_forward is None:
            result = GraphModule.recompile(self)
            self._lazy_needs_recompile = False
            return result
        return self._compiled_forward

    def _needs_recompile(self) -> bool:
        return self._lazy_needs_recompile or self._compiled_forward is None

    @classmethod
    def force_recompile(cls, gm: GraphModule) -> None:
        if isinstance(gm, cls):
            gm.real_recompile()

    def _lazy_forward(self, *args: Any, **kwargs: Any) -> Any:
        self.real_recompile()
        if self._needs_recompile():
            raise AssertionError("recompilation did not produce an executor")
        return self(*args, **kwargs)

    forward = _lazy_forward

    @property
    def code(self) -> str:
        self.real_recompile()
        return super().code

    def __str__(self) -> str:
        self.real_recompile()
        return super().__str__()


__all__ = [
    "_LazyGraphModule",
    "_force_skip_lazy_graph_module",
    "_get_graph_module_cls",
    "_make_graph_module",
    "_unwrap_lazy_graph_module",
    "_use_lazy_graph_module",
]
