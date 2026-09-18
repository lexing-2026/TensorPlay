"""Debug compiler backends.

They isolate frontend problems from lowering problems: each one runs the
captured :class:`GraphModule` as-is (or deliberately breaks it) instead of
generating code.  All are registered with the ``debug`` tag, so
``list_backends()`` hides them unless ``exclude_tags=None`` is passed.

* ``eager`` — run the captured graph on its Python executor.
* ``eager_noexcept`` — as ``eager``, but any exception raised by the graph
  is reported as a compiler failure (checks that capture emits runnable
  graphs).
* ``eager_debug`` — run node by node through the graph interpreter, so an
  error names the failing node and its original source location.
* ``*_TESTING_ONLY`` — inject compile-time, run-time and accuracy failures on
  ``relu`` for exercising error reporting and minimization tooling.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..graph import GraphModule
from .registry import register_debug_backend as register_backend

log = logging.getLogger(__name__)


@register_backend
def eager(
    gm: GraphModule, example_inputs: list[Any], **kwargs: Any
) -> Callable[..., Any]:
    if kwargs:
        log.warning("eager backend ignoring extra kwargs %s", kwargs)
    return gm.forward


@register_backend
def eager_noexcept(
    gm: GraphModule, example_inputs: list[Any], **kwargs: Any
) -> Callable[..., Any]:
    if kwargs:
        log.warning("eager_noexcept backend ignoring extra kwargs %s", kwargs)

    def inner(*args: Any, **call_kwargs: Any) -> Any:
        try:
            return gm(*args, **call_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Unexpected exception when running generated GraphModule"
            ) from exc

    return inner


@register_backend
def eager_debug(
    gm: GraphModule, example_inputs: list[Any], **kwargs: Any
) -> Callable[..., Any]:
    if kwargs:
        log.warning("eager_debug backend ignoring extra kwargs %s", kwargs)

    from ..graph.interpreter import Interpreter

    def inner(*args: Any, **call_kwargs: Any) -> Any:
        return Interpreter(gm).run(*args, **call_kwargs)

    return inner


# --------------------------------------------------------------------------
# Deliberately broken backends for testing error reporting
# --------------------------------------------------------------------------


class ReluCompileError(Exception):
    pass


class TestingOnlyCompileError(Exception):
    pass


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _is_relu(node: Any) -> bool:
    if node.op == "call_method":
        return node.target == "relu"
    if node.op != "call_function":
        return False
    import tensorplay
    import tensorplay.nn.functional as F

    return node.target is tensorplay.relu or node.target is F.relu


@register_backend
def relu_compile_error_TESTING_ONLY(
    gm: GraphModule, example_inputs: list[Any], **kwargs: Any
) -> GraphModule:
    for node in gm.graph.nodes:
        if _is_relu(node):
            raise ReluCompileError
    return gm


@register_backend
def relu_runtime_error_TESTING_ONLY(
    gm: GraphModule, example_inputs: list[Any], **kwargs: Any
) -> GraphModule:
    for node in gm.graph.nodes:
        if _is_relu(node):
            node.op = "call_function"
            node.target = _assert
            node.args = (False, "ReluRuntimeError")
            node.kwargs = {}
    gm.recompile()
    return gm


@register_backend
def relu_accuracy_error_TESTING_ONLY(
    gm: GraphModule, example_inputs: list[Any], **kwargs: Any
) -> GraphModule:
    import tensorplay

    for node in gm.graph.nodes:
        if _is_relu(node):
            node.op = "call_function"
            node.target = tensorplay.add
            node.args = (node.args[0], 1)
            node.kwargs = {}
    gm.recompile()
    return gm


@register_backend
def non_leaf_compile_error_TESTING_ONLY(
    gm: GraphModule, example_inputs: list[Any], **kwargs: Any
) -> GraphModule:
    # Only graphs doing real work are checked, so trivial pass-through
    # regions still compile.
    for node in gm.graph.nodes:
        if node.op == "call_function":
            break
    else:
        return gm
    for value in example_inputs:
        if not getattr(value, "is_leaf", True):
            raise TestingOnlyCompileError
    return gm
