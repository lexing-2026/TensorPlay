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
* ``aot_eager`` / ``aot_eager_default_partitioner`` — ahead-of-time autograd
  with pass-through compilers: forward and backward graphs are traced and
  partitioned (min-cut / default partitioner) and then run as traced, which
  isolates autograd tracing and partitioning problems from code generation.
* ``*_TESTING_ONLY`` — inject compile-time, run-time and accuracy failures on
  ``relu`` for exercising error reporting and minimization tooling.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ...graph import GraphModule
from .._core.registry import register_debug_backend as register_backend

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

    from ...graph.interpreter import Interpreter

    def inner(*args: Any, **call_kwargs: Any) -> Any:
        return Interpreter(gm).run(*args, **call_kwargs)

    return inner


def make_eager_backend_with_function_mode(mode: Any) -> Callable[..., Any]:
    return make_eager_backend_with_function_modes([mode])


def make_eager_backend_with_function_modes(modes: Any) -> Callable[..., Any]:
    """Eager backend that runs the graph under the given function modes.

    The modes are entered around each call instead of being traced, for
    regions whose modes must observe the executed operators.
    """

    from contextlib import ExitStack

    modes = list(modes)

    def fn(gm: GraphModule, example_inputs: list[Any], **kwargs: Any) -> Callable[..., Any]:
        def wrapper(*args: Any, **call_kwargs: Any) -> Any:
            with ExitStack() as stack:
                for mode in modes:
                    stack.enter_context(mode)
                return gm.forward(*args, **call_kwargs)

        return wrapper

    return fn


# --------------------------------------------------------------------------
# Ahead-of-time autograd with pass-through compilers
# --------------------------------------------------------------------------


def boxed_nop(fx_g: GraphModule, example_inputs: list[Any]) -> Callable[..., Any]:
    """Run a traced graph as is, taking its inputs as one list it may clear."""

    from ...graph.graph import _BoxedCodeGen

    fx_g.graph.set_codegen(_BoxedCodeGen())
    fx_g.recompile()
    forward_fn = fx_g.forward

    def run(args: Any) -> Any:
        return forward_fn(args)

    run._boxed_call = True  # type: ignore[attr-defined]
    return run


def boxed_nop_with_mode(fx_g: GraphModule, example_inputs: list[Any], *, mode: Any) -> Callable[..., Any]:
    run_graph = boxed_nop(fx_g, example_inputs)

    def run(args: Any) -> Any:
        with mode:
            return run_graph(args)

    run._boxed_call = True  # type: ignore[attr-defined]
    return run


def aot_eager(
    gm: GraphModule,
    example_inputs: list[Any],
    fw_compiler: Callable[..., Any] | None = None,
    bw_compiler: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    from .._core.aot_autograd import min_cut_rematerialization_partition
    from .._core.common import aot_autograd

    return aot_autograd(
        fw_compiler=fw_compiler or boxed_nop,
        bw_compiler=bw_compiler or boxed_nop,
        partition_fn=min_cut_rematerialization_partition,
        keep_inference_input_mutations=True,
    )(gm, example_inputs, **kwargs)


register_backend(name="aot_eager", compiler_fn=aot_eager)


def aot_eager_default_partitioner(
    gm: GraphModule, example_inputs: list[Any], **kwargs: Any
) -> Callable[..., Any]:
    from .._core.common import aot_autograd

    return aot_autograd(fw_compiler=boxed_nop, keep_inference_input_mutations=True)(
        gm, example_inputs, **kwargs
    )


register_backend(
    name="aot_eager_default_partitioner", compiler_fn=aot_eager_default_partitioner
)


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
