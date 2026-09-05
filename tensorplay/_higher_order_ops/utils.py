"""Shared helpers for higher-order operators.

The entry points here mirror the contract of a traced higher-order call:
``setup_compilation_env`` prepares the capture state and yields the backend
that inner ``compile`` invocations should target.  The remaining helpers
provide the dispatch-role plumbing shared by every operator in this package:
autograd guards, subgraph re-tracing, mutation detection, mode redirection,
backward-state partitioning, and lifted-argument validation.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Iterator

import tensorplay
from tensorplay import Tensor
from tensorplay._higher_order_ops._hop_base import _AutoDispatchBelowAutograd


@dataclass
class UnsupportedAliasMutationException(RuntimeError):
    reason: str


def autograd_not_implemented_inner(
    operator: Callable[..., Any], delayed_error: bool, *args: Any, **kwargs: Any
) -> Any:
    """If autograd is enabled and any of the arguments require grad this will either
    raise an error or return a DelayedError depending on the value of delayed.

    Args:
        operator: The Operator to call with the *args and **kwargs with
        op_name: The name of the Operator
        delayed_error: If True, return a DelayedError instead of raising an error
        args: The flattened operands to the Operator
        kwargs: The keyword arguments to the Operator

    Raises:
        RuntimeError: If autograd is enabled and any of the arguments to the Operator
    """
    with _AutoDispatchBelowAutograd():
        result = operator(*args, **kwargs)
        flat_operands, _ = tensorplay.utils._pytree.tree_flatten(args)
        if tensorplay.is_grad_enabled() and any(
            f.requires_grad for f in flat_operands if isinstance(f, Tensor)
        ):
            if delayed_error:
                delayed_error_fn = _make_delayed_error(operator)
                return tensorplay.utils._pytree.tree_map_only(
                    Tensor, delayed_error_fn, result
                )
            raise RuntimeError(f"Autograd not implemented for {operator}")
        return result


def _make_delayed_error(operator: Callable[..., Any]) -> Callable[[Tensor], Tensor]:
    """Build a node that raises when its backward is first evaluated."""
    functions_mod = getattr(tensorplay._C, "_functions", None)
    if functions_mod is not None and hasattr(functions_mod, "DelayedError"):
        err_fn = functions_mod.DelayedError(f"Autograd not implemented for {operator}", 1)

        def delayed(tensor: Tensor) -> Tensor:
            if tensorplay.is_floating_point(tensor) or tensorplay.is_complex(tensor):
                tensor = tensor.detach()
                tensor.requires_grad = True
            return err_fn(tensor)

        return delayed

    def immediate(tensor: Tensor) -> Tensor:
        raise RuntimeError(f"Autograd not implemented for {operator}")

    return immediate


def autograd_not_implemented(op: Callable[..., Any], deferred_error: bool) -> Callable:
    def inner(*args, **kwargs):
        return autograd_not_implemented_inner(op, deferred_error, *args, **kwargs)

    return inner


def _maybe_run_with_interpreter(fn):
    maybe_interpreted_fn = fn
    from tensorplay.graph import traceback as fx_traceback

    if isinstance(fn, tensorplay.graph_module.GraphModule) and (
        fx_traceback.should_preserve_node_meta
    ):
        # Running graph with interpreter is needed for propagating the stack_trace
        def graph_with_interpreter(*args):
            from tensorplay.graph import interpreter

            gm = fn
            with set_traceback_preserve_node_meta():
                graph = interpreter.Interpreter(gm).run(*args)
            return graph

        return graph_with_interpreter
    else:
        return maybe_interpreted_fn


@contextlib.contextmanager
def set_traceback_preserve_node_meta(preserve_node_meta: bool = True):
    from tensorplay.graph import traceback as fx_traceback

    prev = fx_traceback.should_preserve_node_meta
    fx_traceback.should_preserve_node_meta = preserve_node_meta
    try:
        yield
    finally:
        fx_traceback.should_preserve_node_meta = prev


def reenter_make_fx(fn, subgraph_decomp_table=None):
    """Callee of a HOP that re-enters the active graph capture to trace ``fn``
    into its own subgraph."""
    from tensorplay.graph.experimental import proxy_tensor

    @functools.wraps(fn)
    def wrapped(*args):
        current_tracer = proxy_tensor._CURRENT_MAKE_GRAPH_TRACER.get()
        if current_tracer is None:
            raise AssertionError(
                "Cannot reenter make_fx when we're not under a make_fx tracing session"
            )
        if subgraph_decomp_table is None:
            gm = current_tracer.trace_subgraph(_maybe_run_with_interpreter(fn), *args)
        else:
            gm = current_tracer.trace_subgraph_custom_decomp(
                _maybe_run_with_interpreter(fn), subgraph_decomp_table, *args
            )
        return gm

    return wrapped


def _maybe_reenter_make_fx(fn, subgraph_decomp_table=None):
    """Like :func:`reenter_make_fx`, but outside an active capture it traces a
    standalone subgraph instead of erroring."""
    from tensorplay.graph.experimental import proxy_tensor

    if proxy_tensor._CURRENT_MAKE_GRAPH_TRACER.get() is not None:
        return reenter_make_fx(fn, subgraph_decomp_table=subgraph_decomp_table)

    @functools.wraps(fn)
    def wrapped(*args):
        return make_fx(fn, subgraph_decomp_table)(*args)

    return wrapped


def make_fx(
    f,
    decomposition_table=None,
    tracing_mode="real",
    _allow_non_fake_inputs=False,
    *,
    pre_dispatch=False,
    record_module_stack=False,
    _allow_fake_constant=False,
    _error_on_data_dependent_ops=True,
    record_stack_traces=False,
    proxy_module_inputs=False,
    _disable_function_metadata_mode=False,
):
    from tensorplay.graph.experimental.proxy_tensor import make_graph

    return make_graph(
        f,
        decomposition_table,
        tracing_mode,
        _allow_non_fake_inputs,
        pre_dispatch=pre_dispatch,
        record_module_stack=record_module_stack,
        _allow_fake_constant=_allow_fake_constant,
        _error_on_data_dependent_ops=_error_on_data_dependent_ops,
        record_stack_traces=record_stack_traces,
        proxy_module_inputs=proxy_module_inputs,
        _disable_function_metadata_mode=_disable_function_metadata_mode,
    )


_INPLACE_SUFFIX = "_"


def _target_mutates_inplace(target: Any) -> bool:
    """Whether a graph call target is an in-place operation by naming
    convention: in-place operators carry a trailing underscore."""
    name = getattr(target, "__name__", None)
    if name is None:
        name = str(target)
    return name.endswith(_INPLACE_SUFFIX)


def _collect_fake_inputs(inputs: Sequence[Any]) -> list[Any]:
    """Snapshot the example values of the traced inputs, unwrapping proxies."""
    from tensorplay.graph.node import Node
    from tensorplay.graph.proxy import Proxy

    inputs_fake: list[Any] = []
    for inp in inputs:
        if isinstance(inp, (Proxy, Node)):
            node = inp.node if isinstance(inp, Proxy) else inp
            val = node.meta.get("example_value", node)
            inputs_fake.append(val)
        else:
            inputs_fake.append(inp)
    return inputs_fake


def _graph_mutated_inputs(gm: Any, inputs: Sequence[Any]) -> list[int]:
    """Index the traced inputs that the graph writes in place.

    The capture is replayed on placeholder tensors; any call target that
    follows the in-place naming convention and consumes a graph placeholder
    marks that placeholder as mutated.
    """
    del inputs
    mutated: list[int] = []
    graph = getattr(gm, "graph", None)
    if graph is None:
        return mutated
    placeholder_nodes = [
        node for node in graph.nodes if node.op == "placeholder"
    ]
    node_to_input_idx = {id(node): i for i, node in enumerate(placeholder_nodes)}
    for node in graph.nodes:
        if node.op not in ("call_function", "call_method"):
            continue
        if not _target_mutates_inplace(node.target):
            continue
        for arg in node._input_nodes.values():
            idx = node_to_input_idx.get(id(arg))
            if idx is not None and idx not in mutated:
                mutated.append(idx)
    return mutated


def _as_graph_module(gm: Any, inputs: Sequence[Any], pre_dispatch: bool = False) -> Any:
    """Materialize a callable into a GraphModule when it is not one yet."""
    from tensorplay.graph_module import GraphModule as _GM

    if isinstance(gm, _GM):
        return gm
    return make_fx(gm)(*inputs)


def potential_input_alias_or_mutation(gm: Any, inputs: Sequence[Any], pre_dispatch: bool = False):
    """Return the alias maps and mutated-input list for a traced graph.

    The callable is captured first when it is not already a graph module.
    Only input mutation is detected in this build; alias maps stay empty
    because storage identity is not tracked by the tracer.
    """
    gm = _as_graph_module(gm, inputs, pre_dispatch)
    mutated = _graph_mutated_inputs(gm, inputs)
    return (dict(), dict(), dict()), mutated


def has_potential_input_alias_or_mutation(gm, inputs, pre_dispatch=False):
    (
        (
            inp_inp_alias_map,
            inp_out_alias_map,
            out_out_alias_map,
        ),
        inp_mutation,
    ) = potential_input_alias_or_mutation(gm, inputs, pre_dispatch)
    return (
        any(
            (
                len(inp_inp_alias_map) > 0,
                len(inp_out_alias_map) > 0,
                len(out_out_alias_map) > 0,
            )
        ),
        len(inp_mutation) > 0,
    )


def _has_potential_branch_input_mutation(gm, inputs, pre_dispatch=False):
    (
        (_, _, _),
        inp_mutation,
    ) = potential_input_alias_or_mutation(gm, inputs, pre_dispatch)

    return len(inp_mutation) > 0


def redirect_to_mode(hop: Any, mode: Any):
    """Utility for redispatching HOP to underlying mode

    Args:
        hop: The HOP to redispatch
        mode: The mode to redispatch to

    Returns:
        A decorated function that implements the HOP for the given mode
    """

    @hop.py_impl(mode)
    def impl(mode, *args, **kwargs):
        return mode.__torch_dispatch__(hop, [], args, kwargs)

    return impl


def save_values_for_backward(ctx: Any, args: Sequence[Any]) -> None:
    """Partition a mixed tensor / non-tensor pytree for backward.

    Tensors go through ``ctx.save_for_backward``; every other value is stored
    on the context directly, with a position map to reassemble the original
    ordering in :func:`saved_values`.
    """
    allowed_types = (Tensor, int, type(None))
    for arg in args:
        if not isinstance(arg, allowed_types):
            raise AssertionError(f"Invalid arg types in {args}")
    partitioned_args: list[Any] = [[], []]
    pos = []
    for arg in args:
        idx = 0 if isinstance(arg, Tensor) else 1
        partitioned_args[idx].append(arg)
        pos.append(idx)

    if hasattr(ctx, "non_tensor_args"):
        raise AssertionError("ctx already has non_tensor_args attribute.")
    if hasattr(ctx, "pos"):
        raise AssertionError("ctx already has pos attribute.")
    ctx.save_for_backward(*partitioned_args[0])
    ctx.non_tensor_args = partitioned_args[1]
    ctx.pos = pos


def saved_values(ctx: Any) -> tuple[Any, ...]:
    args = []
    t_idx = 0
    s_idx = 0
    saved_tensors = ctx.saved_tensors
    for p in ctx.pos:
        if p == 0:
            args.append(saved_tensors[t_idx])
            t_idx += 1
        else:
            args.append(ctx.non_tensor_args[s_idx])
            s_idx += 1
    if t_idx + s_idx != len(ctx.pos):
        raise AssertionError(
            f"t_idx ({t_idx}) + s_idx ({s_idx}) != len(ctx.pos) ({len(ctx.pos)})"
        )
    return tuple(args)


def validate_subgraph_args_types(lifted_args: tuple[Any, ...] | list[Any]) -> None:
    allowed_types = (Tensor, int)
    if not all(isinstance(arg, allowed_types) for arg in lifted_args):
        raise AssertionError(
            f"{lifted_args} can only be of {allowed_types} but got {tuple(type(arg) for arg in lifted_args)}"
        )


def has_user_subclass(args, allowed_subclasses) -> bool:
    """Check if any tensor arguments are user subclasses.

    This is used to determine if tensor subclasses should get a chance to run
    their own implementation first before falling back to the default implementation.

    Args:
        args: Arguments to check (will be flattened with pytree)
        allowed_subclasses: Tuple of allowed subclass types

    Returns:
        True if user tensor subclasses are found, False otherwise
    """
    flat_args, _ = tensorplay.utils._pytree.tree_flatten(args)

    return any(
        isinstance(a, Tensor)
        and type(a) is not Tensor
        and not isinstance(a, allowed_subclasses)
        for a in flat_args
    )


def split_into_chunks(iterable: Sequence[Any], chunk_sizes: list[int]) -> list[Any]:
    if sum(chunk_sizes) != len(iterable):
        raise AssertionError(
            f"the sum of all chunks ({sum(chunk_sizes)}) needs to match the length of the iterable ({len(iterable)})."
        )
    elements = []
    idx = 0
    for size in chunk_sizes:
        elements.append(iterable[idx : idx + size])
        idx += size
    return elements


@contextlib.contextmanager
def setup_compilation_env() -> Iterator[Any]:
    """
    Context manager that sets up the environment and backend for ``compile``
    invoked inside a higher-order operator or an export region.

    Yields the backend that the inner compile call should pass on.
    """
    from tensorplay.compiler import get_default_backend

    yield get_default_backend()
