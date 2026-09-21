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
from tensorplay._higher_order_ops._hop_base import (
    _AutoDispatchBelowAutograd,
    disable_functional_mode,
    suspend_functionalization,
)


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
    from tensorplay.graph.graph_module import GraphModule as _GraphModule

    if isinstance(fn, _GraphModule) and (
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
    from tensorplay.graph.graph_module import GraphModule as _GM

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
        return mode.__tensorplay_dispatch__(hop, [], args, kwargs)

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


# ---------------------------------------------------------------------------
# Structural control-flow operator helpers (cond / while_loop / map / scan).
# These helpers operate on flattened argument lists produced by the pytree
# flattening that each structural control-flow operator performs on entry.
# ---------------------------------------------------------------------------


def _pytree_api():
    """Late-bound pytree access (avoids import cycles at module scope)."""
    from tensorplay.utils import _pytree

    return _pytree


def _tensor_data_ptr(t: Tensor) -> Any:
    """A hashable storage identity for aliasing checks, or None."""
    get = getattr(t, "data_ptr", None)
    if callable(get):
        try:
            return (get(), t.storage_offset() if hasattr(t, "storage_offset") else 0)
        except Exception:
            return None
    return None


def filter_with_masks(data: Sequence[Any], masks: Sequence[bool]) -> list[Any]:
    """Keep only the positions of ``data`` whose mask entry is True."""
    if len(data) != len(masks):
        raise AssertionError(
            f"data length ({len(data)}) != masks length ({len(masks)})"
        )
    return [item for item, keep in zip(data, masks) if keep]


def fill_none_with_masks(data: Sequence[Any], masks: Sequence[bool]) -> list[Any]:
    """Reinsert None at the positions dropped by :func:`filter_with_masks`."""
    data_iter = iter(data)
    return [next(data_iter) if kept else None for kept in masks]


def create_fn_remove_none(fn: Callable) -> tuple[Callable, list[bool]]:
    """Wrap ``fn`` so its non-Tensor output leaves are dropped.

    Returns ``(wrapped, mask)``: ``wrapped(*args)`` calls ``fn(*args)``,
    flattens the pytree result and returns only its Tensor leaves as a
    list.  ``mask`` is one bool per leaf, True where the leaf is a Tensor;
    it is populated when ``wrapped`` runs, so callers must read it AFTER
    invoking ``wrapped`` and pass it to :func:`fill_none_with_masks` to
    reconstruct the full output with None at the dropped slots.
    """
    mask: list[bool] = []

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> list[Tensor]:
        leaves = _pytree_api().tree_leaves(fn(*args, **kwargs))
        mask.clear()
        mask.extend(isinstance(o, Tensor) for o in leaves)
        return filter_with_masks(leaves, mask)

    return wrapped, mask


def get_tensor_mask(tensor_list: Iterable[Any]) -> list[bool]:
    """One bool per element: whether it is a Tensor."""
    return [bool(isinstance(v, Tensor)) for v in tensor_list]


def mask_list(
    mask: Sequence[bool], inp: Sequence[Any], other: Sequence[Any] | None = None
) -> list[Any]:
    """Filter ``inp`` by ``mask``; with ``other``, replace instead of drop."""
    if len(mask) != len(inp):
        raise AssertionError(
            f"The length of the mask ({len(mask)}) needs to be identical to the "
            f"length of the input ({len(inp)})"
        )
    if other is not None:
        if len(inp) != len(other):
            raise AssertionError(
                f"If an input and an other list is provided, they need to have "
                f"the same length ({len(inp)} != {len(other)})"
            )
        return [i if m else o for m, i, o in zip(mask, inp, other)]
    return [i for m, i in zip(mask, inp) if m]


def first_slice_copy(t: Tensor, dim: int = 0) -> Tensor:
    """A copy of the first slice along ``dim`` (zeros when the dim is empty)."""
    if t.shape[dim] == 0:
        shape = list(t.shape)
        del shape[dim]
        return t.new_zeros(shape)
    return t.select(dim, 0).clone()


def unique_graph_id(proxy_mode: Any, prefix: str) -> tuple[int, str]:
    """A name and its index unused on the capture root, for attaching subgraphs."""
    root = getattr(proxy_mode.tracer, "root", None)
    next_name = None
    i = 0
    while not next_name:
        candidate = f"{prefix}_{i}"
        if root is not None and hasattr(root, candidate):
            i += 1
        else:
            next_name = candidate
    return i, next_name


def _from_fun(t: Any) -> Any:
    """A metadata stand-in for ``t`` that shares no state with the original.

    Used when a callable is traced into a subgraph: the capture only reads
    metadata, so an uninitialized allocation with matching shape, stride,
    dtype and device keeps the original values out of the traced body.
    """
    if isinstance(t, Tensor):
        if t.dtype != tensorplay.bool:
            stand_in = tensorplay.empty_strided(
                t.size(),
                t.stride(),
                dtype=t.dtype,
                device=t.device,
            )
            if t.requires_grad:
                stand_in.requires_grad_(True)
            return stand_in
        return t.clone()
    return t


def _unstack_pytree(xs: Any) -> list[Any]:
    """Split a pytree whose leaves share a leading dim into per-index pytrees."""
    pytree = _pytree_api()
    flat_xs, inspec = pytree.tree_flatten(xs)
    if not all(isinstance(x, Tensor) for x in flat_xs):
        raise RuntimeError(f"Leaves of xs must be Tensor {flat_xs}")

    if not all(x.shape[0] == flat_xs[0].shape[0] for x in flat_xs):
        raise RuntimeError(
            "Leaves of xs must have same leading dimension size "
            f"{[x.shape for x in flat_xs]}"
        )

    return [pytree.tree_unflatten(tuple, inspec) for tuple in zip(*flat_xs)]


def _stack_pytree(pytrees: Sequence[Any]) -> Any:
    """Stack per-index pytrees back into one pytree with a leading dim."""
    pytree = _pytree_api()
    flat_out = []
    out_spec = None
    for pt in pytrees:
        flat_pt, out_spec = pytree.tree_flatten(pt)
        flat_out.append(flat_pt)
    if out_spec is None:
        raise AssertionError("out_spec cannot be None")
    stacked_out = []
    for leaves in zip(*flat_out):
        if all(isinstance(leaf, Tensor) for leaf in leaves):
            stacked_out.append(tensorplay.stack(list(leaves)))
        elif all(leaf is None for leaf in leaves):
            # A backward body can return None where the forward input did
            # not require grad; keep the slot instead of failing the stack.
            stacked_out.append(None)
        else:
            raise RuntimeError(f"Cannot stack {leaves}.")
    return pytree.tree_unflatten(stacked_out, out_spec)


def _clone_aliasing_output(
    inputs: Sequence[Any], outputs: Sequence[Any]
) -> list[Any]:
    """Copy outputs that share storage with an input or a previous output.

    A gradient that aliases an operand or another gradient (a view returned
    by the differentiated body) would let later in-place accumulation on the
    caller's side corrupt values it does not own, so every such alias is
    broken with an explicit copy.
    """
    seen_ptrs = set()
    for t in inputs:
        if isinstance(t, Tensor):
            ptr = _tensor_data_ptr(t)
            if ptr is not None:
                seen_ptrs.add(ptr)
    final_outputs = []
    for out in outputs:
        if isinstance(out, Tensor):
            ptr = _tensor_data_ptr(out)
            if ptr is not None and ptr in seen_ptrs:
                out = out.clone()
                ptr = _tensor_data_ptr(out)
            if ptr is not None:
                seen_ptrs.add(ptr)
        final_outputs.append(out)
    return final_outputs


def check_meta_consistency(
    lhs_list: Sequence[Any],
    rhs_list: Sequence[Any],
    lhs_name: str,
    rhs_name: str,
) -> None:
    """Raise unless two flat value lists carry matching tensor metadata."""

    def _describe(t: Any) -> str:
        if isinstance(t, Tensor):
            return (
                f"shape {tuple(t.shape)}, dtype {t.dtype}, device {t.device}"
            )
        return repr(t)

    if len(lhs_list) != len(rhs_list):
        raise RuntimeError(
            f"Expected {lhs_name} and {rhs_name} to have the same number of "
            f"outputs but got lhs: {list(lhs_list)} and rhs: {list(rhs_list)}."
        )
    for i, (lhs, rhs) in enumerate(zip(lhs_list, rhs_list)):
        if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
            if (
                tuple(lhs.shape) != tuple(rhs.shape)
                or lhs.dtype != rhs.dtype
                or str(lhs.device) != str(rhs.device)
            ):
                raise RuntimeError(
                    f"Expected {lhs_name} and {rhs_name} to have the same "
                    f"metadata but pair[{i}] differ: {_describe(lhs)} vs "
                    f"{_describe(rhs)}."
                )
        elif isinstance(lhs, Tensor) != isinstance(rhs, Tensor):
            raise RuntimeError(
                f"Expected {lhs_name} and {rhs_name} to have the same types "
                f"but pair[{i}] differ: {_describe(lhs)} vs {_describe(rhs)}."
            )


def check_input_alias_and_mutation_return_outputs(
    gm: Any,
) -> tuple[dict[int, int], dict[int, int], dict[int, int], list[int], list[Any]]:
    """Structural alias/mutation report for a traced graph.

    Input mutation is detected by the in-place naming convention on graph
    calls (see :func:`_graph_mutated_inputs`); input-output aliasing is
    detected for placeholder outputs that pass through the graph unchanged.
    Output-output storage aliasing is not observable structurally in this
    build and stays empty.  Returns
    ``(inp_inp_alias, inp_out_alias, out_out_alias, mutated_inputs, outputs)``
    where ``outputs`` are the flattened output leaves.
    """
    graph = getattr(gm, "graph", None)
    if graph is None:
        return {}, {}, {}, [], []
    from tensorplay.graph.node import Node

    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    ph_index = {n.name: i for i, n in enumerate(placeholders)}
    mutated = _graph_mutated_inputs(gm, [])

    output_node = next((n for n in graph.nodes if n.op == "output"), None)
    outputs: list[Any] = []
    if output_node is not None:
        outputs = list(_pytree_api().tree_flatten(output_node.args[0])[0])
    inp_out_alias = {}
    for i, out in enumerate(outputs):
        if isinstance(out, Node) and out.op == "placeholder":
            inp_out_alias[ph_index[out.name]] = i
    return {}, inp_out_alias, {}, mutated, outputs


def check_input_alias_and_mutation(
    gm: Any, fake_args: Sequence[Any]
) -> tuple[dict[int, int], dict[int, int], dict[int, int], list[int]]:
    """Alias maps and mutated-input list for a traced graph."""
    inp_inp, inp_out, out_out, mutated, _ = check_input_alias_and_mutation_return_outputs(
        gm
    )
    return inp_inp, inp_out, out_out, mutated


def materialize_as_graph(
    fn: Callable,
    args: Sequence[Any],
    include_key_set: Any = None,
    exclude_key_set: Any = None,
    force_enable_grad: bool = False,
) -> Any:
    """Trace ``fn`` on metadata stand-ins into a standalone GraphModule.

    ``include_key_set`` / ``exclude_key_set`` are accepted for call-site
    compatibility; this build has no dispatch key sets, so the surrounding
    role state is irrelevant to the produced graph.  The callable must be a
    plain dataflow function: bodies that consult the autograd engine (for
    example a backward closure) cannot be captured and stay eager.
    """
    from tensorplay.graph.experimental.proxy_tensor import (
        disable_proxy_modes_tracing as _disable_tracing,
    )

    del include_key_set, exclude_key_set
    with suspend_functionalization(), disable_functional_mode():
        with _disable_tracing():
            stand_ins = [_from_fun(arg) for arg in args]
            if force_enable_grad:
                with tensorplay.enable_grad():
                    return _maybe_reenter_make_fx(fn)(*stand_ins)
            return _maybe_reenter_make_fx(fn)(*stand_ins)


def create_bw_fn(
    fn: Callable, args: Sequence[Any], return_fw_outputs: bool = False
) -> Callable:
    """Build the vector-Jacobian closure of ``fn`` over flat ``args``.

    For a function invoked as ``fw_out = fn(*args)``, the returned closure
    computes::

        grad_args = bw_fn(*args, *grad_out)

    with the following invariants:
      1. ``args`` plus the flattened ``fw_out`` leaves stand in 1-1
         correspondence with the closure's flat arguments.
      2. ``grad_args`` has 1-1 correspondence with ``args``.
      3. a Tensor argument whose gradient is not reachable (None) still
         receives a zero tensor of the argument's shape and dtype.
      4. gradients aliasing the closure's inputs or each other are copied.

    The closure recomputes the forward on detached leaf copies under
    enabled-grad, so it is safe to call from inside a backward and supports
    nested (higher-order) differentiation.
    """
    n_primals = len(args)
    pytree = _pytree_api()

    def _detach_leaf(t: Any) -> Any:
        if isinstance(t, Tensor) and (
            tensorplay.is_floating_point(t) or tensorplay.is_complex(t)
        ):
            leaf = t.detach()
            leaf.requires_grad_(True)
            return leaf
        return t

    def flat_fn(*args_and_grad_outs: Any) -> list[Any]:
        primals = args_and_grad_outs[:n_primals]
        tangents = list(args_and_grad_outs[n_primals:])
        if len(tangents) != len(primals) and not tangents:
            raise AssertionError(
                "backward closure requires at least one cotangent"
            )
        with tensorplay.enable_grad():
            leaves = tuple(_detach_leaf(p) for p in primals)
            fw_out = fn(*leaves)
            flat_out, _ = pytree.tree_flatten(fw_out)
            tang_it = iter(tangents)
            out_tensors: list[Tensor] = []
            out_cots: list[Tensor] = []
            for o in flat_out:
                if not isinstance(o, Tensor):
                    continue
                cot = next(tang_it, None)
                if cot is None:
                    cot = tensorplay.zeros_like(o)
                if o.requires_grad:
                    out_tensors.append(o)
                    out_cots.append(cot)
            diff_idx = [
                i for i, l in enumerate(leaves) if isinstance(l, Tensor) and l.requires_grad
            ]
            if out_tensors and diff_idx:
                raw = tensorplay.autograd.grad(
                    out_tensors,
                    [leaves[i] for i in diff_idx],
                    grad_outputs=out_cots,
                    allow_unused=True,
                )
            else:
                raw = [None] * len(diff_idx)
            by_pos = dict(zip(diff_idx, raw))
            grad_args: list[Any] = []
            for i, l in enumerate(leaves):
                grad = by_pos.get(i)
                if grad is None:
                    grad = tensorplay.zeros_like(l) if isinstance(l, Tensor) else None
                grad_args.append(grad)
        grad_args = _clone_aliasing_output(args_and_grad_outs, grad_args)
        if return_fw_outputs:
            return [*flat_out, *grad_args]
        return grad_args

    return flat_fn


def _resolve_real_sample(value: Any, tracer: Any = None) -> Any:
    """The concrete value flowing behind a graph placeholder proxy, or None.

    Under a graph capture a tensor argument reaches an operator as a proxy;
    the tracer records the concrete value that fed each placeholder, which
    is what subgraph tracing and example-output computation need.
    Intermediate (non-placeholder) proxies have no recorded sample.
    """
    from tensorplay.graph.node import Node
    from tensorplay.graph.proxy import Proxy

    if not isinstance(value, Proxy):
        return value
    owner = tracer if tracer is not None else value.tracer
    samples = getattr(owner, "_node_samples", None)
    if not samples:
        return None
    resolved = samples.get(value.node.name)
    if resolved is None or isinstance(resolved, (Proxy, Node)):
        return None
    return resolved


def _path_getitem(value: Any, path: tuple[int, ...]) -> Any:
    """Index ``value`` through a fixed path of positions."""
    for idx in path:
        value = value[idx]
    return value


def _bind_example_output(example: Any, out_proxy: Any, tracer: Any) -> Any:
    """Bind an example result to a freshly created operator node's proxy.

    Returns the value the surrounding capture should keep flowing: the node
    proxy itself for scalar results, or a matching tree of per-element
    accessor nodes for sequence results, so downstream unpacking and element
    use continue symbolically inside the outer graph.  The example values
    are attached as node metadata.
    """
    from tensorplay.graph.experimental.proxy_tensor import track_tensor

    if isinstance(example, (tuple, list)):
        elems = []
        for i, item in enumerate(example):
            item_proxy = tracer.create_proxy(
                "call_function", _path_getitem, (out_proxy, (i,)), {}
            )
            if isinstance(item, (tuple, list)):
                elems.append(_bind_example_output(item, item_proxy, tracer))
            else:
                if isinstance(item, Tensor):
                    track_tensor(item, item_proxy, constant=None, tracer=tracer)
                elems.append(item_proxy)
        return tuple(elems) if isinstance(example, tuple) else elems
    if isinstance(example, Tensor):
        track_tensor(example, out_proxy, constant=None, tracer=tracer)
    return out_proxy


class IdentityFunctionalizeCtx:
    """Default functionalization context.

    Tensor wrappers do not exist in this build, so unwrapping and wrapping are
    identities and redispatch continues at the same operator.
    """

    mode = None

    def unwrap_tensors(self, x: Any) -> Any:
        return x

    def wrap_tensors(self, x: Any) -> Any:
        return x

    def redispatch_to_next(self):
        import contextlib

        return contextlib.nullcontext()

    def functionalize(self, fn: Callable) -> Callable:
        return fn

