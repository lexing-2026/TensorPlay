"""Structural conditional operator.

``cond`` selects between two branch functions under a predicate tensor.
The operator is registry-based: the composite eager registration picks a
branch and runs it, the autograd registration routes through a
:class:`CondAutogradOp` formula whose backward is itself a ``cond`` over
per-branch vector-Jacobian closures, and a graph capture records the call
as one opaque node carrying both traced branch subgraphs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import tensorplay
from tensorplay import Tensor
from tensorplay._higher_order_ops._hop_base import (
    HigherOrderOperator,
    _AutoDispatchBelowAutograd,
    register_fake,
)
from tensorplay._higher_order_ops.utils import (
    UnsupportedAliasMutationException,
    _has_potential_branch_input_mutation,
    _maybe_run_with_interpreter,
    _resolve_real_sample,
    check_meta_consistency,
    create_bw_fn,
    create_fn_remove_none,
    fill_none_with_masks,
    reenter_make_fx,
    save_values_for_backward,
    saved_values,
    unique_graph_id,
    validate_subgraph_args_types,
)
from tensorplay.graph.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    get_proxy_mode,
    track_tensor_tree,
    unwrap_proxy,
)


def _is_capture_value(value: Any) -> bool:
    """Whether ``value`` is a symbolic argument of an active graph capture."""
    from tensorplay.graph.node import Node
    from tensorplay.graph.proxy import Proxy

    return isinstance(value, (Proxy, Node))


def _any_requires_grad(values: Any) -> bool:
    return any(isinstance(t, Tensor) and t.requires_grad for t in values)


class CondOp(HigherOrderOperator):
    """``cond(pred, true_fn, false_fn, operands)`` as a registered operator."""

    def __init__(self) -> None:
        super().__init__("cond")

    def __call__(
        self,
        pred: Any,
        true_fn: Callable,
        false_fn: Callable,
        operands: tuple[Any, ...],
    ) -> Any:
        if get_proxy_mode() is None:
            # Under a capture the operands arrive as symbolic proxies; the
            # concrete types are checked by the eager entry point instead.
            validate_subgraph_args_types(operands)
        return super().__call__(pred, true_fn, false_fn, operands)


cond_op = CondOp()


def cond(
    pred: bool | int | float | Tensor,
    true_fn: Callable,
    false_fn: Callable,
    operands: tuple | list = (),
) -> Any:
    """Conditionally applies ``true_fn`` or ``false_fn``.

    ``cond`` is a structural control flow operator: like a Python
    if-statement, but with restrictions on the branches and operands that
    make the choice capturable as a single graph node.

    Assuming the constraints on the arguments are met, ``cond`` is
    equivalent to the following::

        def cond(pred, true_fn, false_fn, operands):
            if pred:
                return true_fn(*operands)
            else:
                return false_fn(*operands)

    Args:
        pred (bool, int, float or Tensor): A boolean expression, or a
          single-element boolean ``Tensor`` selecting the branch.
        true_fn (Callable): The branch applied when the predicate holds.
        false_fn (Callable): The branch applied otherwise.  Both branches
          must take ``operands`` and return values of the same tree
          structure, with matching tensor dtypes, devices and shapes.
        operands (tuple of Tensors): Inputs handed to the selected branch.
          Defaults to ``()``.

    Example::

        def true_fn(x):
            return x.cos()

        def false_fn(x):
            return x.sin()

        result = cond(x.sum() > 0, true_fn, false_fn, (x,))

    Restrictions:
        - The predicate must be a Python boolean or a single-element
          boolean ``Tensor``.
        - A Python-constant predicate runs (or traces) only the selected
          branch; pass a tensor predicate to keep both branches alive in a
          captured graph.
        - The branch functions must not mutate their operands; intermediate
          in-place results inside a branch are fine.
    """
    if not callable(true_fn) or not callable(false_fn):
        raise RuntimeError("Expect both branches to be callable.")
    if not isinstance(operands, (tuple, list)):
        raise RuntimeError(
            "Expect operands to be a tuple of tensors, but got "
            f"{type(operands)}."
        )
    if isinstance(pred, (bool, int, float)):
        # Constant predicate: only the selected branch is evaluated (and,
        # under a capture, only that branch is inlined).
        return true_fn(*operands) if pred else false_fn(*operands)

    _validate_input(pred, operands)

    return cond_op(pred, true_fn, false_fn, tuple(operands))


def _validate_input(pred: Any, operands: Any) -> None:
    if not isinstance(pred, Tensor) and not _is_capture_value(pred):
        raise RuntimeError(f"Expected pred to be bool or tensor, but got {pred}.")
    if isinstance(pred, Tensor):
        if pred.dtype != tensorplay.bool:
            raise RuntimeError(
                f"Expected pred to be a boolean tensor, but got dtype {pred.dtype}."
            )
        if pred.numel() != 1:
            raise RuntimeError(
                f"Expected pred to be bool or single-element tensor, but got {pred}."
            )
    if get_proxy_mode() is None:
        pytree = tensorplay.utils._pytree
        if pytree.tree_any(lambda t: not isinstance(t, Tensor), operands):
            raise RuntimeError(
                "Expect operands to be a tuple of tensors, but got "
                f"{operands}."
            )


@cond_op.py_impl("CompositeExplicitAutograd")
def cond_op_dense(pred: Any, true_fn: Callable, false_fn: Callable, operands: tuple):
    if not isinstance(operands, (tuple, list)):
        raise AssertionError(
            f"Cond operands must be a list or tuple of tensors and ints {operands}"
        )
    for operand in operands:
        if not isinstance(operand, (Tensor, int)) and not _is_capture_value(operand):
            raise AssertionError(
                f"Dense implementation operands must be a list of tensors and ints {operands}"
            )
    if isinstance(pred, Tensor):
        if pred.numel() != 1:
            raise RuntimeError(
                f"Expected pred to be a single-element tensor, but got {pred}."
            )
        take_true = bool(pred.item())
    else:
        take_true = bool(pred)
    if take_true:
        return true_fn(*operands)
    return false_fn(*operands)


class CondAutogradOp(tensorplay.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        pred: Any,
        true_fn: Callable,
        false_fn: Callable,
        *operands: Tensor,
    ) -> Any:
        ctx._pred = pred
        ctx._true_bw_fn = create_bw_fn(true_fn, operands)
        ctx._false_bw_fn = create_bw_fn(false_fn, operands)
        save_values_for_backward(ctx, operands)
        with _AutoDispatchBelowAutograd():
            return cond_op(pred, true_fn, false_fn, operands)

    @staticmethod
    def backward(ctx: Any, *flat_grads: Any) -> tuple[Any, ...]:
        operands = saved_values(ctx)
        args = tuple(operands) + tuple(flat_grads)
        # Each wrapper drops the non-Tensor leaves of its branch's gradient
        # tree; the mask of whichever branch the predicate selects is the
        # one populated at run time, and both share the output signature.
        true_wrapped, true_mask = create_fn_remove_none(ctx._true_bw_fn)
        false_wrapped, false_mask = create_fn_remove_none(ctx._false_bw_fn)
        grads = cond_op(ctx._pred, true_wrapped, false_wrapped, args)
        mask = true_mask if len(true_mask) else false_mask
        return (None, None, None, *fill_none_with_masks(grads, mask))


@cond_op.py_autograd_impl
def cond_autograd(pred: Any, true_fn: Callable, false_fn: Callable, operands: tuple):
    if not tensorplay.is_grad_enabled() or not _any_requires_grad(operands):
        # Nothing to differentiate: run the composite selection directly.
        return cond_op_dense(pred, true_fn, false_fn, operands)
    return CondAutogradOp.apply(pred, true_fn, false_fn, *operands)


def trace_cond(
    proxy_mode: Any,
    func_overload: Callable,
    pred: Any,
    true_fn: Callable,
    false_fn: Callable,
    operands: tuple,
) -> Any:
    """Capture a ``cond`` call as one node carrying both branch subgraphs."""
    if not isinstance(operands, (list, tuple)):
        raise AssertionError(
            f"Cond operands must be a list or tuple of tensors and ints {operands}"
        )

    tracer = proxy_mode.tracer
    real_pred = _resolve_real_sample(pred, tracer)
    real_operands = tuple(
        _resolve_real_sample(o, tracer) for o in operands
    )
    # Concrete samples let the branch bodies trace against executable
    # metadata and give the node a real example output; unresolvable
    # intermediate proxies still trace (the sub-tracer hands the body fresh
    # placeholders) but skip the example.
    have_samples = real_pred is not None and not any(
        _is_capture_value(orig) and resolved is None
        for orig, resolved in zip(operands, real_operands)
    )

    with disable_proxy_modes_tracing():
        trace_operands = (
            real_operands if all(o is not None for o in real_operands) else tuple(operands)
        )
        true_graph = reenter_make_fx(true_fn)(*trace_operands)
        false_graph = reenter_make_fx(false_fn)(*trace_operands)

    flat_true_outs = _flat_graph_outputs(true_graph)
    flat_false_outs = _flat_graph_outputs(false_graph)
    if len(flat_true_outs) != len(flat_false_outs):
        raise RuntimeError(
            "Expected cond branches to return the same number of outputs "
            f"but got: true branch returns {len(flat_true_outs)} item(s), "
            f"false branch returns {len(flat_false_outs)} item(s)."
        )

    i, true_name = unique_graph_id(proxy_mode, prefix="true_graph")
    false_name = f"false_graph_{i}"
    true_graph.meta["hop_graph_name"] = true_name
    false_graph.meta["hop_graph_name"] = false_name

    args = (pred, true_graph, false_graph, tuple(operands))
    proxy_args = unwrap_proxy(args)
    out_proxy = tracer.create_proxy("call_function", func_overload, proxy_args, {})

    if have_samples:
        take_true = (
            bool(real_pred.item()) if isinstance(real_pred, Tensor) else bool(real_pred)
        )
        branch = true_fn if take_true else false_fn
        with disable_proxy_modes_tracing():
            example = branch(*real_operands)
        if isinstance(example, Tensor):
            return track_tensor_tree(
                example, out_proxy, constant=None, tracer=proxy_mode.tracer
            )
    return out_proxy


def _flat_graph_outputs(graph: Any) -> list[Any]:
    """The flattened output leaves recorded by a traced subgraph."""
    pytree = tensorplay.utils._pytree
    from tensorplay.graph.node import Node

    output_node = next((n for n in graph.graph.nodes if n.op == "output"), None)
    if output_node is None:
        raise AssertionError("no output node found in traced subgraph")
    return [
        leaf
        for leaf in pytree.tree_flatten(output_node.args[0])[0]
        if isinstance(leaf, (Node, Tensor))
    ]


@cond_op.py_impl("ProxyDispatchMode")
def _cond_proxy_mode(
    mode: Any, pred: Any, true_fn: Callable, false_fn: Callable, operands: tuple
) -> Any:
    if mode is None:
        raise AssertionError("Mode should always be enabled for python fallback key")
    return trace_cond(mode, cond_op, pred, true_fn, false_fn, operands)


@cond_op.py_functionalize_impl
def cond_func(
    ctx: Any, pred: Any, true_fn: Callable, false_fn: Callable, inputs: tuple
) -> Any:
    """Functionalization rule: unwrap, check for input mutation, redispatch."""
    unwrapped_inputs = ctx.unwrap_tensors(inputs)
    unwrapped_pred = ctx.unwrap_tensors(pred)
    with ctx.redispatch_to_next():
        functional_true = ctx.functionalize(_maybe_run_with_interpreter(true_fn))
        functional_false = ctx.functionalize(_maybe_run_with_interpreter(false_fn))
        pre_dispatch = getattr(ctx, "mode", None) is not None and ctx.mode.pre_dispatch
        for branch, branch_name in [(true_fn, "cond_true"), (false_fn, "cond_false")]:
            mutates = _has_potential_branch_input_mutation(
                branch, unwrapped_inputs, pre_dispatch
            )
            if mutates:
                raise UnsupportedAliasMutationException(
                    f"Mutations detected in {branch_name}"
                )
        cond_return = cond_op(
            unwrapped_pred, functional_true, functional_false, unwrapped_inputs
        )
        return ctx.wrap_tensors(cond_return)


@register_fake(cond_op)
def cond_fake(
    pred: Any, true_fn: Callable, false_fn: Callable, operands: tuple
) -> Any:
    """Abstract evaluation: both branches run and their outputs must agree."""
    pytree = tensorplay.utils._pytree
    flat_true_outs, true_spec = pytree.tree_flatten(true_fn(*operands))
    flat_false_outs, false_spec = pytree.tree_flatten(false_fn(*operands))
    if str(true_spec) != str(false_spec):
        raise RuntimeError(
            "Unmatched output spec from cond branches: true branch tree_spec "
            f"{true_spec} vs false branch tree_spec {false_spec}."
        )
    check_meta_consistency(flat_true_outs, flat_false_outs, "true branch", "false branch")
    return pytree.tree_unflatten(flat_true_outs, true_spec)
