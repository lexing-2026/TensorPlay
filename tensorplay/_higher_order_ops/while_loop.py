"""Structural while-loop operator.

``while_loop`` runs ``body_fn(*carried_inputs)`` while ``cond_fn`` returns a
true scalar, preserving the loop as one operator under a graph capture.
The eager composite registration is the Python loop itself; the autograd
formula runs a stack-output variant forward (one stacked buffer per carry)
and derives the backward as a reversed loop over the recorded per-iteration
inputs, so the backward graph does not grow with the iteration count.
"""

from __future__ import annotations

import functools
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
    check_meta_consistency,
    create_bw_fn,
    filter_with_masks,
    fill_none_with_masks,
    autograd_not_implemented,
    reenter_make_fx,
    split_into_chunks,
    unique_graph_id,
    validate_subgraph_args_types,
    _resolve_real_sample,
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


class WhileLoopOp(HigherOrderOperator):
    """``while_loop(cond_fn, body_fn, carried_inputs)`` as a registered operator."""

    def __init__(self) -> None:
        super().__init__("while_loop")

    def __call__(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        carried_inputs: tuple[Any, ...],
        additional_inputs: tuple[Any, ...],
    ) -> Any:
        if not isinstance(carried_inputs, (tuple, list)):
            raise RuntimeError(
                f"carried_inputs must be a tuple or list, got {type(carried_inputs)}"
            )
        if not isinstance(additional_inputs, (tuple, list)):
            raise RuntimeError(
                f"additional_inputs must be a tuple or list, got {type(additional_inputs)}"
            )
        if get_proxy_mode() is None:
            validate_subgraph_args_types(carried_inputs)
            validate_subgraph_args_types(additional_inputs)
        return super().__call__(cond_fn, body_fn, carried_inputs, additional_inputs)


while_loop_op = WhileLoopOp()


class WhileLoopStackOutputOp(HigherOrderOperator):
    """``while_loop`` variant whose forward returns every iteration stacked.

    Its semantics are::

        def while_loop_stack_output(cond_fn, body_fn, carried_inputs, additional):
            outs = []
            while cond_fn(*carried_inputs, *additional):
                out = body_fn(*carried_inputs, *additional)
                outs.append(out)
            return torch.stack(outs)  # per carry, along dim 0

    The plain operator's autograd formula runs this variant to record the
    per-iteration values its backward needs.
    """

    def __init__(self) -> None:
        super().__init__("while_loop_stack_output")

    def __call__(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        carried_inputs: tuple[Any, ...],
        additional_inputs: tuple[Any, ...],
    ) -> Any:
        if not isinstance(carried_inputs, (tuple, list)):
            raise RuntimeError(
                f"carried_inputs must be a tuple or list, got {type(carried_inputs)}"
            )
        if not isinstance(additional_inputs, (tuple, list)):
            raise RuntimeError(
                f"additional_inputs must be a tuple or list, got {type(additional_inputs)}"
            )
        if get_proxy_mode() is None:
            validate_subgraph_args_types(carried_inputs)
            validate_subgraph_args_types(additional_inputs)
        return super().__call__(cond_fn, body_fn, carried_inputs, additional_inputs)


while_loop_stack_output_op = WhileLoopStackOutputOp()


def while_loop(
    cond_fn: Callable,
    body_fn: Callable,
    carried_inputs: Any,
) -> Any:
    """Run ``body_fn(*carried_inputs)`` while ``cond_fn`` stays true.

    ``while_loop`` is a structural control flow operator: like a Python
    while-statement, but with restrictions on the callables and carries that
    keep the loop capturable as a single graph node.

    It is equivalent to the following::

        def while_loop(cond_fn, body_fn, carried_inputs):
            val = carried_inputs
            while cond_fn(*val):
                val = body_fn(*val)
            return val

    Args:
        cond_fn (Callable): Returns a boolean scalar ``Tensor`` or a Python
          boolean; decides whether to keep looping.
        body_fn (Callable): Takes the carries and returns a tuple of the same
          length, with matching tensor metadata.
        carried_inputs (tuple of Tensors or ints): Initial loop state.  An
          int carry must stay an int; the loop count is not statically known,
          so an int output of a captured loop is a fresh unknown value.

    Example::

        def cond_fn(it, x):
            return it.sum() < 10

        def body_fn(it, x):
            return it + 1, x.sin()

        while_loop(cond_fn, body_fn, (tensorplay.zeros(1), tensorplay.randn(3, 4)))

    Restrictions:
        - ``body_fn`` must return the same number of values with the same
          metadata as the carries.
        - Neither callable may mutate the carries in place or alias them in
          its outputs; clone before returning instead.
        - Captured external tensors become additional inputs of the operator.
    """
    if not callable(cond_fn) or not callable(body_fn):
        raise RuntimeError("Expect cond_fn and body_fn to be callable.")
    # A flat tuple of tensors and ints passes the callables straight
    # through: the operator invokes them positionally, and fixed signatures
    # stay traceable (varargs wrappers are not capturable by the frontend).
    if isinstance(carried_inputs, (tuple, list)) and (
        get_proxy_mode() is not None
        or all(isinstance(t, (Tensor, int)) for t in carried_inputs)
    ):
        return while_loop_op(
            cond_fn, body_fn, tuple(carried_inputs), ()
        )

    pytree = tensorplay.utils._pytree
    flat_inputs, in_spec = pytree.tree_flatten((carried_inputs, ()))
    if not all(isinstance(t, (Tensor, int)) for t in flat_inputs):
        raise RuntimeError(
            "Expect carried_inputs to be a tuple of possibly nested "
            "dict/list/tuple that only consists of tensor or int leaves, "
            f"but got {carried_inputs}."
        )

    def flat_cond_fn(*flat_args: Any) -> Any:
        carried, additional = pytree.tree_unflatten(flat_args, in_spec)
        return cond_fn(*carried, *additional)

    def flat_body_fn(*flat_args: Any) -> Any:
        carried, additional = pytree.tree_unflatten(flat_args, in_spec)
        return body_fn(*carried, *additional)

    return while_loop_op(flat_cond_fn, flat_body_fn, tuple(flat_inputs), ())


def _validate_cond_output(pred: Any) -> None:
    if isinstance(pred, Tensor):
        if pred.dim() != 0 or pred.dtype != tensorplay.bool:
            raise RuntimeError(
                f"cond_fn must return a boolean scalar tensor or a boolean but got {pred}"
            )
    elif not isinstance(pred, bool):
        raise RuntimeError(
            f"cond_fn must return a boolean scalar tensor or a boolean but got {pred}"
        )


def while_loop_dense(
    cond_fn: Callable,
    body_fn: Callable,
    carried_inputs: tuple[Any, ...],
    additional_inputs: tuple[Any, ...],
    stack_output: bool = False,
) -> Any:
    """Eager loop; with ``stack_output`` every iteration is also stacked."""
    if not isinstance(carried_inputs, (tuple, list)):
        raise RuntimeError(
            f"carried_inputs must be a tuple or list but got {type(carried_inputs)}"
        )

    carried_vals = tuple(carried_inputs)
    should_loop = cond_fn(*carried_vals, *additional_inputs)
    _validate_cond_output(should_loop)

    if not should_loop:
        if stack_output:
            return tuple(
                val.unsqueeze(0).clone() if isinstance(val, Tensor) else val
                for val in carried_vals
            )
        return tuple(
            val.clone() if isinstance(val, Tensor) else val for val in carried_vals
        )

    outputs: list[list[Any]] = [[] for _ in carried_vals]
    while should_loop:
        out = body_fn(*carried_vals, *additional_inputs)
        if not isinstance(out, tuple):
            raise AssertionError(f"body_fn should return a tuple but got {type(out)}")
        if len(out) != len(carried_inputs):
            raise AssertionError(
                f"body_fn should return the same number of elements as "
                f"carried_inputs, got {len(out)} vs {len(carried_inputs)}"
            )
        if stack_output:
            for i, o in enumerate(out):
                outputs[i].append(o)
        carried_vals = out
        should_loop = cond_fn(*carried_vals, *additional_inputs)
        _validate_cond_output(should_loop)

    if stack_output:
        stacked = []
        for i, out in enumerate(outputs):
            if out and all(isinstance(o, Tensor) for o in out):
                stacked.append(tensorplay.stack(out, dim=0))
            else:
                # A non-tensor carry cannot be stacked; its last value stands
                # in for every iteration (it never joins a gradient path).
                stacked.append(out[-1] if out else carried_vals[i])
        return tuple(stacked)
    return carried_vals


# The backward of a while-loop is a reversed loop over the recorded forward:
#
#     gx = gy_last * bw(y_last, y_prev) * ... * bw(y_1, y_0) * bw(y_0, x)
#
# where ``bw(y_i, y_{i-1})`` is the single-step vector-Jacobian of the body.
# The implementation carries (grad_of_state, accumulated_grad_of_inputs)
# backwards and sums the per-step input gradients, mirroring the chain above.
class WhileLoopAutogradOp(tensorplay.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        cond_fn: Callable,
        body_fn: Callable,
        num_carried_inputs: int,
        num_additional_inputs: int,
        *carries_and_inputs: Any,
    ) -> Any:
        carries, additional_inputs = split_into_chunks(
            carries_and_inputs, [num_carried_inputs, num_additional_inputs]
        )
        with _AutoDispatchBelowAutograd():
            fw_outputs = while_loop_stack_output_op(
                cond_fn, body_fn, carries, additional_inputs
            )
        ctx.fw_cond_fn = cond_fn
        ctx.fw_body_fn = body_fn
        ctx.carries = tuple(carries)
        ctx.additional_inputs = tuple(additional_inputs)
        ctx.fw_outputs = fw_outputs
        return tuple(ckp[-1] if isinstance(ckp, Tensor) else ckp for ckp in fw_outputs)

    @staticmethod
    def backward(ctx: Any, *grads: Any) -> tuple[Any, ...]:
        bw_body_fn = create_bw_fn(ctx.fw_body_fn, tuple(ctx.carries) + tuple(ctx.additional_inputs))

        carries_tensor_masks = [
            bool(isinstance(t, Tensor) and tensorplay.is_floating_point(t))
            for t in ctx.carries
        ]
        additional_inputs_tensor_masks = [
            bool(isinstance(t, Tensor) and tensorplay.is_floating_point(t))
            for t in ctx.additional_inputs
        ]

        init_idx = tensorplay.zeros((), dtype=tensorplay.int64)
        init_grad_carries = tuple(
            grad.clone() if isinstance(grad, Tensor) else grad
            for grad in filter_with_masks(grads, carries_tensor_masks)
        )
        init_grad_additional_inputs = tuple(
            tensorplay.zeros_like(t)
            for need_keep, t in zip(additional_inputs_tensor_masks, ctx.additional_inputs)
            if need_keep
        )

        # The input each iteration i saw: the initial carry for i == 0 and
        # iteration i-1's output otherwise. Only tensor carries stack;
        # non-tensor keeps its last value for every iteration.
        fw_carries = [
            (
                tensorplay.cat([carry.unsqueeze(0), carries[:-1]])
                if isinstance(carry, Tensor) and isinstance(carries, Tensor)
                else carries
            )
            for carry, carries in zip(ctx.carries, ctx.fw_outputs)
        ]
        for fw_carry, carry in zip(fw_carries, ctx.carries):
            if isinstance(fw_carry, Tensor):
                fw_carry.requires_grad_(carry.requires_grad)

        # Loop count comes from any stacked tensor carry; without one there
        # is no differentiable path and the loop degenerates to zero grads.
        loop_ref = next(
            (c for c in fw_carries if isinstance(c, Tensor)), None
        )
        num_carry = len(ctx.carries)
        num_additional = len(ctx.additional_inputs)

        def cond_fn(idx: Any, *args: Any) -> Any:
            return idx < loop_ref.size(0)

        def body_fn(*args: Any) -> Any:
            idx = args[0]
            grad_carries = args[1 : 1 + num_carry]
            grad_additional_inputs = args[1 + num_carry : 1 + num_carry + num_additional]
            reversed_idx = loop_ref.size(0) - idx - 1
            selected_fw_carries = [
                (
                    ckp.select(0, int(reversed_idx.item()))
                    if isinstance(ckp, Tensor)
                    else ckp
                )
                for ckp in fw_carries
            ]
            step_out = bw_body_fn(
                *selected_fw_carries, *ctx.additional_inputs, *grad_carries
            )
            cur_grad_carries, cur_grad_additional = split_into_chunks(
                step_out, [num_carry, num_additional]
            )
            cur_grad_carries = filter_with_masks(cur_grad_carries, carries_tensor_masks)
            cur_grad_additional = filter_with_masks(
                cur_grad_additional, additional_inputs_tensor_masks
            )
            return (
                idx + 1,
                *cur_grad_carries,
                *(
                    cur_grad + grad
                    for cur_grad, grad in zip(
                        cur_grad_additional, grad_additional_inputs
                    )
                ),
            )

        n_grad_carries = sum(1 for m in carries_tensor_masks if m)
        _, final_grad_carries, final_grad_additional = split_into_chunks(
            while_loop_op(
                cond_fn,
                body_fn,
                (init_idx, *init_grad_carries, *init_grad_additional_inputs),
                (),
            ),
            [1, n_grad_carries, num_additional],
        )
        return (
            None,
            None,
            None,
            None,
            *fill_none_with_masks(final_grad_carries, carries_tensor_masks),
            *fill_none_with_masks(
                final_grad_additional, additional_inputs_tensor_masks
            ),
        )


@while_loop_op.py_autograd_impl
def while_loop_autograd(
    cond_fn: Callable,
    body_fn: Callable,
    carried_inputs: tuple,
    additional_inputs: tuple,
) -> Any:
    if not tensorplay.is_grad_enabled() or not (
        _any_requires_grad(carried_inputs) or _any_requires_grad(additional_inputs)
    ):
        return while_loop_dense(cond_fn, body_fn, carried_inputs, additional_inputs)
    return WhileLoopAutogradOp.apply(
        cond_fn,
        body_fn,
        len(carried_inputs),
        len(additional_inputs),
        *carried_inputs,
        *additional_inputs,
    )


def trace_while_loop(
    proxy_mode: Any,
    op: Any,
    cond_fn: Callable,
    body_fn: Callable,
    carried_inputs: tuple,
    additional_inputs: tuple,
    stack_output: bool = False,
) -> Any:
    """Capture a ``while_loop`` call as one node with both subgraphs attached."""
    tracer = proxy_mode.tracer
    real_carried = tuple(_resolve_real_sample(c, tracer) for c in carried_inputs)
    real_additional = tuple(
        _resolve_real_sample(a, tracer) for a in additional_inputs
    )
    have_samples = not any(
        _is_capture_value(orig) and real is None
        for orig, real in zip(carried_inputs, real_carried)
    )

    with disable_proxy_modes_tracing():
        trace_carried = (
            real_carried if all(c is not None for c in real_carried)
            else tuple(carried_inputs)
        )
        cond_graph = reenter_make_fx(cond_fn)(*trace_carried, *additional_inputs)
        body_graph = reenter_make_fx(body_fn)(*trace_carried, *additional_inputs)

    i, cond_name = unique_graph_id(proxy_mode, prefix="while_loop_cond_graph")
    body_name = f"while_loop_body_graph_{i}"
    cond_graph.meta["hop_graph_name"] = cond_name
    body_graph.meta["hop_graph_name"] = body_name

    args = (cond_graph, body_graph, carried_inputs, additional_inputs)
    proxy_args = unwrap_proxy(args)
    out_proxy = tracer.create_proxy("call_function", op, proxy_args, {})

    if have_samples:
        with disable_proxy_modes_tracing():
            example = while_loop_dense(
                cond_fn, body_fn, real_carried, real_additional,
                stack_output=stack_output,
            )
        if isinstance(example, Tensor):
            return track_tensor_tree(
                example, out_proxy, constant=None, tracer=proxy_mode.tracer
            )
    return out_proxy


@while_loop_op.py_impl("ProxyDispatchMode")
def _while_loop_proxy_mode(
    mode: Any,
    cond_fn: Callable,
    body_fn: Callable,
    carried_inputs: tuple,
    additional_inputs: tuple,
) -> Any:
    return trace_while_loop(
        mode, while_loop_op, cond_fn, body_fn, carried_inputs, additional_inputs
    )


@while_loop_op.py_functionalize_impl
def while_loop_func(
    ctx: Any,
    cond_fn: Callable,
    body_fn: Callable,
    carried_inputs: tuple,
    additional_inputs: tuple,
) -> Any:
    """Functionalization rule: unwrap, reject input mutation, redispatch."""
    from tensorplay._higher_order_ops.utils import (
        UnsupportedAliasMutationException,
        _has_potential_branch_input_mutation,
    )

    unwrapped_carried = ctx.unwrap_tensors(carried_inputs)
    unwrapped_additional = ctx.unwrap_tensors(additional_inputs)
    unwrapped_inputs = tuple(unwrapped_carried) + tuple(unwrapped_additional)
    with ctx.redispatch_to_next():
        functional_cond_fn = ctx.functionalize(cond_fn)
        functional_body_fn = ctx.functionalize(body_fn)
        for fn, fn_name in [(cond_fn, "cond_fn"), (body_fn, "body_fn")]:
            if _has_potential_branch_input_mutation(fn, unwrapped_inputs):
                raise UnsupportedAliasMutationException(
                    f"Mutations detected in {fn_name}"
                )
        ret = while_loop_op(
            functional_cond_fn,
            functional_body_fn,
            unwrapped_carried,
            unwrapped_additional,
        )
        return ctx.wrap_tensors(ret)


@while_loop_op.py_impl("ProxyDispatchMode")
def _while_loop_stack_proxy_mode(
    mode: Any,
    cond_fn: Callable,
    body_fn: Callable,
    carried_inputs: tuple,
    additional_inputs: tuple,
) -> Any:
    return trace_while_loop(
        mode,
        while_loop_stack_output_op,
        cond_fn,
        body_fn,
        carried_inputs,
        additional_inputs,
        stack_output=True,
    )


# ---------------------------------------------------------------------------
# Stack-output variant registrations
# ---------------------------------------------------------------------------

while_loop_stack_output_op.py_impl("CompositeExplicitAutograd")(
    functools.partial(while_loop_dense, stack_output=True)
)

while_loop_stack_output_op.py_autograd_impl(
    autograd_not_implemented(while_loop_stack_output_op, deferred_error=True)
)

while_loop_stack_output_op.py_functionalize_impl(
    functools.partial(while_loop_func)
)


@register_fake(while_loop_op)
def while_loop_fake(
    mode: Any,
    cond_fn: Callable,
    body_fn: Callable,
    carried_inputs: tuple,
    additional_inputs: tuple,
) -> Any:
    """Abstract evaluation: one body step must match the carry metadata."""
    body_outs = body_fn(*carried_inputs, *additional_inputs)
    if not isinstance(body_outs, tuple):
        raise AssertionError(
            f"body_fn should return a tuple but got {type(body_outs)}"
        )
    if len(body_outs) != len(carried_inputs):
        raise AssertionError(
            f"body_fn should return the same number of elements as "
            f"carried_inputs, got {len(body_outs)} vs {len(carried_inputs)}"
        )
    check_meta_consistency(
        carried_inputs, body_outs, "carried_inputs", "body_output"
    )
    return body_outs
