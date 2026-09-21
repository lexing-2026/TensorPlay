"""Structural map operator.

``map`` applies a callable to every slice of the leading dimension of its
inputs and stacks the results.  Iterations are independent: each one sees a
storage-disjoint slice, so in-place writes inside the body are race-free.
The autograd formula maps the vector-Jacobian closure of the body over the
same leading dimension, keeping the backward graph one body wide.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import tensorplay
from tensorplay import Tensor
from tensorplay._higher_order_ops._hop_base import (
    HigherOrderOperator,
    _AutoDispatchBelowAutograd,
    register_fake,
)
from tensorplay._higher_order_ops.utils import (
    _resolve_real_sample,
    _stack_pytree,
    _unstack_pytree,
    create_bw_fn,
    filter_with_masks,
    fill_none_with_masks,
    first_slice_copy,
    reenter_make_fx,
    save_values_for_backward,
    saved_values,
    split_into_chunks,
    unique_graph_id,
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


class MapImpl(HigherOrderOperator):
    """``map_impl(f, xs, pos_args)`` as a registered operator."""

    def __init__(self) -> None:
        super().__init__("map_impl")

    def __call__(
        self,
        f: Callable[..., Any],
        xs: Sequence[Any],
        pos_args: Sequence[Any],
    ) -> Any:
        if not isinstance(xs, (tuple, list)):
            raise RuntimeError(f"xs must be a tuple or list, got {type(xs)}")
        if not isinstance(pos_args, (tuple, list)):
            raise RuntimeError(
                f"pos_args must be a tuple or list, got {type(pos_args)}"
            )
        return super().__call__(f, xs, pos_args)


map_impl = MapImpl()


def map(
    f: Callable[..., Any],
    xs: Any,
    *args: Any,
) -> Any:
    """Apply ``f`` to each slice along dim 0 of ``xs`` and stack the results.

    Intuitively, the semantics are::

        out = []
        for idx in range(xs.size(0)):
            out.append(f(xs.select(0, idx), *args))
        return torch.stack(out)

    Args:
        f (Callable): Takes one slice of each mapped input plus the additional
          arguments and returns a pytree.
        xs (Tensor or pytree of Tensors): The inputs mapped over; every leaf
          must agree on the size of its leading dimension.
        *args: Additional arguments handed to every step unchanged.  They
          are shared between iterations, so mutating them makes iterations
          depend on each other; use :func:`scan` or :func:`while_loop` when
          sequential iteration is the contract.

    Returns:
        The stacked output of every step, with the loop dimension prepended.

    Example::

        def f(xs):
            return xs[0] + xs[1] + const

        const = tensorplay.randn(3)
        xs = [tensorplay.randn(2, 3), tensorplay.randn(2, 3)]
        out = map(f, xs)  # shape [2, 3]
    """
    pytree = tensorplay.utils._pytree
    flat_xs, xs_spec = pytree.tree_flatten(xs)
    flat_args, args_spec = pytree.tree_flatten(args)
    if not all(isinstance(t, Tensor) for t in flat_xs):
        raise RuntimeError(f"Mapped xs can only consist of tensors. Got xs {flat_xs}.")
    if not flat_xs:
        raise RuntimeError("map() requires at least one mapped tensor in xs.")

    shapes = [tuple(x.shape) for x in flat_xs]
    leading_dim_size = shapes[0][0]
    if leading_dim_size == 0:
        raise RuntimeError("Leading dimensions of mapped xs cannot be 0.")
    if any(cur_shape[0] != leading_dim_size for cur_shape in shapes):
        raise RuntimeError(
            f"Leading dimensions of mapped xs must be consistent. Got shapes {shapes}."
        )

    num_xs = len(flat_xs)

    def wrapped_fn(*flat_args: Any) -> Any:
        xs_tree = pytree.tree_unflatten(flat_args[:num_xs], xs_spec)
        args_tree = pytree.tree_unflatten(flat_args[num_xs:], args_spec)
        return f(xs_tree, *args_tree)

    return map_impl(wrapped_fn, tuple(flat_xs), tuple(flat_args))


@map_impl.py_impl("CompositeExplicitAutograd")
def map_dense(
    f: Callable,
    xs: Sequence[Any],
    pos_args: Sequence[Any],
) -> Any:
    """Eager evaluation: one call per slice, stacked at the end."""
    pytrees = [f(*inp, *pos_args) for inp in _unstack_pytree(xs)]
    return _stack_pytree(pytrees)


class MapAutogradOp(tensorplay.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        f: Callable,
        num_mapped_args: int,
        *flat_args: Any,
    ) -> Any:
        ctx._f = f
        ctx._num_mapped_args = num_mapped_args
        ctx._num_pos_args = len(flat_args) - num_mapped_args
        save_values_for_backward(ctx, flat_args)
        with _AutoDispatchBelowAutograd():
            return map_impl(
                f,
                tuple(flat_args[:num_mapped_args]),
                tuple(flat_args[num_mapped_args:]),
            )

    @staticmethod
    def backward(ctx: Any, *flat_grads: Any) -> tuple[Any, ...]:
        fw_args = saved_values(ctx)
        num_mapped_args = ctx._num_mapped_args
        num_pos_args = ctx._num_pos_args
        num_grads = len(flat_grads)

        fw_mapped_args, pos_args = split_into_chunks(
            fw_args, [num_mapped_args, num_pos_args]
        )
        bw_f = create_bw_fn(ctx._f, tuple(fw_args))

        grads_tensor_masks: list[Any] = []

        def bw_f_wrapper(*args: Any) -> Any:
            nonlocal grads_tensor_masks
            fw_m_args, bw_f_tangents, pos_a = split_into_chunks(
                args, [num_mapped_args, num_grads, num_pos_args]
            )
            gradients = bw_f(*fw_m_args, *pos_a, *bw_f_tangents)
            grads_tensor_masks = [
                True if isinstance(out, Tensor) else out for out in gradients
            ]
            return filter_with_masks(gradients, grads_tensor_masks)

        # The backward maps the same leading dimension: each step receives
        # slices of the forward mapped inputs and of the stacked gradients.

        with _AutoDispatchBelowAutograd():
            grads = map_impl(
                bw_f_wrapper,
                tuple(fw_mapped_args) + tuple(flat_grads),
                tuple(pos_args),
            )

        return (None, None, *fill_none_with_masks(grads, grads_tensor_masks))


@map_impl.py_autograd_impl
def map_autograd(
    f: Callable,
    xs: Sequence[Any],
    pos_args: Sequence[Any],
) -> Any:
    if not tensorplay.is_grad_enabled() or not (
        _any_requires_grad(xs) or _any_requires_grad(pos_args)
    ):
        return map_dense(f, xs, pos_args)
    return MapAutogradOp.apply(f, len(xs), *xs, *pos_args)


def trace_map(
    proxy_mode: Any,
    func_overload: Callable,
    f: Callable,
    xs: Sequence[Any],
    pos_args: Sequence[Any],
) -> Any:
    """Capture a ``map`` call as one node carrying the body subgraph."""
    tracer = proxy_mode.tracer
    real_xs = tuple(_resolve_real_sample(x, tracer) for x in xs)
    real_pos = tuple(_resolve_real_sample(a, tracer) for a in pos_args)
    have_samples = not any(
        _is_capture_value(orig) and real is None
        for orig, real in zip(xs, real_xs)
    )

    with disable_proxy_modes_tracing():
        trace_xs = (
            real_xs if all(x is not None for x in real_xs) else tuple(xs)
        )
        example_input = [
            first_slice_copy(x) if isinstance(x, Tensor) else x for x in trace_xs
        ]
        body_graph = reenter_make_fx(f)(*example_input, *pos_args)

    _, graph_name = unique_graph_id(proxy_mode, prefix="map_body_graph")
    body_graph.meta["hop_graph_name"] = graph_name

    args = (body_graph, tuple(xs), tuple(pos_args))
    proxy_args = unwrap_proxy(args)
    out_proxy = tracer.create_proxy("call_function", func_overload, proxy_args, {})

    if have_samples:
        with disable_proxy_modes_tracing():
            example = map_dense(f, real_xs, real_pos)
        if isinstance(example, Tensor):
            return track_tensor_tree(
                example, out_proxy, constant=None, tracer=proxy_mode.tracer
            )
    return out_proxy


@map_impl.py_impl("CompositeExplicitAutograd")
def map_dense(
    f: Callable,
    xs: Sequence[Any],
    pos_args: Sequence[Any],
) -> Any:
    """Eager evaluation: one call per slice, stacked at the end."""
    pytrees = [f(*inp, *pos_args) for inp in _unstack_pytree(xs)]
    return _stack_pytree(pytrees)


class MapAutogradOp(tensorplay.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        f: Callable,
        num_mapped_args: int,
        *flat_args: Any,
    ) -> Any:
        ctx._f = f
        ctx._num_mapped_args = num_mapped_args
        ctx._num_pos_args = len(flat_args) - num_mapped_args
        save_values_for_backward(ctx, flat_args)
        with _AutoDispatchBelowAutograd():
            return map_impl(
                f,
                tuple(flat_args[:num_mapped_args]),
                tuple(flat_args[num_mapped_args:]),
            )

    @staticmethod
    def backward(ctx: Any, *flat_grads: Any) -> tuple[Any, ...]:
        fw_args = saved_values(ctx)
        num_mapped_args = ctx._num_mapped_args
        num_pos_args = ctx._num_pos_args
        num_grads = len(flat_grads)

        fw_mapped_args, pos_args = split_into_chunks(
            fw_args, [num_mapped_args, num_pos_args]
        )
        bw_f = create_bw_fn(ctx._f, tuple(fw_args))

        grads_tensor_masks: list[Any] = []

        def bw_f_wrapper(*args: Any) -> Any:
            nonlocal grads_tensor_masks
            fw_m_args, bw_f_tangents, pos_a = split_into_chunks(
                args, [num_mapped_args, num_grads, num_pos_args]
            )
            gradients = bw_f(*fw_m_args, *pos_a, *bw_f_tangents)
            grads_tensor_masks = [
                True if isinstance(out, Tensor) else out for out in gradients
            ]
            return filter_with_masks(gradients, grads_tensor_masks)

        # The backward maps the same leading dimension: each step receives
        # slices of the forward mapped inputs and of the stacked gradients.

        with _AutoDispatchBelowAutograd():
            grads = map_impl(
                bw_f_wrapper,
                tuple(fw_mapped_args) + tuple(flat_grads),
                tuple(pos_args),
            )

        return (None, None, *fill_none_with_masks(grads, grads_tensor_masks))


@map_impl.py_autograd_impl
def map_autograd(
    f: Callable,
    xs: Sequence[Any],
    pos_args: Sequence[Any],
) -> Any:
    if not tensorplay.is_grad_enabled() or not (
        _any_requires_grad(xs) or _any_requires_grad(pos_args)
    ):
        return map_dense(f, xs, pos_args)
    return MapAutogradOp.apply(f, len(xs), *xs, *pos_args)


def trace_map(
    proxy_mode: Any,
    func_overload: Callable,
    f: Callable,
    xs: Sequence[Any],
    pos_args: Sequence[Any],
) -> Any:
    """Capture a ``map`` call as one node carrying the body subgraph."""
    tracer = proxy_mode.tracer
    with disable_proxy_modes_tracing():
        example_input = [
            first_slice_copy(x) if isinstance(x, Tensor) else x for x in xs
        ]
        body_graph = reenter_make_fx(f)(*example_input, *pos_args)
        example = func_overload(f, tuple(xs), tuple(pos_args))

    _, graph_name = unique_graph_id(proxy_mode, prefix="map_body_graph")
    body_graph.meta["hop_graph_name"] = graph_name

    args = (body_graph, tuple(xs), tuple(pos_args))
    proxy_args = unwrap_proxy(args)
    out_proxy = tracer.create_proxy("call_function", func_overload, proxy_args, {})
    return track_tensor_tree(
        example, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@map_impl.py_impl("CompositeExplicitAutograd")
def map_dense(
    f: Callable,
    xs: Sequence[Any],
    pos_args: Sequence[Any],
) -> Any:
    """Eager evaluation: one call per slice, stacked at the end."""
    pytrees = [f(*inp, *pos_args) for inp in _unstack_pytree(xs)]
    return _stack_pytree(pytrees)


class MapAutogradOp(tensorplay.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        f: Callable,
        num_mapped_args: int,
        *flat_args: Any,
    ) -> Any:
        ctx._f = f
        ctx._num_mapped_args = num_mapped_args
        ctx._num_pos_args = len(flat_args) - num_mapped_args
        save_values_for_backward(ctx, flat_args)
        with _AutoDispatchBelowAutograd():
            return map_impl(
                f,
                tuple(flat_args[:num_mapped_args]),
                tuple(flat_args[num_mapped_args:]),
            )

    @staticmethod
    def backward(ctx: Any, *flat_grads: Any) -> tuple[Any, ...]:
        fw_args = saved_values(ctx)
        num_mapped_args = ctx._num_mapped_args
        num_pos_args = ctx._num_pos_args
        num_grads = len(flat_grads)

        fw_mapped_args, pos_args = split_into_chunks(
            fw_args, [num_mapped_args, num_pos_args]
        )
        bw_f = create_bw_fn(ctx._f, tuple(fw_args))

        grads_tensor_masks: list[Any] = []

        def bw_f_wrapper(*args: Any) -> Any:
            nonlocal grads_tensor_masks
            fw_m_args, bw_f_tangents, pos_a = split_into_chunks(
                args, [num_mapped_args, num_grads, num_pos_args]
            )
            gradients = bw_f(*fw_m_args, *pos_a, *bw_f_tangents)
            grads_tensor_masks = [
                True if isinstance(out, Tensor) else out for out in gradients
            ]
            return filter_with_masks(gradients, grads_tensor_masks)

        # The backward maps the same leading dimension: each step receives
        # slices of the forward mapped inputs and of the stacked gradients.

        with _AutoDispatchBelowAutograd():
            grads = map_impl(
                bw_f_wrapper,
                tuple(fw_mapped_args) + tuple(flat_grads),
                tuple(pos_args),
            )

        return (None, None, *fill_none_with_masks(grads, grads_tensor_masks))


@map_impl.py_autograd_impl
def map_autograd(
    f: Callable,
    xs: Sequence[Any],
    pos_args: Sequence[Any],
) -> Any:
    if not tensorplay.is_grad_enabled() or not (
        _any_requires_grad(xs) or _any_requires_grad(pos_args)
    ):
        return map_dense(f, xs, pos_args)
    return MapAutogradOp.apply(f, len(xs), *xs, *pos_args)


def trace_map(
    proxy_mode: Any,
    func_overload: Callable,
    f: Callable,
    xs: Sequence[Any],
    pos_args: Sequence[Any],
) -> Any:
    """Capture a ``map`` call as one node carrying the body subgraph."""
    tracer = proxy_mode.tracer
    with disable_proxy_modes_tracing():
        example_input = [
            first_slice_copy(x) if isinstance(x, Tensor) else x for x in xs
        ]
        body_graph = reenter_make_fx(f)(*example_input, *pos_args)

    with disable_proxy_modes_tracing():
        example = func_overload(f, tuple(xs), tuple(pos_args))

    _, graph_name = unique_graph_id(proxy_mode, prefix="map_body_graph")
    body_graph.meta["hop_graph_name"] = graph_name

    args = (body_graph, tuple(xs), tuple(pos_args))
    proxy_args = unwrap_proxy(args)
    out_proxy = tracer.create_proxy("call_function", func_overload, proxy_args, {})
    return track_tensor_tree(
        example, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@map_impl.py_impl("ProxyDispatchMode")
def map_proxy_dispatch_mode(
    mode: Any,
    f: Callable,
    xs: Sequence[Any],
    pos_args: Sequence[Any],
) -> Any:
    return trace_map(mode, map_impl, f, xs, pos_args)


@register_fake(map_impl)
def map_fake(
    f: Callable,
    xs: Sequence[Any],
    pos_args: Sequence[Any],
) -> Any:
    """Abstract evaluation: one step's output, expanded along the loop dim."""
    first_row = [first_slice_copy(x) if isinstance(x, Tensor) else x for x in xs]
    example_output = f(*first_row, *pos_args)
    batch_size = xs[0].size(0)

    def expand_with_batch(t: Any) -> Any:
        if isinstance(t, Tensor):
            return (
                t.unsqueeze(0)
                .expand(batch_size, *t.shape)
                .clone()
            )
        return t

    return tensorplay.utils._pytree.tree_map(expand_with_batch, example_output)


@map_impl.py_functionalize_impl
def map_functionalize(
    ctx: Any,
    f: Callable,
    xs: Sequence[Any],
    pos_args: Sequence[Any],
) -> Any:
    """Functionalization rule: unwrap, reject input mutation, redispatch."""
    from tensorplay._higher_order_ops.utils import (
        UnsupportedAliasMutationException,
        _has_potential_branch_input_mutation,
    )

    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_args = ctx.unwrap_tensors(pos_args)
    wrapped_fn = ctx.functionalize(f)

    with ctx.redispatch_to_next():
        example_inputs = (
            *(
                first_slice_copy(x) if isinstance(x, Tensor) else x
                for x in unwrapped_xs
            ),
            *unwrapped_args,
        )
        if _has_potential_branch_input_mutation(wrapped_fn, example_inputs):
            raise UnsupportedAliasMutationException("Mutations detected in map")
        map_return = map_impl(wrapped_fn, unwrapped_xs, unwrapped_args)
        return ctx.wrap_tensors(map_return)

