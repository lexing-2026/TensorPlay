"""Structural scan operator.

``scan`` threads a carry through a sequence: each step consumes the previous
carry and one slice of the inputs, producing the next carry and one slice of
the output.  The operator is registry-based like the other control flow
operators; its eager composite registration is the sequential loop, and a
graph capture records the call as one opaque node carrying the traced
combine subgraph.

The backward runs through the unrolled sequence the eager composite
executed: autograd records every combine call, so gradients are exact at the
cost of a graph proportional to the scan length.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import tensorplay
from tensorplay import Tensor
from tensorplay._higher_order_ops._hop_base import (
    HigherOrderOperator,
    register_fake,
)
from tensorplay._higher_order_ops.utils import (
    _resolve_real_sample,
    check_meta_consistency,
    first_slice_copy,
    get_tensor_mask,
    mask_list,
    reenter_make_fx,
    split_into_chunks,
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


def _wrap_combine_fn_flat(
    combine_fn: Callable,
    spec_init: Any,
    spec_xs: Any,
    num_init_leaves: int,
) -> Callable:
    """Flatten the two pytree arguments of ``combine_fn(carry, x)``."""

    def wrapped(*args: Any) -> Any:
        carry = tensorplay.utils._pytree.tree_unflatten(
            args[:num_init_leaves], spec_init
        )
        xs = tensorplay.utils._pytree.tree_unflatten(args[num_init_leaves:], spec_xs)
        return combine_fn(carry, xs)

    return wrapped


def _extract_carry_and_out(flat_out: list[Any], num_carry: int) -> tuple[list, list]:
    return split_into_chunks(flat_out, [num_carry, len(flat_out) - num_carry])


def _stack_y(y: Tensor, scan_length: int) -> Tensor:
    """A stacked placeholder for one output leaf over the scan dimension."""
    return y.unsqueeze(0).repeat(*([scan_length] + [1] * y.dim()))


def scan(
    combine_fn: Callable,
    init: Any,
    xs: Any,
    *,
    dim: int = 0,
    reverse: bool = False,
    length: int | None = None,
) -> tuple[Any, Any]:
    """Perform an inclusive scan with a combine function.

    Each step calls ``combine_fn(carry, x)`` where ``carry`` is the previous
    carry (``init`` for the first step) and ``x`` is one slice of ``xs``
    along ``dim``.  The call returns ``(next_carry, y)``; the carries chain
    through the sequence and the ``y`` values stack into the output.

    Args:
        combine_fn (Callable): ``(carry, x) -> (carry, y)``; must be pure --
          no lifted arguments, no side effects, no aliasing or mutation of
          its inputs.
        init (Tensor or pytree of Tensors): The initial carry; must match the
          pytree structure and tensor metadata of the first output of
          ``combine_fn``.
        xs (Tensor, pytree of Tensors or None): The sequence to scan; every
          leaf must agree on the size along ``dim``.  ``None`` together with
          ``length`` runs a counter loop whose steps receive ``x=None``.
        dim (int): The dimension to scan over, default 0.
        reverse (bool): Scan the sequence in reverse, default False.
        length (int or None): Number of iterations when ``xs`` carries no
          tensors; otherwise an optional consistency check against
          ``xs.shape[dim]``.

    Returns:
        ``(final_carry, stacked_ys)``.  A scan length of zero returns the
        initial carry untouched and empty stacked outputs.

    Example::

        def add(carry, x):
            next_carry = x
            return next_carry, (carry + x).clone()

        i0 = tensorplay.zeros(1)
        xs = tensorplay.arange(5.0)
        # returns the last element and the running partial sums
        last, cumsum = scan(add, init=i0, xs=xs)
    """
    pytree = tensorplay.utils._pytree
    leaves_init, spec_init = pytree.tree_flatten(init)
    leaves_xs_orig, spec_xs = pytree.tree_flatten(xs)

    def _tensor_like(leaf: Any) -> bool:
        # Under a graph capture the leaves arrive as proxies; they stand for
        # tensors in every check below.
        return isinstance(leaf, Tensor) or _is_capture_value(leaf)

    xs_has_tensors = any(_tensor_like(leaf) for leaf in leaves_xs_orig)

    if length is not None:
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise RuntimeError(f"scan() length must be a non-negative integer, got {length!r}")
        if not xs_has_tensors:
            if length == 0:
                return init, _empty_output_for_length_zero(combine_fn, init)
            # Counter mode: a dummy sequence of the requested length whose
            # slices are discarded; the wrapped body receives x=None.
            leaves_xs_orig = [tensorplay.zeros(length, dtype=tensorplay.int64)]
            spec_xs = pytree.tree_structure(None)
            _user_combine_fn = combine_fn

            def combine_fn(carry: Any, _ignored: Any) -> Any:  # noqa: F811
                return _user_combine_fn(carry, None)

    elif not xs_has_tensors:
        return init, []

    if not callable(combine_fn):
        raise RuntimeError(f"Combine_fn must be a callable, but got {combine_fn}")

    for leaf in leaves_init:
        if not _tensor_like(leaf):
            raise RuntimeError(f"All init leaves must be a Tensor but got {leaf}")
    for leaf in leaves_xs_orig:
        if not _tensor_like(leaf):
            raise RuntimeError(f"All xs leaves must be a Tensor but got {leaf}")
    # Rank, size-agreement and length checks compare symbolic values under a
    # capture, so they run on the eager path only.
    if get_proxy_mode() is None:
        if any(leaf.dim() <= dim for leaf in leaves_xs_orig):
            raise RuntimeError("All xs leaves must have at least 'dim + 1' dimensions")
        if any(
            leaf.size(dim) != leaves_xs_orig[0].size(dim) for leaf in leaves_xs_orig[1:]
        ):
            raise RuntimeError("All xs leaves must have the same scan dimension size")
        if length is not None and xs_has_tensors and leaves_xs_orig[0].size(dim) != length:
            raise RuntimeError(
                f"scan() length={length} does not match xs size along dim={dim}: "
                f"{leaves_xs_orig[0].size(dim)}"
            )

    # The scan always runs along dim 0 internally; move the user's dim there
    # and back, flipping for reverse scans.
    leaves_xs = [
        tensorplay.movedim(leaf, dim, 0) if dim != 0 else leaf
        for leaf in leaves_xs_orig
    ]
    if reverse:
        leaves_xs = [tensorplay.flip(leaf, [0]) for leaf in leaves_xs]

    # A flat tuple pair passes the callable straight through: the operator
    # invokes it positionally, and fixed signatures stay traceable (varargs
    # wrappers are not capturable by the frontend).
    def _flat_tensor_likes(value: Any) -> bool:
        if _tensor_like(value):
            return True
        return isinstance(value, (tuple, list)) and all(
            _tensor_like(leaf) for leaf in value
        )

    trivial_shapes = _flat_tensor_likes(init) and _flat_tensor_likes(xs)
    if trivial_shapes:
        carry, out = scan_op(
            combine_fn, tuple(leaves_init), tuple(leaves_xs), ()
        )
        flat_combine = combine_fn
    else:
        flat_combine = _wrap_combine_fn_flat(
            combine_fn, spec_init, spec_xs, len(leaves_init)
        )
        carry, out = scan_op(flat_combine, tuple(leaves_init), tuple(leaves_xs), ())

    if reverse:
        out = pytree.tree_map(lambda elem: elem.flip([0]), out)
    if dim != 0:
        out = pytree.tree_map(
            lambda elem: tensorplay.movedim(elem, 0, dim) if dim < elem.dim() else elem,
            out,
        )
    return carry, out


def _empty_output_for_length_zero(combine_fn: Callable, init: Any) -> Any:
    """Probe the body once to learn the stacked output structure."""
    pytree = tensorplay.utils._pytree
    _, sample_y = combine_fn(init, None)
    flat_y, spec_y = pytree.tree_flatten(sample_y)
    results: list[Any] = []
    for leaf in flat_y:
        if isinstance(leaf, Tensor):
            results.append(
                tensorplay.empty(
                    [0] + list(leaf.shape), dtype=leaf.dtype, device=leaf.device
                )
            )
        elif leaf is None:
            results.append(leaf)
        else:
            raise AssertionError(
                f"Expected leaf to be a Tensor or None, got {type(leaf)}"
            )
    return pytree.tree_unflatten(results, spec_y)


class ScanOp(HigherOrderOperator):
    """``scan_op(combine_fn, init, xs, additional_inputs)`` as a registered operator."""

    def __init__(self) -> None:
        super().__init__("scan")

    def __call__(
        self,
        combine_fn: Callable,
        init: tuple[Any, ...],
        xs: tuple[Any, ...],
        additional_inputs: tuple[Any, ...],
    ) -> Any:
        if not isinstance(init, (tuple, list)):
            raise RuntimeError(f"init must be a tuple or list, got {type(init)}")
        if not isinstance(xs, (tuple, list)):
            raise RuntimeError(f"xs must be a tuple or list, got {type(xs)}")
        if not isinstance(additional_inputs, (tuple, list)):
            raise RuntimeError(
                f"additional_inputs must be a tuple or list, got {type(additional_inputs)}"
            )
        if get_proxy_mode() is None:
            validate_subgraph_args_types(init)
            validate_subgraph_args_types(xs)
        return super().__call__(combine_fn, init, xs, additional_inputs)


scan_op = ScanOp()


@scan_op.py_impl("CompositeExplicitAutograd")
def scan_op_dense(
    combine_fn: Callable,
    init: tuple[Any, ...],
    xs: tuple[Any, ...],
    additional_inputs: tuple[Any, ...],
) -> Any:
    """Eager evaluation: the sequential loop, stacking each step's output."""
    carry = tuple(init)
    if len(xs) == 0:
        return (*carry, *())

    scan_length = xs[0].size(0)
    proto_xs = [first_slice_copy(x) for x in xs]
    flat_out = list(
        combine_fn(*carry, *proto_xs, *additional_inputs)
    )
    carry, out_0 = _extract_carry_and_out(flat_out, len(init))
    out_tensor_mask = get_tensor_mask(out_0)

    if scan_length == 0:
        return (*init, *(None for _ in out_tensor_mask))

    collected: list[list[Tensor]] = [
        [o] for o, m in zip(out_0, out_tensor_mask) if m
    ]
    for i in range(1, scan_length):
        slices = [elem.select(0, i) for elem in xs]
        flat_out = list(
            combine_fn(*carry, *slices, *additional_inputs)
        )
        carry, out = _extract_carry_and_out(flat_out, len(init))
        masked = mask_list(out_tensor_mask, out)
        for buf, o in zip(collected, masked):
            buf.append(o)

    stacked = [tensorplay.stack(buf, dim=0) for buf in collected]
    outs_expanded = [stacked.pop(0) if m else None for m in out_tensor_mask]
    return (*carry, *outs_expanded)


def trace_scan(
    proxy_mode: Any,
    func_overload: Callable,
    combine_fn: Callable,
    init: tuple[Any, ...],
    xs: tuple[Any, ...],
    additional_inputs: tuple[Any, ...],
) -> Any:
    """Capture a ``scan`` call as one node with the combine subgraph attached."""
    tracer = proxy_mode.tracer
    real_init = tuple(_resolve_real_sample(i, tracer) for i in init)
    real_xs = tuple(_resolve_real_sample(x, tracer) for x in xs)
    real_additional = tuple(
        _resolve_real_sample(a, tracer) for a in additional_inputs
    )
    have_samples = not any(
        _is_capture_value(orig) and real is None
        for orig, real in zip(init + xs, real_init + real_xs)
    )

    with disable_proxy_modes_tracing():
        trace_init = (
            real_init if all(i is not None for i in real_init) else tuple(init)
        )
        sample_init = tuple(
            i.clone() if isinstance(i, Tensor) else i for i in trace_init
        )
        trace_xs = (
            real_xs if all(x is not None for x in real_xs) else tuple(xs)
        )
        sample_xs = [
            first_slice_copy(x) if isinstance(x, Tensor) else x for x in trace_xs
        ]
        combine_graph = reenter_make_fx(combine_fn)(
            *sample_init, *sample_xs, *real_additional
        )

    _, combine_name = unique_graph_id(proxy_mode, prefix="scan_combine_graph")
    combine_graph.meta["hop_graph_name"] = combine_name

    args = (combine_graph, init, xs, additional_inputs)
    proxy_args = unwrap_proxy(args)
    out_proxy = tracer.create_proxy("call_function", func_overload, proxy_args, {})

    if have_samples:
        with disable_proxy_modes_tracing():
            example = scan_op_dense(
                combine_fn, real_init, real_xs, real_additional
            )
        if isinstance(example, Tensor):
            return track_tensor_tree(
                example, out_proxy, constant=None, tracer=proxy_mode.tracer
            )
    return out_proxy


@scan_op.py_impl("ProxyDispatchMode")
def scan_proxy_mode(
    mode: Any,
    combine_fn: Callable,
    init: tuple[Any, ...],
    xs: tuple[Any, ...],
    additional_inputs: tuple[Any, ...],
) -> Any:
    return trace_scan(
        mode, scan_op, combine_fn, init, xs, additional_inputs
    )


@register_fake(scan_op)
def scan_fake(
    mode: Any,
    combine_fn: Callable,
    init: tuple[Any, ...],
    xs: tuple[Any, ...],
    additional_inputs: tuple[Any, ...],
) -> Any:
    """Abstract evaluation: one combine step, its y stacked over the length."""
    scan_length = xs[0].size(0) if len(xs) else 0
    carry, outputs = _extract_carry_and_out(
        list(
            combine_fn(
                *init,
                *[first_slice_copy(x) for x in xs],
                *additional_inputs,
            )
        ),
        len(init),
    )
    for t in outputs:
        if not isinstance(t, Tensor) and t is not None:
            raise AssertionError(
                f"Expected leaf to be a Tensor or None, got {type(t)}"
            )
    check_meta_consistency(init, carry, "init", "carry")
    out = (
        *carry,
        *(_stack_y(t, scan_length) if isinstance(t, Tensor) else t for t in outputs),
    )
    return out


@scan_op.py_functionalize_impl
def scan_functionalize(
    ctx: Any,
    combine_fn: Callable,
    init: tuple[Any, ...],
    xs: tuple[Any, ...],
    additional_inputs: tuple[Any, ...],
) -> Any:
    """Functionalization rule: unwrap, reject input mutation, redispatch."""
    from tensorplay._higher_order_ops.utils import (
        UnsupportedAliasMutationException,
        _has_potential_branch_input_mutation,
    )

    unwrapped_init = ctx.unwrap_tensors(init)
    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_additional = ctx.unwrap_tensors(additional_inputs)
    wrapped_fn = ctx.functionalize(combine_fn)

    with ctx.redispatch_to_next():
        sample_inputs = (
            *unwrapped_init,
            *(
                first_slice_copy(x) if isinstance(x, Tensor) else x
                for x in unwrapped_xs
            ),
            *unwrapped_additional,
        )
        if _has_potential_branch_input_mutation(wrapped_fn, sample_inputs):
            raise UnsupportedAliasMutationException("Mutations detected in scan")
        ret = scan_op(
            wrapped_fn,
            unwrapped_init,
            unwrapped_xs,
            unwrapped_additional,
        )
        return ctx.wrap_tensors(ret)
