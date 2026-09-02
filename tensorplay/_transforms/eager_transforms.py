"""Composable function transforms, evaluated eagerly.

Every transform here takes a callable and returns a callable -- the derivative
of a function is itself a function -- which is what lets them nest.

``grad`` and ``vjp`` read the reverse-mode graph directly.  ``jvp`` and the
transforms layered on it (``jacfwd``, ``linearize``, ``hessian``) get
directional derivatives from that same graph: the vector-Jacobian product is
*linear* in its cotangent, so differentiating it with respect to that cotangent
recovers the Jacobian-vector product.  One evaluation rule set therefore serves
both directions.

Tensors handed to a transform are re-seeded as graph leaves for the duration of
the call.  Whether the results stay attached to the caller's graph depends on
whether the caller's own inputs were already differentiable -- which is what
makes ``grad(grad(f))`` compose while ``grad(f)`` applied to a plain tensor
still hands back plain tensors.
"""

from __future__ import annotations

import functools
import contextlib
import contextvars
from typing import Any, Callable, Optional, Union

import tensorplay
from tensorplay.utils._pytree import tree_flatten, tree_leaves, tree_map, tree_unflatten

from .utils import argnums_t, exposed_in
from .vmap import (
    doesnt_support_saved_tensors_hooks,
    get_chunk_sizes,
    vmap_impl,
)

__all__ = [
    "vjp",
    "jvp",
    "jacrev",
    "jacfwd",
    "hessian",
    "linearize",
    "functionalize",
    "debug_unwrap",
]

_transform_depth = contextvars.ContextVar("tensorplay_transform_depth", default=0)


@contextlib.contextmanager
def transform_increment_nesting():
    """Tracks transform nesting without changing ordinary grad mode."""
    token = _transform_depth.set(_transform_depth.get() + 1)
    try:
        yield _transform_depth.get()
    finally:
        _transform_depth.reset(token)


def _wraps_without_transform_attrs(func: Callable) -> Callable:
    """Copies user metadata while omitting transform bookkeeping fields."""

    def decorator(wrapper: Callable) -> Callable:
        result = functools.wraps(func)(wrapper)
        for name in tuple(result.__dict__):
            if name.startswith("_tensorplay_transform"):
                result.__dict__.pop(name, None)
        return result

    return decorator


def _is_differentiable(value: object) -> bool:
    return isinstance(value, tensorplay.Tensor) and value.requires_grad


def _any_differentiable(value: Any) -> bool:
    return any(_is_differentiable(leaf) for leaf in tree_leaves(value))


def _set_tensor_requires_grad(value: tensorplay.Tensor) -> tensorplay.Tensor:
    return value.requires_grad_(True)


def _wrap_tensor_for_grad(value: Any, level: int) -> Any:
    del level
    return value


def _wrap_all_tensors(value: Any, level: int) -> Any:
    return tree_map(lambda leaf: _wrap_tensor_for_grad(leaf, level), value)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _safe_zero_index(value: tuple[Any, ...]) -> Any:
    if len(value) != 1:
        raise AssertionError(f"expected tuple of length 1, got length {len(value)}")
    return value[0]


def assert_flat_tuple_of_tensors(values: Any, api: str, argname: str) -> None:
    if not isinstance(values, tuple):
        raise RuntimeError(f"{api}: Expected {argname} to be a tuple of Tensors.")
    assert_non_empty_list_of_tensors(values, api, argname)


def safe_unpack_dual(value: Any, strict: bool = False) -> tuple[Any, Any]:
    """Unpacks a native forward-AD value, materializing an independent zero."""
    if not isinstance(value, tensorplay.Tensor):
        raise RuntimeError(
            f"jvp(f, primals, tangents): expected f(*args) to return only "
            f"tensors, got unsupported type {type(value)}"
        )
    tangent, primal = tensorplay.autograd.forward_ad.unpack_dual(value)
    if tangent is None:
        if strict:
            raise RuntimeError(
                "jvp(f, primals, tangents, strict=True): The output of f is "
                "independent of the inputs. This is not allowed with strict=True."
            )
        tangent = tensorplay.zeros_like(primal)
    return primal, tangent


# ---------------------------------------------------------------- argnums --


def _validate_and_wrap_argnum(argnum: Any, num_args: int) -> int:
    if not isinstance(argnum, int) or isinstance(argnum, bool):
        raise RuntimeError(f"argnum must be int, got: {type(argnum)}")
    if 0 <= argnum < num_args:
        return argnum
    if -num_args <= argnum < 0:
        return argnum + num_args
    raise RuntimeError(f"Got argnum={argnum}, but only {num_args} positional inputs")


def _check_unique_non_empty(argnums: argnums_t) -> None:
    if isinstance(argnums, tuple):
        if len(argnums) == 0:
            raise RuntimeError("argnums must be non-empty")
        if len(set(argnums)) != len(argnums):
            raise RuntimeError(f"argnums elements must be unique, got {argnums}")


def _validate_and_wrap_argnums(argnums: argnums_t, num_args: int) -> argnums_t:
    if isinstance(argnums, int):
        return _validate_and_wrap_argnum(argnums, num_args)
    if isinstance(argnums, tuple):
        return tuple(_validate_and_wrap_argnum(argnum, num_args) for argnum in argnums)
    raise RuntimeError(f"argnums must be int or Tuple[int, ...], got: {type(argnums)}")


def _normalize_argnums(argnums: argnums_t, num_args: int) -> argnums_t:
    """Resolves negative indices once, so slicing and substitution agree."""
    if not isinstance(argnums, (int, tuple)) or isinstance(argnums, bool):
        raise RuntimeError(f"argnums must be int or Tuple[int, ...], got: {type(argnums)}")
    argnums = _validate_and_wrap_argnums(argnums, num_args)
    _check_unique_non_empty(argnums)
    return argnums


def _slice_argnums(args: tuple[Any, ...], argnums: argnums_t, as_tuple: bool = True):
    argnums = _normalize_argnums(argnums, len(args))
    if isinstance(argnums, int):
        return (args[argnums],) if as_tuple else args[argnums]
    return tuple(args[i] for i in argnums)


def _replace_args(old_args, new_args, argnums: argnums_t):
    """Substitutes ``new_args`` back into the positions named by ``argnums``."""
    if isinstance(argnums, int):
        if len(new_args) != 1:
            raise RuntimeError(
                f"new_args should be of size 1, was of size {len(new_args)}"
            )
        return tuple(
            new_args[0] if i == argnums else old_args[i] for i in range(len(old_args))
        )
    if isinstance(argnums, tuple):
        if len(new_args) != len(argnums):
            raise RuntimeError(
                "new_args should have the same size as argnums. "
                f"Argnums size {len(argnums)}, new_args size {len(new_args)}"
            )
        replaced = list(old_args)
        for position, argnum in enumerate(argnums):
            replaced[argnum] = new_args[position]
        return tuple(replaced)
    raise RuntimeError(f"argnums must be int or Tuple[int, ...], got: {type(argnums)}")


def _as_tuple_for_replace(diff_args, argnums: argnums_t) -> tuple[Any, ...]:
    return (diff_args,) if isinstance(argnums, int) else diff_args


# ------------------------------------------------------------ graph seeding --


def _unwrap_transform_layers(value: Any) -> tuple[Any, list[tuple[int, int]]]:
    """Return the physical value and the transform wrappers around it."""

    layers: list[tuple[int, int]] = []
    while isinstance(value, tensorplay.Tensor) and tensorplay._C._transform_is_batched(value):
        level = tensorplay._C._transform_batch_level(value)
        unwrapped, batch_dim = tensorplay._C._transform_unwrap(value, level)
        if batch_dim is None:
            break
        layers.append((int(batch_dim), int(level)))
        value = unwrapped
    return value, layers


def _rewrap_transform_layers(
    value: tensorplay.Tensor, layers: list[tuple[int, int]]
) -> tensorplay.Tensor:
    """Restore transform wrappers after a physical autograd operation."""

    for batch_dim, level in reversed(layers):
        value = tensorplay._C._transform_make_batched(value, batch_dim, level)
    return value


def _create_differentiable(inps, api: str = "transform"):
    """Re-seeds every tensor leaf so it can be differentiated against.

    A tensor that already requires grad is handed back untouched: it belongs to
    an enclosing graph, and detaching it would sever the composition that makes
    nested transforms work.
    """

    def create_differentiable(x):
        if isinstance(x, tensorplay.Tensor):
            physical, layers = _unwrap_transform_layers(x)
            if physical.requires_grad:
                differentiable = physical
            else:
                differentiable = physical.detach().requires_grad_(True)
            return _rewrap_transform_layers(differentiable, layers)
        raise ValueError(
            f"{api}: Expected all inputs to be Tensors, got {type(x)} instead"
        )

    return tree_map(create_differentiable, inps)


def _any_requires_grad(inps) -> bool:
    return any(
        isinstance(leaf, tensorplay.Tensor) and leaf.requires_grad
        for leaf in tree_leaves(inps)
    )


def _undo_create_differentiable(inps, keep_graph: bool):
    """Detaches results unless an enclosing transform still needs the graph."""
    if keep_graph:
        return inps

    def unwrap(x):
        if not isinstance(x, tensorplay.Tensor):
            return x
        physical, layers = _unwrap_transform_layers(x)
        return _rewrap_transform_layers(physical.detach(), layers)

    return tree_map(unwrap, inps)


def _autograd_grad(
    outputs,
    inputs,
    grad_outputs=None,
    retain_graph: bool = True,
    create_graph: bool = True,
):
    """``autograd.grad`` with unused inputs and non-differentiable outputs
    reported as explicit zeros instead of ``None``."""
    physical_outputs_and_layers = tuple(_unwrap_transform_layers(out) for out in outputs)
    physical_outputs = tuple(item[0] for item in physical_outputs_and_layers)
    physical_inputs_and_layers = tuple(_unwrap_transform_layers(inp) for inp in inputs)
    physical_inputs = tuple(item[0] for item in physical_inputs_and_layers)

    if grad_outputs is None:
        diff_outputs = tuple(out for out in physical_outputs if out.requires_grad)
        physical_grad_outputs = tuple(
            tensorplay.ones_like(output)
            if original.numel() == 1 and output.numel() != 1
            else None
            for original, output in zip(outputs, physical_outputs)
            if output.requires_grad
        )
        cotangent_layers = tuple(() for _ in diff_outputs)
        diff_output_layers = tuple(
            output_layers
            for output, (_, output_layers) in zip(
                physical_outputs, physical_outputs_and_layers
            )
            if output.requires_grad
        )
    else:
        physical_cotangents_and_layers = tuple(
            (None, [])
            if cotangent is None
            else _unwrap_transform_layers(cotangent)
            for cotangent in grad_outputs
        )
        pairs = tuple(
            (out, cotangent, output_layers, cotangent_layers_for_output)
            for out, (_, output_layers), (cotangent, cotangent_layers_for_output) in zip(
                physical_outputs,
                physical_outputs_and_layers,
                physical_cotangents_and_layers,
            )
            if out.requires_grad
        )
        if len(pairs) == 0:
            diff_outputs = ()
            physical_grad_outputs = ()
            cotangent_layers = ()
            diff_output_layers = ()
        else:
            diff_outputs = tuple(pair[0] for pair in pairs)
            physical_grad_outputs = tuple(pair[1] for pair in pairs)
            cotangent_layers = tuple(pair[3] for pair in pairs)
            diff_output_layers = tuple(pair[2] for pair in pairs)

    if len(diff_outputs) == 0:
        return tuple(
            _rewrap_transform_layers(tensorplay.zeros_like(inp), layers)
            for inp, layers in physical_inputs_and_layers
        )

    output_levels = tuple(
        {level for _, level in output_layers} for output_layers in diff_output_layers
    )

    def _next_extra_level(layers):
        for output_level_set, cotangent_layer_set in zip(output_levels, layers):
            for _, level in cotangent_layer_set:
                if level not in output_level_set:
                    return level
        return None

    def _native_grad(cotangents):
        return tensorplay.autograd.grad(
            diff_outputs,
            physical_inputs,
            cotangents,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=True,
        )

    def _batched_grad(cotangents, layers):
        level = _next_extra_level(layers)
        if level is None:
            return _native_grad(cotangents)

        batch_dim = next(
            dim
            for cotangent_layers_for_output in layers
            for dim, layer_level in cotangent_layers_for_output
            if layer_level == level
        )
        batch_size = next(
            cotangent.size(batch_dim)
            for cotangent, cotangent_layers_for_output in zip(cotangents, layers)
            if cotangent is not None
            and any(layer_level == level for _, layer_level in cotangent_layers_for_output)
        )
        gradients = [[] for _ in physical_inputs]
        for index in range(batch_size):
            sliced_cotangents = []
            sliced_layers = []
            for cotangent, cotangent_layers_for_output in zip(cotangents, layers):
                matching_dim = next(
                    (
                        dim
                        for dim, layer_level in cotangent_layers_for_output
                        if layer_level == level
                    ),
                    None,
                )
                if matching_dim is None:
                    sliced_cotangents.append(cotangent)
                    sliced_layers.append(cotangent_layers_for_output)
                    continue
                sliced_cotangents.append(
                    None
                    if cotangent is None
                    else cotangent.select(matching_dim, index)
                )
                sliced_layers.append(
                    tuple(
                        (
                            dim - 1
                            if dim > matching_dim
                            else dim,
                            layer_level,
                        )
                        for dim, layer_level in cotangent_layers_for_output
                        if layer_level != level
                    )
                )
            per_sample = _batched_grad(tuple(sliced_cotangents), tuple(sliced_layers))
            for slot, gradient in zip(gradients, per_sample):
                slot.append(gradient)

        return tuple(
            tensorplay.stack(
                [
                    tensorplay.zeros_like(inp) if gradient is None else gradient
                    for gradient in per_input
                ],
                dim=0,
            )
            for per_input, (inp, _) in zip(gradients, physical_inputs_and_layers)
        )

    grad_inputs = _batched_grad(physical_grad_outputs, cotangent_layers)
    extra_layers = []
    for output_level_set, cotangent_layer_set in zip(output_levels, cotangent_layers):
        extra_layers = [
            (0, level)
            for _, level in cotangent_layer_set
            if level not in output_level_set
        ]
        if extra_layers:
            break
    extra_layer_count = len(extra_layers)
    return tuple(
        _rewrap_transform_layers(
            tensorplay.zeros_like(inp) if grad_input is None else grad_input,
            extra_layers
            + [(dim + extra_layer_count, level) for dim, level in layers],
        )
        for grad_input, (inp, layers) in zip(
            grad_inputs, physical_inputs_and_layers
        )
    )


# ------------------------------------------------------------- assertions --


def assert_non_empty_tensor_output(output, api: str) -> None:
    if (len(output) == 1 and output[0] is None) or len(output) < 1:
        raise RuntimeError(
            f"{api}: Expected f to be a function that has non-empty output (got "
            f"output = {output})"
        )
    for out in output:
        if not isinstance(out, tensorplay.Tensor):
            raise RuntimeError(
                f"{api}: expected f(*primals) to return only tensors, got "
                f"unsupported type {type(out)}"
            )


def assert_non_empty_list_of_tensors(output, api: str, argname: str) -> None:
    if len(output) == 0:
        raise RuntimeError(f"{api}: Expected {argname} to contain at least one Tensor.")
    for out in output:
        if isinstance(out, tensorplay.Tensor):
            continue
        raise RuntimeError(f"{api}: Expected {argname} to only contain Tensors, got {type(out)}")


def assert_output_is_tensor_or_tensors(output, api: str) -> None:
    if isinstance(output, tensorplay.Tensor):
        return
    if not isinstance(output, tuple):
        raise RuntimeError(
            f"{api}: Expected output of f to be a Tensor or Tensors, got {type(output)}"
        )
    if len(output) == 0:
        raise RuntimeError(f"{api}: Expected output of f to be a non-empty tuple of Tensors.")
    for out in output:
        if isinstance(out, tensorplay.Tensor):
            continue
        raise RuntimeError(
            f"{api}: Expected output of f to be a Tensor or Tensors, got {type(out)}"
        )


def error_if_complex(func_name: str, args, is_input: bool) -> None:
    flat_args = tree_leaves(args)
    for idx, arg in enumerate(flat_args):
        if isinstance(arg, tensorplay.Tensor) and arg.is_complex():
            input_or_output = "inputs" if is_input else "outputs"
            raise RuntimeError(
                f"{func_name}: Expected all {input_or_output} to be real but "
                f"received complex tensor at flattened input idx: {idx}"
            )


def safe_unflatten(tensor, dim: int, shape):
    """``unflatten`` that also handles the zero-dimensional case."""
    if len(shape) == 0:
        if tensor.shape[dim] != 1:
            raise AssertionError("expected a singleton dimension to squeeze")
        return tensor.squeeze(dim)
    return tensor.unflatten(dim, list(shape))


def _vmap(func, in_dims=0, out_dims=0, randomness="error", *, chunk_size=None):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return vmap_impl(func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs)

    return wrapped


# --------------------------------------------------------------------- vjp --


@exposed_in("tensorplay.func")
def vjp(func: Callable, *primals, has_aux: bool = False):
    """Evaluates ``func`` at ``primals`` and returns a function computing the
    vector-Jacobian product.

    Args:
        func (Callable): a Python function taking one or more tensor arguments.
        primals (Tensors): positional arguments to evaluate ``func`` at.  The
            returned function differentiates with respect to all of them.
        has_aux (bool): whether ``func`` returns ``(output, aux)``, where
            ``aux`` is carried through undifferentiated.

    Returns:
        ``(output, vjp_fn)``, or ``(output, vjp_fn, aux)`` when ``has_aux``.
        ``vjp_fn`` takes a cotangent with the same structure as ``output`` and
        returns the gradients with respect to ``primals``.

    Example:

        >>> x = tensorplay.randn(5)
        >>> out, vjp_fn = vjp(tensorplay.sin, x)
        >>> (grad,) = vjp_fn(tensorplay.ones_like(out))
    """
    with transform_increment_nesting():
        return _vjp_with_argnums(func, *primals, has_aux=has_aux)


def _vjp_with_argnums(
    func: Callable,
    *primals,
    argnums: Optional[argnums_t] = None,
    has_aux: bool = False,
):
    aux: Any = None
    if argnums is None:
        keep_graph = _any_requires_grad(primals) or _transform_depth.get() > 1
    else:
        argnums = _normalize_argnums(argnums, len(primals))
        keep_graph = (
            _any_requires_grad(_slice_argnums(primals, argnums))
            or _transform_depth.get() > 1
        )

    with tensorplay.enable_grad():
        if argnums is None:
            diff_primals = _create_differentiable(primals, "vjp(f, *primals)")
            call_args = diff_primals
        else:
            diff_primals = _create_differentiable(
                _slice_argnums(primals, argnums, as_tuple=False), "vjp(f, *primals)"
            )
            call_args = _replace_args(
                primals, _as_tuple_for_replace(diff_primals, argnums), argnums
            )
        primals_out = func(*call_args)

        if has_aux:
            if not (isinstance(primals_out, tuple) and len(primals_out) == 2):
                raise RuntimeError(
                    "vjp(f, *primals): output of function f should be a tuple: "
                    "(output, aux) if has_aux is True"
                )
            primals_out, aux = primals_out
            aux = _undo_create_differentiable(aux, keep_graph)

        flat_primals_out, primals_out_spec = tree_flatten(primals_out)
        assert_non_empty_tensor_output(flat_primals_out, "vjp(f, *primals)")
        flat_diff_primals, primals_spec = tree_flatten(diff_primals)
        results = _undo_create_differentiable(primals_out, keep_graph)

        for primal_out in flat_primals_out:
            if primal_out.is_floating_point() or primal_out.is_complex():
                continue
            raise RuntimeError(
                "vjp(f, ...): All outputs of f must be floating-point or complex "
                f"Tensors, got Tensor with dtype {primal_out.dtype}"
            )

    def wrapper(cotangents, retain_graph: bool = True, create_graph: Optional[bool] = None):
        if create_graph is None:
            create_graph = tensorplay.is_grad_enabled()
        flat_cotangents, cotangents_spec = tree_flatten(cotangents)
        if primals_out_spec != cotangents_spec:
            raise RuntimeError(
                "Expected pytree structure of cotangents to be the same as "
                f"pytree structure of outputs to the function. cotangents: "
                f"{cotangents_spec}, primal output: {primals_out_spec}"
            )
        result = _autograd_grad(
            flat_primals_out,
            flat_diff_primals,
            flat_cotangents,
            retain_graph=retain_graph,
            create_graph=create_graph,
        )
        return tree_unflatten(result, primals_spec)

    if has_aux:
        return results, wrapper, aux
    return results, wrapper


# ------------------------------------------------------------ basis vectors --


def _chunked_standard_basis_for_(tensors, tensor_numels, chunk_size: Optional[int] = None):
    """Yields the identity basis over the concatenated flattened ``tensors``.

    Each yielded chunk holds ``chunk`` rows of that identity, reshaped so that
    row ``i`` has the shape of the tensor it belongs to.  Splitting the basis
    into chunks bounds the peak memory of a Jacobian computation without
    changing its result.
    """
    if len(tensors) != len(tensor_numels):
        raise AssertionError("tensors and tensor_numels must have the same length")
    if len(tensors) == 0:
        raise AssertionError("expected at least one tensor")
    if chunk_size is not None and chunk_size <= 0:
        raise AssertionError("chunk_size must be positive")

    total_numel = sum(tensor_numels)
    if chunk_size is not None and chunk_size < total_numel:
        chunk_numels = get_chunk_sizes(total_numel, chunk_size)
    else:
        chunk_size = total_numel
        chunk_numels = [total_numel]

    starts = []
    running = 0
    for numel in tensor_numels:
        starts.append(running)
        running += numel

    for chunk_idx, chunk_numel in enumerate(chunk_numels):
        row_offset = chunk_idx * chunk_size
        chunks = []
        for tensor, numel, start in zip(tensors, tensor_numels, starts):
            rows = tensorplay.arange(
                row_offset, row_offset + chunk_numel, device=tensor.device
            ).reshape([chunk_numel, 1])
            cols = tensorplay.arange(
                start, start + numel, device=tensor.device
            ).reshape([1, numel])
            block = (rows == cols).to(tensor.dtype)
            chunks.append(block.reshape([chunk_numel] + list(tensor.shape)))
        yield tuple(chunks)


def _construct_standard_basis_for(tensors, tensor_numels):
    """The whole identity basis in one chunk."""
    for chunk in _chunked_standard_basis_for_(tensors, tensor_numels, chunk_size=None):
        return chunk
    raise AssertionError("unreachable: the basis generator always yields once")


def _regroup_jacobians(flat_jac, flat_primals, flat_output, flat_output_numels):
    """Turns one stacked Jacobian per input into one Jacobian per (output, input).

    ``flat_jac[i]`` arrives with shape ``(sum(output numels), *primal_i.shape)``:
    every basis cotangent stacked along the leading axis.  Splitting that axis
    per output and restoring each output's own shape gives the usual
    ``(*out.shape, *in.shape)`` layout.
    """
    per_input = []
    for primal, jac in zip(flat_primals, flat_jac):
        pieces = jac.split(list(flat_output_numels), dim=0)
        per_input.append(
            [
                piece.reshape(list(out.shape) + list(primal.shape))
                for out, piece in zip(flat_output, pieces)
            ]
        )
    return tuple(
        tuple(per_input[i][o] for i in range(len(flat_primals)))
        for o in range(len(flat_output))
    )


# ------------------------------------------------------------------ jacrev --


@exposed_in("tensorplay.func")
def jacrev(
    func: Callable,
    argnums: Union[int, tuple[int, ...]] = 0,
    *,
    has_aux: bool = False,
    chunk_size: Optional[int] = None,
    _preallocate_and_copy: bool = False,
):
    """Returns a function computing the Jacobian of ``func`` by reverse mode.

    Reverse mode costs one backward pass per *output* element, so ``jacrev`` is
    the cheaper direction when the output is smaller than the input.

    Args:
        func (Callable): a Python function returning one or more tensors.
        argnums (int or Tuple[int]): which positional arguments to
            differentiate with respect to.  Default: 0.
        has_aux (bool): whether ``func`` returns ``(output, aux)``.
        chunk_size (int, optional): compute the Jacobian ``chunk_size`` rows at
            a time to bound peak memory.  ``None`` computes it in one shot.

    Example:

        >>> jacobian = jacrev(tensorplay.sin)(tensorplay.randn(5))
        >>> jacobian.shape
        tensorplay.Size(5, 5)
    """
    if not (chunk_size is None or chunk_size > 0):
        raise ValueError("jacrev: `chunk_size` should be greater than 0.")

    @_wraps_without_transform_attrs(func)
    def wrapper_fn(*args):
        with transform_increment_nesting():
            error_if_complex("jacrev", args, is_input=True)
            vjp_out = _vjp_with_argnums(func, *args, argnums=argnums, has_aux=has_aux)
            if has_aux:
                output, vjp_fn, aux = vjp_out
            else:
                output, vjp_fn = vjp_out

            flat_output, output_spec = tree_flatten(output)
            error_if_complex("jacrev", flat_output, is_input=False)
            flat_output_numels = tuple(out.numel() for out in flat_output)

            primals = _slice_argnums(args, argnums)
            flat_primals, primals_spec = tree_flatten(primals)
            keep_graph = _any_requires_grad(flat_primals) or _transform_depth.get() > 1

            def cotangent_pass(basis_chunk):
                basis = tree_unflatten(list(basis_chunk), output_spec)
                return vjp_fn(basis, retain_graph=True, create_graph=keep_graph)

            stacked = None
            for basis_chunk in _chunked_standard_basis_for_(
                flat_output, flat_output_numels, chunk_size=chunk_size
            ):
                if chunk_size == 1:
                    basis = tree_unflatten(
                        [piece.squeeze(0) for piece in basis_chunk], output_spec
                    )
                    chunk_jac = tree_leaves(
                        vjp_fn(basis, retain_graph=True, create_graph=keep_graph)
                    )
                    chunk_jac = [piece.unsqueeze(0) for piece in chunk_jac]
                else:
                    chunk_jac = tree_leaves(_vmap(cotangent_pass)(tuple(basis_chunk)))
                if stacked is None:
                    stacked = [[piece] for piece in chunk_jac]
                else:
                    for slot, piece in zip(stacked, chunk_jac):
                        slot.append(piece)
            if stacked is None:
                raise RuntimeError("jacrev: unable to construct an output basis")
            if _preallocate_and_copy and len(stacked) > 0 and len(stacked[0]) > 1:
                flat_jac = []
                for pieces, primal in zip(stacked, flat_primals):
                    total_rows = sum(piece.shape[0] for piece in pieces)
                    result = tensorplay.zeros(
                        [total_rows] + list(primal.shape),
                        dtype=primal.dtype,
                        device=primal.device,
                    )
                    offset = 0
                    for piece in pieces:
                        width = piece.shape[0]
                        result[offset : offset + width].copy_(piece)
                        offset += width
                    flat_jac.append(result)
            else:
                flat_jac = [
                    pieces[0] if len(pieces) == 1 else tensorplay.cat(pieces, dim=0)
                    for pieces in stacked
                ]

            jac_outs_ins = _regroup_jacobians(
                flat_jac, flat_primals, flat_output, flat_output_numels
            )
            jac_outs_ins = tuple(
                tree_unflatten(list(jac_ins), primals_spec) for jac_ins in jac_outs_ins
            )
            if isinstance(argnums, int):
                jac_outs_ins = tuple(jac_ins[0] for jac_ins in jac_outs_ins)

            result = tree_unflatten(list(jac_outs_ins), output_spec)
            if has_aux:
                return result, aux
            return result

    return wrapper_fn


# --------------------------------------------------------------------- jvp --


@exposed_in("tensorplay.func")
def jvp(
    func: Callable,
    primals: Any,
    tangents: Any,
    *,
    strict: bool = False,
    has_aux: bool = False,
):
    """Evaluates ``func`` at ``primals`` together with its directional
    derivative along ``tangents``.

    Args:
        func (Callable): a Python function taking one or more tensor arguments.
        primals (Tensors): a tuple of positional arguments to evaluate at.
        tangents (Tensors): the direction to differentiate along.  Must have
            the same python structure, shapes and dtypes as ``primals``.
        strict (bool): raise instead of returning zeros when the output turns
            out to be independent of the inputs.
        has_aux (bool): whether ``func`` returns ``(output, aux)``.

    Returns:
        ``(output, jvp_out)``, or ``(output, jvp_out, aux)`` when ``has_aux``.

    Example:

        >>> x = tensorplay.randn(5)
        >>> out, tangent_out = jvp(tensorplay.sin, (x,), (tensorplay.ones(5),))
    """
    if not isinstance(primals, tuple):
        raise RuntimeError(
            f"jvp(f, primals, tangents): Expected primals to be a tuple. "
            f"E.g. it should be valid to call f(*primals)."
        )
    diff_args = primals
    flat_primals, primals_spec = tree_flatten(diff_args)
    flat_tangents, tangents_spec = tree_flatten(tangents)
    if primals_spec != tangents_spec:
        raise RuntimeError(
            "jvp(f, primals, tangents): Expected primals and tangents to have "
            "the same python structure. For example, if primals is a tuple of "
            "3 tensors, tangents also must be. Got primals with structure "
            f"{primals_spec} and tangents with structure {tangents_spec}"
        )
    with transform_increment_nesting():
        return _jvp_with_argnums(
            func,
            *primals,
            tangents=tangents,
            argnums=None,
            strict=strict,
            has_aux=has_aux,
        )


def _jvp_with_argnums(
    func: Callable,
    *primals,
    tangents,
    argnums: Optional[argnums_t] = None,
    strict: bool = False,
    has_aux: bool = False,
):
    """The Jacobian-vector product, obtained from the reverse-mode graph.

    ``v -> J(x)^T v`` is linear in ``v``, so its derivative with respect to
    ``v`` in the direction ``t`` is ``J(x) t`` -- no separate forward rule set
    is needed.  The cotangent placeholders below stand in for ``v``.
    """
    if not isinstance(tangents, tuple):
        raise RuntimeError(
            "jvp(f, primals, tangents): Expected tangents to be a tuple of Tensors"
        )
    aux: Any = None

    if argnums is None:
        diff_primals = primals
    else:
        argnums = _normalize_argnums(argnums, len(primals))
        # as_tuple keeps the structure the same whether argnums is an int or a
        # tuple, so the tangents supplied by jacfwd always line up.
        diff_primals = _slice_argnums(primals, argnums)
    flat_primals, primals_spec = tree_flatten(diff_primals)
    flat_tangents, tangents_spec = tree_flatten(tangents)
    if primals_spec != tangents_spec:
        raise RuntimeError(
            "jvp(f, primals, tangents): Expected primals and tangents to have "
            f"the same python structure. Got primals with structure "
            f"{primals_spec} and tangents with structure {tangents_spec}"
        )
    assert_non_empty_list_of_tensors(flat_primals, "jvp(f, primals, tangents)", "primals")
    assert_non_empty_list_of_tensors(flat_tangents, "jvp(f, primals, tangents)", "tangents")

    keep_graph = (
        _any_requires_grad(flat_primals)
        or _any_requires_grad(flat_tangents)
        or _transform_depth.get() > 1
    )

    for index, (primal, tangent) in enumerate(zip(flat_primals, flat_tangents)):
        if primal.shape != tangent.shape:
            raise RuntimeError(
                f"jvp(f, primals, tangents): tangent at flattened index {index} "
                f"has shape {tangent.shape}, but the corresponding primal has "
                f"shape {primal.shape}"
            )
        if primal.dtype != tangent.dtype:
            raise RuntimeError(
                f"jvp(f, primals, tangents): tangent at flattened index {index} "
                f"has dtype {tangent.dtype}, but the corresponding primal has "
                f"dtype {primal.dtype}"
            )

    with tensorplay.enable_grad():
        flat_diff_primals = list(
            _create_differentiable(flat_primals, "jvp(f, primals, tangents)")
        )
        new_diff = tree_unflatten(flat_diff_primals, primals_spec)
        if argnums is None:
            call_args = new_diff
        else:
            call_args = _replace_args(primals, new_diff, argnums)
        primals_out = func(*call_args)

        if has_aux:
            if not (isinstance(primals_out, tuple) and len(primals_out) == 2):
                raise RuntimeError(
                    "jvp(f, primals, tangents): output of function f should be a "
                    "tuple: (output, aux) if has_aux is True"
                )
            primals_out, aux = primals_out
            aux = _undo_create_differentiable(aux, keep_graph)

        flat_primals_out, primals_out_spec = tree_flatten(primals_out)
        assert_non_empty_tensor_output(flat_primals_out, "jvp(f, primals, tangents)")

        placeholders = tuple(
            tensorplay.zeros_like(out).requires_grad_(True) for out in flat_primals_out
        )
        cotangent_map = _autograd_grad(
            flat_primals_out, flat_diff_primals, placeholders, create_graph=True
        )
        if strict and any(not output.requires_grad for output in flat_primals_out):
            raise RuntimeError(
                "jvp(f, primals, tangents, strict=True): The output of f is "
                "independent of the inputs. This is not allowed with strict=True."
            )
        flat_jvp_out = _autograd_grad(
            cotangent_map, placeholders, flat_tangents, create_graph=keep_graph
        )

        primals_out = _undo_create_differentiable(primals_out, keep_graph)
        jvp_out = _undo_create_differentiable(
            tree_unflatten(list(flat_jvp_out), primals_out_spec), keep_graph
        )

    if has_aux:
        return primals_out, jvp_out, aux
    return primals_out, jvp_out


# ------------------------------------------------------------------ jacfwd --


@exposed_in("tensorplay.func")
def jacfwd(
    func: Callable,
    argnums: argnums_t = 0,
    has_aux: bool = False,
    *,
    randomness: str = "error",
):
    """Returns a function computing the Jacobian of ``func`` by forward mode.

    Forward mode costs one pass per *input* element, so ``jacfwd`` is the
    cheaper direction when the input is smaller than the output -- the mirror
    image of :func:`jacrev`.

    Args:
        func (Callable): a Python function returning one or more tensors.
        argnums (int or Tuple[int]): which positional arguments to
            differentiate with respect to.  Default: 0.
        has_aux (bool): whether ``func`` returns ``(output, aux)``.
        randomness (str): how the underlying map treats random operations;
            one of ``"error"``, ``"different"`` or ``"same"``.

    Example:

        >>> jacobian = jacfwd(tensorplay.sin)(tensorplay.randn(5))
    """

    @_wraps_without_transform_attrs(func)
    def wrapper_fn(*args):
        with transform_increment_nesting():
            error_if_complex("jacfwd", args, is_input=True)
            primals = args if argnums is None else _slice_argnums(args, argnums)
            flat_primals, primals_spec = tree_flatten(primals)
            flat_primals_numels = tuple(primal.numel() for primal in flat_primals)
            flat_basis = _construct_standard_basis_for(flat_primals, flat_primals_numels)
            basis = tree_unflatten(list(flat_basis), primals_spec)

            def push_jvp(basis):
                output = _jvp_with_argnums(
                    func, *args, tangents=basis, argnums=argnums, has_aux=has_aux
                )
                if has_aux:
                    _, jvp_out, aux = output
                    return jvp_out, aux
                _, jvp_out = output
                return jvp_out

            results = _vmap(push_jvp, randomness=randomness)(basis)
            if has_aux:
                results, aux = results
                aux = tree_map(lambda first_el: first_el[0], aux)

            jac_outs, spec = tree_flatten(results)
            error_if_complex("jacfwd", jac_outs, is_input=False)
            jac_outs_ins = tuple(
                tuple(
                    safe_unflatten(jac_out_in, -1, primal.shape)
                    for primal, jac_out_in in zip(
                        flat_primals,
                        jac_out.movedim(0, -1).split(list(flat_primals_numels), dim=-1),
                    )
                )
                for jac_out in jac_outs
            )
            jac_outs_ins = tuple(
                tree_unflatten(list(jac_ins), primals_spec) for jac_ins in jac_outs_ins
            )
            if isinstance(argnums, int):
                jac_outs_ins = tuple(jac_ins[0] for jac_ins in jac_outs_ins)

            result = tree_unflatten(list(jac_outs_ins), spec)
            if has_aux:
                return result, aux
            return result

    return wrapper_fn


# ----------------------------------------------------------------- hessian --


@exposed_in("tensorplay.func")
def hessian(func: Callable, argnums: argnums_t = 0):
    """Returns a function computing the Hessian of ``func``.

    Composing forward mode over reverse mode is the cheaper ordering for the
    square Jacobian-of-a-gradient that a Hessian is.

    Example:

        >>> def f(x):
        ...     return x.sin().sum()
        >>> hess = hessian(f)(tensorplay.randn(5))
        >>> hess.shape
        tensorplay.Size(5, 5)
    """
    return jacfwd(jacrev(func, argnums), argnums)


# --------------------------------------------------------------- linearize --


@exposed_in("tensorplay.func")
def linearize(func: Callable, *primals) -> tuple[Any, Callable]:
    """Evaluates ``func`` at ``primals`` and returns its linear approximation
    there.

    The primal evaluation happens once, here; the returned ``jvp_fn`` reuses
    that linearization, so applying it to many tangents costs only the tangent
    passes.  Use it in place of repeated :func:`jvp` calls at a fixed point.

    Returns:
        ``(output, jvp_fn)``.  ``jvp_fn`` takes tangents matching the structure
        of ``primals`` and returns the directional derivative of the output.

    Example:

        >>> x = tensorplay.randn(5)
        >>> out, jvp_fn = linearize(tensorplay.sin, x)
        >>> tangent_out = jvp_fn(tensorplay.ones(5))
    """
    flat_primals, primals_spec = tree_flatten(primals)
    assert_non_empty_list_of_tensors(flat_primals, "linearize(f, *primals)", "primals")
    keep_graph = _any_requires_grad(flat_primals)

    with tensorplay.enable_grad():
        flat_diff_primals = list(
            _create_differentiable(flat_primals, "linearize(f, *primals)")
        )
        primals_out = func(*tree_unflatten(flat_diff_primals, primals_spec))
        flat_primals_out, primals_out_spec = tree_flatten(primals_out)
        assert_non_empty_tensor_output(flat_primals_out, "linearize(f, *primals)")
        placeholders = tuple(
            tensorplay.zeros_like(out).requires_grad_(True) for out in flat_primals_out
        )
        cotangent_map = _autograd_grad(
            flat_primals_out, flat_diff_primals, placeholders, create_graph=True
        )
    output = _undo_create_differentiable(primals_out, keep_graph)

    def jvp_fn(*tangents):
        flat_tangents, tangents_spec = tree_flatten(tangents)
        if tangents_spec != primals_spec:
            raise RuntimeError(
                "linearize(f, *primals)(*tangents): Expected tangents to have "
                f"the same python structure as primals. Got primals with "
                f"structure {primals_spec} and tangents with structure "
                f"{tangents_spec}"
            )
        for primal, tangent in zip(flat_primals, flat_tangents):
            if primal.shape != tangent.shape:
                raise RuntimeError(
                    "linearize(f, *primals)(*tangents): Expected tangents to "
                    f"have the same shape as primals. Got shape {tangent.shape} "
                    f"for a primal of shape {primal.shape}"
                )
        with tensorplay.enable_grad():
            flat_jvp_out = _autograd_grad(
                cotangent_map,
                placeholders,
                flat_tangents,
                retain_graph=True,
                create_graph=keep_graph,
            )
        return _undo_create_differentiable(
            tree_unflatten(list(flat_jvp_out), primals_out_spec), keep_graph
        )

    return output, jvp_fn


# ---------------------------------------------------------- functionalize --


@exposed_in("tensorplay.func")
def functionalize(func: Callable, *, remove: str = "mutations") -> Callable:
    """Returns a version of ``func`` that leaves its arguments untouched.

    ``func`` may write into its inputs in place; the wrapper hands it private
    copies instead, so the caller's tensors -- and any views onto them -- come
    back exactly as they went in, while the returned value is unchanged.

    Args:
        func (Callable): the function to make side-effect free.
        remove (str): ``"mutations"`` copies every tensor argument.
            ``"mutations_and_views"`` additionally detaches the copies from any
            aliasing they arrived with, so writes through a view of an argument
            cannot reach another argument either.

    Example:

        >>> def f(x):
        ...     x.add_(1)
        ...     return x
        >>> x = tensorplay.zeros(3)
        >>> out = functionalize(f)(x)
        >>> x  # unchanged
        tensor([0., 0., 0.])
    """
    if remove not in ("mutations", "mutations_and_views"):
        raise RuntimeError(
            f"functionalize(f, remove='{remove}'): remove must be one of "
            f"'mutations' or 'mutations_and_views'"
        )

    def copy_leaf(value):
        if not isinstance(value, tensorplay.Tensor):
            return value
        if remove == "mutations_and_views":
            # A contiguous copy shares storage with nothing, so writes through
            # one argument can no longer be observed through another.
            return value.detach().clone().contiguous().requires_grad_(value.requires_grad)
        return value.clone()

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return func(*tree_map(copy_leaf, args), **tree_map(copy_leaf, kwargs))

    return wrapped


@exposed_in("tensorplay.func")
def debug_unwrap(tensor: Any, *, recurse: bool = True) -> Any:
    """Returns the plain tensor underlying a transform's temporary wrapper.

    Transforms in this build hand ordinary tensors to the function they wrap,
    so there is nothing to strip and the argument is returned unchanged.  The
    entry point exists so that debugging code written against it keeps working
    if a wrapper representation is introduced later.
    """
    del recurse
    return tensor


# ------------------------------------------------------------------- grad --


def grad_and_value_impl(func: Callable, argnums, has_aux, args, kwargs) -> Callable:
    with tensorplay.enable_grad():
        argnums = _normalize_argnums(argnums, len(args))
        diff_args = _slice_argnums(args, argnums, as_tuple=False)
        keep_graph = _any_requires_grad(diff_args)
        diff_args = _create_differentiable(diff_args, "grad")
        args = _replace_args(args, _as_tuple_for_replace(diff_args, argnums), argnums)

        output = func(*args, **kwargs)
        aux: Any = None
        if has_aux:
            if not (isinstance(output, tuple) and len(output) == 2):
                raise RuntimeError(
                    "grad_and_value(f)(*args): output of function f should be a "
                    "tuple: (output, aux) if has_aux is True"
                )
            output, aux = output

        if not isinstance(output, tensorplay.Tensor):
            raise RuntimeError(
                "grad_and_value(f)(*args): Expected f(*args) to return a Tensor, "
                f"got {type(output)}"
            )
        if output.dim() != 0:
            raise RuntimeError(
                "grad_and_value(f)(*args): Expected f(*args) to return a scalar "
                f"Tensor, got tensor with {output.dim()} dims. Maybe you wanted "
                "to use the vjp or jacrev APIs instead?"
            )

        flat_diff_args, spec = tree_flatten(diff_args)
        flat_grad_input = _autograd_grad((output,), flat_diff_args, create_graph=True)
        grad_input = tree_unflatten(list(flat_grad_input), spec)

        grad_input = _undo_create_differentiable(grad_input, keep_graph)
        output = _undo_create_differentiable(output, keep_graph)
        if has_aux:
            aux = _undo_create_differentiable(aux, keep_graph)

    if has_aux:
        return grad_input, (output, aux)
    return grad_input, output


def grad_impl(func: Callable, argnums, has_aux, args, kwargs):
    results = grad_and_value_impl(func, argnums, has_aux, args, kwargs)
    if has_aux:
        grad_input, (_, aux) = results
        return grad_input, aux
    grad_input, _ = results
    return grad_input
