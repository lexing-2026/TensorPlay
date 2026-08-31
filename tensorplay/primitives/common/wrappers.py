# mypy: allow-untyped-defs
from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, TypeVar

import tensorplay
from . import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    TensorLike,
    Number,
    ShapeType,
    check,
    dtype_to_type_ctor,
    elementwise_dtypes,
    extract_shape,
    is_cpu_scalar_tensor,
    same_shape,
    type_to_dtype,
)

__all__ = [
    "backwards_not_supported",
    "elementwise_unary_scalar_wrapper",
    "out_wrapper",
]


def _maybe_convert_to_dtype(a, dtype):
    if isinstance(a, TensorLike):
        if a.dtype != dtype:
            return a.to(dtype)
        return a
    if isinstance(a, Number):
        return dtype_to_type_ctor(dtype)(a)
    if isinstance(a, Sequence):
        return tuple(_maybe_convert_to_dtype(x, dtype) for x in a)
    if a is None:
        return None
    raise ValueError(
        f"Received unsupported type {type(a)}. Expected TensorLike, Number, or Sequence."
    )


def _resize_output_check(out, shape):
    if same_shape(out.shape, shape):
        return False
    if out.numel() != 0:
        warnings.warn(
            f"An output with one or more elements was resized since it had shape "
            f"{str(out.shape)} which does not match the required output shape "
            f"{str(shape)}. This behavior is deprecated, and in a future release "
            f"outputs will not be resized unless they have zero elements.",
            stacklevel=2,
        )
    return True


def _maybe_resize_out(out, shape, memory_format=None):
    if _resize_output_check(out, shape):
        return out.resize_(shape)
    return out


def is_cpu_scalar(x) -> bool:
    return x.dim() == 0 and x.device.type == "cpu"


def check_copy_devices(*, copy_from, copy_to) -> None:
    if copy_from.device != copy_to.device:
        raise RuntimeError(
            f"Attempting to copy from device {copy_from.device} to device "
            f"{copy_to.device}, but cross-device copies are not allowed!"
        )


def _safe_copy_out(*, copy_from, copy_to, exact_dtype: bool = False):
    if not is_cpu_scalar_tensor(copy_from):
        check_copy_devices(copy_from=copy_from, copy_to=copy_to)
    if exact_dtype:
        check(
            copy_from.dtype == copy_to.dtype,
            lambda: f"Expected out tensor to have dtype {copy_from.dtype} "
            f"but got {copy_to.dtype} instead",
        )
    else:
        from . import can_safe_cast_to

        check(
            can_safe_cast_to(cast_from=copy_from.dtype, cast_to=copy_to.dtype),
            lambda: f"Attempting to cast from {copy_from.dtype} to out tensor with "
            f"dtype {copy_to.dtype}, but this can't be cast because it is not safe!",
        )
    return copy_to.copy_(copy_from)


def out_wrapper(
    *out_dtype_matches_op_input,
    exact_dtype: bool = False,
    pass_is_out: bool = False,
    out_dtype: Any = None,
    out_layout: Any = None,
    out_device: Any = None,
):
    """Decorates an operation with ``out`` keyword handling.

    The wrapped function gains an ``out=`` keyword that accepts either a
    single tensor or a sequence of tensors. Outputs are allocated from the
    metadata described by the forwarded ``shape``/``dtype`` arguments (taken
    from the operation signature) whenever no ``out`` is supplied.
    """
    allow_ops_with_scalar_tensors = False

    def _out_wrapper(fn: Callable) -> Callable:
        sig = inspect_signature(fn)
        result_shape_param = "shape" if "shape" in sig.parameters else "size"
        out_dtype_param = None
        if out_dtype is not None:
            out_dtype_param = out_dtype
        elif out_dtype_matches_op_input:
            out_dtype_param = "dtype"

        @wraps(fn)
        def _fn(*args, out=None, **kwargs):
            if out is not None:
                # Single tensor or a (already consolidated) sequence of tensors.
                outputs = out if isinstance(out, (tuple, list)) else (out,)
                if len(outputs) != 1:
                    raise RuntimeError(
                        f"Expected {1} outputs, but got {len(outputs)} outputs!"
                    )
                shape = kwargs.get(result_shape_param, None)
                if shape is None:
                    bound = sig.bind_partial(*args, **kwargs)
                    shape = bound.arguments.get(result_shape_param, None)
                result = fn(*args, **kwargs)
                if isinstance(result, (tuple, list)):
                    result = result[0]
                _safe_copy_out(copy_from=result, copy_to=outputs[0], exact_dtype=exact_dtype)
                if pass_is_out:
                    return (result, outputs[0])
                return outputs[0]

            # Compute the output shape for allocation from the op arguments.
            bound = sig.bind_partial(*args, **kwargs)
            shape = bound.arguments.get(result_shape_param, None)
            if shape is None:
                tensors = [a for a in args if isinstance(a, TensorLike)]
                shape = extract_shape(*tensors, allow_cpu_scalar_tensors=True)

            dtype = None
            if out_dtype_param is not None:
                dtype = (
                    bound.arguments.get(out_dtype_param)
                    or kwargs.get("dtype")
                )
            if dtype is None:
                tensors = [
                    a for a in args if isinstance(a, TensorLike)
                    and not (allow_ops_with_scalar_tensors and is_cpu_scalar_tensor(a))
                ]
                if tensors:
                    dtype = tensors[0].dtype
            if dtype is None:
                dtype = tensorplay.get_default_dtype()

            out_t = tensorplay.empty(shape, dtype=dtype, device=out_device)
            result = fn(*args, out=out_t, **kwargs)
            if pass_is_out:
                return result, out_t
            return result

        _fn.__signature__ = sig  # type: ignore[attr-defined]
        _fn._torch_decompositions_out_wrapper = True  # type: ignore[attr-defined]
        return _fn

    return _out_wrapper


def inspect_signature(fn):
    import inspect

    return inspect.signature(fn)


def _maybe_remove_out_wrapper(fn: Callable):
    import inspect

    return inspect.unwrap(
        fn,
        stop=lambda f: not hasattr(f, "_torch_decompositions_out_wrapper"),
    )


def backwards_not_supported(prim):
    """Wraps a primitive into an autograd boundary that rejects backward."""

    class _BackwardsNotSupported:
        pass

    @wraps(prim)
    def _autograd_impl(*args, **kwargs):
        return prim(*args, **kwargs)

    return _autograd_impl


def elementwise_unary_scalar_wrapper(fn):
    """Allows unary operators that accept tensors to work with Python numbers."""

    @wraps(fn)
    def _fn(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], Number):
            dtype = type_to_dtype(type(args[0]))
            args_ = list(args)
            args_[0] = tensorplay.tensor(args[0], dtype=dtype)
            result = fn(*args_, **kwargs)
            if not isinstance(result, tensorplay.Tensor):
                raise AssertionError(f"Expected tensor, got {type(result)}")
            return result.item()
        return fn(*args, **kwargs)

    return _fn
