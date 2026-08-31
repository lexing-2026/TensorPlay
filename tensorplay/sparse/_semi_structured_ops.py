"""Python dispatch operations for semi-structured sparse values."""

from __future__ import annotations

import contextlib
from typing import Any

__all__ = [
    "fallback_dispatcher",
    "semi_sparse_addmm",
    "semi_sparse_clone",
    "semi_sparse_detach",
    "semi_sparse_detach_",
    "semi_sparse_indices",
    "semi_sparse_is_same_size",
    "semi_sparse_linear",
    "semi_sparse_mm",
    "semi_sparse_scaled_mm",
    "semi_sparse_t",
    "semi_sparse_to",
    "semi_sparse_to_copy",
    "semi_sparse_to_dense",
    "semi_sparse_transpose",
    "semi_sparse_values",
    "semi_sparse_view",
]


@contextlib.contextmanager
def no_dispatch():
    yield


def fallback_dispatcher(
    func: Any,
    types: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> Any:
    del types
    if not callable(func):
        raise NotImplementedError("the requested operation has no fallback")
    with no_dispatch():
        return func(*args, **(kwargs or {}))


def _require(value: Any) -> Any:
    if not hasattr(value, "packed") or not hasattr(value, "shape"):
        raise TypeError(f"expected a semi-structured value, got {type(value).__name__}")
    return value


def _is_semi(value: Any) -> bool:
    return hasattr(value, "packed") and callable(getattr(value, "_mm", None))


def _value_at(
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    index: int,
    *names: str,
    default: Any = None,
) -> Any:
    if index < len(args):
        return args[index]
    values = kwargs or {}
    for name in names:
        if name in values:
            return values[name]
    return default


def _scalar_kwargs(kwargs: dict[str, Any] | None) -> tuple[Any, Any]:
    values = kwargs or {}
    return values.get("alpha", 1), values.get("beta", 1)


def semi_sparse_values(
    func: Any,
    types: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Any:
    del func, types, kwargs
    if len(args) != 1:
        raise ValueError("values expects one argument")
    value = _require(args[0])
    if value.packed is None:
        raise ValueError("the compressed value has no packed storage")
    if value.meta is None:
        rows, cols = value.shape
        kept = rows * cols // 2
        return value.packed.view(-1)[:kept].view(rows, -1)
    return value.packed.detach()


def semi_sparse_indices(
    func: Any,
    types: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Any:
    del func, types, kwargs
    if len(args) != 1:
        raise ValueError("indices expects one argument")
    value = _require(args[0])
    if value.packed is None:
        raise ValueError("the compressed value has no packed storage")
    if value.meta is None:
        rows, cols = value.shape
        kept = rows * cols // 2
        return value.packed.view(-1)[kept:].view(rows, -1)
    return value.meta


def semi_sparse_is_same_size(
    func: Any,
    types: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> bool:
    del func, types
    if len(args) != 2:
        raise ValueError("is_same_size expects two arguments")
    left = _require(args[0])
    right = _value_at(args, kwargs, 1, "other", "input")
    if right is None or not hasattr(right, "shape"):
        raise TypeError("is_same_size expects tensor-like arguments")
    return tuple(left.shape) == tuple(right.shape)


def semi_sparse_t(
    func: Any,
    types: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Any:
    del func, types, kwargs
    if len(args) != 1:
        raise ValueError("t expects one argument")
    return _require(args[0]).t()


def semi_sparse_transpose(
    func: Any,
    types: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Any:
    del func, types
    value = _require(_value_at(args, kwargs, 0, "self", "input"))
    dim0 = _value_at(args, kwargs, 1, "dim0")
    dim1 = _value_at(args, kwargs, 2, "dim1")
    if dim0 is None or dim1 is None:
        raise ValueError("transpose expects a value and two dimensions")
    return value.transpose(dim0, dim1)


def semi_sparse_view(
    func: Any,
    types: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Any:
    del func, types
    value = _require(_value_at(args, kwargs, 0, "self", "input"))
    shape = _value_at(args, kwargs, 1, "shape")
    if shape is None:
        raise ValueError("view expects a value and shape")
    return value.view(shape)


def semi_sparse_detach(
    func: Any,
    types: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> Any:
    del func, types, kwargs
    if len(args) != 1:
        raise ValueError("detach expects one argument")
    return _require(args[0]).detach()


def semi_sparse_detach_(
    func: Any,
    types: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> Any:
    del func, types, kwargs
    if len(args) != 1:
        raise ValueError("detach_ expects one argument")
    value = _require(args[0])
    value.requires_grad = False
    return value


def semi_sparse_mm(
    func: Any,
    types: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Any:
    del func, types
    if (kwargs or {}).get("out") is not None:
        raise NotImplementedError("out is not supported for semi-structured matmul")
    left = _value_at(args, kwargs, 0, "self", "input", "mat1")
    right = _value_at(args, kwargs, 1, "mat2", "other")
    if left is None or right is None:
        raise ValueError("mm expects two arguments")
    if left.dim() != 2 or right.dim() != 2:
        raise NotImplementedError("semi-structured matmul only supports 2-D operands")
    if _is_semi(left):
        return left._mm(right)
    if _is_semi(right):
        return right.t()._mm(left, should_transpose_dense=True)
    raise TypeError("one matrix operand must be semi-structured")


def semi_sparse_addmm(
    func: Any,
    types: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Any:
    del func, types
    bias = _value_at(args, kwargs, 0, "input", "self")
    left = _value_at(args, kwargs, 1, "mat1")
    right = _value_at(args, kwargs, 2, "mat2")
    if bias is None or left is None or right is None:
        raise ValueError("addmm expects input, mat1, and mat2")
    if left.dim() != 2 or right.dim() != 2:
        raise NotImplementedError("semi-structured matmul only supports 2-D operands")
    if bias.dim() != 1:
        raise NotImplementedError(
            "semi-structured addmm only supports a one-dimensional bias"
        )
    alpha, beta = _scalar_kwargs(kwargs)
    if _is_semi(left) and _is_semi(right):
        raise ValueError("two compressed operands are not supported")
    if _is_semi(left):
        return left._mm(right, bias=bias, alpha=alpha, beta=beta)
    if _is_semi(right):
        return right.t()._mm(
            left,
            bias=bias,
            should_transpose_dense=True,
            alpha=alpha,
            beta=beta,
        )
    raise TypeError("one matrix operand must be semi-structured")


def semi_sparse_linear(
    func: Any,
    types: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Any:
    del func, types
    value = _value_at(args, kwargs, 0, "input")
    weight = _value_at(args, kwargs, 1, "weight")
    bias = _value_at(args, kwargs, 2, "bias")
    if value is None or weight is None:
        raise ValueError("linear expects input and weight")
    if not _is_semi(weight):
        raise NotImplementedError("linear requires a semi-structured weight")
    return weight._linear(value, bias)


def semi_sparse_scaled_mm(
    func: Any,
    types: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Any:
    del func, types
    values = list(args)
    if len(values) < 7:
        names = (
            "self",
            "other",
            "self_scale",
            "other_scale",
            "bias",
            "scale_result",
            "out_dtype",
        )
        values.extend(
            _value_at(args, kwargs, index, names[index])
            for index in range(len(values), len(names))
        )
    if len(values) < 7:
        raise ValueError("scaled matrix multiplication requires seven arguments")
    left, right, left_scale, right_scale, bias, result_scale, out_dtype = values[:7]
    del bias, result_scale
    if not _is_semi(left):
        raise TypeError("the first operand must be semi-structured")
    result = left._mm(right)
    result = result * left_scale * right_scale
    return result if out_dtype is None else result.to(dtype=out_dtype)


def semi_sparse_clone(
    func: Any,
    types: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Any:
    del func, types
    value = _require(_value_at(args, kwargs, 0, "self", "input"))
    return value.clone(**(kwargs or {}))


def semi_sparse_to_copy(
    func: Any,
    types: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> Any:
    del func, types
    if not args:
        raise ValueError("to_copy expects one argument")
    value = _require(args[0])
    return value.to(*args[1:], **(kwargs or {}))


def semi_sparse_to(
    func: Any,
    types: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> Any:
    return semi_sparse_to_copy(func, types, args, kwargs)


def semi_sparse_to_dense(
    func: Any,
    types: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Any:
    del func, types, kwargs
    if len(args) != 1:
        raise ValueError("to_dense expects one argument")
    return _require(args[0]).to_dense()
