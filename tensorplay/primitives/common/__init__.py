# mypy: allow-untyped-defs
"""Shared utilities for the primitives subsystem: dtype relations, layout
helpers, shape/stride validation, and the elementwise type-promotion rules
used by the primitive and reference operation layers.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from enum import Enum
from functools import reduce
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, TypeGuard, TypeVar, Union

import tensorplay

_T = TypeVar("_T")
_S = TypeVar("_S")

Tensor: TypeAlias = tensorplay.Tensor
dtype: TypeAlias = tensorplay.dtype
device: TypeAlias = tensorplay.device

ShapeType: TypeAlias = Union[tensorplay.Size, list[int], tuple[int, ...]]
StrideType: TypeAlias = Union[list[int], tuple[int, ...]]
DimsType: TypeAlias = Union[int, list[int], tuple[int, ...]]
DimsSequenceType: TypeAlias = Union[list[int], tuple[int, ...]]
NumberTypeType: TypeAlias = Union[type[bool], type[int], type[float], type[complex]]
NumberType: TypeAlias = Union[bool, int, float, complex]
RealNumberType: TypeAlias = Union[bool, int, float]

Number = (bool, int, float, complex)
Dim = int
IntWithoutSymInt = int
FloatWithoutSymFloat = float
BoolLikeType: TypeAlias = Union[bool, int]
DeviceLikeType: TypeAlias = Union[str, tensorplay.device, int]
TensorLike: TypeAlias = Tensor
TensorLikeType: TypeAlias = Tensor
TensorSequenceType: TypeAlias = Sequence[Tensor]

_integer_dtypes = (
    tensorplay.uint8, tensorplay.uint16, tensorplay.uint32, tensorplay.uint64,
    tensorplay.int8, tensorplay.int16, tensorplay.int32, tensorplay.int64,
)
_low_precision_dtypes = (
    tensorplay.float16, tensorplay.bfloat16,
    tensorplay.complex32, tensorplay.bcomplex32,
)
_complex_dtypes = (
    tensorplay.complex32, tensorplay.bcomplex32,
    tensorplay.complex64, tensorplay.complex128,
)
_float_dtypes = (
    tensorplay.float16, tensorplay.bfloat16, tensorplay.float32, tensorplay.float64,
)
_boolean_dtypes = (tensorplay.bool,)
# Partial order of dtypes from lowest to highest; siblings within one tier
# are unordered relative to each other.
_ordered_dtypes = (
    (tensorplay.bool,),
    (tensorplay.uint8, tensorplay.int8),
    (tensorplay.int16,),
    (tensorplay.int32,),
    (tensorplay.int64,),
    (tensorplay.float16, tensorplay.bfloat16),
    (tensorplay.float32,),
    (tensorplay.float64,),
    (tensorplay.complex32, tensorplay.bcomplex32),
    (tensorplay.complex64,),
    (tensorplay.complex128,),
)
_ordered_types = (bool, int, float, complex)


def check(b: bool, s: Callable[[], str] | str, exc_type: type[Exception] = RuntimeError) -> None:
    """Raise ``exc_type`` with message ``s`` when the condition fails.

    The message may be a callable producing a string, deferring formatting
    work to the failure case.
    """
    if not b:
        msg = s() if callable(s) else s
        raise exc_type(msg)


def _maybe_get_pytype(t):
    return t


def same_shape(a: ShapeType, b: ShapeType, *, allow_rhs_unbacked: bool = False) -> bool:
    if len(a) != len(b):
        return False
    return all(x == y for x, y in zip(a, b))
def compare_tensor_meta(
    a, b, check_sizes=True, check_strides=False, *,
    allow_rhs_unbacked: bool = False, check_conj: bool = True,
) -> None:
    """Checks that two tensor likes have the same shape, dtype and device."""
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")
    if not isinstance(b, TensorLike):
        raise AssertionError(f"b must be TensorLike, got {type(b)}")
    if check_sizes and not same_shape(a.shape, b.shape):
        raise AssertionError(f"Shapes {a.shape} and {b.shape} are not equal!")
    if a.dtype != b.dtype:
        raise AssertionError(f"Dtypes {a.dtype} and {b.dtype} are not equal!")
    same_device = a.device == b.device or (
        a.device.type == b.device.type == "cuda"
        and str(a.device).split(":")[0] == str(b.device).split(":")[0]
    )
    if not same_device:
        raise AssertionError(f"Devices {a.device} and {b.device} are not equal!")
    if check_strides:
        same_strides, idx = check_significant_strides(a, b, only_cuda=False)
        if not same_strides:
            raise AssertionError(
                f"Stride mismatch! {a.stride()} vs {b.stride()} (at {idx})!")
        if a.storage_offset() != b.storage_offset():
            raise AssertionError(
                f"Storage offset mismatch! {a.storage_offset()} vs {b.storage_offset()}!")
    if check_conj and hasattr(a, "is_conj") and a.is_conj() != b.is_conj():
        raise AssertionError(f"Conj mismatch! {a.is_conj()} vs {b.is_conj()}")
    if hasattr(a, "is_neg") and a.is_neg() != b.is_neg():
        raise AssertionError(f"Neg mismatch! {a.is_neg()} vs {b.is_neg()}")


def _check_strides_helper(a, b, *, only_cuda=True, significant_only=True,
                          allow_rhs_unbacked=False) -> tuple[bool, int | None]:
    # Only compares strides that are meaningful: dimensions of length > 1
    # on tensors with more than one element.
    if (not only_cuda or a.device.type == "cuda" or b.device.type == "cuda") and a.numel() > 0:
        for idx in range(a.ndim):
            check_dim = not significant_only or a.shape[idx] > 1
            if a.stride()[idx] != b.stride()[idx] and check_dim:
                return False, idx
    return True, None


def check_significant_strides(a, b, *, only_cuda=True, allow_rhs_unbacked=False):
    return _check_strides_helper(a, b, only_cuda=only_cuda, significant_only=True,
                                 allow_rhs_unbacked=allow_rhs_unbacked)


def check_all_strides(a, b, *, only_cuda=True):
    return _check_strides_helper(a, b, only_cuda=only_cuda, significant_only=False)


def check_contiguous_sizes_strides(sizes, strides, false_if_dde: bool = False) -> bool:
    """Accepts both spellings of the contiguous layout: a length-1 dimension
    may report the next-lower dimension's stride."""
    expected_stride = 1
    expected_stride_max = 1
    for x, y in reversed(tuple(zip(sizes, strides))):
        if x == 1:
            continue
        if y != expected_stride and y != expected_stride_max:
            return False
        expected_stride_max *= max(x, 1)
        expected_stride *= x
    return True


def is_contiguous(a: TensorLikeType, false_if_dde: bool = False) -> bool:
    return a.is_contiguous()


def is_channels_last_contiguous_2d(a: Tensor, false_if_dde: bool = False) -> bool:
    return a.ndim == 4 and tuple(a.stride()) == make_channels_last_2d_strides_for(a.shape)


def is_channels_last_contiguous_3d(a: Tensor, false_if_dde: bool = False) -> bool:
    return a.ndim == 5 and tuple(a.stride()) == make_channels_last_3d_strides_for(a.shape)


def validate_memory_format(memory_format: Any) -> None:
    fmt = getattr(memory_format, "name", memory_format)
    valid = ("contiguous", "channels_last", "channels_last_3d", "preserve_format",
             tensorplay.contiguous_format, tensorplay.channels_last,
             tensorplay.channels_last_3d)
    if fmt not in valid:
        raise ValueError(f"Invalid memory format {memory_format}!")


def is_contiguous_for_memory_format(a: Tensor, memory_format: Any = "contiguous") -> bool:
    fmt = getattr(memory_format, "name", memory_format)
    if fmt in ("contiguous", "preserve_format"):
        return is_contiguous(a)
    if fmt == "channels_last":
        return is_channels_last_contiguous_2d(a)
    if fmt == "channels_last_3d":
        return is_channels_last_contiguous_3d(a)
    raise NotImplementedError(f"Unsupported memory format {memory_format}")


def is_contiguous_or_false(a) -> bool:
    return is_contiguous(a, false_if_dde=True)


def is_channels_last_contiguous_or_false_2d(a: Tensor) -> bool:
    return is_channels_last_contiguous_2d(a, false_if_dde=True)


def is_channels_last_contiguous_or_false_3d(a: Tensor) -> bool:
    return is_channels_last_contiguous_3d(a, false_if_dde=True)


def is_contiguous_for_memory_format_or_false(a: Tensor, memory_format: Any) -> bool:
    try:
        return is_contiguous_for_memory_format(a, memory_format)
    except (NotImplementedError, ValueError):
        return False


def is_channels_last_contiguous(a: Tensor) -> bool:
    return is_channels_last_contiguous_2d(a)


def is_channels_last_contiguous_or_false(a: Tensor) -> bool:
    return is_channels_last_contiguous_or_false_2d(a)


class K(NamedTuple):
    L: Any
    p: Any


def _is_non_overlapping_and_dense_or_false(sizes, strides) -> bool:
    if len(sizes) == 0:
        return True
    if len(sizes) != len(strides):
        return False
    if any(stride < 1 for stride in strides):
        return False
    sorted_pairs = sorted(
        ((s, stride) for s, stride in zip(sizes, strides) if s != 1),
        key=lambda x: x[1],
    )
    expected = 1
    for size, stride in sorted_pairs:
        if stride != expected:
            return False
        expected *= size
    return True


def is_non_overlapping_and_dense_or_false(a: Tensor) -> bool:
    return _is_non_overlapping_and_dense_or_false(a.shape, a.stride())


def compute_elementwise_output_logical_to_physical_perm(*tensors, _skip_checks: bool = False):
    """Returns a permutation from the logical output layout to physical memory.

    Dimensions are ordered by how many participating tensors prefer stride-1
    placement along them; dimensions where more tensors are contiguous come
    first (they stay innermost physically). Returns
    ``(perm, has_ambiguous_order)``.
    """
    if not _skip_checks:
        check_same_shape(*tensors, allow_cpu_scalar_tensors=True)
    tensors = tuple(
        a for a in tensors if isinstance(a, TensorLike) and not is_cpu_scalar_tensor(a)
    )
    if len(tensors) == 0:
        return (), False
    ndim = tensors[0].ndim
    if ndim == 0:
        return (), False
    if ndim == 1:
        return [0], False

    def should_swap(x: int, y: int) -> int:
        # -1 keeps the order, 0 is ambiguous, 1 swaps. Dimensions where more
        # tensors have stride 1 are preferred in innermost positions.
        x_pref = sum(1 for t in tensors if t.stride()[x] == 1)
        y_pref = sum(1 for t in tensors if t.stride()[y] == 1)
        if x_pref != y_pref:
            return 1 if x_pref < y_pref else -1
        return 0

    perm = list(range(ndim))
    for i in range(1, ndim):  # stable insertion sort on the predicate
        key = perm[i]
        j = i - 1
        while j >= 0 and should_swap(perm[j], key) == 1:
            perm[j + 1] = perm[j]
            j -= 1
        perm[j + 1] = key

    ambiguous = any(
        should_swap(perm[i], perm[j]) != -1
        for i, j in zip(range(ndim - 1), range(1, ndim))
    )
    return list(reversed(perm)), ambiguous


def compute_elementwise_output_strides(*tensors) -> tuple[int, ...]:
    """Computes the output strides for elementwise operations."""
    if len(tensors) == 0:
        raise ValueError("Can't compute elementwise output strides for zero tensors!")
    check_same_shape(*tensors, allow_cpu_scalar_tensors=True)
    tensors = tuple(
        a for a in tensors if isinstance(a, TensorLike) and not is_cpu_scalar_tensor(a)
    )
    if len(tensors) == 0:
        return ()
    ndim = tensors[0].ndim
    shape = tensors[0].shape
    if ndim == 0:
        return ()
    if ndim == 1:
        return (1,)
    if len(tensors) == 1:
        if is_non_overlapping_and_dense_or_false(tensors[0]):
            return tuple(tensors[0].stride())
        return tuple(tensorplay.empty_like(tensors[0]).stride())
    logical_to_physical_perm, _ = compute_elementwise_output_logical_to_physical_perm(
        *tensors, _skip_checks=True
    )
    permuted_shape = apply_perm(shape, logical_to_physical_perm)
    new_strides = make_contiguous_strides_for(permuted_shape)
    permuted_strides = apply_perm(new_strides, invert_perm(logical_to_physical_perm))
    return tuple(permuted_strides)


def apply_perm(inp, perm):
    permuted_inp = [-1] * len(inp)
    for idx, x in enumerate(perm):
        permuted_inp[idx] = inp[x]
    return permuted_inp


def invert_perm(perm):
    new_perm = [-1] * len(perm)
    for idx, x in enumerate(perm):
        new_perm[x] = idx
    return new_perm


def validate_dim_length(length: int) -> None:
    if length < 0:
        raise AssertionError(f"length must be non-negative, got {length}")


def _is_seq_like(x) -> bool:
    """A shape/stride provider is any sized, indexable, iterable object.

    The framework's ``Size`` type (a pybind sequence) does not register as a
    ``collections.abc.Sequence``, so duck typing is used here.
    """
    return hasattr(x, "__len__") and hasattr(x, "__getitem__")


def validate_shape(shape: ShapeType) -> None:
    if not _is_seq_like(shape):
        raise AssertionError(f"shape must be a Sequence, got {type(shape)}")
    for length in shape:
        validate_dim_length(length)


def validate_strides(strides: StrideType) -> None:
    if not _is_seq_like(strides):
        raise AssertionError(f"strides must be a Sequence, got {type(strides)}")
    for stride in strides:
        if stride < 0:
            raise AssertionError(f"stride must be non-negative, got {stride}")


def validate_idx(rank: int, idx: int) -> None:
    if not isinstance(idx, int):
        raise AssertionError(f"idx must be an int, got {type(idx)}")
    if idx < -rank or idx >= rank:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-rank}, {rank - 1}], but got {idx})"
        )


def validate_dimension_indices(rank: int, indices: DimsSequenceType) -> None:
    for idx in indices:
        validate_idx(rank, idx)


def validate_exclusive_idx(rank: int, ex_idx: int) -> None:
    if not (isinstance(ex_idx, int) and -rank <= ex_idx < rank and ex_idx != 0):
        raise IndexError(
            f"Exclusive dimension index must be in [-{rank}, {rank}) and not zero, "
            f"but got {ex_idx}"
        )


def canonicalize_dim(rank: int, idx: int, wrap_scalar: bool = True) -> int:
    if rank < 0:
        raise IndexError(f"Rank cannot be negative but got {rank}")
    if rank == 0:
        if not wrap_scalar:
            raise IndexError(f"Dimension specified as {idx} but tensor has no dimensions")
        rank = 1
    if 0 <= idx < rank:
        return idx
    _idx = idx + rank if idx < 0 else idx
    if _idx < 0 or _idx >= rank:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-rank}, {rank - 1}], but got {idx})"
        )
    return _idx


def canonicalize_dims(rank, indices, wrap_scalar=True):
    if isinstance(indices, int):
        return canonicalize_dim(rank, indices, wrap_scalar)
    return tuple(canonicalize_dim(rank, x, wrap_scalar) for x in indices)


def is_valid_permutation(rank: int, perm: DimsSequenceType) -> bool:
    if not isinstance(perm, Sequence):
        return False
    if len(perm) != rank or len(set(perm)) != rank:
        return False
    return all(0 <= x < rank for x in perm)


def is_same_shape(a: Sequence, b: Sequence) -> BoolLikeType:
    return list(a) == list(b)


def is_cpu_scalar_tensor(a: object) -> TypeGuard[TensorLike]:
    return isinstance(a, TensorLike) and a.ndim == 0 and a.device.type == "cpu"


def is_cpu_scalar(a: TensorLikeType) -> bool:
    return isinstance(a, TensorLike) and a.ndim == 0 and a.device.type == "cpu"


def check_same_device(*args, allow_cpu_scalar_tensors: bool) -> None:
    """Checks that all tensors in args have the same device.

    Zero-dimensional CPU scalar tensors are exempt from the check when
    ``allow_cpu_scalar_tensors`` is true.
    """
    device = None
    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                continue
            if device is None:
                device = arg.device
            elif device != arg.device:
                raise RuntimeError(
                    f"Tensor on device {arg.device} is not on the expected "
                    f"device {device}!"
                )
        else:
            raise RuntimeError(
                f"Unexpected type when checking for same device, {type(arg)}!"
            )


def canonicalize_device(device: DeviceLikeType) -> tensorplay.device:
    if isinstance(device, tensorplay.device):
        return device
    return tensorplay.device(device)


def check_same_shape(*args, allow_cpu_scalar_tensors: bool) -> None:
    """Checks that all tensors in args have the same shape.

    Zero-dimensional CPU scalar tensors are exempt from the check when
    ``allow_cpu_scalar_tensors`` is true.
    """
    shape = None
    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                continue
            if shape is None:
                shape = arg.shape
            elif not is_same_shape(shape, arg.shape):
                raise RuntimeError(
                    f"Shape {arg.shape} is not the expected shape {shape}!"
                )
        else:
            raise RuntimeError(
                f"Unexpected type when checking for same shape, {type(arg)}!"
            )


def extract_shape(*args, allow_cpu_scalar_tensors: bool) -> ShapeType | None:
    """Acquires a common shape from the tensor arguments, if it exists.

    Zero-dimensional CPU scalar tensors only contribute a shape when no
    higher-rank tensor participates; ``None`` is returned when the tensors
    disagree or no usable shape exists.
    """
    shape = None
    scalar_shape = None
    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                scalar_shape = arg.shape
                continue
            if shape is None:
                shape = arg.shape
            elif not is_same_shape(shape, arg.shape):
                return None
        else:
            return None
    return shape if shape is not None else scalar_shape


def extract_dims_from_varargs(dim, args=()):
    if isinstance(dim, int):
        dim = (dim,)
    if len(args) != 0:
        if dim is not None:
            raise RuntimeError("Received a non-empty varargs and dim argument!")
        dim = args
    return dim


def extract_shape_from_varargs(shape, validate=True):
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = shape[0]
    if validate:
        validate_shape(shape)
    return shape


def infer_size_shapes(a: ShapeType, b: ShapeType) -> tuple[int, ...]:
    ndim = max(len(a), len(b))
    expandedSizes = [0] * ndim
    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dimA = len(a) - 1 - offset
        dimB = len(b) - 1 - offset
        sizeA = a[dimA] if dimA >= 0 else 1
        sizeB = b[dimB] if dimB >= 0 else 1
        check(
            (sizeA == sizeB) or (sizeA == 1) or (sizeB == 1),
            lambda: (
                f"The size of tensor a ({sizeA}) must match the size of "
                f"tensor b ({sizeB}) at non-singleton dimension {i}"
            ),
        )
        expandedSizes[i] = sizeB if sizeA == 1 else sizeA
    return tuple(expandedSizes)


def infer_size(shape: ShapeType, numel: int) -> tuple[int, ...]:
    """Infers the size of a dim with size -1, if it exists.

    Also checks that the new shape is compatible with the number of elements.
    """
    dim = None
    newsize = 1
    for i, d in enumerate(shape):
        if d == -1:
            check(dim is None, lambda: "only one dimension can be inferred")
            dim = i
        else:
            check(d >= 0, lambda: f"invalid shape dimension {d}")
            newsize *= d
    if dim is None:
        check(
            numel == newsize,
            lambda: f"shape '{list(shape)}' is invalid for input of size {numel}",
        )
    else:
        check(
            newsize != 0,
            lambda: (
                "cannot reshape tensor of 0 elements into shape "
                f"{list(shape)} because the unspecified dimension size -1 "
                "can be any value and is ambiguous"
                if numel == 0
                else f"shape '{list(shape)}' is invalid for input of size {numel}"
            ),
        )
        check(
            numel % newsize == 0,
            lambda: f"shape '{list(shape)}' is invalid for input of size {numel}",
        )
        shape = list(shape)
        shape[dim] = numel // newsize
        check(shape[dim] >= 0)
    return tuple(shape)


def is_boolean_dtype(dtype) -> bool:
    return dtype in _boolean_dtypes


def is_integer_dtype(dtype) -> bool:
    return dtype in _integer_dtypes


def is_low_precision_dtype(dtype) -> bool:
    return dtype in _low_precision_dtypes


def is_float_dtype(dtype) -> bool:
    return dtype in _float_dtypes


def is_complex_dtype(dtype) -> bool:
    return dtype in _complex_dtypes


def is_grad_dtype(dtype) -> bool:
    """Dtypes whose backward computation is supported by convention."""
    return (
        is_float_dtype(dtype)
        or is_complex_dtype(dtype)
        or dtype in (tensorplay.uint8, tensorplay.int8, tensorplay.int16,
                     tensorplay.int32, tensorplay.int64, tensorplay.bool)
    )


def corresponding_real_dtype(dtype):
    return {
        tensorplay.complex32: tensorplay.float16,
        tensorplay.bcomplex32: tensorplay.bfloat16,
        tensorplay.complex64: tensorplay.float32,
        tensorplay.complex128: tensorplay.float64,
    }[dtype]


def corresponding_complex_dtype(dtype):
    return {
        tensorplay.float16: tensorplay.complex32,
        tensorplay.bfloat16: tensorplay.bcomplex32,
        tensorplay.float32: tensorplay.complex64,
        tensorplay.float64: tensorplay.complex128,
    }[dtype]


def dtype_to_type(dtype) -> type:
    """Computes the corresponding Python type ("type kind") for a dtype."""
    if not isinstance(dtype, tensorplay.dtype):
        raise AssertionError(f"Expected tensorplay.dtype, got {type(dtype)}")
    if dtype is tensorplay.bool:
        return bool
    if dtype in _integer_dtypes:
        return int
    if dtype.is_floating_point:
        return float
    if dtype in _complex_dtypes:
        return complex
    raise ValueError("Invalid dtype!")


def dtype_to_type_ctor(dtype):
    """Computes the Python type constructor for a given dtype."""
    if not isinstance(dtype, tensorplay.dtype):
        raise AssertionError(f"Expected tensorplay.dtype, got {type(dtype)}")
    if dtype is tensorplay.bool:
        return bool
    if dtype in _integer_dtypes:
        return int
    if dtype.is_floating_point:
        return float
    if dtype in _complex_dtypes:
        return complex
    raise ValueError("Invalid dtype!")


def type_to_dtype(typ: type) -> tensorplay.dtype:
    """Computes the corresponding dtype for a Number type."""
    if not isinstance(typ, type):
        raise AssertionError(f"Expected type, got {type(typ)}")
    if typ is bool:
        return tensorplay.bool
    if typ is int:
        return tensorplay.long
    if typ is float:
        return tensorplay.get_default_dtype()
    if typ is complex:
        return corresponding_complex_dtype(tensorplay.get_default_dtype())
    raise ValueError(f"Invalid type {typ}!")


def get_dtype(x):
    if isinstance(x, tensorplay.Tensor):
        return x.dtype
    if isinstance(x, Number):
        return type_to_dtype(type(x))
    raise RuntimeError("Only tensors and numbers are supported!")


def check_fp_or_complex(dtype, fn_name: str, allow_low_precision_dtypes: bool = True) -> None:
    """Checks that the dtype is floating point or complex.

    When allow_low_precision_dtypes is False, half precision dtypes are
    rejected as well.
    """
    check(
        is_float_dtype(dtype) or is_complex_dtype(dtype),
        lambda: f"{fn_name}: Expected a floating point or complex tensor as input. Got {dtype}",
    )
    check(
        allow_low_precision_dtypes or not is_low_precision_dtype(dtype),
        lambda: f"{fn_name}: Half precision dtypes not supported. Got {dtype}",
    )


def check_is_matrix(A: TensorLikeType, f_name: str, arg_name: str = "A") -> None:
    check(
        len(A.shape) >= 2,
        lambda: f"{f_name}: The input tensor {arg_name} must have at least 2 dimensions.",
    )


def get_higher_type(a: type, b: type) -> type:
    """Returns the higher of two Number types (bool -> int -> float -> complex)."""
    a, b = _maybe_get_pytype(a), _maybe_get_pytype(b)
    if a not in _ordered_types or b not in _ordered_types:
        raise RuntimeError(f"Expected builtin numeric types, found {a}, {b}")
    if a is b:
        return a
    for typ in _ordered_types:
        if a is typ:
            return b
        if b is typ:
            return a
    raise ValueError("Unknown Python scalar type!")


def get_higher_dtype(a, b=None):
    """Computes the lowest dtype weakly higher than both a and b."""

    def _extract_dtype(x):
        if x is None:
            return None
        if isinstance(x, tensorplay.dtype):
            return x
        if isinstance(x, TensorLike):
            return x.dtype
        if isinstance(x, Number):
            return type_to_dtype(type(x))
        raise RuntimeError("Unexpected type given to _extract_dtype!")

    a, b = _extract_dtype(a), _extract_dtype(b)
    if a is b:
        return a
    if a is None:
        return b
    if b is None:
        return a
    for tier in _ordered_dtypes:
        if a in tier and b in tier:
            return b if _dtype_order(a) < _dtype_order(b) else a
    order_a, order_b = _dtype_order(a), _dtype_order(b)
    return a if order_a > order_b else b


def _dtype_order(dt) -> int:
    for i, tier in enumerate(_ordered_dtypes):
        if dt in tier:
            return i
    raise ValueError(f"Unknown dtype {dt}!")


def check_pin_memory(pin_memory: bool) -> None:
    if not isinstance(pin_memory, bool):
        raise ValueError("pin_memory must be a bool")


def check_layout(layout) -> None:
    if layout is not None and layout is not tensorplay.strided:
        raise ValueError(f"Only strided layout is supported, got {layout}")


def is_weakly_lesser_type(a: type, b: type) -> bool:
    """True when the type of a is weakly lower in the hierarchy than b."""
    def _match(x, tier):
        if tier is int:
            return x is int or x is bool
        if tier is float:
            return x is float or x is int or x is bool
        if tier is complex:
            return x is complex or x is float or x is int or x is bool
        return x is tier

    return _match(a, b)


def can_safe_cast_to(*, cast_to, cast_from) -> bool:
    for fn in (is_complex_dtype, is_float_dtype, is_integer_dtype, is_boolean_dtype):
        if fn(cast_to):
            return True
        if fn(cast_from):
            return False
    raise ValueError(f"Received unknown dtypes {cast_to}, {cast_from}!")


def check_same_dtype(*args) -> None:
    dtypes = [a.dtype for a in args if isinstance(a, TensorLike)]
    if len(dtypes) == 0:
        return
    first = dtypes[0]
    for other in dtypes[1:]:
        if other != first:
            raise TypeError(
                f"Expected all tensors to have the same dtype, but found {first} and {other}!"
            )


def get_computation_dtype(dtype):
    return {
        tensorplay.float16: tensorplay.float32,
        tensorplay.bfloat16: tensorplay.float32,
        tensorplay.complex32: tensorplay.complex64,
        tensorplay.bcomplex32: tensorplay.complex64,
    }.get(dtype, dtype)


_cpu_acc_type_map = {
    tensorplay.float16: tensorplay.float32,
    tensorplay.bfloat16: tensorplay.float32,
}


def get_acc_type(dtype, device=None) -> tensorplay.dtype:
    """Accumulation dtype for reductions; CPU prefers fp32 for half types."""
    dev = device if isinstance(device, tensorplay.device) else tensorplay.device("cpu")
    if dev.type == "cpu":
        return _cpu_acc_type_map.get(dtype, dtype)
    return get_computation_dtype(dtype)


class ELEMENTWISE_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = 0
    NO_OPMATH = 1
    INT_TO_FLOAT = 2
    COMPLEX_TO_FLOAT = 3
    BOOL_TO_LONG = 4
    ALWAYS_BOOL = 5


class REDUCTION_OUTPUT_TYPE_KIND(Enum):
    SAME = 0
    COMPLEX_TO_FLOAT = 1
    KEEP_PROMOTED_TYPE = 2
    ALWAYS_BOOL = 3


class RETURN_TYPE(Enum):
    NEW = (0,)
    VIEW = (1,)
    INPLACE = (2,)
    NONE = (3,)


def number_type(x: NumberType) -> NumberTypeType:
    if isinstance(x, bool):
        return bool
    if isinstance(x, int):
        return int
    if isinstance(x, float):
        return float
    if isinstance(x, complex):
        return complex
    raise RuntimeError(f"Unexpected number type {type(x)}")


def elementwise_dtypes(*_args, type_promotion_kind) -> tuple[dtype, dtype]:
    """Computes the computation and result dtypes for elementwise promotion.

    The promotion first selects the "highest" ordered Python type
    (bool -> int -> float -> complex) among all arguments. The result dtype
    is then the highest participating tensor dtype within that type class,
    preferring tensors with one or more dimensions over zero-dimension
    tensors; when no tensor participates, the default dtype of the selected
    class is used. The computation dtype additionally maps low precision
    floating point and complex dtypes up ("op math") unless the promotion
    kind disables it.
    """
    args = _args
    highest_type: type = bool
    for x in args:
        if not isinstance(x, (*Number, TensorLike)):
            raise ValueError(
                f"Unexpected type {str(type(x))} when computing elementwise "
                f"type promotion!"
            )
        if isinstance(x, Number):
            highest_type = get_higher_type(highest_type, number_type(x))
        else:
            highest_type = get_higher_type(highest_type, dtype_to_type(x.dtype))

    result_dtype = None

    def _find_highest_dtype_filtered(args, filter, *, float_as_complex=False):
        zero_dim_tensor_dtype = None
        one_plus_dim_tensor_dtype = None
        for x in args:
            if isinstance(x, TensorLike) and filter(x.dtype):
                _dtype = x.dtype
                if float_as_complex and is_float_dtype(_dtype):
                    _dtype = corresponding_complex_dtype(_dtype)
                if x.ndim == 0:
                    zero_dim_tensor_dtype = get_higher_dtype(zero_dim_tensor_dtype, _dtype)
                else:
                    one_plus_dim_tensor_dtype = get_higher_dtype(
                        one_plus_dim_tensor_dtype, _dtype
                    )
        if one_plus_dim_tensor_dtype is not None:
            return one_plus_dim_tensor_dtype
        return zero_dim_tensor_dtype

    if highest_type is float:
        result_dtype = _find_highest_dtype_filtered(args, is_float_dtype)
        result_dtype = tensorplay.get_default_dtype() if result_dtype is None else result_dtype
    elif highest_type is complex:
        result_dtype = _find_highest_dtype_filtered(
            args, lambda x: is_float_dtype(x) or is_complex_dtype(x),
            float_as_complex=True,
        )
        if result_dtype is None:
            result_dtype = corresponding_complex_dtype(tensorplay.get_default_dtype())
    elif highest_type is int:
        result_dtype = _find_highest_dtype_filtered(args, is_integer_dtype)
        result_dtype = tensorplay.long if result_dtype is None else result_dtype
    else:
        result_dtype = tensorplay.bool

    kind = type_promotion_kind
    if kind is ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT:
        return get_computation_dtype(result_dtype), result_dtype
    elif kind is ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH:
        return result_dtype, result_dtype
    elif kind is ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT:
        if is_integer_dtype(result_dtype) or is_boolean_dtype(result_dtype):
            result_dtype = tensorplay.get_default_dtype()
        return get_computation_dtype(result_dtype), result_dtype
    elif kind is ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT:
        computation_dtype = get_computation_dtype(result_dtype)
        if is_complex_dtype(result_dtype):
            result_dtype = corresponding_real_dtype(result_dtype)
        return computation_dtype, result_dtype
    elif kind is ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG:
        if is_boolean_dtype(result_dtype):
            return tensorplay.long, tensorplay.long
        return get_computation_dtype(result_dtype), result_dtype
    elif kind is ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
        return get_computation_dtype(result_dtype), tensorplay.bool
    else:
        raise ValueError(f"Unknown type promotion kind {str(type_promotion_kind)}")


def reduction_dtypes(arg, output_dtype_kind, dtype=None) -> tuple[dtype, dtype | None]:
    """Computation and result dtypes for a reduction with the given kind."""
    inp_dtype = dtype if dtype is not None else arg.dtype
    computation_dtype = get_computation_dtype(inp_dtype)
    if output_dtype_kind in (
        REDUCTION_OUTPUT_TYPE_KIND.SAME,
        REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ):
        result_dtype = dtype if dtype else arg.dtype
        if (
            output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
            and is_complex_dtype(result_dtype)
        ):
            result_dtype = corresponding_real_dtype(result_dtype)
    elif output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.KEEP_PROMOTED_TYPE:
        result_dtype = None
    else:
        result_dtype = tensorplay.bool
    return computation_dtype, result_dtype


# Row-major (or column-major for the last two dims) contiguous strides.
def make_contiguous_strides_for(shape: ShapeType, row_major: bool = True) -> tuple[int, ...]:
    """Returns the strides of a contiguous tensor for the given shape.

    With row_major=False the last two dimensions describe a batch of
    Fortran-contiguous matrices, as required by column-major linear algebra
    backends.
    """
    validate_shape(shape)
    if not shape:
        return ()
    multiplier = 1
    strides = []
    for length in reversed(shape):
        strides.append(multiplier)
        multiplier *= max(length, 1)
    result = tuple(reversed(strides))
    if row_major:
        return result
    if len(shape) < 2:
        return result
    return result[:-2] + (1, max(shape[-2], 1))


def make_channels_last_1d_strides_for(shape: ShapeType) -> tuple[int, ...]:
    validate_shape(shape)
    if len(shape) != 3:
        raise RuntimeError(f"Expected a 1D tensor, got shape {shape}!")
    N, C, L = shape
    return (C * L, 1, C) if C else (C, 1, C * L)


def make_channels_last_2d_strides_for(shape: ShapeType) -> tuple[int, ...]:
    validate_shape(shape)
    if len(shape) != 4:
        raise RuntimeError(f"Expected a 2D tensor, got shape {shape}!")
    N, C, H, W = shape
    return (H * W * C, 1, W * C, C) if C else (H * W * C, C, W * C, C)


def make_channels_last_3d_strides_for(shape: ShapeType) -> tuple[int, ...]:
    validate_shape(shape)
    if len(shape) != 5:
        raise RuntimeError(f"Expected a 3D tensor, got shape {shape}!")
    N, C, D, H, W = shape
    if C:
        return (D * H * W * C, 1, H * W * C, W * C, C)
    return (D * H * W * C, C, H * W * C, W * C, C)


def make_channels_last_strides_for(shape: ShapeType) -> tuple[int, ...]:
    ndim = len(shape)
    if ndim == 3:
        return make_channels_last_1d_strides_for(shape)
    if ndim == 4:
        return make_channels_last_2d_strides_for(shape)
    if ndim == 5:
        return make_channels_last_3d_strides_for(shape)
    raise RuntimeError(f"Expected a 3-5D tensor, got shape {shape}!")


def compute_reduction_output_shape(shape: ShapeType, dimensions: Sequence) -> tuple[int, ...]:
    for idx in dimensions:
        validate_idx(len(shape), idx)
    return tuple(shape[idx] for idx in range(len(shape)) if idx not in dimensions)


def validate_no_repeating_dims(dims: Sequence) -> None:
    if len(set(dims)) != len(dims):
        raise RuntimeError("Duplicate value in 'dim'!")


def reduction_dims(shape: ShapeType, dims: Sequence | None) -> tuple[int, ...]:
    if dims is None:
        return tuple(range(len(shape)))
    dims = tuple(canonicalize_dim(len(shape), idx) for idx in dims)
    validate_no_repeating_dims(dims)
    return dims


def set_correction(
    math_mode: bool, use_deterministic_highest_precision: bool, *, scale_dtype=None
) -> bool:
    if math_mode and not use_deterministic_highest_precision:
        return False
    return scale_dtype is None or not is_float_dtype(scale_dtype)


def compute_required_storage_length(shape, strides, storage_offset) -> int:
    """Minimum storage size, in elements, that holds the given geometry."""
    if reduce(operator.mul, shape, 1) == 0:
        return 0
    max_offset = sum((x - 1) * y for x, y in zip(shape, strides))
    # +1 accounts for the first element the offsets are taken from.
    return 1 + storage_offset + max_offset


def check_in_bounds_for_storage(shape, strides, storage_offset, storage_size) -> None:
    if len(shape) != len(strides):
        raise RuntimeError("Shape and strides must have the same length!")
    required = compute_required_storage_length(shape, strides, storage_offset)
    if required > storage_size:
        raise RuntimeError(
            f"Can't view self of size {shape} with strides {strides} and offset "
            f"{storage_offset}: required storage size {required} is larger than "
            f"the storage size {storage_size}!"
        )


def are_strides_like_channels_last_or_false(shape, strides) -> bool:
    ndim = len(shape)
    if ndim == 4:
        return strides == make_channels_last_2d_strides_for(shape) and shape[1] != 0
    if ndim == 5:
        return strides == make_channels_last_3d_strides_for(shape) and shape[1] != 0
    return False


def suggest_memory_format(x: TensorLikeType) -> Any:
    if is_contiguous(x):
        return tensorplay.contiguous_format
    if are_strides_like_channels_last_or_false(x.shape, x.stride()):
        return tensorplay.channels_last
    return None


def prod(xs: Sequence[NumberType]) -> NumberType:
    return reduce(operator.mul, xs, 1)


def is_expandable_to(shape: ShapeType, desired: ShapeType) -> bool:
    """Checks if a shape can be broadcast-expanded to another shape."""
    if len(shape) > len(desired):
        return False
    for i in range(len(shape)):
        if shape[-i - 1] != desired[-i - 1] and shape[-i - 1] != 1:
            return False
    return True


def mask_tensor(mask: TensorLikeType, t: TensorLikeType) -> TensorLikeType:
    """Broadcasts mask over t and fills masked positions with zero."""
    mask = mask.expand(t.shape)
    return tensorplay.where(mask, t, tensorplay.zeros_like(t))


def dtype_or_default(dtype=None):
    return dtype if dtype is not None else tensorplay.get_default_dtype()


def device_or_default(device=None):
    return device if device is not None else tensorplay.device("cpu")


def layout_or_default(layout=None):
    return layout if layout is not None else tensorplay.strided


def clone_preserve_strides(x: TensorLikeType) -> TensorLikeType:
    buffer = tensorplay.empty_strided(
        x.shape, x.stride(), dtype=x.dtype, device=x.device
    )
    return tensorplay.copy_(buffer, x)


def alert_not_deterministic(caller: str) -> None:
    import warnings

    if tensorplay.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            f"{caller} does not have a deterministic implementation, but you set "
            f"'tensorplay.use_deterministic_algorithms(True)'."
        )
    warnings.warn(
        f"{caller} does not have a deterministic implementation, but you set "
        f"'tensorplay.use_deterministic_algorithms(True, warn_only=True)'. "
        f"You can file an issue to help us prioritize adding deterministic support "
        f"for this operation.",
        UserWarning,
    )


class CUDARngStateHelper:
    """Register-scope helper capturing the RNG offset context for kernels."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def get_tensorplay_op(fn, name: str):
    """Resolves the eager operator named ``name`` for logging wrappers."""
    return getattr(tensorplay, name, None)
