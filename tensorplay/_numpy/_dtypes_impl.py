"""Dtype machinery over the framework's own dtype objects.

Here ``dtype`` is always a ``tensorplay.dtype``; this module knows nothing
about scalar types, wrapper dtypes or anything like that.
"""

from collections import namedtuple

import tensorplay


# defaults : mimic NumPy, allow user control
DefaultDTypes = namedtuple(
    "DefaultDTypes", ["float_dtype", "complex_dtype", "int_dtype"]
)

# Framework-level defaults for the compat layer.  They can be overridden
# through the module-level names below before the first call to
# default_dtypes().
_DEFAULT_FLOAT = "float64"
_DEFAULT_COMPLEX = "complex128"
_DEFAULT_INT = "int64"

# a global state, initialized on first use
_default_dtypes: DefaultDTypes | None = None


def default_dtypes() -> DefaultDTypes:
    global _default_dtypes
    if _default_dtypes is None:
        _default_dtypes = DefaultDTypes(
            float_dtype=getattr(tensorplay, _DEFAULT_FLOAT),
            complex_dtype=getattr(tensorplay, _DEFAULT_COMPLEX),
            int_dtype=getattr(tensorplay, _DEFAULT_INT),
        )
        if not isinstance(_default_dtypes.float_dtype, tensorplay.dtype):
            raise AssertionError(
                f"float_dtype must be a tensorplay.dtype, got {type(_default_dtypes.float_dtype)}"
            )
        if not isinstance(_default_dtypes.complex_dtype, tensorplay.dtype):
            raise AssertionError(
                f"complex_dtype must be a tensorplay.dtype, got {type(_default_dtypes.complex_dtype)}"
            )
        if not isinstance(_default_dtypes.int_dtype, tensorplay.dtype):
            raise AssertionError(
                f"int_dtype must be a tensorplay.dtype, got {type(_default_dtypes.int_dtype)}"
            )
    return _default_dtypes


def get_default_dtype_for(dtype: tensorplay.dtype) -> tensorplay.dtype:
    """Default scalar type given sctype category."""
    if dtype == tensorplay.bool:
        return dtype
    if dtype.is_complex:
        return default_dtypes().complex_dtype
    if dtype.is_floating_point:
        return default_dtypes().float_dtype
    # else, it must be (some) integer
    return default_dtypes().int_dtype


from . import _casting_dicts as _cd


def can_cast_impl(
    from_torch_dtype: tensorplay.dtype, to_torch_dtype: tensorplay.dtype, casting: str
) -> bool:
    return _cd._can_cast_dict[casting][from_torch_dtype][to_torch_dtype]


def result_type_impl(*tensors: tensorplay.Tensor) -> tensorplay.dtype:
    # dtypes here are the framework's own
    dtyp = tensors[0].dtype
    if len(tensors) == 1:
        return dtyp

    for curr in tensors[1:]:
        dtyp = _cd._result_type_dict[dtyp][curr.dtype]

    return dtyp


def python_type_for_torch(dtyp: tensorplay.dtype) -> type[bool | int | float | complex]:
    """Get the Python scalar type for a dtype."""
    if dtyp.is_floating_point:
        typ = float
    elif dtyp.is_complex:
        typ = complex
    elif dtyp == tensorplay.bool:
        typ = bool
    else:
        typ = int
    return typ


# ### NEP 50 helpers ###

_SCALAR_TYPES = (int, bool, float, complex)

_SCALAR_AND_SYMBOLIC_TYPES = (
    *_SCALAR_TYPES,
    tensorplay.SymInt,
    tensorplay.SymFloat,
    tensorplay.SymBool,
)

_NEP50_FUNCS_TENSOR_ONLY = (
    "minimum",
    "maximum",
    "logaddexp",
    "logaddexp2",
    "lcm",
    "gcd",
    "hypot",
    "heaviside",
    "fmod",
    "fmin",
    "fmax",
    "copysign",
    "arctan2",
)


def is_scalar(x: object) -> bool:
    return isinstance(x, _SCALAR_TYPES)


def is_scalar_or_symbolic(x: object) -> bool:
    return isinstance(x, _SCALAR_AND_SYMBOLIC_TYPES)


def _dtype_for_scalar(py_type: type) -> tensorplay.dtype:
    return {
        bool: tensorplay.bool,
        tensorplay.SymBool: tensorplay.bool,
        int: tensorplay.int64,
        tensorplay.SymInt: tensorplay.int64,
        float: tensorplay.float64,
        tensorplay.SymFloat: tensorplay.float64,
        complex: tensorplay.complex128,
    }[py_type]


def _dtype_for_scalar_or_tensor(
    x: tensorplay.Tensor | bool | int | float | complex,
) -> tensorplay.dtype:
    return x.dtype if isinstance(x, tensorplay.Tensor) else _dtype_for_scalar(type(x))


def is_float_or_fp_tensor(x: tensorplay.Tensor | bool | int | float | complex) -> bool:
    return _dtype_for_scalar_or_tensor(x).is_floating_point


def is_complex_or_complex_tensor(
    x: tensorplay.Tensor | bool | int | float | complex,
) -> bool:
    return _dtype_for_scalar_or_tensor(x).is_complex


def _category(dtype: tensorplay.dtype) -> int:
    return {
        tensorplay.bool: 0,
        tensorplay.SymBool: 0,
        # int
        tensorplay.uint8: 1,
        tensorplay.int8: 1,
        tensorplay.int16: 1,
        tensorplay.int32: 1,
        tensorplay.int64: 1,
        tensorplay.SymInt: 1,
        # float
        tensorplay.float16: 2,
        tensorplay.float32: 2,
        tensorplay.float64: 2,
        tensorplay.SymFloat: 2,
        # complex
        tensorplay.complex64: 3,
        tensorplay.complex128: 3,
    }[dtype]


def nep50_to_tensors(
    x1: tensorplay.Tensor | bool | int | float | complex,
    x2: tensorplay.Tensor | bool | int | float | complex,
    handle_weaks: bool,
    function_name: str,
) -> tuple[
    tensorplay.Tensor | bool | int | float | complex,
    tensorplay.Tensor | bool | int | float | complex,
]:
    """If either of inputs is a python scalar, type-promote with NEP 50."""

    def to_tensor(
        scalar: tensorplay.Tensor | bool | int | float | complex,
        dtype: tensorplay.dtype | None = None,
    ) -> tensorplay.Tensor:
        if dtype is None:
            dtype = _dtype_for_scalar(type(scalar))
            dtype = get_default_dtype_for(dtype)
        return tensorplay.as_tensor(scalar, dtype=dtype)

    x1_is_weak = not isinstance(x1, tensorplay.Tensor)
    x2_is_weak = not isinstance(x2, tensorplay.Tensor)
    if not handle_weaks or (x1_is_weak and x2_is_weak):
        x1 = to_tensor(x1) if x1_is_weak else x1
        x2 = to_tensor(x2) if x2_is_weak else x2
        return x1, x2

    # scalar <op> tensor: NEP 50
    if x1_is_weak == x2_is_weak:
        raise AssertionError(
            f"Expected exactly one weak type, got x1_is_weak={x1_is_weak}, x2_is_weak={x2_is_weak}"
        )

    weak, not_weak = (x1, x2) if x1_is_weak else (x2, x1)
    if not isinstance(not_weak, tensorplay.Tensor):
        raise AssertionError("the non-weak operand must be a tensor")

    # find the dtype for the weak's type
    weak_dtype = _dtype_for_scalar(type(weak))

    cat_weak = _category(weak_dtype)
    cat_not_weak = _category(not_weak.dtype)

    dt = not_weak.dtype if cat_weak <= cat_not_weak else None

    # special-case complex + float32
    if weak_dtype.is_complex and not_weak.dtype == tensorplay.float32:
        dt = tensorplay.complex64

    # detect overflows: an out-of-range Python integer is rejected instead of
    # wrapping around, as NEP 50 mandates.
    #
    # Note that we only check if each element of the binop overflows,
    # not the result. Consider, e.g. `uint8(100) + 200`. Operands are OK
    # in uint8, but the result overflows and wrap around 255.  No
    # RuntimeWarning is emitted for that case.
    if cat_weak == 1 and cat_not_weak == 1:
        # integers
        iinfo = tensorplay.iinfo(not_weak.dtype)
        # weak is an integer here (guaranteed by cat_weak == 1), not the full union.
        # pyrefly: ignore[unsupported-operation]
        if not (iinfo.min <= weak <= iinfo.max):
            raise OverflowError(
                f"Python integer {weak} out of bounds for {not_weak.dtype}"
            )
    if weak_dtype != dt or function_name in _NEP50_FUNCS_TENSOR_ONLY:
        # finally, can make `weak` into a 0D tensor, if both parameters are required to be tensor.
        weak = to_tensor(weak, dt)

    return (weak, not_weak) if x1_is_weak else (not_weak, weak)
