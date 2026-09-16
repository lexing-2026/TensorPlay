"""Framework elementwise kernels for binary ufuncs, renamed to match numpy.
This listing is further exported to public symbols in the `tensorplay._numpy/_ufuncs.py` module.
"""

import tensorplay
from tensorplay import (  # noqa: F401
    add,
    arctan2,
    bitwise_and,
    bitwise_left_shift as left_shift,
    bitwise_or,
    bitwise_right_shift as right_shift,
    bitwise_xor,
    copysign,
    divide,
    eq as equal,
    float_power,
    floor_divide,
    fmax,
    fmin,
    fmod,
    gcd,
    greater,
    greater_equal,
    heaviside,
    hypot,
    lcm,
    ldexp,
    less,
    less_equal,
    logaddexp,
    logaddexp2,
    logical_and,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    multiply,
    nextafter,
    not_equal,
    pow as power,
    remainder,
    remainder as mod,
    subtract,
    true_divide,
)

from . import _dtypes_impl, _util


# adapt behavior where the framework differs from numpy
def matmul(x: tensorplay.Tensor, y: tensorplay.Tensor) -> tensorplay.Tensor:
    # work around:
    #  - RuntimeError: expected scalar type Int but found Double
    #  - RuntimeError: "addmm_impl_cpu_" not implemented for 'Bool'
    #  - RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
    dtype = _dtypes_impl.result_type_impl(x, y)
    is_bool = dtype == tensorplay.bool
    is_half = (x.dtype == tensorplay.float16 or y.dtype == tensorplay.float16) and (
        x.is_cpu or y.is_cpu
    )

    work_dtype = dtype
    if is_bool:
        work_dtype = tensorplay.uint8
    if is_half:
        work_dtype = tensorplay.float32

    x = _util.cast_if_needed(x, work_dtype)
    y = _util.cast_if_needed(y, work_dtype)

    result = tensorplay.matmul(x, y)

    if work_dtype != dtype:
        result = result.to(dtype)

    return result


# divmod: remainder semantics follow floor division

def divmod(x: tensorplay.Tensor, y: tensorplay.Tensor) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    return x // y, x % y
