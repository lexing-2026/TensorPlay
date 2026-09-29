"""Small helpers for asking a backend what it will accept.

A launch carries two different kinds of name: the names the kernel declares as
parameters, and the names the backend reads as options.  Which names are which
is not knowable by reading the launch -- a name is a parameter or an option
depending on the backend that will run it -- so it is asked.

The backend is reached by asking it to parse options, which is the only thing
it exposes for the purpose.  A name it does not recognize is an error rather
than something to pass along, because a launch whose extra names nobody reads
is a launch that quietly does not do what it says.
"""

from __future__ import annotations

import sympy

import tensorplay as tp


from .triton_compat import JITFunction, libdevice, math, tl, triton
from .triton_compat import math as tl_math


def _triton_jit(fn):
    return fn if triton is None else triton.jit(fn)


def set_driver_to_cpu():
    """Make the host backend the one launches are made through.

    A host backend may not be installed at all, which is not fatal here: a
    kernel that cannot be launched is still worth writing, and refusing to
    write it would turn a missing optional piece into a failure to compile.
    """

    import warnings

    import triton.backends
    import triton.runtime.driver

    driver = triton.runtime.driver
    backend = triton.backends.backends.get("cpu", None)
    if backend is None:
        warnings.warn(
            "could not find an active host backend; generated kernels will not "
            "be executable"
        )
        return
    if isinstance(driver.active, backend.driver):
        return
    driver.set_active(backend.driver())


def _is_backend_active(name, backend):
    """Whether this backend is the one the machine can run.

    The backend knows whether it has a device, but it can be wrong about it
    when the check runs in a subprocess where the device is not visible to
    the library that would find it.  So where the answer would be surprising,
    the device is asked directly instead.
    """

    if backend.driver.is_active():
        return True
    if name == "nvidia":
        return tp.cuda.is_available() and tp.version.hip is None
    if name == "amd":
        return tp.cuda.is_available() and tp.version.hip is not None
    return False


def set_driver_to_gpu():
    """Make the device backend the one launches are made through.

    Compiling and launching name a target, and the runtime has to be holding
    the matching driver before it will.  Which backend that is depends on the
    machine, so it is asked for rather than assumed -- and a backend already
    active is left alone, because setting one up again is not free.
    """

    import triton
    import triton.backends
    import triton.runtime.driver

    driver = triton.runtime.driver
    for name, backend in triton.backends.backends.items():
        if name == "cpu" or not _is_backend_active(name, backend):
            continue
        # The active driver may be a lazy proxy, in which case the object it
        # stands for is what tells whether it is already this backend's.
        active = driver.active
        if isinstance(active, backend.driver) or (
            hasattr(active, "_obj")
            and isinstance(active._obj, backend.driver)
        ):
            return
        driver.set_active(backend.driver())
        return
    raise RuntimeError("could not find an active device backend")


def get_backend_options_for_target(target, options=None):
    """Every option name the backend for ``target`` recognizes."""

    options = {} if options is None else dict(options)
    backend = triton.compiler.compiler.make_backend(target)
    return backend.parse_options(options).__dict__


def _is_concrete_backend_option_value(value) -> bool:
    """Whether a value is fixed now rather than worked out during the launch.

    An option the backend reads while it is launching has to be a value, not
    an expression: there is nothing to evaluate an expression against at that
    point.  So a symbolic value -- anything standing for a number, whether this
    project's own or a symbolic library's -- is not one.
    """

    if isinstance(
        value,
        (
            tp.Tensor,
            tp.SymInt,
            tp.SymFloat,
            tp.SymBool,
            sympy.Expr,
        ),
    ):
        return False
    if isinstance(value, (tuple, list)):
        return all(_is_concrete_backend_option_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            _is_concrete_backend_option_value(key)
            and _is_concrete_backend_option_value(item)
            for key, item in value.items()
        )
    return True


def try_filter_backend_options_for_target(target, options, kernel_arg_names=()):
    """Split launch names into the ones the backend reads as options.

    A name the backend does not read and the kernel does not declare belongs to
    neither, and is an error: passing it along would let a launch name something
    that nothing reads, and the kernel would run with an option the caller
    believed it had set.  The check is skipped entirely when there is nothing
    to split, which is the common case -- a launch that sets no options.
    """

    parsed_options = get_backend_options_for_target(target)
    kernel_arg_names = tuple(kernel_arg_names)
    filtered_options = {
        name: value for name, value in options.items() if name in parsed_options
    }
    invalid_options = [
        name
        for name in options
        if name not in parsed_options and name not in kernel_arg_names
    ]
    if invalid_options:
        raise RuntimeError(
            "launch names must be kernel parameters or backend options: "
            f"{sorted(invalid_options)!r}"
        )
    dynamic_options = [
        name
        for name, value in filtered_options.items()
        if not _is_concrete_backend_option_value(value)
    ]
    if dynamic_options:
        raise RuntimeError(
            f"backend options must be values, not expressions: {sorted(dynamic_options)!r}"
        )
    return filtered_options

def get_constexprs(kernel: JITFunction) -> list[int]:
    """Which of a kernel's parameters were fixed when it was written.

    Returned as positions rather than names, because that is what a launch needs
    to decide: a parameter at one of these positions is not a value the caller
    passes, whatever the launch is called and whatever order the names are
    written in.
    """

    return [p.num for p in kernel.params if p.is_constexpr]


@_triton_jit
def promote_to_tensor(x):
    return x + tl.zeros((1,), tl.int1)


@_triton_jit
def fp8e4m3fn_to_float32(x):
    x_u32 = x.to(tl.uint32)
    sign = (x_u32 & 0x80) << 24
    exp = (x_u32 >> 3) & 0xF
    mant = x_u32 & 0x7

    normal_bits = sign | ((exp + 120) << 23) | (mant << 20)
    normal = normal_bits.to(tl.float32, bitcast=True)

    subnormal_abs = mant.to(tl.float32) * 0.001953125
    subnormal_bits = subnormal_abs.to(tl.uint32, bitcast=True) | sign
    subnormal = subnormal_bits.to(tl.float32, bitcast=True)

    nan = (sign | 0x7FF00000).to(tl.float32, bitcast=True)
    result = tl.where(exp == 0, subnormal, normal)
    return tl.where((exp == 0xF) & (mant == 0x7), nan, result)


@_triton_jit
def div_floor_integer(a, b):
    quot = a // b
    remainder = a % b
    fixed = tl.where(remainder != 0, quot - 1, quot)
    return tl.where((a < 0) != (b < 0), fixed, quot)


@_triton_jit
def remainder_integer(a, b):
    remainder = a % b
    return tl.where((remainder != 0) & ((a < 0) != (b < 0)), remainder + b, remainder)


@_triton_jit
def pow_integer(base, exponent):
    exponent_dtype: tl.constexpr = tl.core.get_int_dtype(
        exponent.dtype.primitive_bitwidth, signed=False
    )
    exp = exponent.to(exponent_dtype)
    result = tl.full(base.shape, 1, base.dtype)
    for _ in tl.static_range(exponent_dtype.primitive_bitwidth):
        result = tl.where((exp & 1) != 0, result * base, result)
        exp = exp >> 1
        base = base * base
    return result


@_triton_jit
def is_floating(x):
    return promote_to_tensor(x).dtype.is_floating()


@_triton_jit
def _prod_accumulate(a, b):
    return a * b


@_triton_jit
def prod(input, axis):
    return tl.reduce(input, axis, _prod_accumulate)


@_triton_jit
def prod_inner_tree(input, axis, reduction_ordering: tl.constexpr):
    return tl.reduce(input, axis, _prod_accumulate, reduction_ordering=reduction_ordering)


@_triton_jit
def minimum(a, b):
    return tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)


@_triton_jit
def maximum(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@_triton_jit
def _minimum_reduce(a, b):
    value = minimum(a, b)
    if is_floating(a):
        value = tl.where(a == b, b, value)
    return value


@_triton_jit
def _maximum_reduce(a, b):
    value = maximum(a, b)
    if is_floating(a):
        value = tl.where(a == b, b, value)
    return value


@_triton_jit
def fmaximum(a, b):
    return tl.maximum(a, b)


@_triton_jit
def min2(a, dim):
    return tl.reduce(a, dim, minimum)


@_triton_jit
def max2(a, dim):
    return tl.reduce(a, dim, maximum)


@_triton_jit
def min2_strict(a, dim):
    return tl.reduce(a, dim, _minimum_reduce)


@_triton_jit
def max2_strict(a, dim):
    return tl.reduce(a, dim, _maximum_reduce)


@_triton_jit
def fmax2(a, dim):
    return tl.reduce(a, dim, fmaximum)


@_triton_jit
def minimum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan & (not b_isnan)
        equal |= a_isnan & b_isnan
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@_triton_jit
def maximum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value > b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan & (not b_isnan)
        equal |= a_isnan & b_isnan
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@_triton_jit
def min_with_index(value, index, dim):
    return tl.reduce((value, index), dim, minimum_with_index)


@_triton_jit
def max_with_index(value, index, dim):
    return tl.reduce((value, index), dim, maximum_with_index)


@_triton_jit
def exp(x, use_fast_math: tl.constexpr):
    if use_fast_math:
        return math.exp(x)
    return libdevice.exp(x)


@_triton_jit
def online_softmax_reduce(
    lhs_max,
    lhs_sum,
    dim,
    use_fast_math: tl.constexpr,
    strict_signed_zero: tl.constexpr,
):
    if strict_signed_zero:
        out_max = max2_strict(lhs_max, dim)
    else:
        out_max = max2(lhs_max, dim)
    out_max_keepdim = tl.expand_dims(out_max, dim)
    delta = tl.where(out_max_keepdim == float("-inf"), 0, lhs_max - out_max_keepdim)
    out_sum = tl.sum(lhs_sum * exp(delta, use_fast_math), dim)
    return out_max, out_sum


@_triton_jit
def online_softmax_combine(
    lhs_max,
    lhs_sum,
    rhs_max,
    use_fast_math: tl.constexpr,
    strict_signed_zero: tl.constexpr,
):
    if strict_signed_zero:
        out_max = _maximum_reduce(lhs_max, rhs_max)
    else:
        out_max = maximum(lhs_max, rhs_max)
    lhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(lhs_max - out_max, use_fast_math)
    )
    rhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(rhs_max - out_max, use_fast_math)
    )
    out_sum = lhs_sum * lhs_scale + rhs_scale
    return out_max, out_sum


@_triton_jit
def online_softmax_combine_with_sum(
    lhs_max,
    lhs_sum,
    rhs_max,
    rhs_sum,
    use_fast_math: tl.constexpr,
    strict_signed_zero: tl.constexpr,
):
    if strict_signed_zero:
        out_max = _maximum_reduce(lhs_max, rhs_max)
    else:
        out_max = maximum(lhs_max, rhs_max)
    lhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(lhs_max - out_max, use_fast_math)
    )
    rhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(rhs_max - out_max, use_fast_math)
    )
    out_sum = lhs_sum * lhs_scale + rhs_sum * rhs_scale
    return out_max, out_sum
