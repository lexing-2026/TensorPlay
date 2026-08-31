"""Primitive operations defined over the eager tensor API.

Each primitive is a strict, non-promoting, non-broadcasting kernel whose only
job is to compute a single op with fully specified metadata.  Reference ops
(decompositions) reduce every public function down to these kernels, so
compilers and tracing can rely on a small, well-specified vocabulary.

Every primitive carries a *meta* function (shape/dtype/stride inference) that
is re-run eagerly before the kernel executes, and is served to the tracing
layers when the operator is captured symbolically.  Primitive kernels never
perform implicit type promotion or broadcasting; callers must do that first
(e.g. via :mod:`tensorplay.primitives.common`).
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from enum import Enum
from functools import partial, reduce
from typing import Any, Optional, Union

import tensorplay
from tensorplay import library as _prims_library
from tensorplay import special as _tp_special
from tensorplay import fft as _tp_fft

from . import common as utils
from .common import (
    Dim,
    DimsSequenceType,
    DimsType,
    IntWithoutSymInt,
    Number,
    RETURN_TYPE,
    ShapeType,
    StrideType,
    TensorLike,
    TensorLikeType,
    TensorSequenceType,
    type_to_dtype,
)
from .common.wrappers import backwards_not_supported
from .debug_prims import register_debug_prims
from .rng_prims import register_rng_prims

# Re-exports kept for callers that refer to the aliases through the package
# root (type annotations and helper consumers).
from .common import (  # noqa: F401
    NumberType,
    NumberTypeType,
    RealNumberType,
    BoolLikeType,
    DeviceLikeType,
    Tensor,
    dtype,
    device,
    check,
    canonicalize_dim,
    canonicalize_dims,
    canonicalize_device,
    check_same_device,
    check_same_shape,
    compute_required_storage_length,
    extract_dims_from_varargs,
    extract_shape,
    extract_shape_from_varargs,
    get_computation_dtype,
    infer_size,
    infer_size_shapes,
    is_cpu_scalar_tensor,
    make_contiguous_strides_for,
    mask_tensor,
    prod,
    reduction_dims,
    validate_dim_length,
    validate_shape,
    validate_strides,
)

__all__ = [
    "RETURN_TYPE",
    "Tensor",
    "TensorLike",
    "TensorLikeType",
    "ShapeType",
    "StrideType",
    "DimsType",
    "DimsSequenceType",
    "Number",
    "NumberType",
]


def TensorMeta(
    tensorlike: Number | TensorLikeType | None = None,
    *,
    shape: ShapeType | None = None,
    strides: StrideType | None = None,
    dtype: Any | None = None,
    device: Any | None = None,
):
    """Produces a zero-element tensor that carries only metadata.

    The returned tensor is never executed; it exists so shape/stride/dtype
    inference can be checked eagerly while staying cheap.
    """
    if isinstance(tensorlike, Number):
        if shape is not None and not isinstance(shape, Sequence):
            raise AssertionError(
                f"shape must be None or a Sequence for Number input, got {type(shape)}"
            )
        if strides is not None and not isinstance(strides, Sequence):
            raise AssertionError(
                f"strides must be None or a Sequence for Number input, got {type(strides)}"
            )
        inferred_shape: tuple[int, ...] = ()
        inferred_strides: tuple[int, ...] = ()
        inferred_dtype = type_to_dtype(type(tensorlike))
        inferred_device = tensorplay.device("cpu")
    elif tensorlike is not None:
        if not isinstance(tensorlike, TensorLike):
            raise AssertionError(f"tensorlike must be a tensor, got {type(tensorlike)}")
        inferred_shape = tensorlike.shape
        inferred_strides = tensorlike.stride()
        inferred_dtype = tensorlike.dtype
        inferred_device = tensorlike.device
    else:
        if shape is None:
            raise AssertionError("shape must be provided when tensorlike is None")
        if strides is None:
            raise AssertionError("strides must be provided when tensorlike is None")
        if dtype is None:
            raise AssertionError("dtype must be provided when tensorlike is None")
        if device is None:
            raise AssertionError("device must be provided when tensorlike is None")

    shape = inferred_shape if shape is None else tuple(shape)
    strides = inferred_strides if strides is None else tuple(strides)
    dtype = inferred_dtype if dtype is None else dtype
    device = inferred_device if device is None else device

    if isinstance(device, str):
        device = tensorplay.device(device)

    return tensorplay.empty_strided(shape, strides, dtype=dtype, device=device)


class ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = 0
    NO_OPMATH = 1
    INT_TO_FLOAT = 2
    COMPLEX_TO_FLOAT = 3
    BOOL_TO_LONG = 4
    ALWAYS_BOOL = 5


# Meta function for elementwise prims.  This intentionally does NOT do full
# type promotion; it only answers "what output metadata does this kernel
# produce" given the promotion rules the kernel is declared with.
def _prim_elementwise_meta(
    *args,
    type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND,
    args_with_fixed_dtypes=None,
):
    if len(args) == 0:
        raise AssertionError("elementwise operation requires at least one argument")

    utils.check_same_dtype(*args)

    args_ = args
    if args_with_fixed_dtypes is not None:
        args_ = list(args_with_fixed_dtypes) + list(args)

    utils.check_same_device(*args_, allow_cpu_scalar_tensors=True)
    utils.check_same_shape(*args_, allow_cpu_scalar_tensors=True)

    l2p_perm, _ = utils.compute_elementwise_output_logical_to_physical_perm(*args_)
    shape = utils.extract_shape(*args_, allow_cpu_scalar_tensors=True)

    # Acquires the dtype
    dtype = None
    scalar_type = None
    for arg in args:
        if isinstance(arg, TensorLike):
            dtype = arg.dtype
            break
        elif isinstance(arg, Number):
            scalar_type = type(arg)

    if dtype is None and scalar_type is not None:
        dtype = utils.type_to_dtype(scalar_type)

    # Acquires the device (if it exists); zero-dimensional CPU scalar tensors
    # do not pin the output device when a larger tensor participates.
    device = None
    number = None
    for arg in args_:
        if isinstance(arg, TensorLike):
            if utils.is_cpu_scalar_tensor(arg):
                if device is None:
                    device = arg.device
                continue
            device = arg.device
            break
        elif isinstance(arg, Number):
            if number is None:
                number = arg

    if device is not None:
        if dtype is None:
            raise AssertionError("dtype must not be None when device is not None")
        if type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
            dtype = tensorplay.bool
        elif type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.INT_TO_FLOAT:
            if utils.is_integer_dtype(dtype) or utils.is_boolean_dtype(dtype):
                dtype = tensorplay.get_default_dtype()
        elif type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT:
            if utils.is_complex_dtype(dtype):
                dtype = utils.corresponding_real_dtype(dtype)

        if shape is None:
            raise AssertionError("shape must not be None when device is not None")
        return _empty_permuted(shape, l2p_perm, device=device, dtype=dtype)

    return TensorMeta(number) if number is not None else TensorMeta(dtype=dtype, device=tensorplay.device("cpu"), shape=(), strides=())


def _empty_permuted(shape, physical_layout, *, device=None, dtype=None):
    """Creates an empty tensor laid out according to a physical dimension order.

    ``physical_layout[p]`` names the logical dimension occupying physical
    slot ``p``: slot 0 is the outermost (largest-stride) dimension and the
    last slot is the innermost (stride-one) dimension. The result is always
    non-overlapping and dense.
    """
    strides = _empty_permuted_layout_strides(shape, physical_layout)
    return tensorplay.empty_strided(shape, strides, device=device, dtype=dtype)


def _complex_only_elementwise_meta(*args, **kwargs):
    utils.check(
        utils.is_complex_dtype(args[0].dtype), lambda: "Only complex dtype is supported"
    )
    return _prim_elementwise_meta(*args, **kwargs)


def _make_elementwise_unary_prim(
    name: str, *, type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND, **kwargs
):
    """Creates an elementwise unary primitive."""
    return _make_prim(
        schema=f"{name}(Tensor self) -> Tensor",
        meta=partial(_prim_elementwise_meta, type_promotion=type_promotion),
        return_type=RETURN_TYPE.NEW,
        **kwargs,
    )


def _make_elementwise_binary_prim(
    name: str, *, type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND, **kwargs
):
    """Creates an elementwise binary primitive."""
    return _make_prim(
        schema=f"{name}(Tensor self, Tensor other) -> Tensor",
        meta=partial(_prim_elementwise_meta, type_promotion=type_promotion),
        return_type=RETURN_TYPE.NEW,
        **kwargs,
    )


def _not_impl(*args, **kwargs):
    raise NotImplementedError


def _make_prim(
    *,
    schema: str,
    return_type: RETURN_TYPE | tuple[RETURN_TYPE, ...],
    meta: Callable,
    impl_aten: Callable,
    doc: str,
    tags: Sequence[Any] | None = None,
    register_conj_neg_fallthrough: bool = False,
):
    """Creates a primitive operation: a named library operator whose kernel runs
    the eager implementation after validating the metadata function."""

    def _prim_impl(*args, **kwargs):
        # Always run the meta function first: the eager implementation often
        # accepts more inputs (promotion, broadcasting) that the primitive is
        # meant to reject.
        meta(*args, **kwargs)
        return impl_aten(*args, **kwargs)

    name = schema.split("(", maxsplit=1)[0]
    qualified = f"prims::{name}"
    full_schema = f"{qualified}{schema[len(name):]}"
    prim_def = tensorplay.library.custom_op(qualified, _prim_impl, schema=full_schema)
    try:
        prim_def.register_fake(meta)
    except Exception:
        # Scalar-returning prims cannot always be summarized by a fake
        # tensor; registration of the eager kernel still proceeds.
        pass

    prim_def._schema = schema
    prim_def._meta = meta
    prim_def._impl_aten = impl_aten
    prim_def._return_type = return_type
    prim_def._doc = doc
    return prim_def

#
# Elementwise unary operations
#

abs = _make_elementwise_unary_prim(
    "abs", impl_aten=tensorplay.abs, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)

acos = _make_elementwise_unary_prim(
    "acos", impl_aten=tensorplay.acos, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

acosh = _make_elementwise_unary_prim(
    "acosh", impl_aten=tensorplay.acosh, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

asin = _make_elementwise_unary_prim(
    "asin", impl_aten=tensorplay.asin, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

asinh = _make_elementwise_unary_prim(
    "asinh", impl_aten=tensorplay.asinh, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atan = _make_elementwise_unary_prim(
    "atan", impl_aten=tensorplay.atan, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atanh = _make_elementwise_unary_prim(
    "atanh", impl_aten=tensorplay.atanh, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

cos = _make_elementwise_unary_prim(
    "cos", impl_aten=tensorplay.cos, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

cosh = _make_elementwise_unary_prim(
    "cosh", impl_aten=tensorplay.cosh, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_j0 = _make_elementwise_unary_prim(
    "bessel_j0", impl_aten=tensorplay.bessel_j0, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_j1 = _make_elementwise_unary_prim(
    "bessel_j1", impl_aten=tensorplay.bessel_j1, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i0 = _make_elementwise_unary_prim(
    "bessel_i0", impl_aten=tensorplay.i0, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i0e = _make_elementwise_unary_prim(
    "bessel_i0e", impl_aten=_tp_special.i0e, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i1 = _make_elementwise_unary_prim(
    "bessel_i1", impl_aten=_tp_special.i1, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i1e = _make_elementwise_unary_prim(
    "bessel_i1e", impl_aten=_tp_special.i1e, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_not = _make_elementwise_unary_prim(
    "bitwise_not", impl_aten=tensorplay.bitwise_not, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _cbrt_aten(a: TensorLikeType) -> TensorLikeType:
    utils.check(
        not a.is_complex(),
        lambda: "cbrt is only defined for floating point tensors",
    )
    return tensorplay.copysign(a.abs() ** (1 / 3), a)


cbrt = _make_elementwise_unary_prim(
    "cbrt", impl_aten=_cbrt_aten, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

ceil = _make_elementwise_unary_prim(
    "ceil", impl_aten=tensorplay.ceil, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _conj_physical_meta(input: TensorLikeType) -> TensorLikeType:
    if not input.dtype.is_complex:
        raise RuntimeError("conj_physical is only defined for complex dtypes")
    strides = utils.compute_elementwise_output_strides(input)
    return TensorMeta(input, strides=strides)


conj_physical = _make_prim(
    schema="conj_physical(Tensor self) -> Tensor",
    meta=_conj_physical_meta,
    impl_aten=tensorplay.conj,
    doc="Returns the physical conjugation of a complex tensor",
    return_type=RETURN_TYPE.NEW,
)


def _clone_meta(
    input: TensorLikeType, *, memory_format: Any = None,
) -> TensorLikeType:
    if memory_format is not None and memory_format != tensorplay.preserve_format:
        return tensorplay.empty(
            input.shape, dtype=input.dtype, device=input.device,
        )
    computed_stride = utils.compute_elementwise_output_strides(input)
    return tensorplay.empty_strided(
        input.shape, computed_stride, dtype=input.dtype, device=input.device,
    )


clone = _make_prim(
    schema="clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor",
    meta=_clone_meta,
    impl_aten=tensorplay.clone,
    doc="Returns the copy of a tensor",
    return_type=RETURN_TYPE.NEW,
    register_conj_neg_fallthrough=True,
)

digamma = _make_elementwise_unary_prim(
    "digamma", impl_aten=_tp_special.digamma if hasattr(tensorplay.special, "digamma") else tensorplay.digamma, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erf = _make_elementwise_unary_prim(
    "erf", impl_aten=tensorplay.erf, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erf_inv = _make_elementwise_unary_prim(
    "erf_inv", impl_aten=_tp_special.erfinv, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erfc = _make_elementwise_unary_prim(
    "erfc", impl_aten=_tp_special.erfc, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erfcx = _make_elementwise_unary_prim(
    "erfcx", impl_aten=_tp_special.erfcx, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

exp = _make_elementwise_unary_prim(
    "exp", impl_aten=tensorplay.exp, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

expm1 = _make_elementwise_unary_prim(
    "expm1", impl_aten=_tp_special.expm1, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

exp2 = _make_elementwise_unary_prim(
    "exp2", impl_aten=_tp_special.exp2, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _fill_meta(a: TensorLikeType, value: Number) -> TensorLikeType:
    return _prim_elementwise_meta(
        a, type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT
    )


# NOTE: fill uses _make_prim directly because it has a value parameter
fill = _make_prim(
    schema="fill(Tensor self, Scalar value) -> Tensor",
    return_type=RETURN_TYPE.NEW,
    meta=_fill_meta,
    impl_aten=tensorplay.fill,
    doc="",
)

floor = _make_elementwise_unary_prim(
    "floor", impl_aten=tensorplay.floor, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

imag = _make_prim(
    schema="imag(Tensor(a) self) -> Tensor(a)",
    meta=partial(
        _complex_only_elementwise_meta,
        type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    ),
    return_type=RETURN_TYPE.VIEW,
    impl_aten=tensorplay.imag,
    doc="",
)

isfinite = _make_elementwise_unary_prim(
    "isfinite", impl_aten=tensorplay.isfinite, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

lgamma = _make_elementwise_unary_prim(
    "lgamma", impl_aten=tensorplay.lgamma, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log = _make_elementwise_unary_prim(
    "log", impl_aten=tensorplay.log, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log1p = _make_elementwise_unary_prim(
    "log1p", impl_aten=tensorplay.log1p, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log2 = _make_elementwise_unary_prim(
    "log2", impl_aten=tensorplay.log2, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log10 = _make_elementwise_unary_prim(
    "log10", impl_aten=tensorplay.log10, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

real = _make_prim(
    schema="real(Tensor(a) self) -> Tensor(a)",
    meta=partial(
        _complex_only_elementwise_meta,
        type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    ),
    return_type=RETURN_TYPE.VIEW,
    impl_aten=tensorplay.real,
    doc="",
)

reciprocal = _make_elementwise_unary_prim(
    "reciprocal", impl_aten=tensorplay.reciprocal, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

ndtri = _make_elementwise_unary_prim(
    "ndtri", impl_aten=_tp_special.ndtri, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

neg = _make_elementwise_unary_prim(
    "neg", impl_aten=tensorplay.neg, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

round = _make_elementwise_unary_prim(
    "round", impl_aten=tensorplay.round, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

rsqrt = _make_elementwise_unary_prim(
    "rsqrt", impl_aten=tensorplay.rsqrt, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sign = _make_elementwise_unary_prim(
    "sign", impl_aten=tensorplay.sign, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

signbit = _make_elementwise_unary_prim(
    "signbit", impl_aten=tensorplay.signbit, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sin = _make_elementwise_unary_prim(
    "sin", impl_aten=tensorplay.sin, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sinh = _make_elementwise_unary_prim(
    "sinh", impl_aten=tensorplay.sinh, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

spherical_bessel_j0 = _make_elementwise_unary_prim(
    "spherical_bessel_j0", impl_aten=_tp_special.spherical_bessel_j0, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sqrt = _make_elementwise_unary_prim(
    "sqrt", impl_aten=tensorplay.sqrt, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

tan = _make_elementwise_unary_prim(
    "tan", impl_aten=tensorplay.tan, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

tanh = _make_elementwise_unary_prim(
    "tanh", impl_aten=tensorplay.tanh, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

trunc = _make_elementwise_unary_prim(
    "trunc", impl_aten=tensorplay.trunc, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

#
# Elementwise binary operations
#

add = _make_elementwise_binary_prim(
    name="add", impl_aten=tensorplay.add, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atan2 = _make_elementwise_binary_prim(
    name="atan2", impl_aten=tensorplay.atan2, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_and = _make_elementwise_binary_prim(
    "bitwise_and", impl_aten=tensorplay.bitwise_and, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_or = _make_elementwise_binary_prim(
    "bitwise_or", impl_aten=tensorplay.bitwise_or, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_xor = _make_elementwise_binary_prim(
    "bitwise_xor", impl_aten=tensorplay.bitwise_xor, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


# div performs truncation division on integer inputs and true division for
# floating and complex inputs.
def _div_aten(a, b):
    is_integral = isinstance(a, (bool, int)) or (
        isinstance(a, tensorplay.Tensor) and utils.is_integer_dtype(a.dtype)
    )
    if is_integral:
        return tensorplay.trunc(tensorplay.true_divide(a, b))
    return tensorplay.true_divide(a, b)


div = _make_elementwise_binary_prim(
    name="div", impl_aten=_div_aten, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

eq = _make_elementwise_binary_prim(
    name="eq", impl_aten=tensorplay.eq, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

fmax = _make_elementwise_binary_prim(
    name="fmax", impl_aten=tensorplay.fmax, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

fmin = _make_elementwise_binary_prim(
    name="fmin", impl_aten=tensorplay.fmin, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

fmod = _make_elementwise_binary_prim(
    name="fmod", impl_aten=tensorplay.fmod, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

gcd = _make_elementwise_binary_prim(
    name="gcd", impl_aten=tensorplay.gcd, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

ge = _make_elementwise_binary_prim(
    name="ge", impl_aten=tensorplay.ge, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

gt = _make_elementwise_binary_prim(
    name="gt", impl_aten=tensorplay.gt, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

hypot = _make_elementwise_binary_prim(
    name="hypot", impl_aten=tensorplay.hypot, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

igamma = _make_elementwise_binary_prim(
    name="igamma", impl_aten=_tp_special.gammainc, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

igammac = _make_elementwise_binary_prim(
    name="igammac", impl_aten=_tp_special.gammaincc, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

le = _make_elementwise_binary_prim(
    name="le", impl_aten=tensorplay.le, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

lt = _make_elementwise_binary_prim(
    name="lt", impl_aten=tensorplay.lt, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

# maximum/minimum kernels reject Python numbers, so scalar operands are
# lifted to zero-dimensional tensors of the other operand's dtype first.
def _maximum_aten(
    a: TensorLikeType | Number, b: TensorLikeType | Number
) -> TensorLikeType:
    if isinstance(a, TensorLike) and isinstance(b, Number):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(b, TensorLike) and isinstance(a, Number):
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)

    return tensorplay.maximum(a, b)  # type: ignore[arg-type]


maximum = _make_elementwise_binary_prim(
    name="maximum", impl_aten=_maximum_aten, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _minimum_aten(
    a: TensorLikeType | Number, b: TensorLikeType | Number
) -> TensorLikeType:
    if isinstance(a, TensorLike) and isinstance(b, Number):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(b, TensorLike) and isinstance(a, Number):
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)

    return tensorplay.minimum(a, b)  # type: ignore[arg-type]


minimum = _make_elementwise_binary_prim(
    name="minimum", impl_aten=_minimum_aten, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

mul = _make_elementwise_binary_prim(
    name="mul", impl_aten=tensorplay.mul, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

ne = _make_elementwise_binary_prim(
    name="ne", impl_aten=tensorplay.ne, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

nextafter = _make_elementwise_binary_prim(
    name="nextafter", impl_aten=tensorplay.nextafter, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

pow = _make_elementwise_binary_prim(
    name="pow", impl_aten=tensorplay.pow, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

remainder = _make_elementwise_binary_prim(
    name="remainder", impl_aten=tensorplay.remainder, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

shift_left = _make_elementwise_binary_prim(
    name="shift_left", impl_aten=tensorplay.bitwise_left_shift, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

shift_right_arithmetic = _make_elementwise_binary_prim(
    name="shift_right_arithmetic", impl_aten=tensorplay.bitwise_right_shift, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sub = _make_elementwise_binary_prim(
    name="sub", impl_aten=tensorplay.sub, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

zeta = _make_elementwise_binary_prim(
    name="zeta", impl_aten=_tp_special.zeta, doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _as_strided_meta(
    a: TensorLikeType, size: ShapeType, stride: StrideType, storage_offset: int
) -> TensorLikeType:
    assert a.device.type in ("cpu", "cuda")
    utils.check(
        len(size) == len(stride),
        lambda: f"size and stride must have the same length, got {len(size)} and {len(stride)}",
    )
    utils.check(
        storage_offset >= 0,
        lambda: f"storage_offset must be non-negative, got {storage_offset}",
    )
    for idx in range(len(size)):
        utils.check(
            stride[idx] >= 0 or size[idx] == 1,
            lambda: f"negative stride {stride[idx]} with size > 1 at dim {idx}",
        )
    utils.check_in_bounds_for_storage(
        size, stride, storage_offset, a.numel()
    )
    return tensorplay.as_strided(a, size, stride, storage_offset)


def _as_strided_aten(
    a: TensorLikeType, size: ShapeType, stride: StrideType, storage_offset: int
) -> TensorLikeType:
    return tensorplay.as_strided(a, size, stride, storage_offset)


_as_strided_doc = """
  Creates a view of the tensor with the given shape, strides, and storage offset.
  """

as_strided = _make_prim(
    schema="as_strided(Tensor(a!) a, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor(a!)",
    meta=_as_strided_meta,
    impl_aten=_as_strided_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_as_strided_doc,
)


def _broadcast_in_dim_meta(
    a: TensorLikeType, shape: ShapeType, broadcast_dimensions: DimsSequenceType
) -> TensorLikeType:
    # Special case of scalar tensor
    if a.ndim == 0 and len(broadcast_dimensions) != 0:
        raise RuntimeError("Cannot broadcast a scalar tensor with multiple dimensions!")

    # Validates that len(broadcast_dimensions) == a.ndim
    if a.ndim != len(broadcast_dimensions):
        msg = (
            f"len(broadcast_dimensions) ({len(broadcast_dimensions)}) "
            f"must be equal to a.ndim ({a.ndim})"
        )
        raise AssertionError(msg)

    # Validates that the shape is broadcastable to
    for idx, new_idx in enumerate(broadcast_dimensions):
        utils.check(
            a.shape[idx] == 1 or a.shape[idx] == shape[new_idx],
            lambda: f"{a.shape[idx]} must be broadcastable to {shape[new_idx]}",
        )

    if len(shape) < a.ndim:
        raise AssertionError(f"len(shape) ({len(shape)}) must be >= a.ndim ({a.ndim})")

    # broadcast_dimensions must be a strictly ascending sequence
    def _greater_than_reduce(acc, x):
        if not isinstance(x, int):
            raise AssertionError(f"broadcast_dimensions element must be Dim, got {type(x)}")
        if x <= acc:
            raise AssertionError(
                f"broadcast_dimensions must be strictly ascending: {x} <= {acc}"
            )
        if x >= len(shape):
            raise AssertionError(
                f"broadcast_dimension {x} out of bounds for shape of length {len(shape)}"
            )
        return x

    reduce(_greater_than_reduce, broadcast_dimensions, -1)

    new_strides = []
    original_idx = 0
    for idx in range(len(shape)):
        if idx in broadcast_dimensions:
            if a.shape[original_idx] == 1:
                if a.shape[original_idx] == shape[idx]:
                    new_strides.append(a.stride()[original_idx])
                else:
                    new_strides.append(0)
            else:
                utils.check(
                    a.shape[original_idx] == shape[idx],
                    lambda: f"non-broadcasting semantics require {a.shape[original_idx]} == {shape[idx]}",
                )
                new_strides.append(a.stride()[original_idx])
            original_idx = original_idx + 1
        else:
            if shape[idx] != 1:
                new_strides.append(0)
            elif original_idx == a.ndim:
                new_strides.append(1)
            else:
                new_strides.append(
                    a.stride()[original_idx] * a.shape[original_idx]
                )

    return a.as_strided(shape, new_strides, a.storage_offset())


def _broadcast_in_dim_aten(a, shape, broadcast_dimensions):
    s = list(shape)
    for broadcast_dimension in broadcast_dimensions:
        s[broadcast_dimension] = -1

    v = a
    for idx, x in enumerate(s):
        if x != -1:
            v = v.unsqueeze(idx)

    return v.expand(shape)


_broadcast_in_dim_doc = """
  Creates a view of a with the specified shape.

  Allows adding dimensions of any length and broadcasting
  dimensions of length one in a to any length.

  The location of the broadcast dimensions must be specified
  using the broadcast_dimensions argument. Changing the
  relative order of dimensions is not supported.
  """

broadcast_in_dim = _make_prim(
    schema="broadcast_in_dim(Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions) -> Tensor(a)",
    meta=_broadcast_in_dim_meta,
    impl_aten=_broadcast_in_dim_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_broadcast_in_dim_doc,
)


def _validate_collapse_args(a: Tensor, start: int, end: int) -> None:
    ndim = max(1, a.dim())
    utils.validate_idx(ndim, start)
    utils.validate_idx(ndim, end)
    utils.check(
        end >= start,
        lambda: f"Attempting to collapse but end, {end}, is less than start, {start}!",
    )


def _collapsed_shape(shape: ShapeType, start: int, end: int) -> tuple[int, ...]:
    shape = (1,) if len(shape) == 0 else tuple(shape)
    dim_length = 1
    for s in shape[start : end + 1]:
        dim_length = dim_length * s
    return shape[0:start] + (dim_length,) + shape[end + 1 :]


def _collapse_view_helper(
    a: TensorLikeType, start: int, end: int, must_be_valid: str | None
) -> tuple[ShapeType | None, StrideType | None]:
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")

    _validate_collapse_args(a, start, end)

    if a.ndim == 0:
        shape = (1,)
        strides = (1,)
    else:
        shape = a.shape
        strides = a.stride()

    if a.ndim == 0 or (end == start):
        return shape, strides

    valid_op = True
    if a.numel() != 0:
        for idx in range(end - 1, start - 1, -1):
            valid_op = valid_op and (
                shape[idx] == 1
                or shape[idx + 1] == 1
                or strides[idx] == strides[idx + 1] * shape[idx + 1]
            )
            if not valid_op:
                break

    valid_op = valid_op or a.numel() == 0

    if must_be_valid:
        utils.check(valid_op, lambda: must_be_valid)
    else:
        if not valid_op:
            return None, None

    # The collapsed dimension's stride is the smallest stride in the range,
    # skipping length-one dimensions.
    stride = strides[end]
    for idx in range(end - 1, start - 1, -1):
        if shape[idx] != 1:
            stride = min(stride, strides[idx])

    # The collapsed dimension's length is the product of the lengths in the
    # range; a zero anywhere makes it zero.
    length = shape[end]
    if length != 0:
        for idx in range(end - 1, start - 1, -1):
            if shape[idx] == 0:
                length = 0
                stride = 0
                break
            length = length * shape[idx]
    else:
        stride = 0

    new_shape = shape[:start] + (length,) + shape[end + 1 :]
    new_strides = strides[:start] + (stride,) + strides[end + 1 :]

    # An empty tensor is restrided as if it were contiguous.
    if a.numel() == 0:
        new_strides = utils.make_contiguous_strides_for(new_shape)

    return new_shape, new_strides


def _collapse_view_meta(
    a: TensorLikeType, start: int, end: int
) -> TensorLikeType:
    shape, strides = _collapse_view_helper(a, start, end, "collapsed view is not valid")
    return a.as_strided(shape, strides, a.storage_offset())


def _collapse_view_aten(a: Tensor, start: int, end: int) -> Tensor:
    shape, strides = _collapse_view_helper(a, start, end, "collapsed view is not valid")
    return a.as_strided(shape, strides, a.storage_offset())


collapse_view = _make_prim(
    schema="collapse_view(Tensor(a) a, int start, int end) -> Tensor(a)",
    meta=_collapse_view_meta,
    impl_aten=_collapse_view_aten,
    return_type=RETURN_TYPE.VIEW,
    doc="",
)


def _conj_meta(a: TensorLikeType) -> TensorLikeType:
    if not a.dtype.is_complex:
        raise RuntimeError("conj is only defined for complex dtypes")
    return TensorMeta(a)


conj = _make_prim(
    schema="conj(Tensor(a) a) -> Tensor(a)",
    meta=_conj_meta,
    impl_aten=tensorplay.conj,
    return_type=RETURN_TYPE.VIEW,
    doc="",
)


def expand_dims(
    a: TensorLikeType, dimensions: DimsSequenceType, ndim: int | None = None
) -> TensorLikeType:
    """Creates a view of a with dimensions of size one inserted.

    ``ndim`` allows interpreting ``a`` as having extra length-one dimensions
    (the caller guarantees they are present), so inserted dimensions may be
    canonicalized against the resulting rank.
    """
    if isinstance(dimensions, int):
        dimensions = (dimensions,)

    # ``ndim`` is the rank after insertion and overrides a.ndim for
    # canonicalizing the requested dimensions.
    out_ndim = a.ndim + len(dimensions) if ndim is None else ndim
    dims = sorted(utils.canonicalize_dims(out_ndim, dimensions))
    if len(set(dims)) != len(dims):
        raise ValueError(f"Received duplicate dimensions to expand in {dimensions}")

    new_shape = list(a.shape)
    for idx in dims:
        new_shape.insert(idx, 1)

    broadcast_dimensions = [idx for idx in range(len(new_shape)) if idx not in dims]
    return broadcast_in_dim(a, new_shape, broadcast_dimensions)



def _split_dim_meta(
    a: TensorLikeType, dim: int, outer_length: int
) -> TensorLikeType:
    utils.validate_idx(a.ndim, dim)
    utils.validate_dim_length(outer_length)
    utils.check(outer_length > 0, lambda: "outer_length must be positive")
    utils.check(
        a.shape[dim] % outer_length == 0,
        lambda: (
            f"Attempting to split dimension of length {a.shape[dim]}, "
            f"but outer length of {outer_length} divides it with a remainder!"
        ),
    )
    inner_length = a.shape[dim] // outer_length
    shape = a.shape[:dim] + (outer_length, inner_length) + a.shape[dim + 1 :]
    strides = (
        a.stride()[:dim]
        + (a.stride()[dim] * inner_length, a.stride()[dim])
        + a.stride()[dim + 1 :]
    )
    return a.as_strided(shape, strides, a.storage_offset())


def _split_dim_aten(a: Tensor, dim: int, outer_length: int) -> Tensor:
    inner_length = a.shape[dim] // outer_length
    new_shape = a.shape[:dim] + (outer_length, inner_length) + a.shape[dim + 1 :]
    return a.view(new_shape)


split_dim = _make_prim(
    schema="split_dim(Tensor(a) a, int dim, SymInt outer_length) -> Tensor(a)",
    meta=_split_dim_meta,
    impl_aten=_split_dim_aten,
    return_type=RETURN_TYPE.VIEW,
    doc="",
)


def _squeeze_meta(a: TensorLikeType, dimensions: Sequence) -> TensorLikeType:
    ndim = a.ndim
    dimensions = utils.canonicalize_dims(ndim, dimensions)
    for d in dimensions:
        utils.check(a.shape[d] == 1, lambda: "squeezing non-singleton dimensions is not allowed")
    new_shape = [s for idx, s in enumerate(a.shape) if idx not in dimensions]
    return a.reshape(new_shape)


squeeze = _make_prim(
    schema="squeeze(Tensor(a) a, int[] dimensions) -> Tensor(a)",
    meta=_squeeze_meta,
    impl_aten=lambda a, dimensions: a.reshape(
        [s for idx, s in enumerate(a.shape) if idx not in utils.canonicalize_dims(a.ndim, dimensions)]
    ),
    return_type=RETURN_TYPE.VIEW,
    doc="",
)


def _transpose_meta(a: TensorLikeType, permutation: DimsSequenceType) -> TensorLikeType:
    if len(permutation) != a.ndim:
        raise RuntimeError(
            f"transpose requires the permutation to have the same length as the "
            f"tensor rank, got {len(permutation)} and {a.ndim}"
        )
    dims = utils.canonicalize_dims(a.ndim, permutation)
    if len(set(dims)) != len(dims):
        raise RuntimeError("transpose permutation must not contain duplicate dims!")
    new_shape = [a.shape[d] for d in dims]
    new_strides = [a.stride()[d] for d in dims]
    return a.as_strided(new_shape, new_strides, a.storage_offset())


def _transpose_aten(a: Tensor, permutation: DimsSequenceType) -> Tensor:
    dims = utils.canonicalize_dims(a.ndim, permutation)
    new_shape = [a.shape[d] for d in dims]
    new_strides = [a.stride()[d] for d in dims]
    return a.as_strided(new_shape, new_strides, a.storage_offset())


transpose = _make_prim(
    schema="transpose(Tensor(a) a, int[] permutation) -> Tensor(a)",
    meta=_transpose_meta,
    impl_aten=_transpose_aten,
    return_type=RETURN_TYPE.VIEW,
    doc="",
)


def _view_of_meta(a: TensorLikeType) -> TensorLikeType:
    return tensorplay.as_strided(a, a.shape, a.stride(), a.storage_offset())


def _view_of_aten(a: Tensor) -> Tensor:
    return a.view(a.shape)


view_of = _make_prim(
    schema="view_of(Tensor(a) a) -> Tensor(a)",
    meta=_view_of_meta,
    impl_aten=_view_of_aten,
    return_type=RETURN_TYPE.VIEW,
    doc="",
)


def _view_element_type_meta(a: TensorLikeType, dtype: Any) -> TensorLikeType:
    utils.check(
        tuple(a.stride()) == utils.make_contiguous_strides_for(a.shape),
        lambda: "view_element_type only supports contiguous tensors",
    )
    return TensorMeta(a, dtype=dtype, strides=utils.make_contiguous_strides_for(a.shape))


def _view_element_type_aten(a: Tensor, dtype: Any) -> Tensor:
    utils.check(
        tuple(a.stride()) == utils.make_contiguous_strides_for(a.shape),
        lambda: "view_element_type only supports contiguous tensors",
    )
    return a.view(dtype)


view_element_type = _make_prim(
    schema="view_of_dtype(Tensor(a) a, ScalarType dtype) -> Tensor(a)",
    meta=_view_element_type_meta,
    impl_aten=_view_element_type_aten,
    return_type=RETURN_TYPE.VIEW,
    doc="",
)


def _as_strided_scatter_meta(
    input: TensorLikeType,
    src: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    storage_offset: int,
) -> TensorLikeType:
    utils.validate_shape(size)
    utils.validate_strides(stride)

    for idx in range(len(size)):
        utils.check(
            stride[idx] >= 0 or size[idx] == 1,
            lambda: f"negative stride {stride[idx]} with size > 1 at dim {idx}",
        )
    utils.check_in_bounds_for_storage(size, stride, storage_offset, input.numel())
    utils.check(
        utils.is_same_shape(src.shape, size),
        lambda: (
            f"expected src to have a size equal to the slice of self. "
            f"src size = {src.shape}, slice size = {size}"
        ),
    )
    return TensorMeta(input)


def _as_strided_scatter_aten(
    input: TensorLikeType,
    src: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    storage_offset: int,
) -> TensorLikeType:
    out = utils.clone_preserve_strides(input)
    out.as_strided(size, stride, storage_offset).copy_(src)
    return out


_as_strided_scatter_doc = """
    Creates a new tensor equivalent to ``out = input.clone()`` after mutation by
    ``out.as_strided(size, stride, storage_offset).copy_(src)``.
"""

as_strided_scatter = _make_prim(
    schema="as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor",
    meta=_as_strided_scatter_meta,
    impl_aten=_as_strided_scatter_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_as_strided_scatter_doc,
)


def _collapse_meta(a: Tensor, start: int, end: int) -> Tensor:
    shape = a.shape
    ndim = max(1, len(shape))
    utils.validate_idx(ndim, start)
    utils.validate_idx(ndim, end)
    utils.check(
        end >= start,
        lambda: f"Attempting to collapse but end, {end}, is less than start, {start}!",
    )
    new_shape = _collapsed_shape(shape, start, end)
    return tensorplay.empty_strided(
        new_shape, utils.make_contiguous_strides_for(new_shape), dtype=a.dtype, device=a.device
    )


def _collapse_aten(a: Tensor, start: int, end: int) -> Tensor:
    shape = a.shape
    ndim = max(1, len(shape))
    utils.validate_idx(ndim, start)
    utils.validate_idx(ndim, end)
    utils.check(
        end >= start,
        lambda: f"Attempting to collapse but end, {end}, is less than start, {start}!",
    )
    new_shape = _collapsed_shape(shape, start, end)
    return a.reshape(new_shape)


collapse = _make_prim(
    schema="collapse(Tensor a, int start, int end) -> Tensor",
    meta=_collapse_meta,
    impl_aten=_collapse_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


def _cat_meta(tensors: TensorSequenceType, dim: int) -> TensorLikeType:
    if len(tensors) == 0:
        raise RuntimeError("Cannot concatenate an empty list of tensors!")
    if len(tensors) == 1:
        return tensors[0]
    if not all(t.device == tensors[0].device for t in tensors[1:]):
        raise RuntimeError("Input tensors must be on the same device!")
    if not all(t.dtype == tensors[0].dtype for t in tensors[1:]):
        raise RuntimeError("Input tensors must have the same dtype!")
    ndim = tensors[0].ndim
    dim = utils.canonicalize_dim(ndim, dim)
    for tensor_idx, tensor in enumerate(tensors):
        if len(tensor.shape) != ndim:
            raise RuntimeError(
                f"All tensors must have the same number of dimensions. "
                f"Expected {ndim} but tensor {tensor_idx} has {len(tensor.shape)}"
            )
    for idx in range(ndim):
        if idx == dim:
            continue
        for tensor_idx, tensor in enumerate(tensors[1:], start=1):
            if tensor.shape[idx] != tensors[0].shape[idx]:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected {tensors[0].shape[idx]} in dimension {idx} but got "
                    f"{tensor.shape[idx]} for tensor number {tensor_idx} in the list"
                )
    shape = list(tensors[0].shape)
    total = 0
    for t in tensors:
        total = total + t.shape[dim]
    shape[dim] = total
    return tensorplay.empty_strided(
        shape, utils.make_contiguous_strides_for(shape), dtype=tensors[0].dtype, device=tensors[0].device
    )


def _cat_aten(tensors: tuple[Tensor, ...] | list[Tensor], dim: int) -> Tensor:
    return tensorplay.cat(tensors, dim=dim)


cat = _make_prim(
    schema="cat(Tensor[] tensors, int dim) -> Tensor",
    meta=_cat_meta,
    impl_aten=_cat_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


def _reshape_meta(a: TensorLikeType, shape: ShapeType):
    if len(shape) == 0 and a.ndim != 0:
        raise AssertionError("Cannot reshape a non-scalar tensor to a scalar shape!")
    utils.check(
        utils.prod(a.shape) == utils.prod(shape),
        lambda: f"shape '{list(shape)}' is invalid for input of size {a.numel()}",
    )
    return TensorMeta(a, shape=shape, strides=utils.make_contiguous_strides_for(shape))


def _reshape_aten(a: Tensor, shape: ShapeType) -> Tensor:
    return tensorplay.reshape(a, shape)


reshape = _make_prim(
    schema="reshape(Tensor a, SymInt[] shape) -> Tensor",
    meta=_reshape_meta,
    impl_aten=_reshape_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


def _rev_meta(a: TensorLikeType, dims: DimsSequenceType) -> TensorLikeType:
    utils.validate_dimension_indices(a.ndim, dims)
    return TensorMeta(
        a, strides=utils.compute_elementwise_output_strides(a)
    )


def _rev_aten(a: Tensor, dims: DimsSequenceType) -> Tensor:
    return tensorplay.flip(a, dims=dims)


rev = _make_prim(
    schema="rev(Tensor a, int[] dims) -> Tensor",
    meta=_rev_meta,
    impl_aten=_rev_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


def _where_meta(
    pred: TensorLikeType, a: TensorLikeType, b: TensorLikeType
) -> TensorLikeType:
    # pred fixes the "input" dtype slot; the output dtype comes from a/b.
    return _prim_elementwise_meta(
        a,
        b,
        type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        args_with_fixed_dtypes=(pred,),
    )


def _where_aten(
    pred: TensorLikeType, a: TensorLikeType, b: TensorLikeType
) -> TensorLikeType:
    return tensorplay.where(pred, a, b)


where = _make_prim(
    schema="where(Tensor pred, Tensor a, Tensor b) -> Tensor",
    meta=_where_meta,
    impl_aten=_where_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


def _convert_element_type_meta(a: TensorLikeType, dtype: Any) -> TensorLikeType:
    return TensorMeta(a, dtype=dtype, strides=utils.compute_elementwise_output_strides(a))


def _convert_element_type_aten(a: Tensor, dtype: Any) -> Tensor:
    return a.to(dtype)


convert_element_type = _make_prim(
    schema="convert_element_type(Tensor a, ScalarType dtype) -> Tensor",
    meta=_convert_element_type_meta,
    impl_aten=_convert_element_type_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


def _device_put_meta(
    a: TensorLikeType,
    device: Any,
    non_blocking: bool = False,
) -> TensorLikeType:
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")
    if not isinstance(device, (str, tensorplay.device)):
        raise AssertionError(f"device must be str or tensorplay.device, got {type(device)}")
    if not isinstance(non_blocking, bool):
        raise AssertionError(f"non_blocking must be bool, got {type(non_blocking)}")
    return TensorMeta(a, device=utils.canonicalize_device(device))


def _device_put_aten(a: Tensor, device: Any, non_blocking: bool = False) -> Tensor:
    return a.to(device, non_blocking=non_blocking)


_device_put_doc = """
  Creates a copy of a tensor on the given device.
  """

device_put = _make_prim(
    schema="device_put(Tensor a, Device device, bool non_blocking=False) -> Tensor",
    meta=_device_put_meta,
    impl_aten=_device_put_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_device_put_doc,
)


_item_doc = """
    Converts a tensor with one element to a Python number.
"""


def _item_meta(a: TensorLikeType):
    return TensorMeta(shape=(), strides=(), dtype=a.dtype, device=a.device)


def _item_aten(*args, **kwargs):
    return tensorplay.Tensor.item(*args, **kwargs)


item = _make_prim(
    schema="item(Tensor a) -> Scalar",
    meta=_item_meta,
    impl_aten=_item_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_item_doc,
)


def _maximum_value_meta(dtype: Any):
    number_type = utils.dtype_to_type(dtype)
    return TensorMeta(number_type(-1))


def _maximum_value_aten(dtype: Any):
    if dtype == tensorplay.bool:
        return True
    elif dtype.is_complex or dtype.is_floating_point:
        return tensorplay.finfo(dtype).max
    else:
        return tensorplay.iinfo(dtype).max


maximum_value = _make_prim(
    schema="maximum_value(ScalarType dtype) -> Scalar",
    meta=_maximum_value_meta,
    impl_aten=_maximum_value_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


def _minimum_value_meta(dtype: Any):
    number_type = utils.dtype_to_type(dtype)
    return TensorMeta(number_type(-1))


def _minimum_value_aten(dtype: Any):
    if dtype == tensorplay.bool:
        return False
    elif dtype.is_complex or dtype.is_floating_point:
        return tensorplay.finfo(dtype).min
    else:
        return tensorplay.iinfo(dtype).min


minimum_value = _make_prim(
    schema="minimum_value(ScalarType dtype) -> Scalar",
    meta=_minimum_value_meta,
    impl_aten=_minimum_value_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


#
# Inplace operators
#


def _copy_to_meta(a: TensorLikeType, b: TensorLikeType):
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")
    if not isinstance(b, TensorLike):
        raise AssertionError(f"b must be TensorLike, got {type(b)}")
    if a.numel() != b.numel():
        msg = f"Attempting to copy {b.numel()} elements to a tensor with {a.numel()} elements!"
        raise RuntimeError(msg)
    return a


def _copy_to_aten(a: Tensor, b: Tensor) -> Tensor:
    return a.copy_(b)


copy_to = _make_prim(
    schema="copy_to(Tensor(a!) a, Tensor b) -> Tensor(a!)",
    meta=_copy_to_meta,
    impl_aten=_copy_to_aten,
    return_type=RETURN_TYPE.INPLACE,
    doc="Copies the data in b to a and returns the modified a.",
    register_conj_neg_fallthrough=True,
)


def _copy_strided_meta(a: TensorLikeType, stride: ShapeType):
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")
    return tensorplay.empty_strided(a.shape, stride, dtype=a.dtype, device=a.device)


def _copy_strided_aten(a: Tensor, stride: ShapeType) -> Tensor:
    out = tensorplay.empty_strided(a.shape, stride, dtype=a.dtype, device=a.device)
    out.copy_(a)
    return out


copy_strided = _make_prim(
    schema="copy_strided(Tensor a, SymInt[] stride) -> Tensor",
    meta=_copy_strided_meta,
    impl_aten=_copy_strided_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


def _resize_meta(a: TensorLikeType, shape: ShapeType):
    return a.resize_(shape)


def _resize_aten(a: Tensor, shape: ShapeType) -> Tensor:
    return a.resize_(shape)


resize = _make_prim(
    schema="resize(Tensor(a!) a, SymInt[] shape) -> Tensor(a!)",
    meta=_resize_meta,
    impl_aten=_resize_aten,
    return_type=RETURN_TYPE.INPLACE,
    doc="",
)


def _reduction_meta(inp, dims, *, output_dtype=None):
    """Meta function for single-output reduction operations."""
    if not isinstance(inp, TensorLike):
        raise AssertionError(f"inp must be TensorLike, got {type(inp)}")
    if output_dtype is None:
        output_dtype = inp.dtype
    output_shape = utils.compute_reduction_output_shape(inp.shape, dims)
    kwargs: dict[str, Any] = {}
    return tensorplay.empty_strided(
        output_shape, utils.make_contiguous_strides_for(output_shape),
        dtype=output_dtype, device=inp.device,
    )


def _var_reduction_meta(inp, dims, correction):
    if utils.is_complex_dtype(inp.dtype):
        output_dtype = utils.corresponding_real_dtype(inp.dtype)
    else:
        output_dtype = inp.dtype
    return _reduction_meta(inp, dims, output_dtype=output_dtype)


_sum_doc = """
    Computes the sum of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_xor_sum_doc = """
    Computes the xor sum of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_prod_doc = """
    Computes the product of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_amax_doc = """
    Computes the maximum value of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_amin_doc = """
    Computes the minimum value of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_var_doc = """
    Computes the biased variance of x over the list of dimensions specified in the dim argument
    """


def _make_reduction_prim(name: str, impl_aten, doc):
    """Creates a reduction primitive."""
    return _make_prim(
        schema=f"{name}(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor",
        meta=_reduction_meta,
        impl_aten=impl_aten,
        return_type=RETURN_TYPE.NEW,
        doc=doc,
    )


def _make_var_reduction_prim(name: str, impl_aten, doc):
    """Creates a variance reduction primitive."""
    return _make_prim(
        schema=f"{name}(Tensor inp, int[]? dims, float? correction=1, *, ScalarType? output_dtype=None) -> Tensor",
        meta=_var_reduction_meta,
        impl_aten=impl_aten,
        return_type=RETURN_TYPE.NEW,
        doc=doc,
    )


def _sum_aten(inp, dims, *, output_dtype=None):
    kwargs = {"dtype": output_dtype} if output_dtype is not None else {}
    if dims is None:
        return tensorplay.sum(inp, **kwargs)
    return tensorplay.sum(inp, dim=tuple(dims), **kwargs)


sum = _make_reduction_prim(
    name="sum",
    impl_aten=_sum_aten,
    doc=_sum_doc,
)


def _xor_sum_aten(inp, dims, *, dtype=None):
    raise NotImplementedError("xor_sum only implemented with inductor")


xor_sum = _make_reduction_prim(name="xor_sum", impl_aten=_xor_sum_aten, doc=_xor_sum_doc)
def _prod_aten(inp, dims, *, dtype=None):
    if dims is not None:
        if len(dims) == 0:
            return tensorplay.clone(inp)
        for d in sorted(dims, reverse=True):
            if d < 0:
                raise AssertionError(f"dimension must be non-negative, got {d}")
            kw = {"dtype": dtype} if dtype is not None else {}
            inp = tensorplay.prod(inp, d, **kw)
        return inp
    kw = {"dtype": dtype} if dtype is not None else {}
    return tensorplay.prod(inp, **kw)


prod = _make_reduction_prim(name="prod", impl_aten=_prod_aten, doc=_prod_doc)


def _var_aten(inp, dims, correction, *, output_dtype=None):
    kwargs = {"dtype": output_dtype} if output_dtype is not None else {}
    if dims is None:
        return tensorplay.var(inp, correction=correction, **kwargs)
    return tensorplay.var(inp, dim=tuple(dims), correction=correction, **kwargs)


var = _make_var_reduction_prim(name="var", impl_aten=_var_aten, doc=_var_doc)


def _amax_aten(inp, dims, *, output_dtype=None):
    kwargs = {"dtype": output_dtype} if output_dtype is not None else {}
    if dims is None:
        return tensorplay.amax(inp, **kwargs)
    return tensorplay.amax(inp, dim=tuple(dims), **kwargs)


def _amin_aten(inp, dims, *, output_dtype=None):
    kwargs = {"dtype": output_dtype} if output_dtype is not None else {}
    if dims is None:
        return tensorplay.amin(inp, **kwargs)
    return tensorplay.amin(inp, dim=tuple(dims), **kwargs)


amax = _make_reduction_prim(name="amax", impl_aten=_amax_aten, doc=_amax_doc)
amin = _make_reduction_prim(name="amin", impl_aten=_amin_aten, doc=_amin_doc)


def _iota_meta(
    length: int,
    start: int,
    step: int,
    dtype: Any,
    device: Any,
    requires_grad: bool,
) -> TensorLikeType:
    utils.check(
        utils.is_integer_dtype(dtype),
        lambda: "prims.iota only supports integer dtypes",
    )
    utils.check(step != 0, lambda: "step must be nonzero")
    utils.check(
        not (requires_grad and (utils.is_integer_dtype(dtype) or utils.is_boolean_dtype(dtype))),
        lambda: "Cannot create integer or boolean tensor with requires_grad=True!",
    )
    return TensorMeta(dtype=dtype, device=device, shape=(length,), strides=(1,))


def _iota_aten(
    length: int,
    start: int,
    step: int,
    dtype: Any,
    device: Any,
    requires_grad: bool,
) -> TensorLikeType:
    utils.check(step != 0, lambda: "step must be nonzero")
    end = start + step * length
    a = tensorplay.arange(start, end, step, dtype=dtype, device=device)
    if a.numel() > length:
        a = a[:length]
    if requires_grad:
        a.requires_grad = True
    return a


iota = _make_prim(
    schema="iota(SymInt length, *, SymInt start, SymInt step, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_iota_meta,
    impl_aten=_iota_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


def _empty_meta(
    shape: ShapeType,
    dtype: Any,
    device: Any,
    requires_grad: bool,
) -> TensorLikeType:
    utils.check(
        not (requires_grad and (utils.is_integer_dtype(dtype) or utils.is_boolean_dtype(dtype))),
        lambda: "Cannot create integer or boolean tensor with requires_grad=True!",
    )
    return TensorMeta(dtype=dtype, device=device, shape=tuple(shape), strides=utils.make_contiguous_strides_for(shape))


def _empty_aten(
    shape: ShapeType,
    dtype: Any,
    device: Any,
    requires_grad: bool,
) -> TensorLikeType:
    return _empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)


def _empty(shape, *, dtype=None, device=None, requires_grad=False):
    a = tensorplay.empty(shape, dtype=dtype, device=device)
    if requires_grad:
        a.requires_grad = True
    return a


empty = _make_prim(
    schema="empty(SymInt[] shape, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_empty_meta,
    impl_aten=_empty_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


def _empty_strided_meta(
    shape: ShapeType,
    strides: StrideType,
    dtype: Any,
    device: Any,
    requires_grad: bool,
) -> TensorLikeType:
    utils.check(
        not (requires_grad and (utils.is_integer_dtype(dtype) or utils.is_boolean_dtype(dtype))),
        lambda: "Cannot create integer or boolean tensor with requires_grad=True!",
    )
    return TensorMeta(dtype=dtype, device=device, shape=tuple(shape), strides=tuple(strides))


def _empty_strided_aten(
    shape: ShapeType,
    strides: StrideType,
    dtype: Any,
    device: Any,
    requires_grad: bool,
) -> TensorLikeType:
    a = tensorplay.empty_strided(shape, strides, dtype=dtype, device=device)
    if requires_grad:
        a.requires_grad = True
    return a


empty_strided = _make_prim(
    schema="empty_strided(SymInt[] shape, SymInt[] strides, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_empty_strided_meta,
    impl_aten=_empty_strided_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


def _empty_permuted_layout_strides(
    shape: ShapeType, physical_layout: DimsSequenceType
) -> tuple[int, ...]:
    """Validates a physical layout and returns the resulting dense strides."""
    shape = tuple(shape)
    dim = len(shape)
    physical_layout = tuple(physical_layout)
    utils.check(
        len(physical_layout) == dim,
        lambda: (
            f"Number of dimensions in the tensor input does not match the "
            f"length of the physical layout; i.e. len(size) = {dim} "
            f"is not equal to len(physical_layout) = {len(physical_layout)}"
        ),
    )
    p_strides = utils.make_contiguous_strides_for([shape[l] for l in physical_layout])
    strides = [0] * dim
    seen_dims = set()
    for p, l in enumerate(physical_layout):
        utils.check(
            0 <= l < dim,
            lambda: (
                f"Dimension out of range (expected to be between 0 and {dim - 1}, "
                f"but got {l} at index {p}). NB: negative dims not currently supported"
            ),
        )
        utils.check(l not in seen_dims, lambda: "Duplicate dim not allowed")
        strides[l] = p_strides[p]
        seen_dims.add(l)
    return tuple(strides)


def _empty_permuted_meta(
    shape: ShapeType,
    physical_layout: StrideType,
    dtype: Any,
    device: Any,
    requires_grad: bool,
) -> TensorLikeType:
    strides = _empty_permuted_layout_strides(shape, physical_layout)
    return TensorMeta(dtype=dtype, device=device, shape=tuple(shape), strides=strides)


def _empty_permuted_aten(
    shape: ShapeType,
    physical_layout: StrideType,
    dtype: Any,
    device: Any,
    requires_grad: bool,
) -> TensorLikeType:
    a = _empty_permuted(tuple(shape), physical_layout, dtype=dtype, device=device)
    if requires_grad:
        a.requires_grad = True
    return a


empty_permuted = _make_prim(
    schema="empty_permuted(SymInt[] shape, int[] physical_layout, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_empty_permuted_meta,
    impl_aten=_empty_permuted_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


def _full_meta(
    shape: ShapeType,
    fill_value: Number,
    dtype: Any,
    device: Any,
    requires_grad: bool,
) -> TensorLikeType:
    utils.check(
        not (requires_grad and (utils.is_integer_dtype(dtype) or utils.is_boolean_dtype(dtype))),
        lambda: "Cannot create integer or boolean tensor with requires_grad=True!",
    )
    return TensorMeta(dtype=dtype, device=device, shape=tuple(shape), strides=utils.make_contiguous_strides_for(shape))


def _full_aten(
    shape: ShapeType,
    fill_value: Number,
    dtype: Any,
    device: Any,
    requires_grad: bool,
) -> TensorLikeType:
    if isinstance(fill_value, complex) and not utils.is_complex_dtype(dtype):
        raise AssertionError("complex fill values require a complex dtype")
    a = tensorplay.full(shape, fill_value, dtype=dtype, device=device)
    if requires_grad:
        a.requires_grad = True
    return a


full = _make_prim(
    schema="full(SymInt[] shape, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_full_meta,
    impl_aten=_full_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


def _full_like_meta(
    a: TensorLikeType,
    fill_value: Number,
    dtype: Any,
    device: Any,
    requires_grad: bool,
) -> TensorLikeType:
    return _full_meta(a.shape, fill_value, dtype, device, requires_grad)


def _full_like_aten(
    a: TensorLikeType,
    fill_value: Number,
    dtype: Any,
    device: Any,
    requires_grad: bool,
) -> TensorLikeType:
    if isinstance(fill_value, complex) and not (dtype.is_complex if hasattr(dtype, "is_complex") else utils.is_complex_dtype(dtype)):
        raise AssertionError("complex fill values require a complex dtype")
    a = tensorplay.full(a.shape, fill_value, dtype=dtype, device=device)
    if requires_grad:
        a.requires_grad = True
    return a


full_like = _make_prim(
    schema="full_like(Tensor a, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_full_like_meta,
    impl_aten=_full_like_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


def _scalar_tensor_meta(
    scalar: Number,
    dtype: Any = None,
    device: Any = None,
) -> TensorLikeType:
    return TensorMeta(
        shape=(),
        strides=(),
        dtype=dtype if dtype is not None else utils.type_to_dtype(type(scalar)),
        device=device if device is not None else tensorplay.device("cpu"),
    )


def _scalar_tensor_aten(
    scalar: Number,
    dtype: Any = None,
    device: Any = None,
) -> TensorLikeType:
    if isinstance(scalar, complex) and (
        dtype is None or not utils.is_complex_dtype(dtype)
    ):
        raise TypeError("Complex scalar requires complex tensor dtype.")
    return tensorplay.tensor(scalar, dtype=dtype, device=device)


scalar_tensor = _make_prim(
    schema="scalar_tensor(Scalar s, *, ScalarType? dtype=None, Device? device=None) -> Tensor",
    meta=_scalar_tensor_meta,
    impl_aten=_scalar_tensor_aten,
    return_type=RETURN_TYPE.NEW,
    doc="",
)


def _normal_meta(
    shape: ShapeType,
    *,
    mean: float,
    std: float,
    dtype: Any,
    device: Any,
    requires_grad: bool,
) -> TensorLikeType:
    utils.check(
        std >= 0,
        lambda: f"expected non-negative standard deviation, but got std={std}",
    )
    utils.check(
        utils.is_float_dtype(dtype) or utils.is_complex_dtype(dtype),
        lambda: f"expected a floating-point or complex dtype, but got dtype={dtype}",
    )
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)


def _normal_aten(
    shape: ShapeType,
    *,
    mean: float,
    std: float,
    dtype: Any,
    device: Any,
    requires_grad: bool,
) -> TensorLikeType:
    utils.check(
        std >= 0,
        lambda: f"expected non-negative standard deviation, but got std={std}",
    )
    utils.check(
        utils.is_float_dtype(dtype) or utils.is_complex_dtype(dtype),
        lambda: f"expected a floating-point or complex dtype, but got dtype={dtype}",
    )
    a = tensorplay.empty(shape, dtype=dtype, device=device)
    with tensorplay.no_grad():
        a.normal_(mean, std)
    if requires_grad:
        a.requires_grad = True
    return a


_normal_doc = """
    Constructs a tensor filled with values drawn from a normal distribution
    with the specified mean and standard deviation.

    Only supports floating-point types.
"""

normal = _make_prim(
    schema="normal(SymInt[] shape, *, Scalar mean, Scalar std, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_normal_meta,
    impl_aten=_normal_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_normal_doc,
)


def _uniform_meta(
    shape: ShapeType,
    *,
    low: float,
    high: float,
    dtype: Any,
    device: Any,
    stride: ShapeType,
) -> TensorLikeType:
    return TensorMeta(shape=shape, strides=stride, dtype=dtype, device=device)


def _uniform_aten(
    shape: ShapeType,
    *,
    low: float,
    high: float,
    dtype: Any,
    device: Any,
    stride: ShapeType,
) -> TensorLikeType:
    a = tensorplay.empty_strided(shape, stride, dtype=dtype, device=device)
    a.uniform_(low, high)
    return a


_uniform_doc = """
    Constructs a tensor filled with values drawn uniformly from low to high.
"""

_uniform_helper = _make_prim(
    schema="uniform(SymInt[] shape, *, Scalar low, Scalar high, ScalarType dtype, Device device, SymInt[] stride) -> Tensor",
    meta=_uniform_meta,
    impl_aten=_uniform_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_uniform_doc,
)


def _svd_meta(
    A: TensorLikeType,
    full_matrices: bool,
    compute_uv: bool,
) -> tuple[TensorLikeType, TensorLikeType, TensorLikeType]:
    utils.check_is_matrix(A, "svd")
    utils.check(
        utils.is_float_dtype(A.dtype),
        lambda: f"svd: Expected a floating point tensor as input. Got {A.dtype}",
    )
    m = A.shape[-2]
    n = A.shape[-1]
    k = min(m, n)
    batch = A.shape[:-2]
    if compute_uv:
        U_shape = batch + (m, m if full_matrices else k)
        V_shape = batch + (n if full_matrices else k, n)
    else:
        U_shape = batch + (0, 0)
        V_shape = batch + (0, 0)
    U_strides = utils.make_contiguous_strides_for(U_shape)
    V_strides = utils.make_contiguous_strides_for(V_shape)
    S_shape = batch + (k,)
    S_strides = utils.make_contiguous_strides_for(S_shape)
    S_dtype = (
        utils.corresponding_real_dtype(A.dtype) if A.dtype.is_complex else A.dtype
    )
    return (
        TensorMeta(shape=U_shape, strides=U_strides, dtype=A.dtype, device=A.device),
        TensorMeta(shape=S_shape, strides=S_strides, dtype=S_dtype, device=A.device),
        TensorMeta(shape=V_shape, strides=V_strides, dtype=A.dtype, device=A.device),
    )


def _svd_aten(
    A: TensorLikeType,
    full_matrices: bool,
    compute_uv: bool,
) -> tuple[TensorLikeType, TensorLikeType, TensorLikeType]:
    return tensorplay.svd(A, full_matrices, compute_uv)


svd = _make_prim(
    schema="svd(Tensor A, bool full_matrices, bool compute_uv) -> (Tensor U, Tensor S, Tensor V)",
    meta=_svd_meta,
    impl_aten=_svd_aten,
    return_type=(RETURN_TYPE.NEW, RETURN_TYPE.NEW, RETURN_TYPE.NEW),
    doc="",
)


def _fft_r_meta(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    onesided: bool,
) -> TensorLikeType:
    dim = utils.canonicalize_dims(input.ndim, dim)
    utils.validate_no_repeating_dims(dim)
    shape = list(input.shape)
    if onesided:
        last_dim = dim[-1]
        shape[last_dim] = shape[last_dim] // 2 + 1
    dtype = utils.corresponding_complex_dtype(input.dtype)
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=input.device)


def _fft_r2c_aten(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    onesided: bool,
) -> TensorLikeType:
    dim = utils.canonicalize_dims(input.ndim, dim)
    last_dim = dim[-1]
    if onesided:
        return _tp_fft.rfft(input, dim=last_dim)
    return _tp_fft.fft(input, dim=last_dim)


_fft_r2c_doc = """
    Performs a real to complex Fast Fourier Transform
"""


fft_r2c = _make_prim(
    schema="fft_r2c(Tensor self, *, int[] dim, bool onesided) -> Tensor",
    meta=_fft_r_meta,
    impl_aten=_fft_r2c_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_fft_r2c_doc,
)


def _fft_c2c_meta(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    forward: bool,
) -> TensorLikeType:
    dim = utils.canonicalize_dims(input.ndim, dim)
    utils.validate_no_repeating_dims(dim)
    shape = input.shape
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=input.dtype, device=input.device)


def _fft_c2c_aten(input, *, dim, forward):
    dim = utils.canonicalize_dims(input.ndim, dim)
    last_dim = dim[-1]
    if forward:
        return _tp_fft.fft(input, dim=last_dim)
    return _tp_fft.ifft(input, dim=last_dim)


fft_c2c = _make_prim(
    schema="fft_c2c(Tensor self, *, int[] dim, bool forward) -> Tensor",
    meta=_fft_c2c_meta,
    impl_aten=_fft_c2c_aten,
    return_type=RETURN_TYPE.NEW,
    doc="Performs either a Fast Fourier Transform, or its inverse",
)


def _fft_c2r_meta(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    last_dim_size: int,
) -> TensorLikeType:
    dim = utils.canonicalize_dims(input.ndim, dim)
    utils.validate_no_repeating_dims(dim)
    shape = list(input.shape)
    shape[dim[-1]] = last_dim_size
    dtype = utils.corresponding_real_dtype(input.dtype)
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=input.device)


def _fft_c2r_aten(input, *, dim, last_dim_size):
    dim = utils.canonicalize_dims(input.ndim, dim)
    last_dim = dim[-1]
    return _tp_fft.irfft(input, n=last_dim_size, dim=last_dim)


_fft_c2r_doc = "Performs a complex to real Inverse Fast Fourier Transform"


fft_c2r = _make_prim(
    schema="fft_c2r(Tensor self, *, int[] dim, SymInt last_dim_size) -> Tensor",
    meta=_fft_c2r_meta,
    impl_aten=_fft_c2r_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_fft_c2r_doc,
)


def _frexp_meta(self: TensorLikeType) -> tuple[TensorLikeType, TensorLikeType]:
    utils.check(
        self.dtype.is_floating_point,
        lambda: "frexp() only supports floating-point dtypes",
    )
    return tensorplay.empty_like(self), tensorplay.empty_like(self, dtype=tensorplay.int32)


frexp = _make_prim(
    schema="frexp(Tensor self) -> (Tensor mantissa, Tensor exponent)",
    meta=_frexp_meta,
    return_type=(RETURN_TYPE.NEW, RETURN_TYPE.NEW),
    impl_aten=tensorplay.frexp,
    doc="",
)


def _make_token_aten() -> TensorLikeType:
    # A scalar placeholder token tensor: this framework does not model
    # ordering side effects with dedicated token buffers, so the token is an
    # opaque zero-dimensional tensor that is sunk by _sink_tokens.
    return tensorplay.empty((), dtype=tensorplay.int32)


_make_token = _make_prim(
    schema="_make_token() -> Tensor",
    meta=_make_token_aten,
    return_type=RETURN_TYPE.NEW,
    impl_aten=_make_token_aten,
    doc="Creates a token used for keeping track of side effects.",
)


def _sink_tokens_aten(tokens) -> None:
    pass


_sink_tokens = _make_prim(
    schema="_sink_tokens(Tensor[] tokens) -> ()",
    meta=_sink_tokens_aten,
    return_type=RETURN_TYPE.NONE,
    impl_aten=_sink_tokens_aten,
    doc="Sink all of the tokens which were previously used for keeping track of side effects.",
)


register_rng_prims()
register_debug_prims()


__all__ += [
    "_make_token",
    "_sink_tokens",
    "_uniform_helper",
    "abs",
    "acos",
    "acosh",
    "add",
    "amax",
    "amin",
    "as_strided",
    "as_strided_scatter",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bessel_i0",
    "bessel_i0e",
    "bessel_i1",
    "bessel_i1e",
    "bessel_j0",
    "bessel_j1",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "broadcast_in_dim",
    "cat",
    "cbrt",
    "ceil",
    "clone",
    "collapse",
    "collapse_view",
    "conj",
    "conj_physical",
    "convert_element_type",
    "copy_strided",
    "copy_to",
    "cos",
    "cosh",
    "device_put",
    "digamma",
    "div",
    "empty",
    "empty_permuted",
    "empty_strided",
    "eq",
    "erf",
    "erf_inv",
    "erfc",
    "erfcx",
    "exp",
    "exp2",
    "expm1",
    "fft_c2c",
    "fft_c2r",
    "fft_r2c",
    "fill",
    "floor",
    "fmax",
    "fmin",
    "fmod",
    "frexp",
    "full",
    "full_like",
    "gcd",
    "ge",
    "gt",
    "hypot",
    "igamma",
    "igammac",
    "imag",
    "iota",
    "isfinite",
    "item",
    "le",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "lt",
    "maximum",
    "maximum_value",
    "minimum",
    "minimum_value",
    "mul",
    "ndtri",
    "ne",
    "neg",
    "nextafter",
    "normal",
    "pow",
    "prod",
    "real",
    "reciprocal",
    "remainder",
    "reshape",
    "resize",
    "rev",
    "round",
    "rsqrt",
    "scalar_tensor",
    "shift_left",
    "shift_right_arithmetic",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "spherical_bessel_j0",
    "split_dim",
    "sqrt",
    "squeeze",
    "sub",
    "sum",
    "svd",
    "tan",
    "tanh",
    "transpose",
    "trunc",
    "var",
    "view_element_type",
    "view_of",
    "where",
    "xor_sum",
    "zeta",
]
