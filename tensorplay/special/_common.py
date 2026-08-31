"""Dtype promotion helpers shared by the special-function families."""
import math

import tensorplay as tp
from tensorplay import Tensor
from tensorplay._C import DType

__all__ = ["INV_SQRT_2", "float_dtype_of", "lgamma_scalar", "promote_pair"]

#: ``1 / sqrt(2)``, the scale that turns ``erfc`` into the normal CDF.
INV_SQRT_2 = 0.5 ** 0.5


def float_dtype_of(t: Tensor) -> DType:
    """Floating result dtype for ``t``: float64 stays, everything else is float32."""
    dt = t.dtype
    if dt in (DType.float32, DType.float64):
        return dt
    return DType.float32


def promote_pair(a, b):
    """Wraps python scalars as tensors and lifts both operands to one float dtype."""
    if not isinstance(a, Tensor):
        a = tp.tensor(float(a))
    if not isinstance(b, Tensor):
        b = tp.tensor(float(b))
    return a.to(float_dtype_of(b)), b


def lgamma_scalar(v: float) -> float:
    """Host-side log-gamma, for the constant terms of the composed formulas."""
    return math.lgamma(v)
