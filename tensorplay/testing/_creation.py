"""Tensor creation utilities for tests."""

import collections.abc
import functools
import math
import warnings
from typing import cast

import tensorplay as tp
from tensorplay import Tensor

__all__ = ["make_tensor"]

_INTEGRAL_TYPES = [
    tp.uint8,
    tp.int8,
    tp.int16,
    tp.int32,
    tp.int64,
    tp.uint16,
    tp.uint32,
    tp.uint64,
]
_FLOATING_TYPES = [tp.float16, tp.bfloat16, tp.float32, tp.float64]
_COMPLEX_TYPES = [tp.complex64, tp.complex128]
_BOOLEAN_OR_INTEGRAL_TYPES = [tp.bool, *_INTEGRAL_TYPES]
_FLOATING_OR_COMPLEX_TYPES = [*_FLOATING_TYPES, *_COMPLEX_TYPES]

# Largest value `randint` can sample: the generator converts the bounds to an
# int64 internally, so the maximum of `int64` itself is out of reach and
# accounted for explicitly.
_INT64_SAMPLING_MAX = tp.iinfo(tp.int64).max


def _uniform_random_(t: Tensor, low: float, high: float) -> Tensor:
    # Fills the tensor with values from the uniform distribution over
    # [low, high).  For dtypes whose representable range is smaller than the
    # requested interval, the range is scaled around the midpoint before and
    # after the draw so the product never overflows.
    if high - low >= tp.finfo(t.dtype).max:
        return t.uniform_(low / 2, high / 2).mul_(2)
    else:
        return t.uniform_(low, high)


def make_tensor(
    *shape: int | tp.Size | list[int] | tuple[int, ...],
    dtype: tp.dtype,
    device: str | tp.device,
    low: float | None = None,
    high: float | None = None,
    requires_grad: bool = False,
    noncontiguous: bool = False,
    exclude_zero: bool = False,
) -> Tensor:
    """Creates a tensor with the given :attr:`shape`, :attr:`device`, and
    :attr:`dtype`, filled with values drawn uniformly from ``[low, high)``.

    If :attr:`low` or :attr:`high` are outside the range of the
    :attr:`dtype`'s representable finite values, they are clamped to the
    lowest or highest representable finite value, respectively. If ``None``,
    they default to ``-9`` and ``9`` respectively (``0`` and ``2`` for
    ``bool``).

    Args:
        shape (Tuple[int, ...]): Single integer or a collection of integers defining the shape
            of the output tensor.
        dtype (tensorplay.dtype): The data type of the returned tensor.
        device (Union[str, tensorplay.device]): The device of the returned tensor.
        low (Optional[Number]): Sets the lower limit of the range of the values in the returned tensor.
        high (Optional[Number]): Sets the upper limit of the range of the values in the returned tensor.
        requires_grad (bool): If ``True``, the returned tensor is set to require gradient.
        noncontiguous (bool): If ``True``, the returned tensor is non-contiguous.
        exclude_zero (bool): If ``True``, zeros in the returned tensor are replaced by the smallest
            normal value of the dtype (``1`` for boolean and integral dtypes).

    Raises:
        ValueError: If :attr:`low` >= :attr:`high`, or if the interval
            ``[low, high)`` does not intersect the dtype's representable range.
        TypeError: For unsupported dtypes.
    """

    def modify_low_high(
        low: float | None,
        high: float | None,
        *,
        lowest_inclusive: float,
        highest_exclusive: float,
        default_low: float,
        default_high: float,
    ) -> tuple[float, float]:
        def clamp(a: float, l: float, h: float) -> float:
            return min(max(a, l), h)

        low = low if low is not None else default_low
        high = high if high is not None else default_high

        if any(isinstance(value, float) and math.isnan(value) for value in [low, high]):
            raise ValueError(
                f"`low` and `high` cannot be NaN, but got {low=} and {high=}"
            )
        elif low >= high:
            raise ValueError(f"`low` must be less than `high`, but got {low} >= {high}")
        elif high < lowest_inclusive or low >= highest_exclusive:
            raise ValueError(
                f"The value interval specified by `low` and `high` is [{low}, {high}), "
                f"but {dtype} only supports [{lowest_inclusive}, {highest_exclusive})"
            )

        low = clamp(low, lowest_inclusive, highest_exclusive)
        high = clamp(high, lowest_inclusive, highest_exclusive)

        if dtype in _BOOLEAN_OR_INTEGRAL_TYPES:
            # Ceil the bounds so no value smaller than `low` is created, and
            # since the sampling upper bound is exclusive.
            return math.ceil(low), math.ceil(high)

        return low, high

    if len(shape) == 1 and isinstance(shape[0], collections.abc.Sequence):
        shape = tuple(shape[0])
    shape = cast(tuple[int, ...], tuple(shape))

    if requires_grad and dtype in _BOOLEAN_OR_INTEGRAL_TYPES:
        raise ValueError(
            f"`requires_grad=True` is not supported for boolean and integral dtypes, "
            f"but got {dtype=}"
        )

    noncontiguous = noncontiguous and functools.reduce(lambda x, y: x * y, shape, 1) > 1
    if noncontiguous:
        # Double the size of the last dimension, so that the final slicing
        # step below yields values that are not adjacent in memory.
        shape = (*shape[:-1], 2 * shape[-1])

    if dtype is tp.bool:
        low, high = cast(
            tuple[int, int],
            modify_low_high(
                low,
                high,
                lowest_inclusive=0,
                highest_exclusive=2,
                default_low=0,
                default_high=2,
            ),
        )
        result = tp.randint(low, high, shape, device=device, dtype=dtype)
    elif dtype in _BOOLEAN_OR_INTEGRAL_TYPES:
        low, high = cast(
            tuple[int, int],
            modify_low_high(
                low,
                high,
                lowest_inclusive=tp.iinfo(dtype).min,
                highest_exclusive=tp.iinfo(dtype).max
                # `randint` converts the bounds to an int64 internally and
                # would overflow for the maximum of `int64` itself.
                + (1 if dtype is not tp.int64 else 0),
                default_low=-9,
                default_high=10,
            ),
        )
        result = tp.randint(low, high, shape, device=device, dtype=dtype)
    elif dtype in _FLOATING_OR_COMPLEX_TYPES:
        low, high = modify_low_high(
            low,
            high,
            lowest_inclusive=tp.finfo(dtype).min,
            highest_exclusive=tp.finfo(dtype).max,
            default_low=-9,
            default_high=9,
        )
        result = tp.empty(shape, device=device, dtype=dtype)
        _uniform_random_(
            tp.view_as_real(result) if dtype in _COMPLEX_TYPES else result, low, high
        )
    else:
        raise TypeError(
            f"The requested dtype '{dtype}' is not supported by make_tensor()."
        )

    if noncontiguous:
        # Offset by 1 to also catch offsetting issues
        result = result[..., 1::2]
    if exclude_zero:
        if dtype in _BOOLEAN_OR_INTEGRAL_TYPES:
            replacement = 1
        else:
            replacement = tp.finfo(dtype).tiny
        result[result == 0] = replacement

    if dtype in _FLOATING_OR_COMPLEX_TYPES:
        result.requires_grad = requires_grad

    return result
