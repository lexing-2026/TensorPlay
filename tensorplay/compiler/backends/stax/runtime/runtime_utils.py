"""Checks the generated wrapper makes about the values it was handed.

A wrapper compiled for one shape, one stride and one element type is only
correct for those.  Handed anything else it computes on a buffer laid out
differently from the one it was written for, and the result is a wrong answer
with nothing to point at.

So the generated source checks before it computes.  The check is in the
generated code rather than around it because the thing that knows what the
kernel was compiled for is the code that was generated for it, and a check
written anywhere else would have to be told that.
"""

from __future__ import annotations

import functools
import operator
from typing import Any

import sympy

import tensorplay as tp


def conditional_product(*args: int) -> int:
    """The product of the arguments that are not zero.

    A zero would make the whole product zero, so an argument that is zero is
    left out rather than multiplied in: what is wanted is the size of what is
    actually there.
    """

    return functools.reduce(operator.mul, [x for x in args if x])


def ceildiv(number: int, denom: int) -> int:
    """The number of steps of ``denom`` needed to reach ``number``."""

    return -(number // -denom)


def is_power_of_2(n: int) -> bool:
    """Whether ``n`` is a power of two."""

    return n > 0 and n & n - 1 == 0


def next_power_of_2(n: int) -> int:
    """The smallest power of two that is at least ``n``."""

    if isinstance(n, sympy.Integer):
        n = int(n)
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def last_power_of_2(n: int) -> int:
    """The largest power of two that is at most ``n``."""

    if isinstance(n, sympy.Integer):
        n = int(n)
    if n <= 0:
        return 1
    return 1 << ((n - 1).bit_length() - 1)


def _describe(value: Any) -> str:
    """Enough about a value to tell two apart in a message."""

    shape = getattr(value, "shape", None)
    if shape is None:
        return f"a value of type {type(value).__name__}"
    return (
        f"a value of shape {tuple(int(s) for s in shape())} "
        f"and stride {tuple(int(s) for s in value.stride())}"
    )


def assert_size_stride(
    name: str,
    size: Any,
    stride: Any,
    op_name: str = "",
) -> None:
    """Raise unless a buffer has the shape and stride it was compiled for.

    The name is the buffer's own, so the message says which argument of which
    call was wrong rather than that something was.
    """

    tensor = _current_buffer(name)
    if tensor is None:
        return
    expected_size = tuple(int(s) for s in size)
    expected_stride = tuple(int(s) for s in stride)
    actual_size = tuple(int(s) for s in tensor.shape)
    actual_stride = tuple(int(s) for s in tensor.stride())
    if actual_size != expected_size or actual_stride != expected_stride:
        where = f" for {op_name}" if op_name else ""
        raise AssertionError(
            f"buffer {name!r}{where} was compiled for shape {expected_size} and "
            f"stride {expected_stride} but arrived with shape {actual_size} and "
            f"stride {actual_stride}"
        )


def assert_tensor_metadata(
    name: str,
    size: Any,
    stride: Any,
    dtype: Any,
    op_name: str = "",
) -> None:
    """Raise unless a value has the shape, stride and element type it was
    compiled for.

    The element type is checked as well because two values of the same shape and
    stride can still be read differently, and a kernel written for one element
    size reading the other produces a number rather than an error.
    """

    tensor = _current_buffer(name)
    if tensor is None:
        return
    assert_size_stride(name, size, stride, op_name)
    if tensor.dtype != dtype:
        where = f" for {op_name}" if op_name else ""
        raise AssertionError(
            f"value {name!r}{where} was compiled for element type {dtype} but "
            f"arrived with element type {tensor.dtype}"
        )


def assert_alignment(name: str, alignment: int) -> None:
    """Raise unless a buffer's memory is aligned as the kernel needs it to be.

    Separate from the shape check because the two fail for different reasons:
    a wrong shape means the caller built something else, while a misaligned
    buffer of the right shape means the caller built the right thing somewhere
    the hardware cannot read it from.
    """

    tensor = _current_buffer(name)
    if tensor is None:
        return
    if tensor.numel() and int(tensor.data_ptr()) % alignment:
        raise AssertionError(
            f"buffer {name!r} is at address {tensor.data_ptr()}, which is not "
            f"a multiple of {alignment}, so a wide load from it would read the "
            "wrong elements"
        )


def _current_buffer(name: str) -> Any:
    """The buffer the generated wrapper is currently working on.

    A miss is not a failure: a name the wrapper never allocated is a name the
    call did not use, and there is nothing to check.
    """

    import sys

    frame = sys._getframe(1)
    while frame is not None:
        if name in frame.f_locals:
            value = frame.f_locals[name]
            if isinstance(value, tp.Tensor):
                return value
        frame = frame.f_back
    return None


__all__ = [
    "assert_alignment",
    "assert_size_stride",
    "assert_tensor_metadata",
]
