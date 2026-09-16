from __future__ import annotations

import functools
from typing import Concatenate, ParamSpec, TYPE_CHECKING, TypeVar

import tensorplay

from . import _dtypes_impl, _util
from ._normalizations import ArrayLike, normalizer


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_P = ParamSpec("_P")
_R = TypeVar("_R")


def upcast(
    func: Callable[Concatenate[tensorplay.Tensor, _P], _R],
) -> Callable[Concatenate[tensorplay.Tensor, _P], _R]:
    """NumPy fft casts inputs to 64 bit and *returns 64-bit results*."""

    @functools.wraps(func)
    def wrapped(tensor: tensorplay.Tensor, *args: _P.args, **kwds: _P.kwargs) -> _R:
        target_dtype = (
            _dtypes_impl.default_dtypes().complex_dtype
            if tensor.is_complex()
            else _dtypes_impl.default_dtypes().float_dtype
        )
        tensor = _util.cast_if_needed(tensor, target_dtype)
        return func(tensor, *args, **kwds)

    return wrapped


@normalizer
@upcast
def fft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> tensorplay.Tensor:
    return tensorplay.fft.fft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def ifft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> tensorplay.Tensor:
    return tensorplay.fft.ifft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def rfft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> tensorplay.Tensor:
    return tensorplay.fft.rfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def irfft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> tensorplay.Tensor:
    return tensorplay.fft.irfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def fftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> tensorplay.Tensor:
    return tensorplay.fft.fftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def ifftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> tensorplay.Tensor:
    return tensorplay.fft.ifftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def rfftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> tensorplay.Tensor:
    return tensorplay.fft.rfftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def irfftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> tensorplay.Tensor:
    return tensorplay.fft.irfftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def fft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> tensorplay.Tensor:
    return tensorplay.fft.fft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def ifft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> tensorplay.Tensor:
    return tensorplay.fft.ifft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def rfft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> tensorplay.Tensor:
    return tensorplay.fft.rfft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def irfft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> tensorplay.Tensor:
    return tensorplay.fft.irfft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def hfft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> tensorplay.Tensor:
    return tensorplay.fft.hfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def ihfft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> tensorplay.Tensor:
    return tensorplay.fft.ihfft(a, n, dim=axis, norm=norm)


@normalizer
def fftfreq(n: int, d: float = 1.0) -> tensorplay.Tensor:
    return tensorplay.fft.fftfreq(n, d)


@normalizer
def rfftfreq(n: int, d: float = 1.0) -> tensorplay.Tensor:
    return tensorplay.fft.rfftfreq(n, d)


@normalizer
def fftshift(x: ArrayLike, axes: int | Sequence[int] | None = None) -> tensorplay.Tensor:
    return tensorplay.fft.fftshift(x, axes)


@normalizer
def ifftshift(x: ArrayLike, axes: int | Sequence[int] | None = None) -> tensorplay.Tensor:
    return tensorplay.fft.ifftshift(x, axes)
