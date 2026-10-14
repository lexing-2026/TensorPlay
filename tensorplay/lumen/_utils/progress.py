#!/usr/bin/env python3

"""Progress reporting utilities.

Exposes a small protocol-compatible surface for progress bars. If tqdm is not
installed, operations fall back to a no-op implementation.
"""

import typing
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    runtime_checkable,
    TextIO,
    TypeVar,
    Union,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

from typing_extensions import Self

T = TypeVar("T")
IterableType = TypeVar("IterableType", covariant=True)


@runtime_checkable
class BaseProgress(Protocol):
    """Protocol defining the base progress bar interfaced with
    context manager support."""

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        exc_traceback: object,
    ) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class IterableProgress(BaseProgress, Iterable[IterableType], Protocol[IterableType]):
    """Protocol for progress bars that support iteration."""

    ...


@runtime_checkable
class Progress(BaseProgress, Protocol):
    """Protocol for progress bars that support manual updates."""

    def update(self, n: float | None = 1) -> bool | None: ...


class DisableErrorIOWrapper(object):
    def __init__(self, wrapped: TextIO) -> None:
        """A wrapper around a TextIO object to ignore write errors."""
        self._wrapped = wrapped

    def __getattr__(self, name: str) -> object:
        return getattr(self._wrapped, name)

    @staticmethod
    def _wrapped_run(
        func: Callable[..., T], *args: object, **kwargs: object
    ) -> Union[T, None]:
        try:
            return func(*args, **kwargs)
        except OSError as e:
            if e.errno != 5:
                raise
        except ValueError as e:
            if "closed" not in str(e):
                raise
        return None

    def write(self, *args: object, **kwargs: object) -> Optional[int]:
        return self._wrapped_run(self._wrapped.write, *args, **kwargs)

    def flush(self, *args: object, **kwargs: object) -> None:
        return self._wrapped_run(self._wrapped.flush, *args, **kwargs)


class NullProgress(IterableProgress[IterableType], Progress):
    """Passthrough class that implements the progress API but does nothing."""

    def __init__(
        self,
        iterable: Optional[Iterable[IterableType]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        self.iterable = iterable

    def __iter__(self) -> Iterator[IterableType]:
        iterable = self.iterable
        if not iterable:
            yield from ()
            return
        for it in iterable:
            yield it

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        exc_traceback: object,
    ) -> None:
        self.close()

    def update(self, n: float | None = 1) -> bool | None:
        return None

    def close(self) -> None:
        pass


@typing.overload
def progress(
    iterable: None = None,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    file: Optional[TextIO] = None,
    mininterval: float = 0.5,
    **kwargs: object,
) -> Progress: ...


@typing.overload
def progress(
    iterable: Iterable[IterableType],
    desc: Optional[str] = None,
    total: Optional[int] = None,
    file: Optional[TextIO] = None,
    mininterval: float = 0.5,
    **kwargs: object,
) -> IterableProgress[IterableType]: ...


def progress(
    iterable: Optional[Iterable[IterableType]] = None,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    file: Optional[TextIO] = None,
    mininterval: float = 0.5,
    **kwargs: object,
) -> Union[Progress, IterableProgress[IterableType]]:
    if tqdm is None:
        return NullProgress(iterable=iterable)
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        file=file,
        mininterval=mininterval,
        **kwargs,
    )