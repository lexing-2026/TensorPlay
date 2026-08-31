"""Completion primitives exposed by the native extension."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, TypeVar

from tensorplay import _C


__all__ = ["Future", "collect_all", "wait_all"]


_T = TypeVar("_T")


class Future(_C.Future, Generic[_T]):
    """A native future value with callback and exception support."""

    def set_exception(self, result: BaseException) -> None:
        if not isinstance(result, Exception):
            raise AssertionError(
                f"{result} is of type {type(result)}, not an Exception."
            )

        def raise_error(_: object) -> None:
            raise result

        self._set_unwrap_func(raise_error)
        self.set_result(result)


def collect_all(futures: Sequence[Future[_T]]) -> Future[list[Future[_T]]]:
    """Return a future completed after every input future is complete."""

    return _C._collect_all(futures)


def wait_all(futures: Sequence[Future[_T]]) -> list[_T]:
    """Wait for all input futures and return their values in input order."""

    return _C._wait_all(futures)
