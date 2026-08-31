from __future__ import annotations

from typing import Any

__all__ = ["CheckpointException"]


def _wrap_exception(exc: BaseException) -> BaseException:
    return exc


def _is_wrapped_exception(obj: Any) -> bool:
    return isinstance(obj, BaseException)


class CheckpointException(Exception):
    def __init__(self, msg: str, failures: dict[int, BaseException]):
        super().__init__(msg)
        self._failures = failures

    @property
    def failures(self) -> dict[int, BaseException]:
        return self._failures

    def __str__(self) -> str:
        if not self._failures:
            return super().__str__()
        return f"{super().__str__()} ({len(self._failures)} failures)"
