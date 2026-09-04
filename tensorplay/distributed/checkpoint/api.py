from __future__ import annotations

import traceback as tb
from typing import Any

WRAPPED_EXCEPTION = tuple[BaseException, tb.StackSummary]

__all__ = ["CheckpointException"]


def _wrap_exception(exc: BaseException) -> WRAPPED_EXCEPTION:
    summary = tb.extract_tb(exc.__traceback__)
    for frame in summary:
        if hasattr(frame, "_code"):
            object.__setattr__(frame, "_code", None)
    return exc, summary


def _is_wrapped_exception(obj: Any) -> bool:
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], BaseException)
        and isinstance(obj[1], tb.StackSummary)
    )


class CheckpointException(BaseException):
    def __init__(self, msg: str, failures: dict[int, WRAPPED_EXCEPTION]):
        super().__init__(msg, failures)
        self._failures = failures

    @property
    def failures(self) -> dict[int, WRAPPED_EXCEPTION]:
        return self._failures

    def __str__(self) -> str:
        if not self._failures:
            return super().__str__()
        output = f"CheckpointException ranks:{self._failures.keys()}\n"
        for rank, failure in self._failures.items():
            if _is_wrapped_exception(failure):
                error, summary = failure
                output += f"Traceback (most recent call last): (RANK {rank})\n"
                output += "".join(tb.format_list(summary))
                output += "".join(tb.format_exception_only(type(error), error))
            else:
                output += f"RANK {rank}: {failure!s}\n"
        return output
