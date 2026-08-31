"""Worker-side failure recording and agent-side failure reporting.

Workers decorated with :func:`record` serialize any unhandled failure into
the error file the agent arranged for them, so multi-process crashes surface
with full tracebacks on the agent side as :class:`ProcessFailure` entries.
"""
import contextlib
import functools
import os
import signal
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

ERROR_FILE_ENV = "TORCHELASTIC_ERROR_FILE"

__all__ = ["ProcessFailure", "ChildFailedError", "record", "SignalException"]


class SignalException(Exception):
    """Raised in the agent when a death signal is delivered."""

    def __init__(self, msg: str, sigval: signal.Signals) -> None:
        super().__init__(msg)
        self.sigval = sigval


@dataclass
class ProcessFailure:
    """Structured failure of one worker process."""

    local_rank: int
    pid: int
    exitcode: int
    error_file: str | None = None
    error_file_data: dict[str, Any] | None = None
    message: str = ""
    timestamp: int = 0

    def __post_init__(self) -> None:
        if self.error_file_data is None and self.error_file:
            try:
                import json

                with open(self.error_file) as f:
                    self.error_file_data = json.load(f)
            except (OSError, ValueError):
                self.error_file_data = None
        if not self.message:
            data = self.error_file_data or {}
            self.message = data.get("message") or (
                f"worker rank {self.local_rank} (pid {self.pid}) exited "
                f"with exitcode {self.exitcode}"
            )

    @property
    def extra_info(self) -> dict[str, Any]:
        return (self.error_file_data or {}).get("extraInfo", {})


class ChildFailedError(Exception):
    """Raised by the launcher when one or more workers failed.

    ``failures`` is a list of ``(role_name, ProcessFailure)`` pairs so the
    caller can report which role failed and why.
    """

    def __init__(self, failures: list[tuple[str, ProcessFailure]] | None = None) -> None:
        self.failures = failures or []
        text = "; ".join(
            f"{role}: {failure.message}" for role, failure in self.failures
        )
        super().__init__(text or "One or more worker processes failed")


def _write_error_file(error_file: str, exc: BaseException) -> None:
    import json

    data = {
        "message": repr(exc),
        "extraInfo": {
            "traceback": traceback.format_exc(),
            "signal": getattr(exc, "sigval", None) and getattr(exc, "sigval").name,
        },
    }
    tmp = f"{error_file}.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(error_file) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(data, f, default=str)
    os.replace(tmp, error_file)


def record(fn=None, *, error_file: str | None = None):
    """Decorator or context manager persisting failures for the agent.

    When used as a decorator, unhandled exceptions are written to the
    worker's arranged error file (the path injected through the worker
    environment unless overridden) before being re-raised.
    """
    if fn is None:
        return contextlib.nullcontext()

    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any):
        target = error_file or os.environ.get(ERROR_FILE_ENV)
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:
            if target and not os.path.isfile(target):
                try:
                    _write_error_file(target, exc)
                except OSError:
                    pass
            raise

    return _wrapper
