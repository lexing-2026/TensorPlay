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
import socket
import time
from string import Template
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

ERROR_FILE_ENV = "TORCHELASTIC_ERROR_FILE"

from .error_handler import ErrorHandler
from .handlers import get_error_handler

__all__ = [
    "ProcessFailure",
    "ChildFailedError",
    "record",
    "SignalException",
    "ErrorHandler",
    "get_error_handler",
]


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
        if self.error_file and os.path.isfile(self.error_file):
            try:
                import json

                with open(self.error_file) as f:
                    self.error_file_data = json.load(f)
                self.message, self.timestamp = self._get_error_data(
                    self.error_file_data
                )
            except Exception:
                raise
        else:
            self._set_no_reply_file()
        if not self.message:
            if self.exitcode < 0:
                self.message = (
                    f"Signal {-self.exitcode} ({self.signal_name()}) received "
                    f"by PID {self.pid}"
                )
            else:
                self.message = (
                    f"Worker rank {self.local_rank} (pid {self.pid}) exited "
                    f"with exit code {self.exitcode} without an error report"
                )

    def _get_error_data(self, error_file_data: dict[str, Any]) -> tuple[Any, int]:
        message = error_file_data["message"]
        if isinstance(message, str):
            timestamp = int(error_file_data.get("timestamp", 0))
        else:
            timestamp = int(message["extraInfo"]["timestamp"])
        return message, timestamp

    def _set_no_reply_file(self) -> None:
        self.error_file = "<N/A>"
        self.error_file_data = {"message": "<NONE>"}
        self.message = ""
        self.timestamp = int(time.time())

    @property
    def exit_code(self) -> int:
        return self.exitcode

    def signal_name(self) -> str:
        if self.exitcode < 0:
            try:
                return signal.Signals(-self.exitcode).name
            except ValueError:
                return "<N/A>"
        return "<N/A>"

    def timestamp_isoformat(self) -> str:
        return datetime.fromtimestamp(self.timestamp).isoformat(sep="_")

    @property
    def extra_info(self) -> dict[str, Any]:
        return (self.error_file_data or {}).get("extraInfo", {})


class ChildFailedError(Exception):
    """Raised by the launcher when one or more workers failed.

    ``failures`` is a list of ``(role_name, ProcessFailure)`` pairs so the
    caller can report which role failed and why.
    """

    def __init__(
        self,
        name_or_failures: str | list[tuple[str, ProcessFailure]] | None = None,
        failures: dict[int, ProcessFailure] | None = None,
    ) -> None:
        if isinstance(name_or_failures, str):
            self.name = name_or_failures
            self.failures = dict(failures or {})
            message = self.format_msg()
        else:
            self.name = "workers"
            self.failures = list(name_or_failures or [])
            message = "; ".join(
                f"{role}: {failure.message}" for role, failure in self.failures
            )
            if not message:
                message = "One or more worker processes failed"
        if not self.failures:
            raise AssertionError
        super().__init__(message)

    def get_first_failure(self):
        if isinstance(self.failures, dict):
            return min(self.failures.items(), key=lambda item: item[1].timestamp)
        return min(self.failures, key=lambda item: item[1].timestamp)[0:2]

    def _format_failure(
        self, idx: int, rank: int, failure: ProcessFailure
    ) -> tuple[str, int]:
        message = failure.message
        if isinstance(message, dict):
            message = (
                message.get("extraInfo", {}).get(
                    "py_callstack", message.get("message", "<N/A>")
                )
            )
        message = str(message).replace("\n", "\n  ")
        signal_name = failure.signal_name()
        suffix = f" ({signal_name})" if signal_name != "<N/A>" else ""
        text = (
            f"[{idx}]:\n  time      : {failure.timestamp_isoformat()}\n"
            f"  host      : {socket.getfqdn()}\n  rank      : {rank} "
            f"(local_rank: {failure.local_rank})\n  exitcode  : {failure.exitcode} "
            f"(pid: {failure.pid}){suffix}\n  error_file: {failure.error_file}\n"
            f"  traceback : {message}"
        )
        return text, max((len(line) for line in text.splitlines()), default=0)

    def format_msg(self, boarder_delim: str = "=", section_delim: str = "-") -> str:
        root_rank, _ = self.get_first_failure()
        rows = []
        root = ""
        width = len(f"{self.name} FAILED")
        items = self.failures.items() if isinstance(self.failures, dict) else []
        for idx, (rank, failure) in enumerate(items):
            rendered, item_width = self._format_failure(idx, rank, failure)
            width = max(width, item_width)
            if rank == root_rank:
                if failure.exitcode < 0:
                    handler = get_error_handler()
                    enrich = getattr(
                        handler, "maybe_enrich_signal_failure_message", None
                    )
                    if enrich is not None:
                        rendered = enrich(rendered, failure.error_file)
                root = rendered
            else:
                rows.append(rendered)
        width = min(width, 60)
        return Template(
            "\n${border}\n${title}\n${section}\nFailures:\n${others}\n"
            "${section}\nRoot Cause (first observed failure):\n${root}\n${border}"
        ).substitute(
            border=boarder_delim * width,
            title=f"{self.name} FAILED",
            section=section_delim * width,
            others="\n".join(rows or ["  <NO_OTHER_FAILURES>"]),
            root=root,
        )


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
