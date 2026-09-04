"""Elastic worker process management."""
from collections.abc import Callable

from tensorplay.distributed.elastic.multiprocessing.errors import (
    ChildFailedError,
    ProcessFailure,
    record,
)

from .api import (
    DefaultLogsSpecs,
    LogsDest,
    LogsSpecs,
    MultiprocessContext,
    PContext,
    RunProcsResult,
    SignalException,
    Std,
    SubprocessContext,
    start_processes as _start_processes,
)
from .redirects import Redirects, Std, to_map
from .tail_log import TailLog, tail_logfile

__all__ = [
    "RunProcsResult",
    "start_processes",
    "PContext",
    "MultiprocessContext",
    "SubprocessContext",
    "SignalException",
    "Std",
    "Redirects",
    "to_map",
    "tail_logfile",
    "TailLog",
    "LogsDest",
    "LogsSpecs",
    "DefaultLogsSpecs",
    "ProcessFailure",
    "ChildFailedError",
    "record",
]


def start_processes(*args, **kwargs):
    return _start_processes(*args, **kwargs)
