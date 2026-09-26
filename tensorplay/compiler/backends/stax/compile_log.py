"""What a build leaves behind for someone reading it afterwards, and how long it took.

Two things are wanted from a compilation that neither the generated code nor a
pass/fail says: the source it produced, and how long each step of building it
took.  Both are optional -- a build that is only being run does not need either,
and writing a file per build costs more than the build does for a small
kernel -- so both are off until something asks for them.

The log a build writes to is separate from the ordinary one on purpose.  A
compiler's ordinary log says what went wrong and is read while it is happening;
the generated source is read after, by someone who was not there, and is
usually wanted for a run that succeeded.  Putting the two on one logger means
turning on the first to see the second, and the volume of the second drowns the
first.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import time
from collections.abc import Callable, Iterator
from typing import Any

#: The logger generated source is written to.  Its own logger, so that reading
#: the source does not require turning on everything the compiler says.
output_code_log = logging.getLogger("tensorplay.stax.output_code")

#: The logger compilation timings are written to.
compile_time_log = logging.getLogger("tensorplay.stax.compile_time")

#: Where a run leaves the files meant to be read afterwards.  Named per run,
#: so two runs do not write over each other's.
_DEBUG_DIR_ENV = "TP_DEBUG_DIR"


def _debug_root() -> str:
    configured = os.environ.get(_DEBUG_DIR_ENV)
    if configured:
        return configured
    from .runtime.cache_dir_utils import cache_dir

    return os.path.join(cache_dir(), "debug")


def _run_dir_name() -> str:
    """A name no two runs share, so their output does not overwrite each other's.

    The clock alone is not enough: two runs started in the same microsecond, or
    a clock that jumped, would collide.  So the process id is part of the name.
    """

    import datetime

    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    return f"run_{stamp}-pid_{os.getpid()}"


def get_debug_dir() -> str:
    """A directory this run can write what it wants read afterwards into.

    The directory is named, not made: a caller that only wants to know where the
    files would go should not leave an empty directory behind by asking.
    """

    return os.path.join(_debug_root(), _run_dir_name())


@contextlib.contextmanager
def debug_dir() -> Iterator[str]:
    """The run's debug directory, made, and removed again if nothing was written.

    An empty directory says a run was debugged when it was not, so one that
    stayed empty is taken away rather than left to be found.
    """

    path = get_debug_dir()
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        try:
            if not os.listdir(path):
                os.rmdir(path)
        except OSError:
            pass


@contextlib.contextmanager
def timed_block(key: str) -> Iterator[None]:
    """How long the block took, filed under ``key``.

    Reported whether or not the block raised, because a step that fails after
    four minutes is exactly the one whose timing is wanted.  The measurement is
    only written when something is listening: timing a build that nobody asked
    about is work the build does for no one.
    """

    if not compile_time_log.isEnabledFor(logging.DEBUG):
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        compile_time_log.debug("%s took %.3f ms", key, (time.perf_counter() - start) * 1e3)


def timed(key: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Report how long the decorated call took, under ``key``.

    A decorator rather than a block so that the thing being measured is the
    whole call, and so that a caller cannot measure a call that has not happened
    by forgetting to wrap it.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with timed_block(key):
                return fn(*args, **kwargs)

        return wrapper

    return decorate


def trace_structured(name: str, payload: dict[str, Any]) -> None:
    """Record one structured event about the build.

    Written as a log line rather than through a tracing library, because the
    consumer is a person reading a file and a log line is what they can read
    without a tool.  The payload is sorted so that two runs that reached the
    same point produce the same line, which is what makes two such lines
    comparable.
    """

    if not compile_time_log.isEnabledFor(logging.DEBUG):
        return
    fields = " ".join(f"{key}={payload[key]!r}" for key in sorted(payload))
    compile_time_log.debug("%s %s", name, fields)


def write_generated_source(path: str, source: str) -> str:
    """Leave the generated source where a person can read it.

    The directory is made if it is not there, and the path returned, so a caller
    that wants to write beside the source has somewhere to write.
    """

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        handle.write(source)
    output_code_log.debug("wrote generated source to %s", path)
    return path


__all__ = [
    "compile_time_log",
    "debug_dir",
    "timed_block",
    "get_debug_dir",
    "output_code_log",
    "timed",
    "trace_structured",
    "write_generated_source",
]
