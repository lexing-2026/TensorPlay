"""Which build is running, and which attempt at it this is.

A build is identified by the frame it came from and how many times that frame
has been built, and that pair has to survive being written down and read back
by something that did not build it -- a cache entry, a log line, a report about
a run that has already finished.  So its written form is the whole of it: a
short string that can be read without the machinery that made it.

The build currently in progress is kept per thread rather than globally,
because builds run in parallel and a build that could see another build's
identity would attribute work to the wrong one.  The attributes are created up
front per thread rather than on demand, because the question is asked on every
call of a compiled function and a missing attribute would cost a raised
exception each time rather than a comparison.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import NamedTuple


COMPILE_ID_PATTERN = re.compile(r"^(?P<frame_id>\d+)/(?P<frame_compile_id>\d+)$")
CA_COMPILE_ID_PATTERN = re.compile(
    r"^!(?P<compiled_autograd_id>\d+)(?:/(?P<frame_id>\d+)/(?P<frame_compile_id>\d+))?$"
)

# [Note: Updating CompileId]
#
# A compile id names one build uniquely, and that property has to hold as the
# code around it changes, because things outside this repository read it.  The
# in-memory form can change freely; only the written form is depended on.
#
# The written form should be:
# 1. A program-level uid: it identifies a compiled graph uniquely.
# 2. Storage efficient: it appears in nearly every entry that is written down.
# 3. Compact: some tools display it directly, so short is better.


@dataclass(frozen=True, kw_only=True, slots=True)
class CompileId:
    frame_id: int | None
    # This id is per-frame, and counts how many times we've compiled this
    # frame.  This could have been a global id but having this be per-frame
    # gives you a better intuitive sense for how many recompiles have occurred
    # so far.
    frame_compile_id: int | None

    # A compiled autograd graph being compiled
    compiled_autograd_id: int | None = None

    def __str__(self) -> str:
        # NOTE: Keep this in sync with both from_string and whatever reads it.
        if self.compiled_autograd_id is not None:
            if (self.frame_id is None) != (self.frame_compile_id is None):
                raise AssertionError(
                    f"frame_id and frame_compile_id must both be None or both be set, "
                    f"got frame_id={self.frame_id}, frame_compile_id={self.frame_compile_id}"
                )
            frame_str = ""
            if self.frame_id is not None:
                frame_str = f"/{self.frame_id}/{self.frame_compile_id}"

            return f"!{self.compiled_autograd_id}{frame_str}"
        else:
            if self.frame_id is None or self.frame_compile_id is None:
                raise AssertionError(
                    f"frame_id and frame_compile_id must not be None, "
                    f"got frame_id={self.frame_id}, frame_compile_id={self.frame_compile_id}"
                )
            return f"{self.frame_id}/{self.frame_compile_id}"

    @classmethod
    def from_string(cls, compile_id: str | None) -> CompileId | None:
        """A compile id from the form it was written in.

        Kept in step with the written form above, so that a written id always
        reads back as the id it was.
        """

        if compile_id is None:
            return None
        try:
            for pattern in (COMPILE_ID_PATTERN, CA_COMPILE_ID_PATTERN):
                if match := pattern.match(compile_id):
                    groups = match.groupdict()
                    for k, v in groups.items():
                        if v is not None:
                            groups[k] = int(v)
                    return cls(**groups)  # type: ignore[arg-type]
            else:
                raise ValueError

        except Exception as e:
            raise ValueError(f"Invalid compile_id '{compile_id}'") from e


class TraceId(NamedTuple):
    compile_id: CompileId
    # This starts off as 0, and every time we restart analysis it goes
    # up by one
    attempt: int

    def __str__(self) -> str:
        # Keep this in step with whatever reads it.
        if self.attempt == 0:
            return str(self.compile_id)
        else:
            return f"{self.compile_id}_{self.attempt}"


class _TLSStorage(threading.local):
    """The build in progress, one per thread.

    Both attributes are created per thread up front rather than on demand, so
    that asking whether there is a compile context costs a comparison instead
    of a raised exception.  A thread that never built anything itself -- a
    worker running code another thread compiled -- would otherwise pay that on
    every call of a compiled function.
    """

    def __init__(self) -> None:
        self.compile_context: CompileContext | None = None


_TLS = _TLSStorage()


class CompileContext:
    """The build in progress on this thread.

    Anything that is written down about a build has to say which build it was,
    and a build that is running in a worker thread has no way to ask except
    through here.
    """

    @staticmethod
    def get() -> CompileContext:
        if _TLS.compile_context is None:
            raise AssertionError("compile_context is not set")
        return _TLS.compile_context

    @staticmethod
    def try_get() -> CompileContext | None:
        return getattr(_TLS, "compile_context", None)

    def __init__(self, compile_id: CompileId | None) -> None:
        if compile_id is not None and not isinstance(compile_id, CompileId):
            raise AssertionError(
                f"compile_id must be None or CompileId, got {type(compile_id)}"
            )
        self.compile_id: CompileId | None = compile_id
        self.attempt = 0
        # Verbose ShapeEnv guards produced.
        self.shape_env_guards: list[str] = []

    @staticmethod
    def current_compile_id() -> CompileId | None:
        self = CompileContext.try_get()
        if self is None:
            return None
        return self.compile_id

    @staticmethod
    def current_trace_id() -> TraceId | None:
        self = CompileContext.try_get()
        if self is None:
            return None
        if self.compile_id is None:
            return None
        return TraceId(self.compile_id, self.attempt)
