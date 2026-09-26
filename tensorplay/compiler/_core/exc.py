"""The exceptions every part of compilation raises through.

Something went wrong somewhere below, and the traceback that says where is
longer than the part of it anyone can act on.  The exception raised here
records where that first actionable line is, so a caller who only wants to
know that compilation failed can catch one thing and read the rest off it.

The line is only known once the failure has happened, so it cannot be
passed in when the exception is raised.  It is found instead by walking the
traceback back from the raise until the frame matches, which is what
:py:meth:`ShortenTraceback.remove_compilation_frames` does.  A failure with
no such frame is left alone rather than cut at the wrong place, since a
traceback that stops early is worse than one that is too long.
"""

from __future__ import annotations

from typing import Any

from .. import config


class ShortenTraceback(RuntimeError):
    """An error raised below compilation, with the frame to act on kept.

    Everything between the raise and the frame that has to be edited is
    machinery, and a report that leads with it sends the reader looking in
    the wrong file.
    """

    def __init__(self, *args: Any, first_useful_frame: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.first_useful_frame = first_useful_frame

    def remove_compilation_frames(self) -> ShortenTraceback:
        """Drop the frames of compilation itself from this error's traceback.

        Returns the error unchanged when there is no frame to cut at, or when
        the reader has asked to see everything.
        """
        tb = self.__traceback__
        if self.first_useful_frame is None or tb is None or config.verbose:
            return self
        while tb.tb_frame is not self.first_useful_frame:
            tb = tb.tb_next
            if tb is None:
                raise AssertionError("internal error, please report a bug")
        return self.with_traceback(tb)


class BackendCompilerFailed(ShortenTraceback):
    """A backend was asked to compile something and raised.

    Which backend was running is part of the message rather than of the
    type, because the same failure reported by two backends is two different
    problems with the same symptom.
    """

    def __init__(
        self,
        backend_fn: Any,
        inner_exception: Exception,
        first_useful_frame: Any,
    ) -> None:
        self.backend_name = getattr(backend_fn, "__name__", "?")
        self.inner_exception = inner_exception
        msg = (
            f"backend={self.backend_name!r} raised:\n"
            f"{type(inner_exception).__name__}: {inner_exception}"
        )
        super().__init__(msg, first_useful_frame=first_useful_frame)
