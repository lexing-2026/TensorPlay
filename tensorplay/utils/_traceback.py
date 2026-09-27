"""What stack a thing was called from, captured cheaply and read later.

Capturing a stack is not free, so this is arranged so that a caller who wants
a stack on every call can afford it: the capture hands back frames already
reduced to what a record needs -- where, what, which line -- rather than
something that has to be symbolized afterwards.  That also means there is
nothing to symbolize, so a captured stack survives being held, stored, or
sent somewhere without dragging the machinery that produced it along.
"""

from __future__ import annotations

import contextlib
import inspect
import os.path
import tempfile
import traceback
from types import TracebackType


__all__ = [
    "CapturedTraceback",
    "format_frame",
    "format_traceback_short",
    "report_compile_source_on_error",
    "shorten_filename",
]


@contextlib.contextmanager
def report_compile_source_on_error():
    """Make a failure inside generated code point at the code that made it.

    Code produced by compiling something has no file of its own, so a
    traceback through it shows ``<string>`` and no line.  Here the source is
    written to a file and the frames are rebuilt to name that file, so the
    ordinary error printer prints something a reader can use.

    The frames are rebuilt rather than edited: a code object cannot be
    modified, so a frame carrying a replacement code object is made instead.
    It prints correctly and is not the original -- running it would not do
    what the original did -- which is why it is only ever used for printing.
    """

    try:
        yield
    except Exception as exc:
        tb = exc.__traceback__

        # Walk the traceback, looking for frames that have
        # source attached
        stack = []
        while tb is not None:
            filename = tb.tb_frame.f_code.co_filename
            source = tb.tb_frame.f_globals.get("__compile_source__")

            if filename == "<string>" and source is not None:
                # Don't delete the temporary file so the user can inspect it
                # TODO: This creates a temporary file for every frame, but we
                # technically only need one per distinct __compile_source__
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".py"
                ) as f:
                    f.write(source)
                # Create a frame.  Python doesn't let you construct
                # FrameType directly, so just make one with compile
                frame = tb.tb_frame
                code = compile("__inspect_currentframe()", f.name, "eval")
                code = code.replace(co_name=frame.f_code.co_name)
                # Python 3.11 only
                if hasattr(frame.f_code, "co_linetable"):
                    # We can't copy ALL of the metadata over, because you
                    # can cause Python to segfault this way.  What exactly
                    # do we need?  We need enough information for
                    # traceback to be able to print the exception
                    # correctly.  The traceback machinery asks the code
                    # object for its positions, and that iterator is built
                    # from the line table and the first line number -- so
                    # copy these we must!
                    code = code.replace(  # type: ignore[call-arg]
                        co_linetable=frame.f_code.co_linetable,  # type: ignore[attr-defined]
                        co_firstlineno=frame.f_code.co_firstlineno,  # type: ignore[attr-defined]
                    )
                fake_frame = eval(
                    code,
                    frame.f_globals,
                    {**frame.f_locals, "__inspect_currentframe": inspect.currentframe},
                )
                fake_tb = TracebackType(None, fake_frame, tb.tb_lasti, tb.tb_lineno)
                stack.append(fake_tb)
            else:
                stack.append(tb)

            tb = tb.tb_next

        # Reconstruct the linked list
        tb_next = None
        for tb in reversed(stack):
            tb.tb_next = tb_next
            tb_next = tb

        raise exc.with_traceback(tb_next)  # noqa: B904


def shorten_filename(fn, *, base=None):
    """Shorten a source path, on the assumption the package prefix is known."""
    if base is None:
        base = os.path.dirname(os.path.dirname(__file__))
    try:
        prefix = os.path.commonpath([fn, base])
    except ValueError:
        return fn
    else:
        return fn[len(prefix) + 1 :]


def format_frame(frame, *, base=None, line=False) -> str:
    """
    Format a FrameSummary in a short way, without printing full absolute path or code.

    The idea is the result fits on a single line.
    """
    extra_line = ""
    if line:
        extra_line = f"{frame.line}  # "
    return f"{extra_line}{shorten_filename(frame.filename, base=base)}:{frame.lineno} in {frame.name}"


def format_traceback_short(tb):
    """Format a TracebackType in a short way, printing only the inner-most frame."""
    return format_frame(traceback.extract_tb(tb)[-1])


class CapturedTraceback:
    """A stack, held as frames rather than as something that must be resolved.

    Held rather than formatted because the caller usually wants to decide
    whether to pay for reading it: a run that records a stack for every call
    and prints none of them should not pay to read any of them.

    The frames are already what a record needs, so there is no resolution
    step and nothing to keep alive to make one possible.
    """

    __slots__ = ["tb", "skip"]

    def __init__(self, tb, skip=0) -> None:
        self.tb = tb
        self.skip = skip

    def cleanup(self) -> None:
        """Let go of the frames, for a caller that has read what it wanted."""

        self.tb = None

    def summary(self):
        """The frames as something that formats like an ordinary stack."""

        if self.tb is None:
            return traceback.StackSummary()

        return _extract_symbolized_tb(self.tb, self.skip)

    def __getstate__(self):
        # Frames name files and functions, but a captured stack is not worth
        # making picklable: nothing downstream needs it after a process boundary,
        # and a half-record reads as a stack with no frames when it is not.
        return (
            None,
            {
                "tb": None,
                "skip": self.skip,
            },
        )

    @staticmethod
    def extract(*, script=False, cpp=False, skip=0):
        """The caller's stack, as frames.

        Python frames by default, which is what a caller inside this library
        wants; the native frames instead when asked, which is what a caller
        that has left the interpreter wants.

        `skip` drops that many innermost frames, for a caller whose own frames
        are not part of the answer -- which is this function's own frame
        unless native frames were asked for, where it cannot be told apart.
        """
        import tensorplay as tp

        if script or cpp:
            if skip != 0:
                raise AssertionError("skip with script/cpp NYI")
            return CapturedTraceback(tp._C._gather_native_traceback())

        return CapturedTraceback(
            tp._C._gather_python_traceback(),
            # Elide extract() frame if we don't have script/cpp frames.  If
            # we do have those frames, it doesn't work so force zero.
            skip + 1,
        )

    def format(self):
        """
        Formats a single CapturedTraceback into a list of
        strings equivalent to the output of traceback.format_list.
        """
        return traceback.format_list(self.summary())

    @staticmethod
    def format_all(tbs):
        """
        Bulk version of CapturedTraceback.format.  Returns a list of list of strings.
        """
        rs: list[list[str]] = []
        for tb in tbs:
            rs.append([] if tb.tb is None else tb.format())
        return rs


def _extract_symbolized_tb(tb, skip):
    """
    Given a captured stack, return a StackSummary of its frames.

    Innermost first, the way a stack is normally read, and with the frames the
    caller said it did not want dropped.
    """
    stack = traceback.StackSummary()
    for f in reversed(tb[skip:]):
        stack.append(traceback.FrameSummary(f[0], f[1], f[2]))
    return stack
