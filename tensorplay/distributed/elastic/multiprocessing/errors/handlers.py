"""Error handlers serializing worker failures to a shared location."""
import abc
import os
import signal
import sys
import traceback


class ErrorHandler(abc.ABC):
    """Policy for capturing and persisting an in-process failure."""

    @abc.abstractmethod
    def record_exception(self, exc: BaseException) -> None:
        ...


class FileErrorHandler(ErrorHandler):
    """Serialize the failure as JSON into a pre-arranged error file.

    The target path is provided by the agent through the worker's
    environment; the file is written with ``O_CREAT|O_EXCL`` so the first
    failing writer wins and late failures do not clobber the first report.
    """

    def __init__(self, error_file: str, map_location: str | None = None) -> None:
        self.error_file = error_file
        os.makedirs(os.path.dirname(error_file), exist_ok=True)

    def record_exception(self, exc: BaseException) -> None:
        from .api import ProcessFailure  # avoid a circular import at module load

        data = {
            "message": repr(exc),
            "extraInfo": {
                "traceback": traceback.format_exc(),
                "timestamp": os.times(),
            },
        }
        with open(self.error_file, "w") as f:
            import json

            json.dump(data, f, default=str)


class SignalExceptionHandler(ErrorHandler):
    """Convert a death signal into a structured failure record."""

    def record_exception(self, exc: BaseException) -> None:
        signal_type = getattr(exc, "sigval", None)
        if signal_type is None:
            return
        from .api import SignalException

        if isinstance(exc, SignalException):
            raise exc
