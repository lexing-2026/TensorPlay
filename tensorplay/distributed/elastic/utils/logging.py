"""Logging helper used across the elastic package."""
import logging
import os
import inspect
import warnings


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger whose level honors the ``LOGLEVEL`` environment variable.

    ``LOGLEVEL`` defaults to ``INFO``. Messages propagate to the root logger
    so agent and worker logs share one destination.
    """
    return _setup_logger(name or _derive_module_name(depth=2))


def _setup_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
    return logger


def _derive_module_name(depth: int = 1) -> str | None:
    try:
        stack = inspect.stack()
        if depth >= len(stack):
            raise AssertionError
        frame_info = stack[depth]
        module = inspect.getmodule(frame_info[0])
        if module is not None:
            return module.__name__
        return os.path.splitext(os.path.basename(frame_info.filename))[0]
    except Exception as exc:
        warnings.warn(
            f"Error deriving logger module name: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
