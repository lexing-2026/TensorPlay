"""Logging helper used across the elastic package."""
import logging
import os


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger whose level honors the ``LOGLEVEL`` environment variable.

    ``LOGLEVEL`` defaults to ``INFO``. Messages propagate to the root logger
    so agent and worker logs share one destination.
    """
    logger = logging.getLogger(name)
    level = os.environ.get("LOGLEVEL", "INFO").upper()
    logger.setLevel(level)
    return logger
