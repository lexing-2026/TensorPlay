"""Logging helpers for explicit sharded tensors."""

import logging

__all__ = ["_get_or_create_logger", "_get_logging_handler"]


def _get_or_create_logger(name: str = "tensorplay.distributed.sharded_tensor") -> logging.Logger:
    return logging.getLogger(name)


def _get_logging_handler() -> logging.Handler:
    return logging.NullHandler()
