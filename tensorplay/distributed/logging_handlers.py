# Ported from torch/distributed/logging_handlers.py.
import logging

__all__: list[str] = []

_log_handlers: dict[str, logging.Handler] = {
    "default": logging.NullHandler(),
}
