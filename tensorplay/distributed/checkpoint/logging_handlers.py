from __future__ import annotations

import logging

from tensorplay.distributed.logging_handlers import _log_handlers

__all__: list[str] = []

DCP_LOGGER_NAME = "dcp_logger"

_log_handlers[DCP_LOGGER_NAME] = logging.NullHandler()
