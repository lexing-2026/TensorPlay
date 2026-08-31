from __future__ import annotations

import logging


class _CheckpointLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        return None
