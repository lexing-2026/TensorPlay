from __future__ import annotations

import logging
from collections.abc import Callable

__all__ = ["FlightRecorderLogger"]


class FlightRecorderLogger:
    _instance: "FlightRecorderLogger | None" = None

    def __new__(cls):
        if cls._instance is None:
            instance = super().__new__(cls)
            instance.logger = logging.getLogger("tensorplay.flight_recorder")
            if not instance.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                instance.logger.addHandler(handler)
            instance.logger.setLevel(logging.INFO)
            cls._instance = instance
        return cls._instance

    def set_log_level(self, level: int) -> None:
        self.logger.setLevel(level)

    @property
    def debug(self) -> Callable:
        return self.logger.debug

    @property
    def info(self) -> Callable:
        return self.logger.info

    @property
    def warning(self) -> Callable:
        return self.logger.warning

    @property
    def error(self) -> Callable:
        return self.logger.error

    @property
    def critical(self) -> Callable:
        return self.logger.critical
