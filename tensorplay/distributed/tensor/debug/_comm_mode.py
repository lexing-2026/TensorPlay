"""Collective and operation counters for a scoped execution."""

from __future__ import annotations

import contextlib
from collections import Counter
from typing import Any

__all__ = ["CommDebugMode"]


class CommDebugMode(contextlib.AbstractContextManager["CommDebugMode"]):
    def __init__(self) -> None:
        self.comm_counts: Counter[str] = Counter()
        self.operation_counts: Counter[str] = Counter()
        self._active = False

    def record_collective(self, name: str, count: int = 1) -> None:
        self.comm_counts[name] += count

    def record_operation(self, name: str, count: int = 1) -> None:
        self.operation_counts[name] += count

    def get_comm_counts(self) -> dict[str, int]:
        return dict(self.comm_counts)

    def get_operation_counts(self) -> dict[str, int]:
        return dict(self.operation_counts)

    def generate_json_dump(self, file_name: str = "comm_mode_log.json", noise_level: int = 3) -> dict[str, Any]:
        del noise_level
        import json

        data = {"collectives": self.get_comm_counts(), "operations": self.get_operation_counts()}
        with open(file_name, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
        return data

    def __enter__(self) -> "CommDebugMode":
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        self._active = False
