"""Small profiling utilities for sharded execution."""

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SimpleProfiler", "_get_sharded_module_tree_with_module_name_to_fqns"]


class SimpleProfiler:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._active: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._active[name] = time.perf_counter()

    def end(self, name: str) -> float:
        start = self._active.pop(name, time.perf_counter())
        elapsed = time.perf_counter() - start
        self.events.append({"name": name, "duration": elapsed})
        return elapsed

    def __enter__(self) -> "SimpleProfiler":
        self.start("scope")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.end("scope")


def _get_sharded_module_tree_with_module_name_to_fqns(root: Any) -> tuple[Any, dict[str, list[str]]]:
    result: dict[str, list[str]] = {}
    for name, module in root.named_modules():
        result[name] = [fqn for fqn, _ in module.named_parameters(recurse=False)]
    return root, result
