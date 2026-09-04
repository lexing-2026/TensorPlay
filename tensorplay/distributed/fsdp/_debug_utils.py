"""Profiling and module-tree inspection helpers."""

import logging
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any

from ._common_utils import _get_module_fsdp_state, clean_tensor_name
from ._common_utils import _apply_to_modules

logger = logging.getLogger(__name__)


class SimpleProfiler:
    class Type(str, Enum):
        ALL = "all"
        ALLGATHER = "all_gather"
        ALLGATHER_OBJ = "all_gather_object"
        RESHARDING = "resharding"
        H2D = "H2D"
        D2H = "D2H"

    results: dict[str, float] = defaultdict(float)
    profiling: set[str] = set()

    @classmethod
    def reset(cls) -> None:
        cls.results.clear()
        cls.profiling.clear()

    @classmethod
    @contextmanager
    def profile(cls, profile_type: str) -> Iterator[None]:
        if profile_type in cls.profiling:
            raise AssertionError(f"{profile_type} is already being profiled")
        cls.profiling.add(profile_type)
        begin = time.monotonic()
        try:
            yield
        finally:
            cls.results[profile_type] += time.monotonic() - begin
            cls.profiling.remove(profile_type)

    @classmethod
    def dump_and_reset(cls, message: str = "") -> None:
        if message:
            logger.info("%s %s", message, cls.results)
        cls.reset()

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

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.end("scope")


def _get_sharded_module_tree_with_module_name_to_fqns(
    model: Any,
) -> tuple[str, dict[str, list[str]]]:
    def module_fn(
        module: Any,
        prefix: str,
        tree_level: int,
        tree_info: list[str],
        module_to_fqns: dict[str, list[str]],
    ) -> None:
        name = prefix[:-1] if prefix.endswith(".") else prefix
        key = f"{name}[{type(module).__name__}]"
        state = _get_module_fsdp_state(module)
        tree_info.append(" " * (tree_level * 4) + key + (" FULLY SHARDED" if state is not None else ""))
        if state is None:
            return
        handles = getattr(state, "_fully_sharded_module_to_handle", {})
        handle = handles.get(module) if isinstance(handles, dict) else None
        if handle is not None:
            flat = getattr(handle, "flat_param", None)
            fqns = getattr(flat, "_fqns", ())
            module_to_fqns[key] = [clean_tensor_name(prefix + fqn) for fqn in fqns]
            return
        for group in getattr(state, "_all_param_groups", lambda: [])():
            for param in getattr(group, "params", ()):
                info = getattr(param, "module_info", None)
                if info is not None:
                    module_to_fqns.setdefault(key, []).append(clean_tensor_name(info.fqn))

    def return_fn(
        tree_info: list[str], module_to_fqns: dict[str, list[str]]
    ) -> tuple[str, dict[str, list[str]]]:
        return "\n".join(tree_info) + ("\n" if tree_info else ""), module_to_fqns

    tree_info: list[str] = []
    module_to_fqns: dict[str, list[str]] = {}
    names = [name for name, _ in model.named_parameters()]
    return _apply_to_modules(
        model, module_fn, return_fn, names, tree_info, module_to_fqns
    )


__all__ = ["SimpleProfiler", "_get_sharded_module_tree_with_module_name_to_fqns"]
