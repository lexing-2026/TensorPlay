from __future__ import annotations

import operator
import pickle
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import tensorplay as tp

BYTES_PER_MB = 1024 * 1024.0


def _tensor_bytes(value: Any) -> int:
    nbytes = getattr(value, "nbytes", None)
    if callable(nbytes):
        return int(nbytes())
    return int(getattr(value, "numel", lambda: 0)()) * int(getattr(value, "element_size", lambda: 1)())


class MemoryProfileDispatchMode:
    """Record a memory sample after each explicitly dispatched operation."""

    def __init__(self, memory_tracker: "MemoryTracker") -> None:
        self.memory_tracker = memory_tracker

    def __enter__(self) -> "MemoryProfileDispatchMode":
        return self

    def __exit__(self, *args: Any) -> None:
        del args

    def __tensorplay_dispatch__(self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> Any:
        del types
        result = func(*args, **(kwargs or {}))
        self.memory_tracker._record_memory_stats(getattr(func, "__name__", repr(func)))
        return result


class MemoryTracker:
    """Collect operator-level memory samples from module execution."""

    def __init__(self) -> None:
        self._hooks: list[Any] = []
        self._operator_names: dict[str, int] = defaultdict(int)
        self.memories_allocated: dict[int, tuple[str, float]] = {}
        self.memories_active: dict[int, tuple[str, float]] = {}
        self.memories_reserved: dict[int, tuple[str, float]] = {}
        self._markers: dict[str, int] = {}
        self._cur_module_name = ""
        self._op_index = 0
        self._num_alloc_retries = 0
        self.profile_mode: MemoryProfileDispatchMode | None = None
        self._root_module: Any = None
        self._current_memory = 0

    def _clear_state(self) -> None:
        self._operator_names.clear()
        self.memories_allocated.clear()
        self.memories_active.clear()
        self.memories_reserved.clear()
        self._markers.clear()
        self._cur_module_name = ""
        self._op_index = 0
        self._num_alloc_retries = 0
        self._current_memory = 0

    def _module_memory(self, module: Any) -> int:
        total = 0
        for value in module.parameters():
            total += _tensor_bytes(value)
        for value in module.buffers():
            total += _tensor_bytes(value)
        return total

    def start_monitor(self, root_module: Any) -> None:
        self._clear_state()
        self._root_module = root_module
        for name, module in root_module.named_modules():
            module._memory_tracker_is_root = module is root_module
            self._hooks.append(module.register_forward_pre_hook(self._create_pre_forward_hook(name)))
            self._hooks.append(module.register_forward_hook(self._create_post_forward_hook(name)))
        self.profile_mode = MemoryProfileDispatchMode(self)
        self.profile_mode.__enter__()

    def stop(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        if self.profile_mode is not None:
            self.profile_mode.__exit__(None, None, None)
            self.profile_mode = None

    def _create_pre_forward_hook(self, name: str) -> Callable[..., Any]:
        def hook(module: Any, inputs: Any) -> None:
            del inputs
            self._cur_module_name = f"{name}.forward"
            self._current_memory = max(self._current_memory, self._module_memory(module))
            if getattr(module, "_memory_tracker_is_root", False):
                self._add_marker("fw_start")
        return hook

    def _create_post_forward_hook(self, name: str) -> Callable[..., Any]:
        def hook(module: Any, inputs: Any, outputs: Any) -> None:
            del name, inputs
            self._current_memory = max(self._current_memory, self._module_memory(module) + sum(_tensor_bytes(value) for value in _walk_tensors(outputs)))
            if getattr(module, "_memory_tracker_is_root", False):
                self._add_marker("fw_bw_boundary")
        return hook

    def _create_backward_hook(self, name: str) -> Callable[..., Any]:
        def hook(module: Any, grad_input: Any, grad_output: Any) -> None:
            del module, grad_input, grad_output
            self._cur_module_name = f"{name}.backward"
        return hook

    def _record_memory_stats(self, fn_name: str) -> None:
        name = f"{self._cur_module_name}.{fn_name}_{self._operator_names[fn_name]}"
        self._operator_names[fn_name] += 1
        value = self._current_memory / BYTES_PER_MB
        self.memories_allocated[self._op_index] = (name, value)
        self.memories_active[self._op_index] = (name, value)
        self.memories_reserved[self._op_index] = (name, value)
        self._op_index += 1

    def _add_marker(self, marker_name: str) -> None:
        self._markers[marker_name] = self._op_index

    def summary(self, top: int = 20) -> None:
        changes: dict[str, float] = defaultdict(float)
        previous = 0.0
        for index in sorted(self.memories_allocated):
            name, current = self.memories_allocated[index]
            changes[name] += current - previous
            previous = current
        print(f"allocation retries: {self._num_alloc_retries}")
        for name, value in sorted(changes.items(), key=operator.itemgetter(1), reverse=True)[:top]:
            print(f"{name}: {value:.3f} MB")

    def show_traces(self, path: str = "") -> None:
        if path:
            self.load(path)
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ImportError("plotting traces requires matplotlib") from error
        x = sorted(self.memories_allocated)
        plt.figure()
        for values, label in ((self.memories_allocated, "allocated"), (self.memories_active, "active"), (self.memories_reserved, "reserved")):
            plt.plot(x, [values[index][1] for index in x], label=label)
        for marker, index in self._markers.items():
            plt.axvline(index, label=marker)
        plt.xlabel("operator calls")
        plt.ylabel("memory (MB)")
        plt.legend()
        if path:
            plt.savefig(path)

    def save_stats(self, path: str) -> None:
        with open(path, "wb") as stream:
            pickle.dump({"memories_allocated": self.memories_allocated, "memories_active": self.memories_active, "memories_reserved": self.memories_reserved, "markers": self._markers, "num_alloc_retries": self._num_alloc_retries}, stream, pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        with open(path, "rb") as stream:
            values = pickle.load(stream)
        self.memories_allocated = values["memories_allocated"]
        self.memories_active = values["memories_active"]
        self.memories_reserved = values["memories_reserved"]
        self._markers = values["markers"]
        self._num_alloc_retries = values["num_alloc_retries"]
        self._op_index = len(self.memories_allocated)


def _walk_tensors(value: Any):
    if isinstance(value, tp.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_tensors(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk_tensors(child)
