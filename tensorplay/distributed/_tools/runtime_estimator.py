from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import tensorplay as tp

from .mod_tracker import ModTracker

__all__ = ["RuntimeEstimator"]


def _walk_tensors(value: Any):
    if isinstance(value, tp.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_tensors(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk_tensors(child)


def _tensor_bytes(value: Any) -> int:
    nbytes = getattr(value, "nbytes", None)
    if callable(nbytes):
        return int(nbytes())
    return int(value.numel()) * int(getattr(value, "element_size", lambda: 1)())


class RuntimeEstimator:
    """Estimate execution time and aggregate it by active module."""

    _no_fallback_kernel: set[Any] = set()
    fake_mode: Any = None
    gpu_type: str | None = None

    def __init__(self, gpu_type: str | None = None) -> None:
        self._gpu_type = gpu_type
        self._estimate: Callable[..., tuple[Any, float]] = self._benchmark_estimate
        self._estimate_mode_type = "operator-level-benchmark"
        self._mod_tracker = ModTracker()
        self.mod_runtimes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.mod_fw_pre_order: list[str] = []
        self.mod_bw_pre_order: list[str] = []
        self.mod_fw_post_order: list[str] = []
        self.mod_bw_post_order: list[str] = []
        self.total_runtime = 0.0

    @classmethod
    def _maybe_run_and_benchmark_fallback_kernel(cls, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], orig_not_implemented_exception: Exception) -> tuple[Any, float]:
        del orig_not_implemented_exception
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000.0
        return result, elapsed

    @classmethod
    def _benchmark_estimate(cls, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any] | None) -> tuple[Any, float]:
        start = time.perf_counter()
        result = func(*args, **(kwargs or {}))
        return result, (time.perf_counter() - start) * 1000.0

    @classmethod
    def _roofline_estimate(cls, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any] | None) -> tuple[Any, float]:
        result = func(*args, **(kwargs or {}))
        bytes_moved = sum(_tensor_bytes(value) for value in _walk_tensors(args)) + sum(_tensor_bytes(value) for value in _walk_tensors(result))
        bandwidth = 900.0e9 if cls.gpu_type else 50.0e9
        return result, bytes_moved / bandwidth * 1000.0

    def display_modulewise_stats(self, depth: int = 2) -> None:
        for name in self.mod_fw_pre_order:
            if name.count(".") + 1 <= depth:
                print(name)
        for name, values in self.mod_runtimes.items():
            if name.count(".") + 1 <= depth:
                print(f"{name} fw: {values.get('fw', 0.0):.3f}ms bw: {values.get('bw', 0.0):.3f}ms")

    def __tensorplay_dispatch__(self, func: Callable[..., Any], types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> Any:
        del types
        result, elapsed = self._estimate(func, args, kwargs)
        phase = "bw" if self._mod_tracker.is_bw else "fw"
        for parent in self._mod_tracker.parents:
            self.mod_runtimes[parent][phase] += elapsed
        self.total_runtime += elapsed
        return result

    def __call__(self, estimate_mode_type: str) -> "RuntimeEstimator":
        if estimate_mode_type == "operator-level-benchmark":
            self._estimate = self._benchmark_estimate
        elif estimate_mode_type == "operator-level-cost-model":
            self._estimate = self._roofline_estimate
        else:
            raise NotImplementedError(f"estimate mode {estimate_mode_type!r} is not supported")
        self._estimate_mode_type = estimate_mode_type
        return self

    def __enter__(self) -> "RuntimeEstimator":
        self.mod_runtimes = defaultdict(lambda: defaultdict(float))
        self.mod_fw_pre_order.clear()
        self.mod_bw_pre_order.clear()
        self.mod_fw_post_order.clear()
        self.mod_bw_post_order.clear()
        self.total_runtime = 0.0
        self._mod_tracker.register_user_hooks(
            pre_fw_hook=lambda module, inputs: self.mod_fw_pre_order.append(self._mod_tracker.get_known_fqn(module) or type(module).__name__),
            post_fw_hook=lambda module, inputs, output: self.mod_fw_post_order.append(self._mod_tracker.get_known_fqn(module) or type(module).__name__),
            pre_bw_hook=lambda module, grad: self.mod_bw_pre_order.append(self._mod_tracker.get_known_fqn(module) or type(module).__name__),
            post_bw_hook=lambda module, grad: self.mod_bw_post_order.append(self._mod_tracker.get_known_fqn(module) or type(module).__name__),
        )
        self._mod_tracker.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._mod_tracker.clear_user_hooks()
        self._mod_tracker.__exit__(*args)
