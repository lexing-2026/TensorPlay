from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, NamedTuple

import tensorplay as tp

from .mod_tracker import ModTracker
from .runtime_estimator import RuntimeEstimator

__all__ = ["SACEstimator", "SACStats", "MSPS", "SACTradeOffStats", "SACGreedyOrderMeta"]


@dataclass
class _SACMetadata:
    func: Any
    time_taken: float
    memory_used: float
    curr_idx: int
    output_ids: tuple[int, ...]
    inplace_info: tuple[int, ...]
    is_view_like: bool
    is_rand_op: bool


@dataclass
class _SACModMetadata:
    start_idx: int
    force_store_random: bool
    sac_metadata: list[_SACMetadata]


@dataclass
class SACStats:
    func_names: list[str]
    runtimes: list[float]
    memory: list[int]
    view_like_ops: list[int]
    rand_ops: list[int]
    saved_autograd_ops: list[int]
    inplace_ops: list[tuple[int, int]]
    force_store_random: bool


class MSPS(NamedTuple):
    func_names: set[str]
    op_idx: int
    memory: int
    runtime: float
    msps: float


@dataclass
class SACTradeOffStats:
    tradeoff_curve: OrderedDict[float, float]
    sac_runtime: float
    sac_memory: int
    n_segments: int
    slopes: list[float]
    intercepts: list[float]
    fit_breaks: list[float]


@dataclass
class SACGreedyOrderMeta:
    inplace_op_groups: dict[int, set[int]]
    random_ops_group: dict[int, set[int]]
    msps_meta: list[MSPS]


def _tensor_bytes(value: Any) -> int:
    nbytes = getattr(value, "nbytes", None)
    if callable(nbytes):
        return int(nbytes())
    return int(getattr(value, "numel", lambda: 0)()) * int(getattr(value, "element_size", lambda: 1)())


def _walk_tensors(value: Any):
    if isinstance(value, tp.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _walk_tensors(child)


class SACEstimator:
    """Collect activation records and build memory/runtime trade-off curves."""

    def __init__(self, gpu_type: str | None = None) -> None:
        self.sac_mod_stats: dict[str, SACStats] = {}
        self.sac_mod_tradeoff_stats: dict[str, SACTradeOffStats] = {}
        self.sac_mod_greedy_order_meta: dict[str, SACGreedyOrderMeta] = {}
        self._mod_tracker = ModTracker()
        self._gpu_type = gpu_type
        self._estimate_runtime = RuntimeEstimator._roofline_estimate
        self._sac_metadata: list[_SACMetadata] = []
        self._sac_mod_metadata: dict[str, _SACModMetadata] = {}
        self._leaf_modules: set[str] = set()

    def _pre_fw_hook(self, mod: Any, inputs: Any) -> None:
        name = self._mod_tracker.get_known_fqn(mod) or type(mod).__name__
        if list(mod.children()):
            self._sac_mod_metadata[name] = _SACModMetadata(len(self._sac_metadata), not any(isinstance(x, tp.Tensor) for x in _walk_tensors(inputs)), [])
        else:
            self._leaf_modules.add(name)

    def _post_fw_hook(self, mod: Any, inputs: Any, outputs: Any) -> None:
        del inputs
        name = self._mod_tracker.get_known_fqn(mod) or type(mod).__name__
        if name in self._leaf_modules:
            return
        metadata = self._sac_mod_metadata.get(name, _SACModMetadata(0, False, []))
        if not metadata.sac_metadata:
            memory = sum(_tensor_bytes(value) for value in _walk_tensors(outputs))
            metadata.sac_metadata.append(_SACMetadata("module_output", 0.0, memory, 0, (), (), False, False))
        self.sac_mod_stats[name] = self._get_sac_stats(metadata.sac_metadata, metadata.force_store_random)
        self.sac_mod_greedy_order_meta[name] = self._get_greedy_order_meta(self.sac_mod_stats[name])

    def _get_sac_stats(self, data: list[_SACMetadata], force_store_random: bool) -> SACStats:
        return SACStats(
            func_names=[getattr(item.func, "__name__", str(item.func)) for item in data],
            runtimes=[float(item.time_taken) for item in data],
            memory=[int(item.memory_used) for item in data],
            view_like_ops=[index for index, item in enumerate(data) if item.is_view_like],
            rand_ops=[index for index, item in enumerate(data) if item.is_rand_op],
            saved_autograd_ops=[],
            inplace_ops=[item.inplace_info for item in data if item.inplace_info],
            force_store_random=force_store_random,
        )

    def _get_greedy_order_meta(self, sac_stats: SACStats) -> SACGreedyOrderMeta:
        groups = {index: {index} for index in sac_stats.rand_ops}
        values = [
            MSPS({name}, index, memory, runtime, memory / runtime if runtime else float("inf"))
            for index, (name, runtime, memory) in enumerate(zip(sac_stats.func_names, sac_stats.runtimes, sac_stats.memory))
        ]
        values.sort(key=lambda item: item.msps, reverse=True)
        return SACGreedyOrderMeta({}, groups, values)

    def _get_sac_tradeoff_pwlf_stats(self, sac_stats: SACStats) -> SACTradeOffStats:
        total_memory = sum(sac_stats.memory)
        total_runtime = sum(sac_stats.runtimes)
        curve = OrderedDict({0.0: 0.0, 1.0: total_runtime})
        return SACTradeOffStats(curve, total_runtime, total_memory, 1, [total_runtime], [0.0], [0.0, 1.0])

    def __tensorplay_dispatch__(self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> Any:
        del types
        start = 0.0
        result = func(*args, **(kwargs or {}))
        output_ids = tuple(id(value) for value in _walk_tensors(result))
        item = _SACMetadata(func, start, sum(_tensor_bytes(value) for value in _walk_tensors(result)), len(self._sac_metadata), output_ids, (), False, False)
        self._sac_metadata.append(item)
        for name, metadata in self._sac_mod_metadata.items():
            if len(self._sac_metadata) > metadata.start_idx:
                metadata.sac_metadata.append(item)
        return result

    def pwlf_sac_tradeoff_curve(self) -> None:
        self.sac_mod_tradeoff_stats = {name: self._get_sac_tradeoff_pwlf_stats(stats) for name, stats in self.sac_mod_stats.items()}

    def display_sac_stats(self, depth: int = 2, print_tabular: bool = False) -> None:
        del print_tabular
        for name, stats in self.sac_mod_stats.items():
            if name.count(".") + 1 <= depth:
                print(name, list(zip(stats.func_names, stats.memory, stats.runtimes)))

    def display_sac_tradeoff_stats(self, depth: int = 2, print_tabular: bool = False) -> None:
        del print_tabular
        self.pwlf_sac_tradeoff_curve()
        for name, stats in self.sac_mod_tradeoff_stats.items():
            if name.count(".") + 1 <= depth:
                print(name, stats.tradeoff_curve)

    def display_modulewise_sac_stats(self, depth: int = 2, print_tabular: bool = False) -> None:
        self.display_sac_stats(depth, print_tabular)

    def __call__(self, estimate_mode_type: str) -> "SACEstimator":
        estimator = RuntimeEstimator(self._gpu_type)(estimate_mode_type)
        self._estimate_runtime = estimator._estimate
        return self

    def __enter__(self) -> "SACEstimator":
        self._sac_metadata.clear()
        self._sac_mod_metadata.clear()
        self.sac_mod_stats.clear()
        self._mod_tracker.register_user_hooks(self._pre_fw_hook, self._post_fw_hook)
        self._mod_tracker.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._mod_tracker.clear_user_hooks()
        self._mod_tracker.__exit__(*args)
