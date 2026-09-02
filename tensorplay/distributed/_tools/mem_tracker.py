from __future__ import annotations

import math
import re
import weakref
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from enum import Enum, auto
from functools import partial
from typing import Any

import tensorplay as tp

from .common_utils import get_untyped_storages
from .mod_tracker import ModTracker

__all__ = ["MemTracker"]

_TOTAL_KEY = "Total"


class _RefType(str, Enum):
    pass


class _State(str, Enum):
    pass


class _MemRefType(_RefType):
    PARAM = "Parameter"
    BUFFER = "Buffer"
    GRAD = "Gradient"
    ACT = "Activation"
    TEMP = "Temp"
    OPT = "Optstate"
    OTH = "Other"


class _ModState(_State):
    PRE_FW = "Pre-Forward"
    POST_FW = "Post-Forward"
    PEAK_FW = "Peak-Forward"
    PRE_BW = "Pre-Backward"
    PRE_FW_AC = "Pre-Forward-AC"
    POST_FW_AC = "Post-Forward-AC"
    POST_BW = "Post-Backward"
    PEAK_BW = "Peak-Backward"


class _ModMemStats:
    def __init__(self, mod_fqn: str):
        self.mod_fqn = mod_fqn
        self.parameter_mem = 0
        self.buffer_mem = 0
        self.input_mem = 0
        self.output_mem = 0
        self.local_peak: dict[Any, int] = {}
        self.snapshots: dict[_ModState, list[dict[Any, dict[str, int]]]] = {}


def _storage_size(storage: Any, value: Any) -> int:
    nbytes = getattr(storage, "nbytes", None)
    if callable(nbytes):
        return int(nbytes())
    size = getattr(storage, "size", None)
    element_size = getattr(value, "element_size", lambda: 1)
    if callable(size):
        return int(size()) * int(element_size())
    return int(getattr(value, "numel", lambda: 0)()) * int(element_size())


class _WeakRefInfo:
    def __init__(self, size: int, element_size: int, device: Any, reftype: _RefType) -> None:
        self.size = int(size)
        self.element_size = int(element_size)
        self.reftype = reftype
        self.device = device
        self.mem_consumed = self._calculate_mem_consumed()

    def _calculate_mem_consumed(self) -> int:
        value = self.size * self.element_size
        minimum = 512 if getattr(self.device, "type", None) in {"cuda", "xpu"} else 1
        return math.ceil(value / minimum) * minimum if value else 0

    def update_mem_consumed(self, st: Any) -> int:
        size = int(st.size())
        if size != self.size:
            self.size = size
            self.mem_consumed = self._calculate_mem_consumed()
        return self.mem_consumed

    @classmethod
    def create_winfo(
        cls,
        st: Any,
        device: Any,
        reftype: _RefType,
        callback: Callable[["_WeakRefInfo", Any], Any] | None = None,
    ) -> tuple["_WeakRefInfo", Any]:
        size = int(st.size()) if callable(getattr(st, "size", None)) else 0
        element_size = int(st.element_size()) if callable(getattr(st, "element_size", None)) else 1
        winfo = cls(size, element_size, device, reftype)
        try:
            reference = weakref.ref(st, partial(callback, winfo) if callback else None)
        except TypeError:
            reference = lambda: st
        return winfo, reference


def _get_mem_divisor(units: str) -> int:
    values = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30}
    if units not in values:
        raise ValueError(f"unsupported memory unit {units!r}")
    return values[units]


def _rounding_fn(value: int, divisor: int, precision: int) -> float | int:
    return value if divisor == 1 else round(value / divisor, precision)


def _print_snapshot(snapshot: dict[Any, dict[str, int]], units: str) -> None:
    if not snapshot:
        print("No memory tracked.")
        return
    divisor = _get_mem_divisor(units)
    for device, values in snapshot.items():
        if not values.get(_TOTAL_KEY, 0):
            continue
        print(f"Device: {device}")
        for key, value in values.items():
            label = key.value if isinstance(key, _RefType) else key
            print(f"\t{label}: {_rounding_fn(value, divisor, 2)} {units}")


def _print_snapshot_tabular(snapshot: dict[Any, dict[str, int]], units: str) -> None:
    if not snapshot:
        print("No memory tracked.")
        return
    divisor = _get_mem_divisor(units)
    keys = list(next(iter(snapshot.values())).keys())
    headings = ["Device"] + [key.value if isinstance(key, _RefType) else str(key) for key in keys]
    print(" | ".join(headings))
    for device, values in snapshot.items():
        row = [str(device)] + [str(_rounding_fn(values.get(key, 0), divisor, 2)) for key in keys]
        print(" | ".join(row))


def _print_state_snapshots(snapshots: dict[_State, list[dict[Any, dict[str, int]]]], units: str) -> None:
    for state, values in snapshots.items():
        print(state.value)
        for index, snapshot in enumerate(values, 1):
            print(f"#{index}:")
            _print_snapshot(snapshot, units)


def _print_state_snapshots_tabular(snapshots: dict[_State, list[dict[Any, dict[str, int]]]], units: str) -> None:
    for state, values in snapshots.items():
        for index, snapshot in enumerate(values, 1):
            print(f"{state.value} #{index}")
            _print_snapshot_tabular(snapshot, units)


class _UpdateType(Enum):
    ADD = auto()
    DEL = auto()
    REF = auto()
    SIZE = auto()


def _walk_tensors(value: Any):
    if isinstance(value, tp.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_tensors(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk_tensors(child)


def _tensor_memory(value: Any) -> int:
    storages = get_untyped_storages(value)
    if storages:
        return sum(_storage_size(storage, value) for storage in storages)
    numel = getattr(value, "numel", lambda: 0)
    element_size = getattr(value, "element_size", lambda: 1)
    return int(numel()) * int(element_size())


class MemTracker:
    """Capture tensor memory by device and by execution phase."""

    def __init__(self) -> None:
        self.memory_tracking: dict[Any, _ModMemStats] = {}
        self._curr_mem_snap: dict[Any, dict[str, int]] = {}
        self._peak_mem_snap: dict[Any, dict[str, int]] = {}
        self._peak_mem: dict[Any, int] = {}
        self._WINFO: dict[int, tuple[_WeakRefInfo, Any]] = {}
        self._mod_tracker = ModTracker()
        self._ref_class: type[_RefType] = _MemRefType
        self._depth = 0
        self._in_opt = False
        self._in_ac = False
        self._ac_mod: weakref.ref[Any] | None = None
        self._tracked_objects: dict[int, tuple[int, Any, Any]] = {}
        self._external_modules: list[Any] = []
        self._external_optimizers: list[Any] = []

    def _empty_device_snapshot(self) -> dict[str, int]:
        snapshot = {key: 0 for key in self._ref_class}
        snapshot[_TOTAL_KEY] = 0
        return snapshot

    def _update_snap(
        self,
        u_type: _UpdateType,
        winfo: _WeakRefInfo,
        old_mem_consumed: int | None = None,
        old_reftype: _RefType | None = None,
    ) -> None:
        values = self._curr_mem_snap.setdefault(winfo.device, self._empty_device_snapshot())
        if u_type is _UpdateType.ADD:
            values[winfo.reftype] = values.get(winfo.reftype, 0) + winfo.mem_consumed
            values[_TOTAL_KEY] += winfo.mem_consumed
        elif u_type is _UpdateType.DEL:
            values[winfo.reftype] = max(0, values.get(winfo.reftype, 0) - winfo.mem_consumed)
            values[_TOTAL_KEY] = max(0, values[_TOTAL_KEY] - winfo.mem_consumed)
        elif u_type is _UpdateType.REF:
            if old_reftype is None:
                raise ValueError("old reference type is required")
            values[old_reftype] = max(0, values.get(old_reftype, 0) - winfo.mem_consumed)
            values[winfo.reftype] = values.get(winfo.reftype, 0) + winfo.mem_consumed
        elif u_type is _UpdateType.SIZE:
            if old_mem_consumed is None:
                raise ValueError("old memory size is required")
            change = winfo.mem_consumed - old_mem_consumed
            values[winfo.reftype] = values.get(winfo.reftype, 0) + change
            values[_TOTAL_KEY] += change
        else:
            raise ValueError(f"unknown update type {u_type}")
        if values[_TOTAL_KEY] <= 0:
            self._curr_mem_snap.pop(winfo.device, None)
        self._update_peak_stats(_ModState.PEAK_FW)

    def _update_and_maybe_create_winfos(self, t: tp.Tensor, reftype: _RefType, update_existing: bool = False) -> set[_WeakRefInfo]:
        result: set[_WeakRefInfo] = set()
        for storage in get_untyped_storages(t):
            key = id(storage)
            previous = self._WINFO.get(key)
            if previous is not None:
                winfo = previous[0]
                old_type = winfo.reftype
                if old_type is not reftype:
                    winfo.reftype = reftype
                    self._update_snap(_UpdateType.REF, winfo, old_reftype=old_type)
                result.add(winfo)
                continue
            if update_existing:
                raise KeyError("storage is not tracked")
            winfo, reference = _WeakRefInfo.create_winfo(storage, t.device, reftype, self._delete_callback)
            self._WINFO[key] = (winfo, reference)
            if winfo.mem_consumed:
                self._update_snap(_UpdateType.ADD, winfo)
            result.add(winfo)
        return result

    def _delete_callback(self, winfo: _WeakRefInfo, w_st: Any) -> None:
        del w_st
        if winfo.mem_consumed:
            self._update_snap(_UpdateType.DEL, winfo)

    def _track_resize(self) -> None:
        return None

    def _restore_resize(self) -> None:
        return None

    def _update_peak_stats(self, peak_state: _State) -> None:
        del peak_state
        for device, snapshot in self._curr_mem_snap.items():
            if snapshot.get(_TOTAL_KEY, 0) > self._peak_mem.get(device, 0):
                self._peak_mem[device] = snapshot[_TOTAL_KEY]
                self._peak_mem_snap[device] = deepcopy(snapshot)
            for module, stats in self.memory_tracking.items():
                if module in self._known_active_modules():
                    current = snapshot[_TOTAL_KEY]
                    if current > stats.local_peak.get(device, 0):
                        stats.local_peak[device] = current
                        if _ModState.PEAK_FW in stats.snapshots:
                            stats.snapshots[_ModState.PEAK_FW][-1][device] = deepcopy(snapshot)

    def _known_active_modules(self) -> set[Any]:
        return {module for module, stats in self.memory_tracking.items() if stats.mod_fqn in self._mod_tracker.parents}

    def _track(self, reftype: _RefType, t: tp.Tensor) -> None:
        self._update_and_maybe_create_winfos(t, reftype)

    def get_tracker_snapshot(self, type: str = "current") -> dict[Any, dict[str, int]]:
        if type == "current":
            return deepcopy(self._curr_mem_snap)
        if type == "peak":
            return deepcopy(self._peak_mem_snap)
        raise ValueError(f"invalid snapshot type {type!r}")

    def _track_module_params_and_buffers(self, module: Any, install_grad_hooks: bool = True) -> tuple[int, int]:
        del install_grad_hooks
        parameter_mem = 0
        for parameter in module.parameters():
            parameter_mem += sum(item.mem_consumed for item in self._update_and_maybe_create_winfos(parameter, _MemRefType.PARAM))
            grad = getattr(parameter, "grad", None)
            if grad is not None:
                self._update_and_maybe_create_winfos(grad, _MemRefType.GRAD)
        buffer_mem = 0
        for buffer in module.buffers():
            buffer_mem += sum(item.mem_consumed for item in self._update_and_maybe_create_winfos(buffer, _MemRefType.BUFFER))
        return parameter_mem, buffer_mem

    def _track_inputs_or_outputs(self, args: Any) -> int:
        return sum(_tensor_memory(value) for value in _walk_tensors(args))

    def _pre_fw_hook(self, module: Any, inputs: Any) -> None:
        name = self._mod_tracker.get_known_fqn(module) or type(module).__name__
        stats = self.memory_tracking.get(module)
        if stats is None:
            stats = _ModMemStats(name)
            stats.parameter_mem, stats.buffer_mem = self._track_module_params_and_buffers(module)
            self.memory_tracking[module] = stats
        stats.input_mem = self._track_inputs_or_outputs(inputs)
        snapshot = self.get_tracker_snapshot()
        stats.local_peak = {device: values[_TOTAL_KEY] for device, values in snapshot.items()}
        stats.snapshots.setdefault(_ModState.PRE_FW, []).append(deepcopy(snapshot))
        stats.snapshots.setdefault(_ModState.PEAK_FW, []).append(deepcopy(snapshot))

    def _post_fw_hook(self, module: Any, inputs: Any, outputs: Any) -> None:
        del inputs
        stats = self.memory_tracking.setdefault(module, _ModMemStats(self._mod_tracker.get_known_fqn(module) or type(module).__name__))
        stats.output_mem = self._track_inputs_or_outputs(outputs)
        for value in _walk_tensors(outputs):
            self._track(_MemRefType.ACT, value)
        stats.snapshots.setdefault(_ModState.POST_FW, []).append(self.get_tracker_snapshot())

    def _pre_bw_hook(self, module: Any, args: Any) -> None:
        if module is None:
            return
        stats = self.memory_tracking.get(module)
        if stats is not None:
            snapshot = self.get_tracker_snapshot()
            stats.snapshots.setdefault(_ModState.PRE_BW, []).append(deepcopy(snapshot))
            stats.snapshots.setdefault(_ModState.PEAK_BW, []).append(deepcopy(snapshot))

    def _post_bw_hook(self, module: Any, args: Any) -> None:
        del args
        stats = self.memory_tracking.get(module)
        if stats is not None:
            stats.snapshots.setdefault(_ModState.POST_BW, []).append(self.get_tracker_snapshot())

    def _track_optimizer_states(self, reftype: _RefType, optimizer: Any) -> None:
        for state in getattr(optimizer, "state", {}).values():
            for value in state.values() if isinstance(state, Mapping) else ():
                if isinstance(value, tp.Tensor):
                    self._track(reftype, value)

    def _register_global_optimizer_hook(self) -> None:
        return None

    def _deregister_param_and_optimizer_hooks(self) -> None:
        return None

    def track_external(self, *external: Any) -> None:
        for obj in external:
            if isinstance(obj, tp.Tensor):
                self._track(_MemRefType.OTH, obj)
            elif hasattr(obj, "named_modules"):
                if obj not in self._external_modules:
                    self._external_modules.append(obj)
                self._track_module_params_and_buffers(obj, install_grad_hooks=False)
            elif hasattr(obj, "state") and hasattr(obj, "param_groups"):
                if obj not in self._external_optimizers:
                    self._external_optimizers.append(obj)
                self._track_optimizer_states(_MemRefType.OPT, obj)
            elif obj is not None:
                raise TypeError(f"unsupported tracked object {type(obj)!r}")

    def display_snapshot(self, type: str = "current", units: str = "B", tabulate: bool = False) -> None:
        snapshot = self.get_tracker_snapshot(type)
        (_print_snapshot_tabular if tabulate else _print_snapshot)(snapshot, units)

    def display_modulewise_snapshots(self, depth: int = 2, units: str = "B", tabulate: bool = False) -> None:
        def key(item: _ModMemStats) -> list[int | str]:
            return [int(part) if part.isdigit() else part for part in re.split(r"([0-9]+)", item.mod_fqn)]
        for stats in sorted(self.memory_tracking.values(), key=key):
            if stats.mod_fqn.count(".") + 1 > depth:
                continue
            print(f"Module: {stats.mod_fqn}")
            (_print_state_snapshots_tabular if tabulate else _print_state_snapshots)(stats.snapshots, units)

    def reset_mod_stats(self) -> None:
        self.memory_tracking.clear()

    def __enter__(self) -> "MemTracker":
        if self._depth == 0:
            self._mod_tracker.register_user_hooks(self._pre_fw_hook, self._post_fw_hook, self._pre_bw_hook, self._post_bw_hook)
            self._mod_tracker.__enter__()
            self._peak_mem_snap = self.get_tracker_snapshot()
            self._peak_mem = {device: values[_TOTAL_KEY] for device, values in self._peak_mem_snap.items()}
        self._depth += 1
        return self

    def __exit__(self, *args: Any) -> None:
        self._depth -= 1
        if self._depth == 0:
            self._deregister_param_and_optimizer_hooks()
            self._mod_tracker.clear_user_hooks()
            self._mod_tracker.__exit__(*args)

    def __tensorplay_dispatch__(self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> Any:
        del types
        result = func(*args, **(kwargs or {}))
        for value in _walk_tensors(result):
            self._track(_MemRefType.ACT if not self._in_opt else _MemRefType.OPT, value)
        return result
