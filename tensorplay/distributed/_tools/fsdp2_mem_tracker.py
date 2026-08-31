from __future__ import annotations

from enum import Enum, auto
from typing import Any, NamedTuple

import tensorplay as tp

from .mem_tracker import _MemRefType, _ModMemStats, _ModState, _RefType, _State, MemTracker

_TOTAL_KEY = "Total"
__all__ = ["FSDPMemTracker"]


class _FSDPRefType(_RefType):
    SHARDED_PARAM = "Sharded Param"
    UNSHARDED_PARAM = "Unsharded Param"
    BUFFER = "Buffer"
    SHARDED_GRAD = "Sharded Grad"
    UNSHARDED_GRAD = "Unsharded Grad"
    ACT = "Activation"
    TEMP = "Temp"
    ALL_GATHER = "All Gather"
    REDUCE_SCATTER = "Reduce Scatter"
    OPT = "OptState"
    INP = "Inputs"


class _SavedFSDPMethods(NamedTuple):
    pre_backward: Any
    post_backward: Any


class _FSDPModState(_State):
    BEF_PRE_FW = "Before Pre-Forward"
    AFT_PRE_FW = "After Pre-Forward"
    BEF_POST_FW = "Before Post-Forward"
    AFT_POST_FW = "After Post-Forward"
    BEF_PRE_BW = "Before Pre-Backward"
    AFT_PRE_BW = "After Pre-Backward"
    BEF_POST_BW = "Before Post-Backward"
    AFT_POST_BW = "After Post-Backward"
    PRE_FW_AC = "Pre-Forward AC"
    POST_FW_AC = "Post-Forward AC"
    PEAK_FW = "Peak Forward"
    PEAK_BW = "Peak Backward"


class _FSDPModMemStats:
    def __init__(self, mod_fqn: str) -> None:
        self.mod_fqn = mod_fqn
        self.local_peak: dict[Any, int] = {}
        self.snapshots: dict[_FSDPModState, list[dict[Any, dict[str, int]]]] = {}


class _FSDPState(Enum):
    PRE_FW = auto()
    FW = auto()
    POST_FW = auto()
    PRE_BW = auto()
    BW = auto()
    POST_BW = auto()


class FSDPMemTracker(MemTracker):
    """Track memory phases for modules that expose sharded parameter state."""

    def __init__(self, mod: Any, optm: Any | None = None) -> None:
        super().__init__()
        if not hasattr(mod, "named_modules"):
            raise TypeError("mod must be a module")
        self._root_mod = mod
        self._optm = optm
        self._fsdp_mod_to_saved_methods: dict[Any, _SavedFSDPMethods] = {}
        self._fsdp_state = _FSDPState.PRE_FW
        self._ref_class = _FSDPRefType

    def _instrument_fsdp_sharded_params_grads(self, fsdp_param_group: Any) -> None:
        for parameter in getattr(fsdp_param_group, "fsdp_params", ()):
            sharded = getattr(parameter, "sharded_param", None)
            if isinstance(sharded, tp.Tensor):
                self._update_and_maybe_create_winfos(sharded, _FSDPRefType.SHARDED_PARAM)
            grad = getattr(sharded, "grad", None)
            if isinstance(grad, tp.Tensor):
                self._update_and_maybe_create_winfos(grad, _FSDPRefType.SHARDED_GRAD)

    def _fsdp_state_pre_forward(self, fsdp_mod: Any, orig_fsdp_state_pre_fw: Any):
        def inner(*args: Any, **kwargs: Any):
            self._fsdp_state = _FSDPState.PRE_FW
            result = orig_fsdp_state_pre_fw(*args, **kwargs)
            name = self._mod_tracker.get_known_fqn(fsdp_mod) or type(fsdp_mod).__name__
            stats = self.memory_tracking.setdefault(fsdp_mod, _FSDPModMemStats(name))
            stats.snapshots.setdefault(_FSDPModState.BEF_PRE_FW, []).append(self.get_tracker_snapshot())
            stats.snapshots.setdefault(_FSDPModState.AFT_PRE_FW, []).append(self.get_tracker_snapshot())
            self._fsdp_state = _FSDPState.FW
            return result
        return inner

    def _fsdp_state_post_forward(self, fsdp_mod: Any, orig_fsdp_state_post_fw: Any):
        def inner(*args: Any, **kwargs: Any):
            self._fsdp_state = _FSDPState.POST_FW
            name = self._mod_tracker.get_known_fqn(fsdp_mod) or type(fsdp_mod).__name__
            stats = self.memory_tracking.setdefault(fsdp_mod, _FSDPModMemStats(name))
            stats.snapshots.setdefault(_FSDPModState.BEF_POST_FW, []).append(self.get_tracker_snapshot())
            result = orig_fsdp_state_post_fw(*args, **kwargs)
            stats.snapshots.setdefault(_FSDPModState.AFT_POST_FW, []).append(self.get_tracker_snapshot())
            return result
        return inner

    def _fsdp_param_group_pre_backward(self, fsdp_param_group: Any, orig_pre_backward: Any):
        def inner(*args: Any, **kwargs: Any):
            self._fsdp_state = _FSDPState.PRE_BW
            self._instrument_fsdp_sharded_params_grads(fsdp_param_group)
            return orig_pre_backward(*args, **kwargs)
        return inner

    def _fsdp_param_group_post_backward(self, fsdp_param_group: Any, orig_post_backward: Any):
        def inner(*args: Any, **kwargs: Any):
            self._fsdp_state = _FSDPState.POST_BW
            result = orig_post_backward(*args, **kwargs)
            self._instrument_fsdp_sharded_params_grads(fsdp_param_group)
            return result
        return inner

    def _instrument_fsdp_module(self) -> None:
        for module in self._root_mod.modules():
            if hasattr(module, "_get_fsdp_state"):
                state = module._get_fsdp_state()
                group = getattr(state, "_fsdp_param_group", None)
                if group is not None:
                    self._instrument_fsdp_sharded_params_grads(group)

    def _instrument_optimizer(self) -> None:
        if self._optm is not None:
            self._track_optimizer_states(_FSDPRefType.OPT, self._optm)

    def _register_module_and_optimizer_hooks(self) -> None:
        self._instrument_fsdp_module()
        self._instrument_optimizer()

    def _deregister_module_and_optimizer_hooks(self) -> None:
        self._fsdp_mod_to_saved_methods.clear()

    def track_inputs(self, inputs: tuple[Any, ...]) -> None:
        for value in inputs:
            if isinstance(value, tp.Tensor):
                self._track(_FSDPRefType.INP, value)

    def track_external(self, *external: Any) -> None:
        super().track_external(*external)

    def __enter__(self) -> "FSDPMemTracker":
        self._register_module_and_optimizer_hooks()
        super().__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        super().__exit__(*args)
        self._deregister_module_and_optimizer_hooks()

    def __torch_dispatch__(self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> Any:
        result = super().__torch_dispatch__(func, types, args, kwargs)
        return result
