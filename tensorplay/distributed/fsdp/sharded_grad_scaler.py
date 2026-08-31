"""Gradient scaling with state tracking for sharded optimizers."""

from collections import defaultdict
from typing import Any, Iterable

from tensorplay.amp.grad_scaler import GradScaler, OptState

__all__ = ["ShardedGradScaler"]


def _refresh_per_optimizer_state() -> dict[str, Any]:
    return {"stage": OptState.READY, "found_inf_per_device": {}}


def _is_supported_device(tensor: Any) -> bool:
    return hasattr(tensor, "device")


class _GeneralMultiDeviceReplicator:
    def __init__(self, master_tensor: Any) -> None:
        self.master = master_tensor
        self._per_device_tensors: dict[Any, Any] = {}

    def get(self, device: Any) -> Any:
        if device not in self._per_device_tensors:
            self._per_device_tensors[device] = self.master.to(device=device)
        return self._per_device_tensors[device]


class ShardedGradScaler(GradScaler):
    def __init__(self, device: str = "cuda", init_scale: float = 2.0**16, backoff_factor: float = 0.5, growth_factor: float = 2.0, growth_interval: int = 2000, enabled: bool = True, process_group: Any = None) -> None:
        super().__init__(device=device, init_scale=init_scale, backoff_factor=backoff_factor, growth_factor=growth_factor, growth_interval=growth_interval, enabled=enabled)
        self.process_group = process_group
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    def scale(self, outputs: Any) -> Any:
        if not self.is_enabled():
            return outputs
        if hasattr(outputs, "device"):
            return super().scale(outputs)
        if isinstance(outputs, (list, tuple)):
            return type(outputs)(self.scale(value) for value in outputs)
        if isinstance(outputs, Iterable):
            return (self.scale(value) for value in outputs)
        raise TypeError("outputs must be a tensor or an iterable of tensors")
