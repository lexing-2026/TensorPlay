"""Gradient scaling with state tracking for sharded optimizers."""

from collections import abc
from collections import defaultdict
from typing import Any, Iterable, overload

import tensorplay as tp
from tensorplay.amp.grad_scaler import GradScaler, OptState
from .. import distributed_core as dist

__all__ = ["ShardedGradScaler"]


def _refresh_per_optimizer_state() -> dict[str, Any]:
    return {"stage": OptState.READY, "found_inf_per_device": {}}


def _is_supported_device(tensor: Any) -> bool:
    return hasattr(tensor, "device") and getattr(tensor, "device", None) is not None


class _GeneralMultiDeviceReplicator:
    def __init__(self, master_tensor: Any) -> None:
        self.master = master_tensor
        self._per_device_tensors: dict[Any, Any] = {}

    def get(self, device: Any) -> Any:
        key = str(device)
        if key not in self._per_device_tensors:
            self._per_device_tensors[key] = self.master.to(
                device=device, non_blocking=True, copy=True
            )
        return self._per_device_tensors[key]


class ShardedGradScaler(GradScaler):
    def __init__(
        self,
        device: str = "cuda",
        init_scale: float = 2.0**16,
        backoff_factor: float = 0.5,
        growth_factor: float = 2.0,
        growth_interval: int = 2000,
        enabled: bool = True,
        process_group: Any = None,
    ) -> None:
        super().__init__(
            device=device,
            init_scale=init_scale,
            backoff_factor=backoff_factor,
            growth_factor=growth_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        if self._enabled:
            self.process_group = process_group
            self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    @overload
    def scale(self, outputs: tp.Tensor) -> tp.Tensor: ...

    @overload
    def scale(self, outputs: Iterable[tp.Tensor]) -> Iterable[tp.Tensor]: ...

    def scale(self, outputs: Any) -> Any:
        if not self._enabled:
            return outputs
        if isinstance(outputs, tp.Tensor):
            if not _is_supported_device(outputs):
                raise AssertionError(f"unsupported tensor device {outputs.device}")
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            if self._scale is None:
                raise AssertionError("scale was not initialized")
            return (outputs * self._scale.to(device=outputs.device)).to(outputs.dtype)
        if isinstance(outputs, abc.Iterable):
            stash: list[_GeneralMultiDeviceReplicator] = []

            def apply_scale(value: Any) -> Any:
                if isinstance(value, tp.Tensor):
                    if not _is_supported_device(value):
                        raise AssertionError(f"unsupported tensor device {value.device}")
                    if not stash:
                        if self._scale is None:
                            self._lazy_init_scale_growth_tracker(value.device)
                        if self._scale is None:
                            raise AssertionError("scale was not initialized")
                        stash.append(_GeneralMultiDeviceReplicator(self._scale))
                    return (value * stash[0].get(value.device)).to(value.dtype)
                if isinstance(value, abc.Iterable):
                    mapped = map(apply_scale, value)
                    return type(value)(mapped) if isinstance(value, (list, tuple)) else mapped
                raise ValueError("outputs must be a tensor or an iterable of tensors")

            return apply_scale(outputs)
        raise ValueError("outputs must be a tensor or an iterable of tensors")

    def _apply_scale(self, outputs: Any) -> Any:
        stash: list[_GeneralMultiDeviceReplicator] = []

        def apply_scale(value: Any) -> Any:
            if isinstance(value, tp.Tensor):
                if not stash:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(value.device)
                    if self._scale is None:
                        raise AssertionError("scale was not initialized")
                    stash.append(_GeneralMultiDeviceReplicator(self._scale))
                return (value * stash[0].get(value.device)).to(value.dtype)
            if isinstance(value, abc.Iterable):
                mapped = map(apply_scale, value)
                return type(value)(mapped) if isinstance(value, (list, tuple)) else mapped
            raise TypeError("outputs must contain tensors")

        return apply_scale(outputs)

    def _unscale_grads_(
        self, optimizer: Any, inv_scale: Any, found_inf: Any, allow_fp16: bool = True
    ) -> dict[Any, Any]:
        inv = _GeneralMultiDeviceReplicator(inv_scale)
        inf = _GeneralMultiDeviceReplicator(found_inf)
        grouped: dict[str, dict[Any, list[Any]]] = defaultdict(lambda: defaultdict(list))
        with tp.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    grad = getattr(param, "grad", None)
                    if grad is None:
                        continue
                    if not allow_fp16 and grad.dtype == tp.float16:
                        raise ValueError("Attempting to unscale FP16 gradients.")
                    if getattr(grad, "is_sparse", False):
                        raise NotImplementedError("sparse gradients are not supported")
                    grouped[str(grad.device)][grad.dtype].append(grad)
            for device_key, by_dtype in grouped.items():
                device = by_dtype[next(iter(by_dtype))][0].device
                for grads in by_dtype.values():
                    tp._amp_foreach_non_finite_check_and_unscale_(
                        grads, inf.get(device), inv.get(device)
                    )
        if not inf._per_device_tensors and self._scale is not None:
            inf.get(self._scale.device)
        return inf._per_device_tensors

    def unscale_(self, optimizer: Any) -> None:
        if not self._enabled:
            return
        self._check_scale_growth_tracker("unscale_")
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError("unscale_() has already been called on this optimizer")
        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step()")
        if self._scale is None:
            raise AssertionError("scale was not initialized")
        inv_scale = tp.full((), 1.0, dtype=tp.float64, device=self._scale.device)
        inv_scale = (inv_scale / self._scale.to(tp.float64)).to(tp.float32)
        found_inf = tp.full((), 0.0, dtype=tp.float32, device=self._scale.device)
        optimizer_state["found_inf_per_device"] = self._unscale_grads_(
            optimizer, inv_scale, found_inf, True
        )
        optimizer_state["stage"] = OptState.UNSCALED
        if dist.is_initialized():
            works = []
            for value in optimizer_state["found_inf_per_device"].values():
                works.append(dist.all_reduce(value, async_op=True, group=self.process_group))
            for work in works:
                if work is not None:
                    work.wait()

    def _amp_update_scale_cpu_(self, found_inf: Any) -> None:
        if self._scale is None or self._growth_tracker is None:
            raise AssertionError("scale and growth tracker must be initialized")
        if float(found_inf.item()) >= 1.0:
            self._scale *= self._backoff_factor
            self._growth_tracker.fill_(0)
            return
        successful = int(self._growth_tracker.item()) + 1
        if successful == self._growth_interval:
            self._scale *= self._growth_factor
            self._growth_tracker.fill_(0)
        else:
            self._growth_tracker.fill_(successful)

    def update(self, new_scale: Any = None) -> None:
        if not self._enabled:
            return
        scale, tracker = self._check_scale_growth_tracker("update")
        if new_scale is not None:
            if isinstance(new_scale, float):
                scale.fill_(new_scale)
            else:
                scale.copy_(new_scale)
        else:
            found = [
                value.to(device=scale.device)
                for state in self._per_optimizer_states.values()
                for value in state["found_inf_per_device"].values()
            ]
            if not found:
                raise AssertionError("No inf checks were recorded prior to update")
            combined = found[0]
            for value in found[1:]:
                combined = combined + value
            if str(getattr(scale.device, "type", scale.device)) == "cpu":
                self._amp_update_scale_cpu_(combined)
            else:
                tp._amp_update_scale_(
                    scale, tracker, combined, self._growth_factor,
                    self._backoff_factor, self._growth_interval
                )
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
