from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import constants as rpc_constants

DeviceType = int | str | Any


@dataclass(frozen=True, order=True)
class _Device:
    type: str
    index: int | None = None

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"

    def __repr__(self) -> str:
        return f"device(type='{self.type}', index={self.index})"


def _parse_device(device: DeviceType) -> _Device:
    if isinstance(device, _Device):
        return device
    if isinstance(device, int):
        return _Device("cuda", int(device))
    value = str(device)
    fields = value.split(":", 1)
    if len(fields) == 1:
        return _Device(fields[0])
    if not fields[0] or not fields[1].isdigit():
        raise ValueError(f"invalid device specification: {device!r}")
    return _Device(fields[0], int(fields[1]))


def _to_device(device: DeviceType) -> _Device:
    parsed = _parse_device(device)
    if parsed.type != "cuda":
        raise ValueError(
            f"set_devices expects CUDA devices, but got device type {parsed.type}"
        )
    return parsed


def _to_device_map(device_map: dict[DeviceType, DeviceType]) -> dict[_Device, _Device]:
    result: dict[_Device, _Device] = {}
    reverse: dict[_Device, _Device] = {}
    for source, target in device_map.items():
        source_device = _parse_device(source)
        target_device = _parse_device(target)
        if target_device in reverse:
            raise ValueError(
                f"device_map only supports one-to-one mapping to {target_device}"
            )
        result[source_device] = target_device
        reverse[target_device] = source_device
    return result


def _to_device_list(devices: list[DeviceType]) -> list[_Device]:
    return [_to_device(device) for device in devices]


class TensorPipeRpcBackendOptions:
    def __init__(
        self,
        *,
        num_worker_threads: int = rpc_constants.DEFAULT_NUM_WORKER_THREADS,
        rpc_timeout: float = rpc_constants.DEFAULT_RPC_TIMEOUT_SEC,
        init_method: str = rpc_constants.DEFAULT_INIT_METHOD,
        device_maps: dict[str, dict[DeviceType, DeviceType]] | None = None,
        devices: list[DeviceType] | None = None,
        _transports: list[Any] | None = None,
        _channels: list[Any] | None = None,
    ) -> None:
        if int(num_worker_threads) <= 0:
            raise ValueError("num_worker_threads must be positive")
        if float(rpc_timeout) < 0:
            raise ValueError("rpc_timeout must be non-negative")
        self.num_worker_threads = int(num_worker_threads)
        self.rpc_timeout = float(rpc_timeout)
        self.init_method = str(init_method)
        self.device_maps = {
            str(worker): _to_device_map(mapping)
            for worker, mapping in (device_maps or {}).items()
        }
        self.devices = _to_device_list(devices or [])
        self._transports = None if _transports is None else list(_transports)
        self._channels = None if _channels is None else list(_channels)

    def set_device_map(self, to: str, device_map: dict[DeviceType, DeviceType]) -> None:
        normalized = _to_device_map(device_map)
        current = self.device_maps.setdefault(str(to), {})
        for source, target in normalized.items():
            old_target = current.get(source)
            if old_target is not None and old_target != target:
                raise ValueError(f"source device {source} already has a different target")
            if target in current.values() and current.get(source) != target:
                raise ValueError(f"target device {target} is already mapped")
            current[source] = target

    def set_devices(self, devices: list[DeviceType]) -> None:
        self.devices = _to_device_list(devices)


__all__ = ["TensorPipeRpcBackendOptions"]
