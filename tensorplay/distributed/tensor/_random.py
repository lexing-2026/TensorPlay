"""Random-state tracking for distributed tensor execution."""

from __future__ import annotations

import contextlib
import math
import warnings
from typing import Any

import tensorplay
import tensorplay.distributed as dist

from ..device_mesh import DeviceMesh, _get_device_handle
from ._dtensor_spec import DTensorSpec
from .placement_types import Shard

__all__ = ["is_rng_supported_mesh", "manual_seed", "OffsetBasedRNGTracker"]

_rng_tracker: Any = None


def is_rng_supported_mesh(device_mesh: DeviceMesh) -> bool:
    device_handle = _get_device_handle(device_mesh.device_type)
    if device_handle and hasattr(device_handle, "set_rng_state"):
        return True
    warnings.warn(
        f"distributed random operations may be incomplete on {device_mesh.device_type}"
    )
    return False


def manual_seed(seed: int, device_mesh: DeviceMesh) -> None:
    device_handle = _get_device_handle(device_mesh.device_type)
    if not device_handle:
        raise NotImplementedError(
            "distributed random state requires a device handle with RNG support"
        )

    global _rng_tracker
    if not _rng_tracker:
        _rng_tracker = OffsetBasedRNGTracker(
            device_mesh.device_type, run_state_sync=False
        )

    if device_mesh.get_coordinate() is not None:
        _rng_tracker._manual_seed(seed)
    else:
        raise RuntimeError(
            "manual_seed requires the current rank to belong to the device mesh"
        )


class _RNGStateTracker:
    def __init__(self, device_type: str = "cuda") -> None:
        self._device_type = device_type
        self._device_handle = _get_device_handle(device_type)
        if not (self._device_handle and self._device_handle.is_available()):
            raise RuntimeError(
                f"{self.__class__.__name__} requires an available RNG device"
            )
        self._states: dict[str, Any] = {}
        self._devices = [self._device_handle.current_device()]
        self._use_distribute_region = True

    @property
    def rng_states(self) -> dict[str, Any]:
        return self._states

    @property
    def distribute_region_enabled(self) -> bool:
        return self._use_distribute_region

    @distribute_region_enabled.setter
    def distribute_region_enabled(self, value: bool) -> None:
        self._use_distribute_region = bool(value)

    def rng_state_is_sync(self, name: str) -> bool:
        return name in self.rng_states

    def get_seed(self, name: str) -> int:
        if name not in self.rng_states:
            raise RuntimeError(
                f"{self.__class__.__name__} does not have random state {name}"
            )
        seed_tensor = self.rng_states[name][0:8].view(tensorplay.int64)
        return int(seed_tensor.item())

    def set_seed(self, name: str, seed: int) -> None:
        seed_tensor = tensorplay.tensor(
            [seed], dtype=tensorplay.uint64, device="cpu"
        ).view(tensorplay.uint8)
        offset_tensor = tensorplay.tensor(
            [0], dtype=tensorplay.uint64, device="cpu"
        ).view(tensorplay.uint8)
        self.rng_states[name] = tensorplay.cat(
            [seed_tensor, offset_tensor], dim=0
        )

    def _distribute_region(self, spec: DTensorSpec):
        del spec
        raise NotImplementedError

    def _manual_seed(self, parallel_seed: int) -> None:
        del parallel_seed
        raise NotImplementedError


class OffsetBasedRNGTracker(_RNGStateTracker):
    def __init__(self, device_type: str = "cuda", run_state_sync: bool = True):
        super().__init__(device_type)
        rng_state = self._device_handle.get_rng_state().to(device_type)
        if run_state_sync:
            dist.broadcast(rng_state, 0)
        self.rng_states["parallel-rng"] = rng_state.to("cpu")

    def _manual_seed(self, parallel_seed: int) -> None:
        self.set_seed("parallel-rng", parallel_seed)

    @contextlib.contextmanager
    def _fork_rng(self):
        cpu_state = tensorplay.get_rng_state()
        device_state = self._device_handle.get_rng_state()
        try:
            yield
        finally:
            self._device_handle.set_rng_state(device_state)
            tensorplay.set_rng_state(cpu_state)

    @contextlib.contextmanager
    def _distribute_region(self, spec: DTensorSpec):
        if not self.rng_state_is_sync("parallel-rng"):
            raise RuntimeError(
                "random state must be synchronized before entering a distributed region"
            )

        if not self.distribute_region_enabled:
            yield
            return

        old_offset = self.get_offset("parallel-rng")
        self._set_pre_op_offset(spec)
        with self._fork_rng():
            self._device_handle.set_rng_state(self.rng_states["parallel-rng"])
            try:
                yield
            finally:
                self._set_post_op_offset(spec, old_offset)

    def get_offset(self, name: str) -> int:
        if name not in self.rng_states:
            raise RuntimeError(
                f"{self.__class__.__name__} does not have random state {name}"
            )
        offset_tensor = self.rng_states[name][8:].view(tensorplay.int64)
        return int(offset_tensor.item())

    def set_offset(self, name: str, offset: int) -> None:
        if name not in self.rng_states:
            raise RuntimeError(
                f"{self.__class__.__name__} does not have random state {name}"
            )
        seed_tensor = self.rng_states[name][0:8]
        offset_tensor = tensorplay.tensor(
            [offset], dtype=tensorplay.uint64, device="cpu"
        ).view(tensorplay.uint8)
        self.rng_states[name] = tensorplay.cat(
            [seed_tensor, offset_tensor], dim=0
        )

    def _set_pre_op_offset(self, spec: DTensorSpec) -> None:
        dtensor_shape = spec.shape
        mesh = spec.mesh
        dim_map: list[int | list[int]] = [-1] * spec.ndim
        for mesh_dim, placement in enumerate(spec.placements):
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                if dim_map[shard_dim] == -1:
                    dim_map[shard_dim] = [mesh_dim]
                else:
                    mesh_dim_list = dim_map[shard_dim]
                    assert isinstance(mesh_dim_list, list)
                    mesh_dim_list.append(mesh_dim)

        mesh_coordinate = mesh.get_coordinate()
        assert mesh_coordinate is not None
        mesh_size = mesh.shape
        shard_idx_by_dim = []
        total_num_shards_by_dim = []
        for mesh_dims in dim_map:
            shard_idx = 0
            total_num_shards = 1
            if isinstance(mesh_dims, list):
                rank_coord = [mesh_coordinate[index] for index in mesh_dims]
                num_shards = [mesh_size[index] for index in mesh_dims]
                for index, size in zip(rank_coord, num_shards):
                    shard_idx = shard_idx * size + index
                    total_num_shards *= size
            shard_idx_by_dim.append(shard_idx)
            total_num_shards_by_dim.append(total_num_shards)

        shard_linear_idx = self._calc_shard_linear_idx(
            shard_idx_by_dim, total_num_shards_by_dim
        )

        local_size_on_rank_0 = list(dtensor_shape)
        for mesh_dim, placement in enumerate(spec.placements):
            if isinstance(placement, Shard):
                mesh_dim_size = mesh.size(mesh_dim)
                shard_dim = placement.dim
                local_size_on_rank_0[shard_dim] = placement.local_shard_size_and_offset(
                    dtensor_shape[shard_dim], mesh_dim_size, 0
                )[0]

        local_size = math.prod(local_size_on_rank_0)
        current_offset = self.get_offset("parallel-rng")
        offset_incr = (shard_linear_idx * local_size + 3) // 4 * 4
        self.set_offset("parallel-rng", current_offset + offset_incr)

    def _set_post_op_offset(self, spec: DTensorSpec, old_offset: int) -> None:
        numel = math.prod(spec.shape)
        numel = (numel + 3) // 4 * 4
        self.set_offset("parallel-rng", old_offset + numel)

    def _calc_shard_linear_idx(
        self, shard_coord: list[int], shard_size: list[int]
    ) -> int:
        shard_linear_idx = 0
        shard_coord_stride = 1
        for index, size in zip(reversed(shard_coord), reversed(shard_size)):
            shard_linear_idx += index * shard_coord_stride
            shard_coord_stride *= size
        return shard_linear_idx
