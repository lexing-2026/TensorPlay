"""State flattening helpers for combining tensor and parameter sharding."""

from __future__ import annotations

from typing import Any

from .._api import DTensor
from .._dtensor_spec import DTensorSpec
from .._shards_wrapper import ShardMetadata, ShardsWrapper
from .._utils import compute_local_shape_and_global_offset
from ..placement_types import Shard
from ._data_parallel_utils import _flatten_tensor, _unflatten_tensor

__all__ = ["DTensorExtensions"]


class DTensorExtensions:
    def __init__(self, device_handle: Any = None) -> None:
        self.device_handle = device_handle
        self.compute_stream = None

    def pre_flatten_transform(self, tensor: Any) -> tuple[Any, DTensorSpec | None]:
        local, spec = _flatten_tensor(tensor)
        return local, None if spec is None else DTensorSpec(spec.device_mesh, spec.placements, None)

    def post_unflatten_transform(self, tensor: Any, param_extension: DTensorSpec) -> Any:
        return _unflatten_tensor(tensor, param_extension)

    def chunk_tensor(self, tensor: Any, rank: int, world_size: int, num_devices_per_node: int, pg: Any, device: Any = None) -> Any:
        del num_devices_per_node, pg, device
        return tensor.chunk(world_size, dim=0)[rank]

    def chunk_dtensor(self, tensor: DTensor, rank: int, device_mesh: Any) -> Any:
        del rank
        return tensor.redistribute(device_mesh=device_mesh, placements=[Shard(0)]).to_local()

    def pre_load_state_dict_transform(self, tensor: Any) -> tuple[Any, list[ShardMetadata]]:
        if not isinstance(tensor, DTensor):
            return tensor, []
        local_shape, offset = compute_local_shape_and_global_offset(tensor.shape, tensor.device_mesh, tensor.placements)
        return tensor.to_local(), [ShardMetadata(tensor.to_local(), offset, local_shape)]

    def all_gather_dtensor(self, tensor: DTensor, parent_mesh: Any = None) -> Any:
        del parent_mesh
        return tensor.full_tensor()
