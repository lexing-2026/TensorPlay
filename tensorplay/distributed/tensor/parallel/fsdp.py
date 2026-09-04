"""State flattening helpers for combining tensor and parameter sharding."""

from __future__ import annotations

from typing import Any

from .._api import DTensor
from .._dtensor_spec import DTensorSpec
from .._utils import compute_local_shape_and_global_offset
from ..placement_types import Replicate, Shard
from ._data_parallel_utils import _flatten_tensor, _unflatten_tensor
from ...fsdp._shard_utils import _all_gather_dtensor, _create_chunk_dtensor, _create_chunk_sharded_tensor
from ..._shard.metadata import ShardMetadata
from ..._shard.sharded_tensor.shard import Shard as LocalShard

__all__ = ["DTensorExtensions"]


class DTensorExtensions:
    def __init__(self, device_handle: Any = None) -> None:
        self.device_handle = device_handle
        self.compute_stream = None

    def pre_flatten_transform(self, tensor: Any) -> tuple[Any, DTensorSpec | None]:
        return _flatten_tensor(tensor)

    def post_unflatten_transform(self, tensor: Any, param_extension: DTensorSpec) -> Any:
        stream = self.compute_stream
        current_stream = getattr(self.device_handle, "current_stream", None)
        if stream is None and callable(current_stream):
            stream = current_stream()
        stream_context = getattr(self.device_handle, "stream", None)
        if stream is not None and callable(stream_context):
            with stream_context(stream):
                return _unflatten_tensor(
                    tensor,
                    param_extension,
                    device_handle=self.device_handle,
                    compute_stream=self.compute_stream,
                )
        return _unflatten_tensor(
            tensor,
            param_extension,
            device_handle=self.device_handle,
            compute_stream=self.compute_stream,
        )

    def chunk_tensor(self, tensor: Any, rank: int, world_size: int, num_devices_per_node: int, pg: Any, device: Any = None) -> Any:
        return _create_chunk_sharded_tensor(
            tensor, rank, world_size, num_devices_per_node, pg, device
        )

    def chunk_dtensor(self, tensor: DTensor, rank: int, device_mesh: Any) -> Any:
        return _create_chunk_dtensor(tensor, rank, device_mesh)

    def pre_load_state_dict_transform(self, tensor: Any) -> tuple[Any, list[LocalShard]]:
        if not isinstance(tensor, DTensor):
            return tensor, []
        local_shape, offset = compute_local_shape_and_global_offset(tensor.shape, tensor.device_mesh, tensor.placements)
        metadata = ShardMetadata(
            list(offset), list(local_shape), f"rank:{tensor.device_mesh.get_rank()}/{tensor.device}"
        )
        return tensor.to_local(), [LocalShard(tensor.to_local(), metadata)]

    def all_gather_dtensor(self, tensor: DTensor, parent_mesh: Any = None) -> Any:
        return _all_gather_dtensor(tensor, parent_mesh)
