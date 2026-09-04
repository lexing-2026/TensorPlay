from __future__ import annotations

import copy
from typing import Any

from .. import distributed_core as dist
from .._shard.metadata import ShardMetadata
from .._shard.sharded_tensor.api import Shard, ShardedTensor
from ..remote_device import _remote_device
from ._traverse import set_element, traverse_state_dict
from .utils import _element_wise_add, _normalize_device_info

__all__ = ["_flatten_sharded_tensors"]


def _flatten_sharded_tensors(state_dict: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def rewrite_dict(path: tuple[Any, ...], value: Any) -> None:
        if not isinstance(value, ShardedTensor):
            set_element(result, path, value)
            return
        shards = value.local_shards()
        if len(shards) == 0:
            return
        if len(shards) != 1:
            set_element(result, path, value)
            return
        outer_shard = shards[0]
        inner_tensor = outer_shard.tensor
        if not isinstance(inner_tensor, ShardedTensor):
            set_element(result, path, value)
            return
        inner_shards = inner_tensor.local_shards()
        if len(inner_shards) != 1:
            raise ValueError("cannot handle an inner tensor with multiple shards")
        inner_shard = inner_shards[0]
        local_shard = Shard(
            tensor=inner_shard.tensor,
            metadata=ShardMetadata(
                shard_offsets=_element_wise_add(
                    outer_shard.metadata.shard_offsets,
                    inner_shard.metadata.shard_offsets,
                ),
                shard_sizes=inner_shard.metadata.shard_sizes,
                placement=f"rank:{dist.get_rank()}/{inner_shard.tensor.device}",
            ),
        )
        tensor_metadata = copy.deepcopy(value.metadata())
        other_rank = 0 if dist.get_rank() > 0 else 1
        device_info = _normalize_device_info(
            inner_shard.tensor.device.type, inner_shard.tensor.device.index or 0
        )
        for index, shard_metadata in enumerate(tensor_metadata.shards_metadata):
            if shard_metadata.shard_offsets == outer_shard.metadata.shard_offsets:
                tensor_metadata.shards_metadata.pop(index)
                break
        for shard_metadata in tensor_metadata.shards_metadata:
            object.__setattr__(
                shard_metadata,
                "placement",
                _remote_device(f"rank:{other_rank}/{device_info}"),
            )
        for inner_metadata in inner_tensor.metadata().shards_metadata:
            if inner_metadata.shard_offsets == inner_shard.metadata.shard_offsets:
                continue
            tensor_metadata.shards_metadata.append(
                ShardMetadata(
                    shard_offsets=_element_wise_add(
                        outer_shard.metadata.shard_offsets,
                        inner_metadata.shard_offsets,
                    ),
                    shard_sizes=inner_metadata.shard_sizes,
                    placement=f"rank:{other_rank}/{device_info}",
                )
            )
        tensor_metadata.shards_metadata.append(local_shard.metadata)
        set_element(
            result,
            path,
            ShardedTensor._init_from_local_shards_and_global_metadata(
                local_shards=[local_shard],
                sharded_tensor_metadata=tensor_metadata,
            ),
        )

    traverse_state_dict(state_dict, rewrite_dict)
    return result
