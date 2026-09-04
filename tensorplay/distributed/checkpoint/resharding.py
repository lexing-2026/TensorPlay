from __future__ import annotations

from .metadata import ChunkStorageMetadata

__all__: list[str] = []


def _check_shard_metadata_pair_overlap(
    shard1: ChunkStorageMetadata, shard2: ChunkStorageMetadata
) -> bool:
    for dimension in range(len(shard1.offsets)):
        if shard1.offsets[dimension] >= shard2.offsets[dimension] + shard2.sizes[dimension]:
            return False
        if shard2.offsets[dimension] >= shard1.offsets[dimension] + shard1.sizes[dimension]:
            return False
    return True


def _shards_get_overlap_region_wrt_saved_tensor(
    saved_shard: ChunkStorageMetadata, current_shard: ChunkStorageMetadata
) -> list[tuple[int, int, int, int]]:
    narrows = []
    for dimension, (
        saved_offset,
        current_offset,
        saved_size,
        current_size,
    ) in enumerate(
        zip(
            saved_shard.offsets,
            current_shard.offsets,
            saved_shard.sizes,
            current_shard.sizes,
        )
    ):
        end = min(saved_offset + saved_size, current_offset + current_size)
        length = end - max(current_offset, saved_offset)
        if saved_offset > current_offset:
            saved_relative = 0
            current_relative = saved_offset - current_offset
        else:
            saved_relative = current_offset - saved_offset
            current_relative = 0
        narrows.append((dimension, saved_relative, current_relative, length))
    return narrows
