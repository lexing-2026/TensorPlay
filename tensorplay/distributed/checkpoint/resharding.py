from __future__ import annotations

from .metadata import ChunkStorageMetadata

__all__: list[str] = []


def _check_shard_metadata_pair_overlap(shard1: ChunkStorageMetadata, shard2: ChunkStorageMetadata) -> bool:
    return all(a < b + bs and b < a + as_ for a, as_, b, bs in zip(shard1.offsets, shard1.sizes, shard2.offsets, shard2.sizes))


def _shards_get_overlap_region_wrt_saved_tensor(saved_shard: ChunkStorageMetadata, current_shard: ChunkStorageMetadata) -> list[tuple[int, int, int, int]]:
    result = []
    for dim, (saved_offset, current_offset, saved_size, current_size) in enumerate(zip(saved_shard.offsets, current_shard.offsets, saved_shard.sizes, current_shard.sizes)):
        end = min(saved_offset + saved_size, current_offset + current_size)
        length = max(0, end - max(saved_offset, current_offset))
        result.append((dim, max(0, current_offset - saved_offset), max(0, saved_offset - current_offset), length))
    return result
