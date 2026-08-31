"""Embedding operations for chunk-sharded weights."""

from typing import Any

from ..chunk_sharding_spec import ChunkShardingSpec
from ._common import _chunk_sharding_spec_check

__all__ = ["sharded_embedding", "_validate_embedding_param", "_handle_col_wise_sharding", "_handle_row_wise_sharding"]


def _validate_embedding_param(weight: Any) -> None:
    if not hasattr(weight, "local_shards"):
        raise TypeError("embedding weight must be sharded")


def sharded_embedding(input: Any, weight: Any, *args: Any, **kwargs: Any) -> Any:
    _validate_embedding_param(weight)
    import tensorplay
    return tensorplay.nn.functional.embedding(input, weight.gather(), *args, **kwargs)


def _handle_col_wise_sharding(*args: Any, **kwargs: Any) -> Any:
    return sharded_embedding(*args, **kwargs)


def _handle_row_wise_sharding(*args: Any, **kwargs: Any) -> Any:
    return sharded_embedding(*args, **kwargs)
