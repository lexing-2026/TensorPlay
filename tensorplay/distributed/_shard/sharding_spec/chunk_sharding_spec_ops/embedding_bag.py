"""Embedding-bag operations for chunk-sharded weights."""

from typing import Any

__all__ = ["sharded_embedding_bag", "_validate_embedding_bag_param", "_handle_col_wise_sharding", "_handle_row_wise_sharding", "_all_gather_embedding_bag_input"]


def _validate_embedding_bag_param(weight: Any) -> None:
    if not hasattr(weight, "local_shards"):
        raise TypeError("embedding-bag weight must be sharded")


def sharded_embedding_bag(input: Any, weight: Any, *args: Any, **kwargs: Any) -> Any:
    _validate_embedding_bag_param(weight)
    import tensorplay
    return tensorplay.nn.functional.embedding_bag(input, weight.gather(), *args, **kwargs)


def _handle_col_wise_sharding(*args: Any, **kwargs: Any) -> Any:
    return sharded_embedding_bag(*args, **kwargs)


def _handle_row_wise_sharding(*args: Any, **kwargs: Any) -> Any:
    return sharded_embedding_bag(*args, **kwargs)


def _all_gather_embedding_bag_input(value: Any, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return value.gather() if hasattr(value, "gather") else value
