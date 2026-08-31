"""Temporary attention placement rules for context-parallel execution."""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

from ..._ops.utils import register_op_strategy
from ...placement_types import Replicate, Shard

__all__ = ["register_cp_sharding_rules", "unregister_cp_sharding_rules"]

SEQ_DIM = 2
_registered: dict[Any, Any] = {}


def _cp_sharding() -> Shard:
    return Shard(SEQ_DIM)


@contextlib.contextmanager
def _single_dim_strategy_context(operation: Any, strategy: Any) -> Iterator[None]:
    old = _registered.get(operation)
    _registered[operation] = strategy
    try:
        yield
    finally:
        if old is None:
            _registered.pop(operation, None)
        else:
            _registered[operation] = old


def register_cp_sharding_rules(operations: Any = ()) -> None:
    for operation in operations:
        register_op_strategy(operation)(_cp_sharding)
        _registered[operation] = _cp_sharding


def unregister_cp_sharding_rules(clear_the_cache: bool = False) -> None:
    del clear_the_cache
    _registered.clear()
