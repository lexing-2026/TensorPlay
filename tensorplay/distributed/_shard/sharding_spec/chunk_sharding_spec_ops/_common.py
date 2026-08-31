"""Operation helpers for chunk sharding specifications."""

from typing import Any

__all__ = ["_chunk_sharding_spec_check", "_register_sharded_op_on_local_tensor", "_handle_col_wise_sharding_base", "_result_distribute_with_col_rearrange", "_handle_max_norm_col_wise", "_all_gather_base_input", "_handle_row_wise_mask"]


def _chunk_sharding_spec_check(spec: Any, tensor: Any = None) -> None:
    if not hasattr(spec, "placements"):
        raise TypeError("expected a chunk sharding specification")
    if tensor is not None and spec.dim >= tensor.dim():
        raise ValueError("sharding dimension is outside tensor rank")


def _register_sharded_op_on_local_tensor(op: Any, fn: Any) -> Any:
    from ..api import _CUSTOM_SHARDING_SPEC_OPS
    _CUSTOM_SHARDING_SPEC_OPS.setdefault("ChunkShardingSpec", {})[op] = fn
    return fn


def _handle_col_wise_sharding_base(*args: Any, **kwargs: Any) -> Any:
    del kwargs
    return args[0] if args else None


def _result_distribute_with_col_rearrange(result: Any, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return result


def _handle_max_norm_col_wise(*args: Any, **kwargs: Any) -> Any:
    del kwargs
    return args[0] if args else None


def _all_gather_base_input(value: Any, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return value.gather() if hasattr(value, "gather") else value


def _handle_row_wise_mask(value: Any, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return value
