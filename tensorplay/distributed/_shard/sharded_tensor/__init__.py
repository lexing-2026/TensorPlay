"""Public explicit sharded tensor constructors."""

import functools
from typing import Any

import tensorplay as tp

from .api import (
    Shard,
    ShardedTensor,
    ShardedTensorBase,
    ShardedTensorMetadata,
    TensorProperties,
    _CUSTOM_SHARDED_OPS,
    _SHARDED_OPS,
)
from .metadata import ShardMetadata

__all__ = [
    "Shard",
    "ShardMetadata",
    "ShardedTensor",
    "ShardedTensorBase",
    "ShardedTensorMetadata",
    "TensorProperties",
    "empty",
    "ones",
    "zeros",
    "full",
    "rand",
    "randn",
    "init_from_local_shards",
    "state_dict_hook",
    "pre_load_state_dict_hook",
    "custom_sharded_op_impl",
]


def _make(spec: Any, size: Any, operation: Any, fill_value: Any = None, **kwargs: Any) -> ShardedTensor:
    shape = tuple(size) if isinstance(size, (list, tuple)) else (size,)
    supported = {"dtype", "device", "pin_memory", "requires_grad"}
    operation_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    if fill_value is None:
        global_value = operation(*shape, **operation_kwargs)
    elif operation is tp.full:
        global_value = operation(shape, fill_value, **operation_kwargs)
    else:
        global_value = operation(*shape, fill_value=fill_value, **operation_kwargs)
    return ShardedTensor._init_from_global_tensor(spec, global_value, kwargs.get("process_group"))


def empty(sharding_spec: Any, *size: Any, **kwargs: Any) -> ShardedTensor:
    return _make(sharding_spec, size, tp.empty, **kwargs)


def ones(sharding_spec: Any, *size: Any, **kwargs: Any) -> ShardedTensor:
    return _make(sharding_spec, size, tp.ones, **kwargs)


def zeros(sharding_spec: Any, *size: Any, **kwargs: Any) -> ShardedTensor:
    return _make(sharding_spec, size, tp.zeros, **kwargs)


def full(sharding_spec: Any, size: Any, fill_value: Any, **kwargs: Any) -> ShardedTensor:
    return _make(sharding_spec, size, tp.full, fill_value=fill_value, **kwargs)


def rand(sharding_spec: Any, *size: Any, **kwargs: Any) -> ShardedTensor:
    return _make(sharding_spec, size, tp.rand, **kwargs)


def randn(sharding_spec: Any, *size: Any, **kwargs: Any) -> ShardedTensor:
    return _make(sharding_spec, size, tp.randn, **kwargs)


def init_from_local_shards(local_shards: list[Shard], sharded_tensor_metadata: ShardedTensorMetadata, process_group: Any = None) -> ShardedTensor:
    return ShardedTensor._init_from_local_shards_and_global_metadata(local_shards, sharded_tensor_metadata, process_group=process_group)


def state_dict_hook(module: Any, destination: dict[str, Any], prefix: str, local_metadata: dict[str, Any]) -> None:
    del local_metadata
    for name, value in module.__dict__.items():
        if isinstance(value, ShardedTensor):
            destination[prefix + name] = value


def pre_load_state_dict_hook(module: Any, state_dict: dict[str, Any], prefix: str, local_metadata: dict[str, Any], strict: bool, missing_keys: list[str], unexpected_keys: list[str], error_msgs: list[str]) -> None:
    del module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs


def custom_sharded_op_impl(op: Any):
    def decorator(fn: Any) -> Any:
        _CUSTOM_SHARDED_OPS[op] = fn
        return fn
    return decorator


def _sharded_op_impl(op: Any, types: Any, args: Any, kwargs: Any) -> Any:
    fn = _CUSTOM_SHARDED_OPS.get(op) or _SHARDED_OPS.get(op)
    if fn is None:
        raise RuntimeError("sharded operation is not registered")
    return fn(types, args, kwargs)
