"""Tensor-like properties for explicit sharded values."""

from typing import Any

__all__ = ["tensor_device", "st_is_meta", "sharded_type_as_check", "same_dtype", "sharded_type_as", "sharded_deepcopy", "sharded_inplace_copy", "sharded_clone", "sharded_detach", "tensor_requires_grad_set"]


def tensor_device(value: Any) -> Any:
    return value.device


def st_is_meta(value: Any) -> bool:
    return getattr(getattr(value, "device", None), "type", None) == "meta"


def sharded_type_as_check(value: Any, other: Any) -> None:
    if not hasattr(value, "to_local") or not hasattr(other, "dtype"):
        raise TypeError("value cannot be cast as requested")


def same_dtype(value: Any, dtype: Any) -> bool:
    return value.dtype == dtype


def sharded_type_as(value: Any, dtype: Any) -> Any:
    for shard in value.local_shards():
        shard.tensor.data = shard.tensor.to(dtype=dtype)
    return value


def sharded_deepcopy(value: Any, memo: dict[int, Any] | None = None) -> Any:
    del memo
    return value.clone()


def sharded_inplace_copy(value: Any, other: Any) -> Any:
    for target, source in zip(value.local_shards(), other.local_shards()):
        target.tensor.copy_(source.tensor)
    return value


def sharded_clone(value: Any) -> Any:
    return value.clone()


def sharded_detach(value: Any) -> Any:
    return value.detach()


def tensor_requires_grad_set(value: Any, requires_grad: bool) -> Any:
    return value.requires_grad_(requires_grad)
