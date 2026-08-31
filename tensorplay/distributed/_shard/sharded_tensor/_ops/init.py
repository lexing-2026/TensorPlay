"""Initialization operations for local shards."""

from typing import Any

__all__ = ["validate_param", "uniform_", "normal_", "kaiming_uniform_", "constant_", "register_tensor_creation_op"]


def validate_param(param: Any) -> Any:
    if not hasattr(param, "local_shards"):
        raise TypeError("expected a sharded tensor")
    return param


def _apply(param: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    validate_param(param)
    for shard in param.local_shards():
        getattr(shard.tensor, method)(*args, **kwargs)
    return param


def uniform_(param: Any, a: float = 0.0, b: float = 1.0) -> Any:
    return _apply(param, "uniform_", a, b)


def normal_(param: Any, mean: float = 0.0, std: float = 1.0) -> Any:
    return _apply(param, "normal_", mean, std)


def kaiming_uniform_(param: Any, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Any:
    import tensorplay.nn.init as init
    validate_param(param)
    for shard in param.local_shards():
        init.kaiming_uniform_(shard.tensor, a=a, mode=mode, nonlinearity=nonlinearity)
    return param


def constant_(param: Any, value: float) -> Any:
    return _apply(param, "fill_", value)


def register_tensor_creation_op(op: Any, fn: Any) -> Any:
    from ..api import _SHARDED_OPS
    _SHARDED_OPS[op] = fn
    return fn
