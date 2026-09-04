"""Placement strategies for random operations."""

from __future__ import annotations

from typing import Any

from .._api import DTensor
from .._dtensor_spec import TensorMeta
from .._op_schema import OpSchema, OpSpec, OpStrategy, StrategyType
from .utils import is_tensor_partial
from .single_dim_strategy import _ShardingPlaceholder, register_single_dim_strategy

__all__ = [
    "_random_inplace_single_dim_strategy",
    "multinomial_single_dim_strategy",
    "random_op_strategy",
    "register_random_ops",
]


def _operation_name(operation: Any) -> str:
    return str(getattr(operation, "__name__", operation)).rsplit(".", 1)[-1]


def _random_inplace_single_dim_strategy(
    operation: Any,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Any]]:
    del kwargs_schema
    self_meta = args_schema[0]
    if not isinstance(self_meta, TensorMeta):
        raise AssertionError("random operation requires tensor metadata")
    num_outputs = 2 if _operation_name(operation) == "native_dropout" else 1
    return [
        [_ShardingPlaceholder(dim)] * (num_outputs + 1)
        for dim in range(len(self_meta.shape))
    ]


def multinomial_single_dim_strategy(
    operation: Any,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Any]]:
    del operation, kwargs_schema
    self_meta = args_schema[0]
    if not isinstance(self_meta, TensorMeta):
        raise AssertionError("multinomial requires tensor metadata")
    return [
        [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)]
        for dim in range(len(self_meta.shape) - 1)
    ]


def random_op_strategy(mesh: Any, op_schema: OpSchema) -> StrategyType:
    del mesh
    self_strategy = op_schema.args_schema[0]
    assert isinstance(self_strategy, OpStrategy)

    random_strategy = OpStrategy([])
    for arg_strategy in self_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if is_tensor_partial(arg_spec):
            raise RuntimeError(f"{op_schema.op} with Partial is not supported yet")
        random_strategy.strategies.append(OpSpec(output_specs=arg_spec))
    return random_strategy


_RANDOM_OPS_READY = False


def register_random_ops() -> None:
    global _RANDOM_OPS_READY
    if _RANDOM_OPS_READY:
        return
    _RANDOM_OPS_READY = True
    for name in (
        "normal_",
        "uniform_",
        "native_dropout",
        "bernoulli_",
        "bernoulli",
        "log_normal_",
        "exponential_",
        "geometric_",
    ):
        DTensor._op_dispatcher.sharding_propagator.register_op_strategy(
            name, random_op_strategy
        )
        register_single_dim_strategy(
            name,
            allow_uneven_sharding=True,
        )(_random_inplace_single_dim_strategy)
    register_single_dim_strategy("multinomial")(
        multinomial_single_dim_strategy
    )
