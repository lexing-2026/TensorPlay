"""Register custom distributed placement strategies."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from .._api import DTensor
from .._op_schema import (
    OpSchema,
    OpStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from .._ops.utils import expand_to_full_mesh_op_strategy
from ..placement_types import Placement

__all__ = ["register_sharding"]


def _strategy_to_spec(strategy: Any) -> Any:
    if isinstance(strategy, OpStrategy):
        if not strategy.strategies:
            raise ValueError("a placement strategy must contain at least one choice")
        return strategy.strategies[0].output_spec
    if isinstance(strategy, TupleStrategy):
        return tuple(_strategy_to_spec(child) for child in strategy.children)
    return strategy


def _return_count(operation: Any) -> int:
    schema = getattr(operation, "_schema", None)
    returns = getattr(schema, "returns", None)
    if returns is None:
        return 1
    return len(tuple(returns))


def _schema_info(operation: Any) -> RuntimeSchemaInfo:
    schema = getattr(operation, "_schema", None)
    arguments = getattr(schema, "arguments", None)
    if arguments is not None:
        static_argnum = 100
        static_kwargs: list[str] = []
        for index, argument in enumerate(arguments):
            argument_type = getattr(argument, "type", None)
            type_name = type(argument_type).__name__
            is_integer = type_name in {"IntType", "OptionalType"} or (
                getattr(argument_type, "name", None) in {"int", "SymInt"}
            )
            if is_integer:
                static_argnum = min(static_argnum, index)
                if getattr(argument, "kwarg_only", False):
                    static_kwargs.append(str(argument.name))
        return RuntimeSchemaInfo(
            static_argnum,
            static_kwargs or None,
            needs_pytree=True,
        )

    static_argnum = 100
    static_kwargs: list[str] = []
    try:
        parameters = tuple(inspect.signature(operation).parameters.values())
    except (TypeError, ValueError):
        parameters = ()
    for index, parameter in enumerate(parameters):
        annotation = parameter.annotation
        if annotation is int or getattr(annotation, "__name__", None) in {
            "SymInt",
            "OptionalInt",
        }:
            static_argnum = min(static_argnum, index)
            if parameter.kind is parameter.KEYWORD_ONLY:
                static_kwargs.append(parameter.name)
    return RuntimeSchemaInfo(
        static_argnum,
        static_kwargs or None,
        needs_pytree=True,
    )


def _custom_strategy(
    custom_sharding_fn: Callable[..., Sequence[tuple[Any, Any]]],
    mesh: Any,
    op_schema: OpSchema,
) -> StrategyType:
    args_schema = tuple(_strategy_to_spec(value) for value in op_schema.args_schema)
    kwargs_schema = {
        key: _strategy_to_spec(value)
        for key, value in op_schema.kwargs_schema.items()
    }
    acceptable_shardings = custom_sharding_fn(*args_schema, **kwargs_schema)
    single_mesh_dim_strategies: list[list[Placement | Any | None]] = []
    for output_specs, input_specs in acceptable_shardings:
        single_mesh_dim_strategies.append(
            list(output_specs) + list(input_specs)
        )
    return expand_to_full_mesh_op_strategy(
        mesh,
        op_schema,
        single_mesh_dim_strategies,
        input_index=_return_count(op_schema.op),
        inplace_op=op_schema.is_inplace_op(),
    )


def register_sharding(
    operation: Any | Sequence[Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that installs a placement strategy for operations."""
    operations = (
        tuple(operation) if isinstance(operation, (list, tuple)) else (operation,)
    )

    def decorator(
        custom_sharding_fn: Callable[..., Sequence[tuple[Any, Any]]],
    ) -> Callable[..., Any]:
        for item in operations:
            DTensor._op_dispatcher.sharding_propagator.register_op_strategy(
                item,
                partial(_custom_strategy, custom_sharding_fn),
                _schema_info(item),
            )
        return custom_sharding_fn

    return decorator
