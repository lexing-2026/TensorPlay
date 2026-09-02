"""Registry and cache for eager placement propagation rules."""

from __future__ import annotations

import inspect
import threading
from functools import lru_cache
from typing import Any, Callable

from ._dtensor_spec import DTensorSpec, TensorMeta
from ._op_schema import (
    OpInfo,
    OpSchema,
    OpStrategy,
    OutputSharding,
    PlacementStrategy,
    TupleStrategy,
)

__all__ = [
    "LocalLRUCache",
    "ShardingPropagator",
    "_length",
    "_select_min_cost_strategy",
    "_select_min_redistribute_cost",
    "_validate_tensor_meta_count",
]


class _LocalLRUCache(threading.local):
    def __init__(self, function: Callable[..., Any]) -> None:
        self._function = function
        self._cache = lru_cache(maxsize=None)(function)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            hash(args)
            hash(tuple(kwargs.items()))
        except TypeError:
            return self._function(*args, **kwargs)
        return self._cache(*args, **kwargs)

    def cache_info(self) -> Any:
        return self._cache.cache_info()

    def cache_clear(self) -> None:
        self._cache.cache_clear()


LocalLRUCache = _LocalLRUCache


def _length(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (tuple, list)):
        return len(value)
    return 1


def _expected_tensor_outputs(op_schema: OpSchema) -> int | None:
    returns = op_schema._returns()
    if not returns:
        return 0
    first = getattr(returns[0], "type", returns[0])
    name = getattr(first, "__name__", type(first).__name__).lower()
    if "list" in name:
        return None
    return len(returns) if "tensor" in name else 0


def _validate_tensor_meta_count(
    op_schema: OpSchema, tensor_meta: TensorMeta | Any
) -> None:
    expected = _expected_tensor_outputs(op_schema)
    if expected is None:
        return
    actual = 0 if tensor_meta is None else 1 if isinstance(tensor_meta, TensorMeta) else _length(tensor_meta)
    if actual != expected:
        raise ValueError(
            f"tensor metadata count mismatch: expected {expected}, got {actual}"
        )


def _select_min_redistribute_cost(
    costs: list[float], strategies: list[PlacementStrategy]
) -> int:
    if not costs or len(costs) != len(strategies):
        raise ValueError("strategy and cost counts must match")
    return min(range(len(costs)), key=lambda index: (float(costs[index]), index))


def _select_min_cost_strategy(
    strategy: OpStrategy, op_schema: OpSchema | None = None
) -> PlacementStrategy:
    if not strategy.strategies:
        raise ValueError("strategy has no choices")
    selected = ShardingPropagator._select_strategy(strategy, op_schema)
    if not isinstance(selected, PlacementStrategy):
        raise ValueError("strategy selection did not return a placement strategy")
    return selected


class ShardingPropagator:
    """Resolve registered rules and retain results for hashable schemas."""

    def __init__(self) -> None:
        self.op_to_rules: dict[Any, Callable[..., Any]] = {}
        self.op_strategy_funcs: dict[Any, Callable[..., Any]] = {}
        self.op_single_dim_strategy_funcs: dict[Any, Any] = {}
        self.op_to_schema_info: dict[Any, Any] = {}
        self.op_to_schema_info_for_single_dim_strategy: dict[Any, Any] = {}
        self.op_to_shape_and_stride_idx: dict[Any, Any] = {}
        self._rules = self.op_to_rules
        self.propagate_op_sharding = _LocalLRUCache(
            self.propagate_op_sharding_non_cached
        )

    def register_sharding_prop_rule(
        self,
        operation: Any,
        rule: Callable[..., Any],
        schema_info: Any = None,
    ) -> Callable[..., Any]:
        self.op_to_rules[operation] = rule
        if schema_info is not None:
            self.op_to_schema_info[operation] = schema_info
        self.propagate_op_sharding.cache_clear()
        return rule

    def register_op_strategy(
        self,
        operation: Any,
        rule: Callable[..., Any],
        schema_info: Any = None,
    ) -> Callable[..., Any]:
        self.op_strategy_funcs[operation] = rule
        if schema_info is not None:
            self.op_to_schema_info[operation] = schema_info
        self.propagate_op_sharding.cache_clear()
        return rule

    def register_single_dim_op_strategy(
        self,
        operation: Any,
        strategy_info: Any,
        schema_info: Any = None,
    ) -> Any:
        self.op_single_dim_strategy_funcs[operation] = strategy_info
        if schema_info is not None:
            self.op_to_schema_info_for_single_dim_strategy[operation] = schema_info
        self.propagate_op_sharding.cache_clear()
        return strategy_info

    @staticmethod
    def _global_rule(operation: Any) -> tuple[str, Callable[..., Any] | None]:
        from ._ops.utils import (
            _PROPAGATION_RULES,
            _STRATEGY_RULES,
            _lookup_builtin_rule,
        )

        rule = _PROPAGATION_RULES.get(operation)
        if rule is not None:
            return "prop", rule
        rule = _STRATEGY_RULES.get(operation)
        if rule is not None:
            return "strategy", rule
        return _lookup_builtin_rule(operation)

    def _find_mesh(self, schema: OpSchema) -> Any:
        for value in schema.args_spec:
            return value.device_mesh
        for value in schema.args_strategy:
            return value.strategies[0].output_spec.device_mesh
        for value in schema.kwargs_schema.values():
            if isinstance(value, DTensorSpec):
                return value.device_mesh
        return None

    @staticmethod
    def _tensor_meta(value: Any) -> TensorMeta | None:
        if not hasattr(value, "shape") or not hasattr(value, "stride"):
            return None
        stride = value.stride() if callable(value.stride) else value.stride
        return TensorMeta(tuple(value.shape), tuple(stride), value.dtype)

    def _propagate_tensor_meta(self, schema: OpSchema) -> Any:
        operation = schema.op
        if not callable(operation):
            return None
        try:
            result = operation(*schema.gen_fake_args(), **schema.gen_fake_kwargs())
        except Exception:
            return None
        if isinstance(result, (tuple, list)):
            values = [self._tensor_meta(item) for item in result]
            return tuple(values) if isinstance(result, tuple) else values
        return self._tensor_meta(result)

    @classmethod
    def _attach_tensor_meta(cls, value: Any, meta: Any) -> Any:
        if isinstance(value, DTensorSpec):
            return value if meta is None else value.shallow_copy_with_tensor_meta(meta)
        if isinstance(value, (tuple, list)) and isinstance(meta, (tuple, list)):
            values = [
                cls._attach_tensor_meta(item, item_meta)
                for item, item_meta in zip(value, meta)
            ]
            return tuple(values) if isinstance(value, tuple) else values
        return value

    @classmethod
    def _attach_result_meta(cls, result: Any, meta: Any) -> Any:
        if isinstance(result, OutputSharding):
            result.output_spec = cls._attach_tensor_meta(result.output_spec, meta)
            return result
        return cls._attach_tensor_meta(result, meta)

    @staticmethod
    def _to_strategy(value: Any) -> Any:
        if isinstance(value, DTensorSpec):
            return OpStrategy([PlacementStrategy(value)])
        if isinstance(value, tuple):
            converted = tuple(ShardingPropagator._to_strategy(item) for item in value)
            if converted and all(
                isinstance(item, (OpStrategy, TupleStrategy)) for item in converted
            ):
                return TupleStrategy(converted)
            return converted
        if isinstance(value, list):
            converted = [ShardingPropagator._to_strategy(item) for item in value]
            if converted and all(
                isinstance(item, (OpStrategy, TupleStrategy)) for item in converted
            ):
                return TupleStrategy(converted)
            return converted
        if isinstance(value, dict):
            return {
                key: ShardingPropagator._to_strategy(item)
                for key, item in value.items()
            }
        return value

    @classmethod
    def _strategy_schema(cls, schema: OpSchema) -> OpSchema:
        return OpSchema(
            op=schema.op,
            args_schema=tuple(cls._to_strategy(item) for item in schema.args),
            kwargs_schema={
                key: cls._to_strategy(item)
                for key, item in schema.kwargs.items()
            },
            schema_info=schema.schema_info,
        )

    @staticmethod
    def _call_strategy(rule: Callable[..., Any], schema: OpSchema, mesh: Any) -> Any:
        try:
            signature = inspect.signature(rule)
            positional = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            ]
            has_varargs = any(
                parameter.kind == parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            positional = []
            has_varargs = True

        names = [parameter.name for parameter in positional]
        if len(positional) >= 3 and names[1:3] == ["args_schema", "kwargs_schema"]:
            return rule(schema.op, schema.args, schema.kwargs)
        if names and (
            names[0] in {"mesh", "device_mesh", "mesh_arg"}
            or (len(names) == 2 and names[1] in {"schema", "op_schema"})
        ):
            return rule(mesh, ShardingPropagator._strategy_schema(schema))
        if names and names[0] in {"schema", "op_schema"}:
            return rule(schema, *schema.args[1:], **schema.kwargs)
        if has_varargs or positional:
            return rule(*schema.args, **schema.kwargs)
        return rule()

    @staticmethod
    def _select_strategy(strategy: OpStrategy, schema: OpSchema | None = None) -> Any:
        if not strategy.strategies:
            return None
        if len(strategy.strategies) == 1:
            return strategy.strategies[0]

        best_index = 0
        best_key: tuple[float, int, int] | None = None
        current_specs = () if schema is None else schema.args_spec
        for index, candidate in enumerate(strategy.strategies):
            cost = candidate.redistribute_cost
            explicit_cost = (
                float("inf")
                if cost is None
                else sum(sum(float(value) for value in row) for row in cost)
            )
            mismatches = 0
            if candidate.input_specs is not None:
                for current, desired in zip(current_specs, candidate.input_specs):
                    if current.placements != desired.placements:
                        mismatches += 1
            key = (
                0.0 if mismatches == 0 else explicit_cost,
                mismatches,
                index,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
        return strategy.strategies[best_index]

    @classmethod
    def _strategy_output(cls, result: Any) -> Any:
        if isinstance(result, OpStrategy):
            selected = cls._select_strategy(result)
            return OutputSharding(
                None if selected is None else selected.output_specs
            )
        if isinstance(result, TupleStrategy):
            outputs = []
            for child in result.childs:
                if not isinstance(child, OpStrategy):
                    return result
                selected = cls._select_strategy(child)
                outputs.append(None if selected is None else selected.output_specs)
            return OutputSharding(tuple(outputs))
        return result

    @staticmethod
    def _strategy_suggestion(
        schema: OpSchema,
        selected: PlacementStrategy,
    ) -> tuple[OpSchema | None, bool]:
        if selected.input_specs is None:
            return None, False
        expected_iter = iter(selected.input_specs)
        needs_redistribute = False

        def replace(value: Any) -> Any:
            nonlocal needs_redistribute
            if isinstance(value, DTensorSpec):
                try:
                    desired = next(expected_iter)
                except StopIteration as error:
                    raise ValueError("strategy input count is smaller than schema input count") from error
                expected = desired
                if value.tensor_meta is not None:
                    expected = desired.shallow_copy_with_tensor_meta(value.tensor_meta)
                if value.placements != desired.placements:
                    needs_redistribute = True
                return expected
            if isinstance(value, tuple):
                return tuple(replace(item) for item in value)
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, dict):
                return {key: replace(item) for key, item in value.items()}
            return value

        expected_args = tuple(replace(value) for value in schema.args_schema)
        expected_kwargs = {
            key: replace(value) for key, value in schema.kwargs_schema.items()
        }
        try:
            next(expected_iter)
        except StopIteration:
            pass
        else:
            raise ValueError("strategy input count is larger than schema input count")
        if not needs_redistribute:
            return None, False
        suggestion = OpSchema(
            schema.op,
            expected_args,
            expected_kwargs,
            schema_info=schema.schema_info,
        )
        return suggestion, True

    def _propagate_schema(self, schema: OpSchema) -> Any:
        operation = schema.op
        single_dim_info = self.op_single_dim_strategy_funcs.get(operation)
        if (
            schema.schema_info is None
            and operation in self.op_to_schema_info
        ):
            schema = OpSchema(
                op=schema.op,
                args_schema=schema.args_schema,
                kwargs_schema=schema.kwargs_schema,
                schema_info=self.op_to_schema_info[operation],
            )
        if (
            schema.schema_info is None
            and operation in self.op_to_schema_info_for_single_dim_strategy
        ):
            schema = OpSchema(
                op=schema.op,
                args_schema=schema.args_schema,
                kwargs_schema=schema.kwargs_schema,
                schema_info=self.op_to_schema_info_for_single_dim_strategy[operation],
            )
        rule = self.op_to_rules.get(operation)
        kind = "prop"
        if rule is None:
            rule = self.op_strategy_funcs.get(operation)
            kind = "strategy"
        if rule is None and single_dim_info is None:
            kind, rule = self._global_rule(operation)
        if rule is None and single_dim_info is None:
            return None
        output_meta = self._propagate_tensor_meta(schema)
        if single_dim_info is not None:
            from ._ops.single_dim_strategy import _expand_single_dim_strategy_to_mesh

            strategy_schema = self._strategy_schema(schema)
            mesh = self._find_mesh(schema)
            if mesh is None:
                mesh = schema.get_mesh_from_args(validate=False)
            expanded = _expand_single_dim_strategy_to_mesh(
                mesh, strategy_schema, single_dim_info, output_meta
            )
            result = expanded(
                schema.op, strategy_schema.args_meta, strategy_schema.kwargs_meta
            )
            kind = "strategy"
        if kind == "strategy":
            if single_dim_info is None:
                result = self._call_strategy(rule, schema, self._find_mesh(schema))
            if isinstance(result, DTensorSpec):
                return OutputSharding(self._attach_tensor_meta(result, output_meta))
            if isinstance(result, OpStrategy):
                if not result.strategies:
                    return OutputSharding(None, failed_reason="strategy has no choices")
                selected = self._select_strategy(result, schema)
                if not isinstance(selected, PlacementStrategy):
                    return OutputSharding(None, failed_reason="invalid strategy result")
                suggestion, needs_redistribute = self._strategy_suggestion(
                    schema, selected
                )
                return OutputSharding(
                    self._attach_tensor_meta(selected.output_specs, output_meta),
                    suggestion,
                    needs_redistribute=needs_redistribute,
                )
            return self._attach_result_meta(self._strategy_output(result), output_meta)
        result = rule(schema)
        if isinstance(result, OutputSharding):
            if result.output_spec is None and result.redistribute_schema is not None:
                retry = rule(result.redistribute_schema)
                if isinstance(retry, OutputSharding):
                    result.output_spec = retry.output_spec
                result.needs_redistribute = True
            result.output_spec = self._attach_tensor_meta(
                result.output_spec, output_meta
            )
            return result
        if isinstance(result, DTensorSpec):
            return OutputSharding(self._attach_tensor_meta(result, output_meta))
        return self._attach_result_meta(result, output_meta)

    def propagate_op_sharding_non_cached(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(operation, OpSchema):
            return self._propagate_schema(operation)
        rule = self.op_to_rules.get(operation)
        if rule is not None:
            return rule(*args, **kwargs)
        strategy = self.op_strategy_funcs.get(operation)
        if strategy is not None:
            return strategy(*args, **kwargs)
        kind, rule = self._global_rule(operation)
        if rule is None:
            return None
        if kind == "strategy" and not args and not kwargs:
            return rule()
        return rule(*args, **kwargs)

    def propagate(self, op_info: OpInfo) -> None:
        if op_info.schema is None:
            raise ValueError("OpInfo.schema is required")
        if op_info.schema.has_symints:
            op_info.output_sharding = self.propagate_op_sharding_non_cached(
                op_info.schema
            )
        else:
            op_info.output_sharding = self.propagate_op_sharding(op_info.schema)

    def clear(self) -> None:
        self.op_to_rules.clear()
        self.op_strategy_funcs.clear()
        self.op_single_dim_strategy_funcs.clear()
        self.op_to_schema_info.clear()
        self.op_to_schema_info_for_single_dim_strategy.clear()
        self.op_to_shape_and_stride_idx.clear()
        self.propagate_op_sharding.cache_clear()
