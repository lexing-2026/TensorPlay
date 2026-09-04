"""Registry and cache for eager placement propagation rules."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Callable

from ._dtensor_spec import DTensorSpec, TensorMeta
from ._utils import compute_local_shape_and_global_offset, compute_local_stride
from ._op_schema import (
    OpInfo,
    OpSchema,
    OpStrategy,
    OutputSharding,
    PlacementStrategy,
    TupleStrategy,
)
from .placement_types import Replicate, _StridedShard, _is_shard_like

__all__ = [
    "LocalLRUCache",
    "ShardingPropagator",
    "_format_unbacked_hinting_log",
    "_get_expected_num_tensor_outputs",
    "_length",
    "_propagate_use_strided_shard_flag",
    "_select_min_cost_strategy",
    "_select_min_redistribute_cost",
    "_validate_tensor_meta_count",
]


class LocalLRUCache(threading.local):
    def __init__(self, function: Callable[..., Any]) -> None:
        self._function = function
        self.cache = lru_cache(maxsize=None)(function)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            hash(args)
            hash(tuple(kwargs.items()))
        except TypeError:
            return self._function(*args, **kwargs)
        return self.cache(*args, **kwargs)

    def cache_info(self) -> Any:
        return self.cache.cache_info()

    def cache_clear(self) -> None:
        self.cache.cache_clear()


_LocalLRUCache = LocalLRUCache


def _length(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, Sequence):
        return len(value)
    return 1


def _get_expected_num_tensor_outputs(operation: Any) -> int | None:
    if isinstance(operation, OpSchema):
        returns = operation._returns()
    else:
        operation_schema = getattr(operation, "_schema", None)
        if operation_schema is None:
            return None
        returns = tuple(getattr(operation_schema, "returns", ()) or ())
    if not returns:
        return 0
    first = getattr(returns[0], "type", returns[0])
    name = getattr(first, "__name__", type(first).__name__).lower()
    if "list" in name:
        return None
    return len(returns) if "tensor" in name else 0


def _expected_tensor_outputs(op_schema: OpSchema) -> int | None:
    return _get_expected_num_tensor_outputs(op_schema)


def _validate_tensor_meta_count(
    op_schema: OpSchema, tensor_meta: TensorMeta | Any
) -> None:
    expected = _get_expected_num_tensor_outputs(op_schema.op)
    if tensor_meta is None:
        actual = 0
    elif isinstance(tensor_meta, TensorMeta):
        actual = 1
    else:
        actual = len(tensor_meta)
    if expected is None:
        if getattr(op_schema.op, "_schema", None) is None and not op_schema._returns():
            return
        if not isinstance(tensor_meta, list):
            raise AssertionError(
                f"tensor metadata for {op_schema.op} must be a list, "
                f"got {type(tensor_meta).__name__}"
            )
        return
    if actual != expected:
        raise AssertionError(
            f"tensor metadata count mismatch for {op_schema.op}: "
            f"expected {expected}, got {actual}"
        )


def _spec_leaves(value: Any) -> list[DTensorSpec]:
    if isinstance(value, DTensorSpec):
        return [value]
    if isinstance(value, dict):
        result: list[DTensorSpec] = []
        for child in value.values():
            result.extend(_spec_leaves(child))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for child in value:
            result.extend(_spec_leaves(child))
        return result
    return []


def _propagate_use_strided_shard_flag(
    op_strategy: OpStrategy,
    op_schema: OpSchema,
) -> None:
    use_strided: bool | None = None
    for spec in _spec_leaves(op_schema.args_schema):
        if not any(isinstance(placement, _StridedShard) for placement in spec.placements):
            continue
        value = bool(getattr(spec, "use_strided_shard_as_shard_order", True))
        use_strided = value if use_strided is None else use_strided or value
    if use_strided is None:
        return

    def rewrite(value: Any) -> Any:
        if not isinstance(value, DTensorSpec):
            if isinstance(value, tuple):
                return tuple(rewrite(item) for item in value)
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            return value
        if not any(isinstance(placement, _StridedShard) for placement in value.placements):
            return value
        current = bool(getattr(value, "use_strided_shard_as_shard_order", True))
        if current == use_strided:
            return value
        shard_order = (
            None
            if use_strided
            else DTensorSpec.compute_default_shard_order(value.placements)
        )
        return DTensorSpec(
            value.mesh,
            value.placements,
            tensor_meta=value.tensor_meta,
            shard_order=shard_order,
            use_strided_shard_as_shard_order=use_strided,
        )

    for strategy in op_strategy.strategies:
        strategy.output_specs = rewrite(strategy.output_specs)
        if strategy.input_specs is not None:
            strategy.input_specs = type(strategy.input_specs)(
                rewrite(spec) for spec in strategy.input_specs
            )


def _format_spec(value: Any) -> Any:
    if isinstance(value, DTensorSpec):
        return str(value)
    if isinstance(value, tuple):
        return tuple(_format_spec(item) for item in value)
    if isinstance(value, list):
        return [_format_spec(item) for item in value]
    return value


def _format_unbacked_hinting_log(
    op_schema: OpSchema,
    strategies: list[PlacementStrategy],
    strategy_index: int,
    replacements: dict[Any, Any],
) -> str:
    strategy = strategies[strategy_index]
    inputs = None if strategy.input_specs is None else tuple(
        _format_spec(spec) for spec in strategy.input_specs
    )
    outputs = _format_spec(strategy.output_specs)
    return (
        f"selected strategy {inputs} -> {outputs} for {op_schema.op} "
        f"with input {tuple(str(spec) for spec in op_schema.args_schema)}, "
        f"using symbolic hints: {replacements}"
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
    if len(strategy.strategies) == 1:
        return strategy.strategies[0]
    costs: list[float] = []
    negative_index = -1
    no_redistribute_index = -1
    zero_cost_index = -1
    current_specs = () if op_schema is None else op_schema.args_spec
    for index, candidate in enumerate(strategy.strategies):
        if candidate.redistribute_cost is None:
            raise AssertionError("strategy costs are required")
        cost = sum(
            float(value)
            for row in candidate.redistribute_cost
            for value in row
        )
        costs.append(cost)
        if cost < 0:
            if negative_index == -1 or cost < costs[negative_index]:
                negative_index = index
            continue
        if cost != 0 or op_schema is None:
            continue
        mismatched = False
        for input_index, current in enumerate(current_specs):
            desired = (
                candidate.output_spec
                if candidate.input_specs is None
                else candidate.input_specs[input_index]
            )
            if current.placements != desired.placements:
                mismatched = True
                break
        if not mismatched:
            no_redistribute_index = index
        elif zero_cost_index == -1:
            zero_cost_index = index
    if negative_index != -1:
        selected_index = negative_index
    elif no_redistribute_index != -1:
        selected_index = no_redistribute_index
    elif zero_cost_index != -1:
        selected_index = zero_cost_index
    else:
        selected_index = _select_min_redistribute_cost(costs, strategy.strategies)
    return strategy.strategies[selected_index]


class ShardingPropagator:
    """Resolve registered rules and retain results for hashable schemas."""

    def __init__(self) -> None:
        self.op_to_rules: dict[Any, Callable[..., Any]] = {}
        self.op_strategy_funcs: dict[Any, Callable[..., Any]] = {}
        self.op_single_dim_strategy_funcs: dict[Any, Any] = {}
        self.op_to_schema_info: dict[Any, Any] = {}
        self.op_to_schema_info_for_single_dim_strategy: dict[Any, Any] = {}
        self.op_to_shape_and_stride_idx: dict[Any, int | tuple[int, int]] = {
            "new_empty": 1,
            "new_full": 1,
            "new_ones": 1,
            "new_zeros": 1,
            "new_empty_strided": (1, 2),
            "expand": 1,
            "expand_copy": 1,
            "reshape": 1,
            "view": 1,
            "view_copy": 1,
            "_unsafe_view": 1,
            "select_backward": 1,
            "slice_backward": 1,
            "upsample_nearest1d_backward": 2,
            "upsample_nearest2d_backward": 2,
            "upsample_nearest3d_backward": 2,
            "_upsample_nearest_exact1d_backward": 2,
            "_upsample_nearest_exact2d_backward": 2,
            "_upsample_nearest_exact3d_backward": 2,
            "_upsample_bilinear2d_aa_backward": 2,
            "_upsample_bicubic2d_aa_backward": 2,
            "_upsample_lanczos2d_aa_backward": 2,
            "upsample_bicubic2d_backward": 2,
            "upsample_bilinear2d_backward": 2,
            "upsample_linear1d_backward": 2,
            "upsample_trilinear3d_backward": 2,
        }
        self.op_to_scalar_shape_adjuster: dict[Any, Callable[..., OpSchema]] = {}
        self.squeeze_op_to_dims_variant: dict[Any, Any] = {
            "squeeze": "squeeze",
            "squeeze_": "squeeze_",
            "squeeze_dims": "squeeze_dims",
            "squeeze_dims_": "squeeze_dims_",
        }
        from ._decompositions import DecompShardingStrategy

        self.decomp_strategy = DecompShardingStrategy(self)
        self._rules = self.op_to_rules
        self.propagate_op_sharding = LocalLRUCache(
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

    @staticmethod
    def _operation_name(operation: Any) -> str:
        if isinstance(operation, str):
            value = operation
        else:
            value = getattr(
                operation,
                "__name__",
                getattr(operation, "name", type(operation).__name__),
            )
        value = str(value).rsplit(".", 1)[-1]
        for suffix in ("_default", "_out", "_functional"):
            if value.endswith(suffix):
                value = value[: -len(suffix)]
        return value

    @classmethod
    def _operation_value(cls, mapping: dict[Any, Any], operation: Any) -> Any:
        value = mapping.get(operation)
        if value is not None:
            return value
        return mapping.get(cls._operation_name(operation))

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

    def _propagate_tensor_meta_non_cached(self, schema: OpSchema) -> Any:
        operation = schema.op
        if self._operation_name(operation) == "equal":
            return None
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

    @lru_cache(maxsize=None)
    def _propagate_tensor_meta_cached(self, schema: OpSchema) -> Any:
        return self._propagate_tensor_meta_non_cached(schema)

    def _propagate_tensor_meta(self, schema: OpSchema) -> Any:
        if schema.has_symints:
            return self._propagate_tensor_meta_non_cached(schema)
        try:
            return self._propagate_tensor_meta_cached(schema)
        except TypeError:
            return self._propagate_tensor_meta_non_cached(schema)

    @classmethod
    def _create_output_spec_with_new_tensor_meta(
        cls, operation: Any, output_specs: Any, output_tensor_meta: Any
    ) -> Any:
        if output_tensor_meta is None:
            return output_specs
        operation_name = str(
            getattr(operation, "name", getattr(operation, "__name__", operation))
        )
        if isinstance(output_specs, DTensorSpec):
            if not isinstance(output_tensor_meta, TensorMeta):
                raise ValueError(
                    f"output spec for {operation_name} has one result but metadata "
                    f"contains {_length(output_tensor_meta)} results"
                )
            return output_specs.shallow_copy_with_tensor_meta(output_tensor_meta)
        if isinstance(output_specs, (tuple, list)):
            if not isinstance(output_tensor_meta, (tuple, list)):
                raise ValueError(
                    f"output specs for {operation_name} require per-result metadata"
                )
            if len(output_specs) != len(output_tensor_meta):
                raise ValueError(
                    f"output spec count for {operation_name} is {len(output_specs)}, "
                    f"metadata count is {len(output_tensor_meta)}"
                )
            values: list[Any] = []
            for index, spec in enumerate(output_specs):
                meta = output_tensor_meta[index]
                if isinstance(spec, DTensorSpec):
                    if meta is None:
                        values.append(None)
                    elif isinstance(meta, TensorMeta):
                        values.append(spec.shallow_copy_with_tensor_meta(meta))
                    else:
                        raise ValueError(
                            f"output {index} for {operation_name} has invalid metadata"
                        )
                else:
                    values.append(spec)
            return tuple(values) if isinstance(output_specs, tuple) else values
        return output_specs

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
    def _to_strategy(value: Any, plain_mesh: Any = None) -> Any:
        if isinstance(value, TensorMeta):
            if plain_mesh is None:
                return value
            ndim = getattr(plain_mesh, "ndim")
            ndim = int(ndim() if callable(ndim) else ndim)
            return OpStrategy(
                [
                    PlacementStrategy(
                        DTensorSpec(
                            plain_mesh,
                            (Replicate(),) * ndim,
                            tensor_meta=value,
                        )
                    )
                ]
            )
        if isinstance(value, DTensorSpec):
            return OpStrategy([PlacementStrategy(value)])
        if isinstance(value, tuple):
            converted = tuple(
                ShardingPropagator._to_strategy(item, plain_mesh) for item in value
            )
            if converted and all(
                isinstance(item, (OpStrategy, TupleStrategy)) for item in converted
            ):
                return TupleStrategy(converted)
            return converted
        if isinstance(value, list):
            converted = [
                ShardingPropagator._to_strategy(item, plain_mesh) for item in value
            ]
            if converted and all(
                isinstance(item, (OpStrategy, TupleStrategy)) for item in converted
            ):
                return TupleStrategy(converted)
            return converted
        if isinstance(value, dict):
            return {
                key: ShardingPropagator._to_strategy(item, plain_mesh)
                for key, item in value.items()
            }
        return value

    @classmethod
    def _wrap_with_op_strategy(cls, schema: OpSchema) -> OpSchema:
        auxiliary_names = {
            "gather",
            "index",
            "index_add",
            "index_copy",
            "index_fill",
            "index_put",
            "index_reduce",
            "index_select",
            "scatter",
            "scatter_add",
            "take",
            "take_along_dim",
        }
        plain_mesh = None

        def find_mesh(value: Any) -> Any:
            if isinstance(value, DTensorSpec):
                return value.mesh
            if isinstance(value, OpStrategy) and value.strategies:
                return value.mesh
            if isinstance(value, (tuple, list)):
                for item in value:
                    mesh = find_mesh(item)
                    if mesh is not None:
                        return mesh
            if isinstance(value, dict):
                for item in value.values():
                    mesh = find_mesh(item)
                    if mesh is not None:
                        return mesh
            return None

        if cls._operation_name(schema.op) in auxiliary_names:
            plain_mesh = find_mesh(schema.args_schema)
            if plain_mesh is None:
                plain_mesh = find_mesh(schema.kwargs_schema)
        return OpSchema(
            op=schema.op,
            args_schema=tuple(
                cls._to_strategy(item, plain_mesh) for item in schema.args
            ),
            kwargs_schema={
                key: cls._to_strategy(item, plain_mesh)
                for key, item in schema.kwargs.items()
            },
            schema_info=schema.schema_info,
        )

    @classmethod
    def _strategy_schema(cls, schema: OpSchema) -> OpSchema:
        return cls._wrap_with_op_strategy(schema)

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
            if len(positional) == 1 and not has_varargs:
                return rule(schema)
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
            selected = _select_min_cost_strategy(result)
            return OutputSharding(
                None if selected is None else selected.output_specs
            )
        if isinstance(result, TupleStrategy):
            outputs = []
            for child in result.childs:
                if not isinstance(child, OpStrategy):
                    return result
                selected = _select_min_cost_strategy(child)
                outputs.append(None if selected is None else selected.output_specs)
            return OutputSharding(tuple(outputs))
        return result

    @classmethod
    def _propagate_tuple_strategy(
        cls, schema: OpSchema, strategy: TupleStrategy, output_meta: Any
    ) -> OutputSharding:
        selected: list[PlacementStrategy] = []
        outputs: list[Any] = []
        for child in strategy.children:
            if not isinstance(child, OpStrategy):
                raise AssertionError("tuple strategy child must be an operation strategy")
            _propagate_use_strided_shard_flag(child, schema)
            choice = _select_min_cost_strategy(child, schema)
            selected.append(choice)
            if choice.output_specs is not None:
                outputs.append(choice.output_spec)

        needs_redistribute = False
        tensor_arg_index = 0
        suggestion_args: list[Any] = []
        for arg in schema.args_schema:
            if isinstance(arg, (tuple, list)) and _spec_leaves(arg):
                children: list[Any] = []
                for child_index, item in enumerate(arg):
                    if not isinstance(item, DTensorSpec):
                        children.append(item)
                        continue
                    desired = selected[child_index].input_spec(tensor_arg_index)
                    expected = desired.shallow_copy_with_tensor_meta(item.tensor_meta)
                    if item.placements != desired.placements:
                        needs_redistribute = True
                    children.append(expected)
                tensor_arg_index += 1
                suggestion_args.append(type(arg)(children))
            elif isinstance(arg, DTensorSpec):
                desired = selected[0].input_spec(tensor_arg_index)
                expected = desired.shallow_copy_with_tensor_meta(arg.tensor_meta)
                if arg.placements != desired.placements:
                    needs_redistribute = True
                suggestion_args.append(expected)
                tensor_arg_index += 1
            else:
                suggestion_args.append(arg)

        suggestion = None
        if needs_redistribute:
            suggestion = OpSchema(
                schema.op,
                tuple(suggestion_args),
                schema.kwargs_schema,
                schema_info=schema.schema_info,
            )
        return OutputSharding(
            tuple(outputs) if output_meta is not None else None,
            suggestion,
            needs_redistribute=needs_redistribute,
        )

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
            if isinstance(value, TensorMeta):
                try:
                    next(expected_iter)
                except StopIteration as error:
                    raise ValueError(
                        "strategy input count is smaller than schema input count"
                    ) from error
                return value
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

    def _adjust_shape_and_stride_args(
        self,
        out_tensor_meta: TensorMeta,
        schema: OpSchema,
        spec: DTensorSpec,
    ) -> OpSchema:
        shape_stride_idx = self._operation_value(
            self.op_to_shape_and_stride_idx, schema.op
        )
        if shape_stride_idx is None:
            raise KeyError(schema.op)
        if isinstance(shape_stride_idx, tuple):
            shape_idx, stride_idx = shape_stride_idx
        else:
            shape_idx, stride_idx = shape_stride_idx, None
        expected_args = list(schema.args_schema)
        local_shape, _ = compute_local_shape_and_global_offset(
            out_tensor_meta.shape,
            spec.mesh,
            spec.placements,
            skip_offset=True,
        )
        expected_args[shape_idx] = local_shape
        if stride_idx is not None:
            expected_args[stride_idx] = compute_local_stride(
                out_tensor_meta.stride, local_shape
            )
        return OpSchema(
            schema.op,
            tuple(expected_args),
            schema.kwargs_schema,
            schema_info=schema.schema_info,
        )

    def _adjust_squeeze_to_global_singletons(
        self, schema: OpSchema
    ) -> OpSchema | None:
        if not schema.args_schema or not isinstance(schema.args_schema[0], DTensorSpec):
            return None
        input_spec = schema.args_schema[0]
        tensor_meta = input_spec.tensor_meta
        if tensor_meta is None:
            raise RuntimeError("squeeze requires tensor metadata")
        global_shape = tensor_meta.shape
        ndim = len(global_shape)

        def normalize(dim: Any) -> int:
            value = int(dim)
            return value if value >= 0 else value + ndim

        def is_singleton(dim: Any) -> bool:
            value = normalize(dim)
            if value < 0 or value >= ndim:
                return False
            try:
                return bool(global_shape[value] == 1)
            except (TypeError, ValueError):
                return False

        operation_name = self._operation_name(schema.op)
        dim = schema.kwargs_schema.get(
            "dim", schema.args_schema[1] if len(schema.args_schema) > 1 else None
        )
        if dim is None:
            target_dims = tuple(
                index
                for index, size in enumerate(global_shape)
                if is_singleton(index)
            )
        elif isinstance(dim, (tuple, list)):
            target_dims = tuple(
                normalize(item) for item in dim if is_singleton(item)
            )
        else:
            value = normalize(dim)
            target_dims = (value,) if is_singleton(value) else ()

        dims_variant = self._operation_value(
            self.squeeze_op_to_dims_variant, schema.op
        )
        if dims_variant is None:
            dims_variant = self.squeeze_op_to_dims_variant.get(operation_name)
        if dims_variant is None:
            return None
        if operation_name in {"squeeze_dims", "squeeze_dims_"}:
            existing = schema.args_schema[1] if len(schema.args_schema) > 1 else ()
            if tuple(existing) == target_dims:
                return None
        return OpSchema(
            dims_variant,
            (input_spec, target_dims),
            {},
            schema_info=schema.schema_info,
        )

    def _propagate_schema(self, schema: OpSchema) -> Any:
        operation = schema.op
        single_dim_info = self._operation_value(
            self.op_single_dim_strategy_funcs, operation
        )
        schema_info = self._operation_value(self.op_to_schema_info, operation)
        if schema.schema_info is None and schema_info is not None:
            schema = OpSchema(
                op=schema.op,
                args_schema=schema.args_schema,
                kwargs_schema=schema.kwargs_schema,
                schema_info=schema_info,
            )
        single_dim_schema_info = self._operation_value(
            self.op_to_schema_info_for_single_dim_strategy, operation
        )
        if schema.schema_info is None and single_dim_schema_info is not None:
            schema = OpSchema(
                op=schema.op,
                args_schema=schema.args_schema,
                kwargs_schema=schema.kwargs_schema,
                schema_info=single_dim_schema_info,
            )
        output_meta = self._propagate_tensor_meta_non_cached(schema)
        rule = self._operation_value(self.op_to_rules, operation)
        kind = "prop"
        decomposed_strategy: OpStrategy | None = None
        if rule is None:
            rule = self._operation_value(self.op_strategy_funcs, operation)
            kind = "strategy"
        if rule is None and single_dim_info is None:
            kind, rule = self._global_rule(operation)
        if rule is None and single_dim_info is None:
            if self.decomp_strategy.has_decomp(operation):
                self.decomp_strategy.ensure_schema_info(operation)
                try:
                    decomposed_strategy = self.decomp_strategy.propagate_strategy(
                        schema, output_meta
                    )
                except Exception as error:
                    raise NotImplementedError(
                        f"decomposition propagation failed for {operation}"
                    ) from error
                if decomposed_strategy is None:
                    return None
                kind = "strategy"
            else:
                return None
        if single_dim_info is not None:
            from ._ops.single_dim_strategy import _expand_single_dim_strategy_to_mesh

            _validate_tensor_meta_count(schema, output_meta)
            strategy_schema = self._wrap_with_op_strategy(schema)
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
                strategy_schema = self._strategy_schema(schema)
                result = (
                    decomposed_strategy
                    if decomposed_strategy is not None
                    else self._call_strategy(
                        rule, strategy_schema, self._find_mesh(schema)
                    )
                )
            if isinstance(result, DTensorSpec):
                return OutputSharding(
                    self._create_output_spec_with_new_tensor_meta(
                        operation, result, output_meta
                    )
                )
            if isinstance(result, OpStrategy):
                if not result.strategies:
                    return OutputSharding(None, failed_reason="strategy has no choices")
                _propagate_use_strided_shard_flag(result, schema)
                selected = self._select_strategy(result, schema)
                if not isinstance(selected, PlacementStrategy):
                    return OutputSharding(None, failed_reason="invalid strategy result")
                suggestion, needs_redistribute = self._strategy_suggestion(
                    schema, selected
                )
                output_specs = selected.output_specs
                if schema._returns():
                    if schema.return_type_tuple_tensor_like():
                        if isinstance(output_specs, DTensorSpec):
                            output_specs = tuple(
                                DTensorSpec(
                                    output_specs.mesh,
                                    output_specs.placements,
                                    output_specs.tensor_meta,
                                    shard_order=output_specs.shard_order,
                                    use_strided_shard_as_shard_order=output_specs.use_strided_shard_as_shard_order,
                                )
                                for _ in schema._returns()
                            )
                    elif not (
                        schema.return_type_tensor()
                        or schema.return_type_list_tensor_like()
                    ):
                        output_specs = None
                if (
                    self._operation_value(self.op_to_shape_and_stride_idx, operation)
                    is not None
                    and isinstance(output_specs, DTensorSpec)
                    and any(
                        _is_shard_like(placement)
                        for placement in output_specs.placements
                    )
                    and isinstance(output_meta, TensorMeta)
                ):
                    suggestion = self._adjust_shape_and_stride_args(
                        output_meta,
                        suggestion or schema,
                        output_specs,
                    )
                    needs_redistribute = True
                scalar_adjuster = self._operation_value(
                    self.op_to_scalar_shape_adjuster, operation
                )
                if scalar_adjuster is not None and selected.input_specs is not None:
                    if any(
                        _is_shard_like(placement)
                        for spec in selected.input_specs
                        for placement in spec.placements
                    ):
                        suggestion = scalar_adjuster(
                            list(selected.input_specs), suggestion or schema
                        )
                        needs_redistribute = True
                if self._operation_value(
                    self.squeeze_op_to_dims_variant, operation
                ) is not None:
                    adjusted = self._adjust_squeeze_to_global_singletons(
                        suggestion or schema
                    )
                    if adjusted is not None:
                        suggestion = adjusted
                        needs_redistribute = True
                return OutputSharding(
                    self._create_output_spec_with_new_tensor_meta(
                        operation, output_specs, output_meta
                    ),
                    suggestion,
                    needs_redistribute=needs_redistribute,
                )
            if isinstance(result, TupleStrategy):
                output_sharding = self._propagate_tuple_strategy(
                    schema, result, output_meta
                )
                output_sharding.output_spec = (
                    self._create_output_spec_with_new_tensor_meta(
                        operation, output_sharding.output_spec, output_meta
                    )
                )
                return output_sharding
            normalized = self._strategy_output(result)
            if isinstance(normalized, OutputSharding):
                normalized.output_spec = self._create_output_spec_with_new_tensor_meta(
                    operation, normalized.output_spec, output_meta
                )
                return normalized
            return self._attach_result_meta(normalized, output_meta)
        result = rule(schema)
        if isinstance(result, OutputSharding):
            if result.output_spec is None and result.redistribute_schema is not None:
                retry = rule(result.redistribute_schema)
                if isinstance(retry, OutputSharding):
                    result.output_spec = retry.output_spec
                result.needs_redistribute = True
            result.output_spec = self._create_output_spec_with_new_tensor_meta(
                operation, result.output_spec, output_meta
            )
            return result
        if isinstance(result, DTensorSpec):
            return OutputSharding(
                self._create_output_spec_with_new_tensor_meta(
                    operation, result, output_meta
                )
            )
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
