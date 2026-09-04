"""Schema-based registration of related operation strategy variants."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from .._dtensor_spec import TensorMeta
from .._op_schema import RuntimeSchemaInfo
from ..placement_types import Placement
from .single_dim_strategy import (
    _ShardingPlaceholder,
    _SingleDimStrategyFunc,
    _SingleDimStrategyInfo,
)

__all__ = ["auto_register_op_variants"]


def _schema_arguments(operation: Any) -> Sequence[Any]:
    return getattr(getattr(operation, "_schema", None), "arguments", ())


def _schema_returns(operation: Any) -> Sequence[Any]:
    return getattr(getattr(operation, "_schema", None), "returns", ())


def _schema_type(value: Any) -> str:
    return str(getattr(value, "type", value))


def _tensor_schema_count(values: Sequence[Any]) -> int:
    return sum(1 for value in values if "Tensor" in _schema_type(value))


def _schema_tensor_output_count(operation: Any) -> int:
    return _tensor_schema_count(_schema_returns(operation))


def _is_write_arg(arg: Any) -> bool:
    alias_info = getattr(arg, "alias_info", None)
    return bool(alias_info is not None and getattr(alias_info, "is_write", False))


def _operation_name(operation: Any) -> str:
    name = getattr(operation, "name", None)
    if callable(name):
        name = name()
    if name is None:
        name = getattr(operation, "__name__", operation)
    return str(name)


def _is_foreach_like_op_name(operation_name: str) -> bool:
    name = operation_name.rsplit("::", 1)[-1].split(".", 1)[0]
    return name.startswith(("_foreach_", "_amp_foreach_", "_fused_"))


def _normalize_schema_type(value: Any) -> str:
    return _schema_type(value).replace("!", "").replace("?", "")


def _schema_args_match(base_args: Sequence[Any], candidate_args: Sequence[Any]) -> bool:
    if len(base_args) != len(candidate_args):
        return False
    return all(
        getattr(base, "name", None) == getattr(candidate, "name", None)
        and bool(getattr(base, "kwarg_only", False))
        == bool(getattr(candidate, "kwarg_only", False))
        and _normalize_schema_type(base) == _normalize_schema_type(candidate)
        for base, candidate in zip(base_args, candidate_args)
    )


def _schema_args_are_same(
    base_args: Sequence[Any], candidate_args: Sequence[Any]
) -> bool:
    return _schema_args_match(base_args, candidate_args)


def _is_explicit_out_arg(arg: Any) -> bool:
    return bool(getattr(arg, "is_out", False))


def _schema_non_alias_tensor_output_indices(operation: Any) -> list[int]:
    return [
        index
        for index, value in enumerate(_schema_returns(operation))
        if "Tensor" in _schema_type(value) and getattr(value, "alias_info", None) is None
    ]


def _schema_written_tensor_arg_count(operation: Any) -> int:
    return sum(
        1
        for arg in _schema_arguments(operation)
        if _is_write_arg(arg) and "Tensor" in _schema_type(arg)
    )


def _functional_variant_tensor_output_count(base_op: Any) -> int:
    return len(_schema_non_alias_tensor_output_indices(base_op)) + _schema_written_tensor_arg_count(base_op)


def _get_overload_packet(operation: Any, name: str) -> Any | None:
    packets = getattr(operation, "_variant_packets", None)
    if isinstance(packets, dict) and name in packets:
        return packets[name]
    packet = getattr(operation, "overloadpacket", None)
    if packet is not None:
        candidate = getattr(packet, name, None)
        if candidate is not None:
            return candidate
    namespace = getattr(operation, "namespace", None)
    if namespace is not None:
        getter = getattr(namespace, "get_overload_packet", None)
        if callable(getter):
            return getter(name)
    return None


def _get_packet_overload(packet: Any, name: str) -> Any | None:
    if isinstance(packet, dict):
        return packet.get(name)
    candidate = getattr(packet, name, None)
    if candidate is not None:
        return candidate
    getter = getattr(packet, "get_overload", None)
    return getter(name) if callable(getter) else None


def _iter_packet_overloads(packet: Any | None) -> list[Any]:
    if packet is None:
        return []
    names = packet.overloads() if callable(getattr(packet, "overloads", None)) else packet
    result = []
    for name in names:
        candidate = _get_packet_overload(packet, name)
        if candidate is not None:
            result.append(candidate)
    return result


def _count_tensor_meta_values(value: object) -> int:
    if isinstance(value, TensorMeta):
        return 1
    if isinstance(value, (list, tuple)):
        return sum(_count_tensor_meta_values(item) for item in value)
    return 0


def _clone_schema_info(
    schema_info: RuntimeSchemaInfo | None, *, needs_pytree: bool | None = None
) -> RuntimeSchemaInfo | None:
    if schema_info is None:
        return None if needs_pytree is None else RuntimeSchemaInfo(needs_pytree=needs_pytree)
    return RuntimeSchemaInfo(
        static_argnum=schema_info.static_argnum,
        static_kwargkey=(
            list(schema_info.static_kwargkey)
            if schema_info.static_kwargkey is not None
            else None
        ),
        needs_pytree=schema_info.needs_pytree if needs_pytree is None else needs_pytree,
    )


def _clone_strategy_info(
    info: _SingleDimStrategyInfo, func: _SingleDimStrategyFunc
) -> _SingleDimStrategyInfo:
    return _SingleDimStrategyInfo(
        func=func,
        allow_unbacked_sharding=info.allow_unbacked_sharding,
        allow_uneven_sharding=info.allow_uneven_sharding,
        full_mesh_strategy_filter=info.full_mesh_strategy_filter,
        different_mesh_args=(
            list(info.different_mesh_args)
            if info.different_mesh_args is not None
            else None
        ),
    )


def _canonical_variant_base_name(operation: Any) -> tuple[str, str]:
    full_name = _operation_name(operation)
    namespace, _, rest = full_name.partition("::")
    if not rest:
        namespace, rest = "", full_name
    base_name = rest.split(".", 1)[0]
    schema = getattr(operation, "_schema", None)
    if base_name.endswith("_functional"):
        base_name = base_name.removesuffix("_functional")
    elif bool(getattr(schema, "is_mutable", False)) and base_name.endswith("_"):
        base_name = base_name.removesuffix("_")
    return namespace, base_name


def _strip_output_args_for_base_call(
    out_op: Any,
    output_arg_names: set[str],
    args_schema: Sequence[Any],
    kwargs_schema: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    base_args: list[Any] = []
    positional_index = 0
    for arg in _schema_arguments(out_op):
        if bool(getattr(arg, "kwarg_only", False)):
            continue
        if positional_index >= len(args_schema):
            break
        value = args_schema[positional_index]
        positional_index += 1
        if getattr(arg, "name", None) not in output_arg_names:
            base_args.append(value)
    return (
        tuple(base_args),
        {name: value for name, value in kwargs_schema.items() if name not in output_arg_names},
    )


def _find_inplace_variant_overloads(base_op: Any) -> list[Any]:
    schema = getattr(base_op, "_schema", None)
    if schema is None or bool(getattr(schema, "is_mutable", False)):
        return []
    if _is_foreach_like_op_name(_operation_name(base_op)):
        return []
    if any(_is_explicit_out_arg(arg) for arg in _schema_arguments(base_op)):
        return []
    _, base_name = _canonical_variant_base_name(base_op)
    packet = _get_overload_packet(base_op, f"{base_name}_")
    return [
        candidate
        for candidate in _iter_packet_overloads(packet)
        if bool(getattr(getattr(candidate, "_schema", None), "is_mutable", False))
        and _schema_tensor_output_count(candidate) == _schema_tensor_output_count(base_op)
        and _schema_args_match(_schema_arguments(base_op), _schema_arguments(candidate))
    ]


def _find_out_variant_overloads(
    base_op: Any,
) -> list[tuple[Any, tuple[str, ...]]]:
    schema = getattr(base_op, "_schema", None)
    if schema is None or _is_foreach_like_op_name(_operation_name(base_op)):
        return []
    _, base_name = _canonical_variant_base_name(base_op)
    if base_name.endswith("_functional") or any(
        _is_explicit_out_arg(arg) for arg in _schema_arguments(base_op)
    ):
        return []
    output_count = _schema_tensor_output_count(base_op)
    if output_count == 0:
        return []
    packet = _get_overload_packet(base_op, base_name)
    result: list[tuple[Any, tuple[str, ...]]] = []
    for candidate in _iter_packet_overloads(packet):
        if candidate is base_op or not bool(getattr(getattr(candidate, "_schema", None), "is_mutable", False)):
            continue
        if _schema_tensor_output_count(candidate) != output_count:
            continue
        output_args = [
            arg for arg in _schema_arguments(candidate) if _is_explicit_out_arg(arg)
        ]
        if len(output_args) != output_count or not all(
            "Tensor" in _schema_type(arg) for arg in output_args
        ):
            continue
        non_output_args = [
            arg for arg in _schema_arguments(candidate) if not _is_explicit_out_arg(arg)
        ]
        if _schema_args_are_same(_schema_arguments(base_op), non_output_args):
            result.append((candidate, tuple(getattr(arg, "name", "") for arg in output_args)))
    return result


def _find_functional_variant_overloads(base_op: Any) -> list[Any]:
    if not any(_is_write_arg(arg) for arg in _schema_arguments(base_op)):
        return []
    if _is_foreach_like_op_name(_operation_name(base_op)) or any(
        _is_explicit_out_arg(arg) for arg in _schema_arguments(base_op)
    ):
        return []
    _, base_name = _canonical_variant_base_name(base_op)
    packet = _get_overload_packet(base_op, f"{base_name}_functional")
    expected_outputs = _functional_variant_tensor_output_count(base_op)
    return [
        candidate
        for candidate in _iter_packet_overloads(packet)
        if not bool(getattr(getattr(candidate, "_schema", None), "is_mutable", False))
        and _schema_args_match(_schema_arguments(base_op), _schema_arguments(candidate))
        and _schema_tensor_output_count(candidate) == expected_outputs
    ]


def _resolve_foreach_elementwise_overload(operation: Any) -> Any:
    return getattr(operation, "base_operation", getattr(operation, "_base_operation", None))


def _find_foreach_variants(base_op: Any) -> list[Any]:
    namespace, base_name = _canonical_variant_base_name(base_op)
    del namespace
    if getattr(base_op, "_schema", None) is None or bool(
        getattr(getattr(base_op, "_schema", None), "is_mutable", False)
    ) or _is_foreach_like_op_name(_operation_name(base_op)):
        return []
    if _schema_tensor_output_count(base_op) == 0:
        return []
    result: list[Any] = []
    for name in (f"_foreach_{base_name}", f"_foreach_{base_name}_"):
        packet = _get_overload_packet(base_op, name)
        for candidate in _iter_packet_overloads(packet):
            if "out" not in _operation_name(candidate) and (
                _resolve_foreach_elementwise_overload(candidate) is base_op
                or _operation_name(_resolve_foreach_elementwise_overload(candidate))
                == _operation_name(base_op)
            ):
                result.append(candidate)
    return result


def _make_same_schema_variant_strategy_fn(
    base_fn: _SingleDimStrategyFunc, base_op: Any
) -> _SingleDimStrategyFunc:
    def strategy(
        _operation: Any, args_schema: Any, kwargs_schema: Any
    ) -> list[list[Placement | _ShardingPlaceholder]]:
        return base_fn(base_op, args_schema, kwargs_schema)

    return strategy


def _make_out_variant_strategy_fn(
    base_fn: _SingleDimStrategyFunc,
    base_op: Any,
    out_op: Any,
    output_arg_names: tuple[str, ...],
) -> _SingleDimStrategyFunc:
    output_arg_name_set = set(output_arg_names)
    output_arg_to_index = {
        name: index for index, name in enumerate(output_arg_names)
    }
    base_num_outputs = _schema_tensor_output_count(base_op)

    def output_arg_placements(
        rule: list[Placement | _ShardingPlaceholder],
        arg_name: str,
        value: object,
    ) -> list[Placement | _ShardingPlaceholder]:
        tensor_count = _count_tensor_meta_values(value)
        if tensor_count == 0:
            return []
        if tensor_count > 1 and len(output_arg_names) == 1:
            return [rule[index] for index in range(base_num_outputs)]
        return [rule[output_arg_to_index[arg_name]]]

    def strategy(
        _operation: Any, args_schema: Any, kwargs_schema: Any
    ) -> list[list[Placement | _ShardingPlaceholder]]:
        base_args, base_kwargs = _strip_output_args_for_base_call(
            out_op, output_arg_name_set, args_schema, kwargs_schema
        )
        base_rules = base_fn(base_op, base_args, base_kwargs)
        rules: list[list[Placement | _ShardingPlaceholder]] = []
        for rule in base_rules:
            output_placements: list[Placement | _ShardingPlaceholder] = []
            positional_index = 0
            for arg in _schema_arguments(out_op):
                if bool(getattr(arg, "kwarg_only", False)):
                    continue
                if positional_index >= len(args_schema):
                    break
                value = args_schema[positional_index]
                positional_index += 1
                name = getattr(arg, "name", "")
                if name in output_arg_to_index:
                    output_placements.extend(output_arg_placements(rule, name, value))
            for name, value in kwargs_schema.items():
                if name in output_arg_to_index:
                    output_placements.extend(output_arg_placements(rule, name, value))
            rules.append([*rule, *output_placements])
        return rules

    return strategy


def _make_functional_variant_strategy_fn(
    base_fn: _SingleDimStrategyFunc, base_op: Any
) -> _SingleDimStrategyFunc:
    mutable_arg_names = {
        getattr(arg, "name", "")
        for arg in _schema_arguments(base_op)
        if _is_write_arg(arg)
    }
    base_num_outputs = _schema_tensor_output_count(base_op)
    non_alias_output_indices = _schema_non_alias_tensor_output_indices(base_op)

    def mutable_input_rule_indices(args_schema: Any, kwargs_schema: Any) -> list[int]:
        result: list[int] = []
        tensor_input_index = 0
        positional_index = 0
        for arg in _schema_arguments(base_op):
            if bool(getattr(arg, "kwarg_only", False)):
                value = kwargs_schema.get(getattr(arg, "name", ""))
            elif positional_index < len(args_schema):
                value = args_schema[positional_index]
                positional_index += 1
            else:
                value = kwargs_schema.get(getattr(arg, "name", ""))
            tensor_count = _count_tensor_meta_values(value)
            if tensor_count == 0:
                continue
            if getattr(arg, "name", "") in mutable_arg_names:
                result.append(base_num_outputs + tensor_input_index)
            tensor_input_index += tensor_count
        return result

    def strategy(
        _operation: Any, args_schema: Any, kwargs_schema: Any
    ) -> list[list[Placement | _ShardingPlaceholder]]:
        mutable_indices = mutable_input_rule_indices(args_schema, kwargs_schema)
        base_rules = base_fn(base_op, args_schema, kwargs_schema)
        result: list[list[Placement | _ShardingPlaceholder]] = []
        for rule in base_rules:
            outputs = [rule[index] for index in non_alias_output_indices]
            outputs.extend(rule[index] for index in mutable_indices)
            result.append([*outputs, *rule[base_num_outputs:]])
        return result

    return strategy


def auto_register_op_variants() -> None:
    from .._api import DTensor

    propagator = DTensor._op_dispatcher.sharding_propagator
    registry = propagator.op_single_dim_strategy_funcs
    already_registered = set(registry)
    for base_op, info in list(registry.items()):
        if getattr(base_op, "_schema", None) is None:
            continue
        schema_info = propagator.op_to_schema_info_for_single_dim_strategy.get(base_op)
        for operation in _find_inplace_variant_overloads(base_op):
            if operation in already_registered:
                continue
            propagator.register_single_dim_op_strategy(
                operation,
                _clone_strategy_info(
                    info, _make_same_schema_variant_strategy_fn(info.func, base_op)
                ),
                _clone_schema_info(schema_info),
            )
            already_registered.add(operation)
        for operation, output_arg_names in _find_out_variant_overloads(base_op):
            if operation in already_registered:
                continue
            propagator.register_single_dim_op_strategy(
                operation,
                _clone_strategy_info(
                    info,
                    _make_out_variant_strategy_fn(
                        info.func, base_op, operation, output_arg_names
                    ),
                ),
                _clone_schema_info(schema_info),
            )
            already_registered.add(operation)
        for operation in _find_functional_variant_overloads(base_op):
            if operation in already_registered:
                continue
            propagator.register_single_dim_op_strategy(
                operation,
                _clone_strategy_info(
                    info, _make_functional_variant_strategy_fn(info.func, base_op)
                ),
                _clone_schema_info(schema_info),
            )
            already_registered.add(operation)
        for operation in _find_foreach_variants(base_op):
            if operation in already_registered:
                continue
            propagator.register_single_dim_op_strategy(
                operation,
                info,
                _clone_schema_info(schema_info, needs_pytree=True),
            )
            already_registered.add(operation)
