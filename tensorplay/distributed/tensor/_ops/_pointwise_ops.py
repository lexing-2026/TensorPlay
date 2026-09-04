"""Placement strategies for elementwise and foreach operations."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .._api import DTensor
from .._dtensor_spec import DTensorSpec
from .._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from ..placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
    _StridedShard,
    _is_shard_like,
)
from .utils import (
    generate_redistribute_costs,
    infer_broadcast_dims_map,
    map_placements_after_broadcast,
    normalize_dim,
)

__all__ = [
    "common_pointwise_strategy",
    "linear_pointwise_strategy",
    "list_linear_pointwise_strategy",
    "list_pointwise_strategy",
    "pointwise_strategy",
    "register_pointwise_ops",
]


_POINTWISE_OP_NAMES = frozenset(
    """
    __ilshift__ __irshift__ __lshift__ __rshift__ _conj abs abs_ acos acos_ acosh
    acosh_ add add_ addcdiv addcdiv_ addcmul addcmul_ angle asin asin_ asinh
    asinh_ atan atan2 atan2_ atan_ atanh atanh_ bitwise_and bitwise_and_
    bitwise_left_shift bitwise_left_shift_ bitwise_not bitwise_not_ bitwise_or
    bitwise_or_ bitwise_right_shift bitwise_right_shift_ bitwise_xor bitwise_xor_
    ceil ceil_ clamp clamp_ clip clip_ conj_physical conj_physical_ copysign
    copysign_ cos cos_ cosh cosh_ deg2rad deg2rad_ digamma digamma_ div div_
    eq erf erf_ erfc erfc_ erfinv erfinv_ exp exp2 exp2_ exp_ expm1 expm1_
    float_power float_power_ floor floor_ fmod fmod_ frac frac_ ge gelu gt hypot
    hypot_ i0 i0_ igamma igamma_ igammac igammac_ isinf isnan isneginf
    isposinf ldexp ldexp_ lt le lerp lerp_ lgamma lgamma_ log log10 log10_
    log1p log1p_ log2 log2_ log_ logaddexp logaddexp2 logical_and logical_and_
    logical_not logical_not_ logical_or logical_or_ logical_xor logical_xor_
    logit logit_ masked_fill maximum mul mul_ mvlgamma mvlgamma_
    native_dropout_backward nan_to_num nan_to_num_ ne neg neg_ nextafter
    nextafter_ polygamma polygamma_ positive pow pow_ reciprocal reciprocal_
    rad2deg rad2deg_ relu relu_ remainder remainder_ round round_ rsqrt rsqrt_
    rsub sgn sgn_ sigmoid sigmoid_ sign sign_ signbit silu sin sin_ sinc
    sinc_ sinh sinh_ sqrt sqrt_ square square_ sub sub_ tan tan_ tanh tanh_
    true_divide trunc trunc_ where xlogy xlogy_ gelu_backward sigmoid_backward
    silu_backward tanh_backward threshold_backward
    """.split()
)

_FOREACH_OP_NAMES = frozenset(
    """
    _foreach_abs _foreach_abs_ _foreach_addcdiv_ _foreach_addcmul
    _foreach_addcmul_ _foreach_clamp_max_ _foreach_clamp_min_ _foreach_div_
    _foreach_div _foreach_lerp_ _foreach_maximum_ _foreach_mul _foreach_mul_
    _foreach_neg _foreach_neg_ _foreach_reciprocal_ _foreach_sub _foreach_sub_
    _foreach_sqrt _foreach_sqrt_ _foreach_zero_ _foreach_exp _foreach_exp_
    _foreach_cos _foreach_cos_ _foreach_log _foreach_log_
    _foreach_log_ _amp_foreach_non_finite_check_and_unscale_
    """.split()
)

_FOREACH_LINEARITY_NAMES = frozenset({"_foreach_add", "_foreach_add_"})
_FUSED_OP_NAMES = frozenset(
    {"_fused_adam_", "_fused_adam", "_fused_adamw_", "_fused_adamw"}
)
_POINTWISE_READY = False


def _broadcast_shape(shapes: Sequence[Sequence[int]]) -> tuple[int, ...]:
    if not shapes:
        return ()
    rank = max(len(shape) for shape in shapes)
    result = [1] * rank
    for shape in shapes:
        offset = rank - len(shape)
        for index, size in enumerate(shape):
            target = offset + index
            size = int(size)
            if size not in (1, result[target]) and result[target] != 1:
                raise ValueError("pointwise input shapes are not broadcastable")
            result[target] = max(result[target], size)
    return tuple(result)


def _strategy_args(args_schema: Sequence[Any]) -> list[OpStrategy]:
    return [value for value in args_schema if isinstance(value, OpStrategy)]


def pointwise_strategy(
    mesh: Any, op_schema: OpSchema, linearity: bool = False
) -> OpStrategy:
    max_shards_strategy_index = -1
    max_shards = -1

    if op_schema.is_inplace_op():
        followed_strategy = op_schema.args_schema[0]
    elif op_schema.is_out_variant_op():
        followed_strategy = op_schema.kwargs_schema["out"]
    else:
        for index, arg_strategy in enumerate(op_schema.args_schema):
            if not isinstance(arg_strategy, OpStrategy):
                continue
            arg_max_shards = arg_strategy.max_num_shards()
            if arg_max_shards > max_shards:
                max_shards_strategy_index = index
                max_shards = arg_max_shards
        followed_strategy = op_schema.args_schema[max_shards_strategy_index]

    if not isinstance(followed_strategy, OpStrategy):
        raise AssertionError(f"no pointwise strategy to follow for {op_schema}")
    return common_pointwise_strategy(
        mesh,
        op_schema.args_schema,
        followed_strategy,
        linearity,
    )


def _output_placement(
    placement: Placement,
    input_ndim: int,
    common_ndim: int,
    linearity: bool,
) -> Placement:
    if _is_shard_like(placement):
        shard_dim = normalize_dim(placement.dim, input_ndim)
        new_shard_dim = common_ndim - input_ndim + shard_dim
        if isinstance(placement, _StridedShard):
            return _StridedShard(new_shard_dim, placement.split_factor)
        return Shard(new_shard_dim)
    if isinstance(placement, Partial) and not linearity:
        return Replicate()
    return placement


def common_pointwise_strategy(
    mesh: Any,
    args_schema: Sequence[Any],
    followed_strategy: OpStrategy,
    linearity: bool,
) -> OpStrategy:
    args_strategies = _strategy_args(args_schema)
    common_shape = _broadcast_shape([arg.shape for arg in args_strategies])
    strategies: list[OpSpec] = []

    for placement_strategy in followed_strategy.strategies:
        spec_to_follow = placement_strategy.output_spec
        out_placements = tuple(
            _output_placement(
                placement,
                len(spec_to_follow.shape),
                len(common_shape),
                linearity,
            )
            for placement in spec_to_follow.placements
        )
        input_specs: list[DTensorSpec] = []
        redistribute_costs: list[list[float]] = []
        for input_arg in args_strategies:
            input_arg_spec = input_arg.strategies[0].output_spec
            input_arg_dims_map = infer_broadcast_dims_map(
                common_shape,
                input_arg_spec.shape,
            )
            input_target_placements = map_placements_after_broadcast(
                out_placements,
                common_shape,
                input_arg_dims_map,
            )
            input_target_spec = DTensorSpec(
                mesh,
                input_target_placements,
                tensor_meta=input_arg_spec.tensor_meta,
            )
            input_specs.append(input_target_spec)
            redistribute_costs.append(
                list(generate_redistribute_costs(input_arg, input_target_spec))
            )

        strategies.append(
            OpSpec(
                output_specs=DTensorSpec(mesh, out_placements),
                input_specs=input_specs,
                redistribute_cost=redistribute_costs,
            )
        )
    return OpStrategy(strategies)


def linear_pointwise_strategy(mesh: Any, op_schema: OpSchema) -> OpStrategy:
    return pointwise_strategy(mesh, op_schema, linearity=True)


def _div_pointwise_strategy(mesh: Any, op_schema: OpSchema) -> OpStrategy:
    return pointwise_strategy(
        mesh,
        op_schema,
        linearity=len(_strategy_args(op_schema.args_schema)) <= 1,
    )


def _tuple_argument_strategies(
    args_schema: Sequence[Any], op_schema: OpSchema
) -> list[TupleStrategy]:
    first_arg = args_schema[0]
    if not isinstance(first_arg, TupleStrategy):
        raise AssertionError(f"list operation requires tuple strategies: {op_schema}")
    strategy_len = len(first_arg.children)
    result: list[TupleStrategy] = []
    for arg_index, arg in enumerate(args_schema):
        if isinstance(arg, TupleStrategy):
            if len(arg.children) != strategy_len:
                raise AssertionError("tuple strategy lengths must match")
            result.append(arg)
        elif isinstance(arg, OpStrategy):
            if arg_index == 0:
                raise RuntimeError(f"list operation requires a tuple first argument: {op_schema}")
            result.append(TupleStrategy([arg] * strategy_len))
    return result


def list_pointwise_strategy(mesh: Any, op_schema: OpSchema) -> StrategyType:
    args_strategies = _tuple_argument_strategies(op_schema.args_schema, op_schema)
    follow_strategy = args_strategies[0]
    list_strategies: list[OpStrategy] = []
    for child_index, child_strategy in enumerate(follow_strategy.children):
        if not isinstance(child_strategy, OpStrategy):
            raise AssertionError("tuple strategy child must be an operation strategy")
        child_args = [
            arg_strategy.children[child_index] for arg_strategy in args_strategies
        ]
        list_strategies.append(
            common_pointwise_strategy(
                mesh,
                child_args,
                child_strategy,
                False,
            )
        )
    return TupleStrategy(list_strategies)


def list_linear_pointwise_strategy(mesh: Any, op_schema: OpSchema) -> StrategyType:
    args_strategies = _tuple_argument_strategies(op_schema.args_schema, op_schema)
    follow_strategy = args_strategies[0]
    list_strategies: list[OpStrategy] = []
    for child_index, child_strategy in enumerate(follow_strategy.children):
        if not isinstance(child_strategy, OpStrategy):
            raise AssertionError("tuple strategy child must be an operation strategy")
        child_args = [
            arg_strategy.children[child_index] for arg_strategy in args_strategies
        ]
        list_strategies.append(
            common_pointwise_strategy(
                mesh,
                child_args,
                child_strategy,
                True,
            )
        )
    return TupleStrategy(list_strategies)


def _register(
    names: Sequence[str],
    function: Callable[..., Any],
    schema_info: RuntimeSchemaInfo,
) -> None:
    propagator = DTensor._op_dispatcher.sharding_propagator
    for name in names:
        propagator.register_op_strategy(name, function, schema_info)


def register_pointwise_ops() -> None:
    global _POINTWISE_READY
    if _POINTWISE_READY:
        return
    _POINTWISE_READY = True
    schema_info = RuntimeSchemaInfo(static_kwargkey=["out"])
    linear_names = {"add", "add_", "to"}
    for name in sorted(_POINTWISE_OP_NAMES):
        if name in {"div", "div_"}:
            function = _div_pointwise_strategy
        elif name in linear_names:
            function = linear_pointwise_strategy
        else:
            function = pointwise_strategy
        _register((name,), function, schema_info)
    _register(
        tuple(sorted(_FOREACH_OP_NAMES - _FOREACH_LINEARITY_NAMES)),
        list_pointwise_strategy,
        RuntimeSchemaInfo(needs_pytree=True),
    )
    _register(
        tuple(sorted(_FOREACH_LINEARITY_NAMES)),
        list_linear_pointwise_strategy,
        RuntimeSchemaInfo(needs_pytree=True),
    )
    _register(
        tuple(sorted(_FUSED_OP_NAMES)),
        list_pointwise_strategy,
        RuntimeSchemaInfo(needs_pytree=True),
    )
