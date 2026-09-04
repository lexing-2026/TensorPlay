from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any, TypeAlias

import tensorplay as tp
from tensorplay import functional as tpF

from ..._dtensor_spec import TensorMeta
from ..._op_schema import RuntimeSchemaInfo
from ..._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from ...placement_types import Placement, Replicate

__all__ = ["register_cp_sharding_rules", "unregister_cp_sharding_rules"]

SEQ_DIM = 2
ArgsType: TypeAlias = tuple[Any, ...]
KwargsType: TypeAlias = dict[str, Any]
SingleDimPlacementList: TypeAlias = list[Placement | _ShardingPlaceholder | None]


@contextlib.contextmanager
def _single_dim_strategy_context(
    operation: Any, strategy_func: Any, schema_info: RuntimeSchemaInfo | None = None
) -> Iterator[tuple[Any, Any]]:
    from ..._api import DTensor

    propagator = DTensor._op_dispatcher.sharding_propagator
    origin_strategy = propagator.op_single_dim_strategy_funcs.get(operation)
    origin_schema = propagator.op_to_schema_info_for_single_dim_strategy.get(operation)
    register_single_dim_strategy(operation, schema_info=schema_info)(strategy_func)
    try:
        yield origin_strategy, origin_schema
    finally:
        if origin_strategy is None:
            propagator.op_single_dim_strategy_funcs.pop(operation, None)
        else:
            propagator.op_single_dim_strategy_funcs[operation] = origin_strategy
        if origin_schema is None:
            propagator.op_to_schema_info_for_single_dim_strategy.pop(operation, None)
        else:
            propagator.op_to_schema_info_for_single_dim_strategy[operation] = origin_schema


def _cp_sharding() -> _ShardingPlaceholder:
    return _ShardingPlaceholder(SEQ_DIM)


def _scaled_dot_product_flash_attention_cp_single_dim_strategy(
    op: Any, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[SingleDimPlacementList]:
    del op, kwargs_schema
    return_debug_mask = len(args_schema) >= 6 and args_schema[5]
    debug_attn_mask_sharding: Placement | _ShardingPlaceholder = (
        _cp_sharding() if return_debug_mask else Replicate()
    )
    return [[
        _cp_sharding(), _cp_sharding(), None, None, None, None,
        Replicate(), None, debug_attn_mask_sharding,
        _cp_sharding(), _cp_sharding(), _cp_sharding(),
    ]]


def _scaled_dot_product_flash_attention_backward_cp_single_dim_strategy(
    op: Any, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[SingleDimPlacementList]:
    del op, kwargs_schema
    num_tensor_inputs = sum(isinstance(arg, TensorMeta) for arg in args_schema)
    cp_strategy: SingleDimPlacementList = [
        _cp_sharding(), _cp_sharding(), _cp_sharding(), _cp_sharding(),
        _cp_sharding(), _cp_sharding(), _cp_sharding(), _cp_sharding(),
        _cp_sharding(),
    ]
    cp_strategy.extend([Replicate()] * (num_tensor_inputs - 6))
    return [cp_strategy]


def _scaled_dot_product_efficient_attention_cp_single_dim_strategy(
    op: Any, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[SingleDimPlacementList]:
    del op, kwargs_schema
    has_attn_bias = args_schema[3] is not None
    cp_strategy: SingleDimPlacementList = [
        _cp_sharding(), _cp_sharding(), None, None,
        _cp_sharding(), _cp_sharding(), _cp_sharding(),
    ]
    if has_attn_bias:
        cp_strategy.append(Replicate())
    return [cp_strategy]


def _scaled_dot_product_efficient_attention_backward_cp_single_dim_strategy(
    op: Any, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[SingleDimPlacementList]:
    del op, kwargs_schema
    has_attn_bias = args_schema[4] is not None
    cp_strategy: SingleDimPlacementList = [
        _cp_sharding(), _cp_sharding(), _cp_sharding(),
        _ShardingPlaceholder(1) if has_attn_bias else None,
        _cp_sharding(), _cp_sharding(), _cp_sharding(), _cp_sharding(),
        _cp_sharding(), _cp_sharding(),
    ]
    if has_attn_bias:
        cp_strategy.insert(8, _ShardingPlaceholder(1))
    cp_strategy.extend([Replicate(), Replicate()])
    return [cp_strategy]


def _scaled_dot_product_cudnn_attention_cp_single_dim_strategy(
    op: Any, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[SingleDimPlacementList]:
    del op, kwargs_schema
    attn_bias_meta = args_schema[3]
    compute_log_sumexp = args_schema[4]
    return_debug_mask = len(args_schema) >= 8 and args_schema[7]
    has_attn_bias = attn_bias_meta is not None
    logsumexp_sharding: Placement | _ShardingPlaceholder = (
        _cp_sharding() if compute_log_sumexp else Replicate()
    )
    debug_attn_mask_sharding = _cp_sharding() if return_debug_mask else None
    cp_strategy: SingleDimPlacementList = [
        _cp_sharding(), logsumexp_sharding, None, None, None, None,
        None, None, debug_attn_mask_sharding,
        _cp_sharding(), _cp_sharding(), _cp_sharding(),
    ]
    if has_attn_bias:
        cp_strategy.append(Replicate())
    return [cp_strategy]


def _scaled_dot_product_cudnn_attention_backward_cp_single_dim_strategy(
    op: Any, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[SingleDimPlacementList]:
    del op, kwargs_schema
    if len(args_schema) < 15:
        raise AssertionError(f"Expected at least 15 args, got {len(args_schema)}")
    for arg in args_schema[:6]:
        if not isinstance(arg, TensorMeta):
            raise AssertionError(f"Expected TensorMeta, got {type(arg)}")
    philox_placements: list[Placement] = []
    for arg in (args_schema[6], args_schema[7]):
        if isinstance(arg, TensorMeta):
            philox_placements.append(Replicate())
        elif not isinstance(arg, tp.Tensor):
            raise AssertionError(f"Expected TensorMeta or Tensor, got {type(arg)}")
    has_attn_bias = args_schema[8] is not None
    if has_attn_bias and not isinstance(args_schema[8], (TensorMeta, tp.Tensor)):
        raise AssertionError(f"Expected TensorMeta or Tensor, got {type(args_schema[8])}")
    cum_seq_placements: list[None] = []
    for arg in (args_schema[9], args_schema[10]):
        if isinstance(arg, TensorMeta):
            cum_seq_placements.append(None)
        elif arg is None or isinstance(arg, tp.Tensor):
            pass
        else:
            raise AssertionError(f"Expected TensorMeta or Tensor, got {type(arg)}")
    cp_sharding: SingleDimPlacementList = [
        _cp_sharding(), _cp_sharding(), _cp_sharding(), _cp_sharding(),
        _cp_sharding(), _cp_sharding(), _cp_sharding(), _cp_sharding(),
        _cp_sharding(),
    ]
    cp_sharding.extend(philox_placements)
    if has_attn_bias and isinstance(args_schema[8], TensorMeta):
        cp_sharding.append(_cp_sharding())
    cp_sharding.extend(cum_seq_placements)
    return [cp_sharding]


_cp_strategy_contexts: dict[Any, Any] = {}
_original_strategies: dict[Any, tuple[Any, Any]] = {}


def register_cp_sharding_rules() -> None:
    if _cp_strategy_contexts:
        return
    cp_strategies = [
        (tpF._scaled_dot_product_flash_attention,
         _scaled_dot_product_flash_attention_cp_single_dim_strategy,
         RuntimeSchemaInfo(5)),
        (tpF._scaled_dot_product_flash_attention_backward,
         _scaled_dot_product_flash_attention_backward_cp_single_dim_strategy,
         None),
        (tpF._scaled_dot_product_efficient_attention,
         _scaled_dot_product_efficient_attention_cp_single_dim_strategy,
         RuntimeSchemaInfo(4)),
        (tpF._scaled_dot_product_efficient_attention_backward,
         _scaled_dot_product_efficient_attention_backward_cp_single_dim_strategy,
         None),
        (tpF._scaled_dot_product_cudnn_attention,
         _scaled_dot_product_cudnn_attention_cp_single_dim_strategy,
         RuntimeSchemaInfo(4)),
        (tpF._scaled_dot_product_cudnn_attention_backward,
         _scaled_dot_product_cudnn_attention_backward_cp_single_dim_strategy,
         None),
    ]
    for operation, strategy_func, schema_info in cp_strategies:
        context = _single_dim_strategy_context(operation, strategy_func, schema_info)
        original_strategy, original_schema = context.__enter__()
        _cp_strategy_contexts[operation] = context
        _original_strategies[operation] = (original_strategy, original_schema)


def unregister_cp_sharding_rules(clear_the_cache: bool = False) -> None:
    for context in _cp_strategy_contexts.values():
        context.__exit__(None, None, None)
    if clear_the_cache:
        from ...debug import _clear_fast_path_sharding_prop_cache, _clear_python_sharding_prop_cache

        _clear_fast_path_sharding_prop_cache()
        _clear_python_sharding_prop_cache()
    _cp_strategy_contexts.clear()
    _original_strategies.clear()
