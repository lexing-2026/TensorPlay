from __future__ import annotations

import contextlib
import itertools
import logging
import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Any, Protocol, TypeAlias

import tensorplay as tp
from tensorplay import functional as tpF
from tensorplay.nn import functional as nnF
from tensorplay.nn.attention.omni_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    BlockMask,
    create_block_mask,
)

from ..._dispatch import _dtensors, unwrap_dtensor
from ..._api import DTensor, distribute_tensor
from ...device_mesh import DeviceMesh
from ...placement_types import Placement, Replicate, Shard
from ...parallel import ParallelStyle
from .... import distributed_core as dist
from .... import _functional_collectives as ft_c
from .....utils._pytree import tree_flatten, tree_unflatten
from ._cp_custom_ops import flex_cp_allgather
from ._load_balancer import _LoadBalancer, _create_default_load_balancer

__all__ = [
    "_CausalBehavior",
    "_context_parallel_shard",
    "_ContextParallel",
    "_cp_options",
    "_disable_context_parallel_dispatcher",
    "_enable_context_parallel_dispatcher",
    "_is_causal_behavior",
    "_RotateMethod",
    "context_parallel",
    "context_parallel_unshard",
    "set_rotate_method",
]


class _CausalBehavior(Enum):
    SKIP = None
    NOT_IS_CAUSAL = False
    IS_CAUSAL = True


class _RotateMethod(Enum):
    ALL_TO_ALL = auto()
    ALL_GATHER = auto()


class _DispatchMode(Enum):
    MONKEY_PATCH = auto()
    MODULE_WRAPPER = auto()


_dispatch_mode = _DispatchMode.MONKEY_PATCH
logger = logging.getLogger(__name__)


@dataclass
class _ContextParallelOptions:
    convert_to_f32: bool = True
    enable_load_balance: bool = True
    rotate_method: _RotateMethod = _RotateMethod.ALL_GATHER


_cp_options = _ContextParallelOptions()


def _is_causal_behavior(
    rank: int, world_size: int, i: int, is_causal: bool
) -> _CausalBehavior:
    if not is_causal:
        return _CausalBehavior.NOT_IS_CAUSAL
    if i == 0:
        return _CausalBehavior.IS_CAUSAL
    source_rank = (rank - i) % world_size
    if source_rank < rank or _cp_options.enable_load_balance:
        return _CausalBehavior.NOT_IS_CAUSAL
    return _CausalBehavior.SKIP


def _maybe_wait(tensor: tp.Tensor) -> tp.Tensor:
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return ft_c.wait_tensor(tensor)
    return tensor


def _partial_update(
    original: tp.Tensor,
    new: tp.Tensor,
    dim: int,
    n_chunks: int,
    idx: int,
    add: bool,
) -> tp.Tensor:
    chunks = list(original.chunk(n_chunks, dim=dim))
    if chunks[idx].shape != new.shape:
        raise AssertionError((original.shape, new.shape, idx))
    if add:
        chunks[idx] += new
    else:
        chunks[idx] = new
    return tp.cat(chunks, dim=dim)


class _SDPAMerger:
    def __init__(self, convert_to_f32: bool, seq_dim: int):
        self._seq_dim = seq_dim
        self._out: tp.Tensor | None = None
        self._lse: tp.Tensor | None = None
        self._should_lse_squeeze = False
        self._convert_to_f32 = convert_to_f32
        self._out_dtype = tp.float32
        self._lse_dtype = tp.float32

    def _merge_one(
        self, block_out: tp.Tensor, block_lse: tp.Tensor, partial: bool
    ) -> None:
        if len(block_lse.shape) < len(block_out.shape):
            block_lse = block_lse.unsqueeze(dim=-1)
            self._should_lse_squeeze = True
        if len(block_lse.shape) != len(block_out.shape):
            raise AssertionError
        if self._lse is None:
            self._lse = block_lse
            self._out = block_out
            return
        if self._out is None:
            raise AssertionError
        cycle = 2
        lse = self._lse.chunk(cycle, dim=self._seq_dim)[1] if partial else self._lse
        out = self._out.chunk(cycle, dim=self._seq_dim)[1] if partial else self._out
        out = out - nnF.sigmoid(block_lse - lse) * (out - block_out)
        lse = lse - nnF.logsigmoid(lse - block_lse)
        if partial:
            self._lse = _partial_update(self._lse, lse, self._seq_dim, cycle, 1, False)
            self._out = _partial_update(self._out, out, self._seq_dim, cycle, 1, False)
        else:
            self._lse = lse
            self._out = out

    def step(self, out: tp.Tensor, lse: tp.Tensor, partial: bool) -> None:
        self._out_dtype = out.dtype
        self._lse_dtype = lse.dtype
        if self._convert_to_f32:
            out = out.to(tp.float32)
            lse = lse.to(tp.float32)
        self._merge_one(out, lse, partial)

    def results(self) -> tuple[tp.Tensor, tp.Tensor]:
        if self._out is None or self._lse is None:
            raise AssertionError
        out = self._out.to(self._out_dtype)
        if self._should_lse_squeeze:
            lse = self._lse.squeeze(-1).to(self._lse_dtype)
        else:
            lse = self._lse.to(self._lse_dtype)
        return out, lse


class _AttentionOp(Protocol):
    def __call__(
        self,
        query: tp.Tensor,
        key: tp.Tensor,
        value: tp.Tensor,
        **kwargs: object,
    ) -> tuple[tp.Tensor, ...]: ...


class _RingRotater(ABC):
    @abstractmethod
    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None: ...

    @abstractmethod
    def exchange_buffers(self, curr_buffer: tp.Tensor) -> None: ...

    @abstractmethod
    def next_buffer(self) -> tp.Tensor: ...


class _AllToAllRotater(_RingRotater):
    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None:
        self._pg = pg
        self._seq_dim = seq_dim
        self._buffer: tp.Tensor | None = None

    def exchange_buffers(self, curr_buffer: tp.Tensor) -> None:
        size = dist.get_world_size(self._pg)
        dsts = list(range(1, size)) + [0]
        self._buffer = ft_c.permute_tensor(curr_buffer.contiguous(), dsts, self._pg)

    def next_buffer(self) -> tp.Tensor:
        if self._buffer is None:
            raise AssertionError
        return _maybe_wait(self._buffer)


class _AllGatherRotater(_RingRotater):
    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None:
        self._pg = pg
        self._seq_dim = seq_dim
        self._aggregated_buffer: tp.Tensor | None = None
        self._idx = 0

    def exchange_buffers(self, curr_buffer: tp.Tensor) -> None:
        self._idx += 1
        if self._aggregated_buffer is None:
            self._aggregated_buffer = ft_c.all_gather_single(
                curr_buffer.contiguous(), gather_dim=0, group=self._pg
            )

    def next_buffer(self) -> tp.Tensor:
        rank = dist.get_rank(self._pg)
        idx = rank - self._idx
        if self._aggregated_buffer is None:
            raise AssertionError
        self._aggregated_buffer = _maybe_wait(self._aggregated_buffer)
        return self._aggregated_buffer.chunk(dist.get_world_size(self._pg))[idx]


def _create_rotater(
    pg: dist.ProcessGroup, seq_dim: int, method: _RotateMethod | None = None
) -> _RingRotater:
    method = _cp_options.rotate_method if method is None else method
    if method == _RotateMethod.ALL_TO_ALL:
        return _AllToAllRotater(pg, seq_dim)
    if method == _RotateMethod.ALL_GATHER:
        return _AllGatherRotater(pg, seq_dim)
    raise NotImplementedError(f"Unknown method {method}")


def _templated_ring_attention(
    group: dist.ProcessGroup,
    seq_dim: int,
    op: _AttentionOp,
    query: tp.Tensor,
    key: tp.Tensor,
    value: tp.Tensor,
    is_causal: bool = False,
    **kwargs: object,
) -> tuple[tp.Tensor, ...]:
    if is_causal and query.size(2) != key.size(2):
        raise NotImplementedError(
            "is_causal requires the same query and context sequence lengths"
        )
    if not is_causal and _cp_options.enable_load_balance:
        raise RuntimeError("Load balancing requires is_causal=True.")
    if not isinstance(group, dist.ProcessGroup):
        raise AssertionError("process group must be single dimension")
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)
    key = key.contiguous()
    value = value.contiguous()
    merger = _SDPAMerger(_cp_options.convert_to_f32, seq_dim)
    saved_rest: list[Any] | None = None
    rotater = _create_rotater(group, 2)
    for i in range(size):
        if i > 0:
            next_kv = rotater.next_buffer()
            key = next_kv[: key.numel()].reshape(key.shape)
            value = next_kv[key.numel() :].reshape(value.shape)
        if i < size - 1:
            rotater.exchange_buffers(tp.cat([key.flatten(), value.flatten()]))
        causal_behavior = _is_causal_behavior(rank, size, i, is_causal)
        if causal_behavior == _CausalBehavior.SKIP:
            continue
        if i == 0 or not _cp_options.enable_load_balance or not is_causal:
            q, k, v, partial = query, key, value, False
        elif i <= rank:
            q, k, v, partial = (
                query,
                key.chunk(2, dim=2)[0],
                value.chunk(2, dim=2)[0],
                False,
            )
        else:
            q, k, v, partial = query.chunk(2, dim=2)[1], key, value, True
        out, logsumexp, *rest = op(
            q, k, v, is_causal=causal_behavior.value, **kwargs
        )
        if saved_rest is None:
            saved_rest = rest
        merger.step(out, logsumexp, partial)
    if saved_rest is None:
        raise AssertionError("No attention operation was executed")
    return *merger.results(), *saved_rest


def _templated_ring_attention_backward(
    group: dist.ProcessGroup,
    seq_dim: int,
    op: _AttentionOp,
    grad_out: tp.Tensor,
    grad_out_name: str,
    query: tp.Tensor,
    key: tp.Tensor,
    value: tp.Tensor,
    out: tp.Tensor,
    logsumexp: tp.Tensor,
    is_causal: bool,
    **kwargs: Any,
) -> tuple[tp.Tensor, ...]:
    if not is_causal and _cp_options.enable_load_balance:
        raise RuntimeError("Load balancing requires is_causal=True.")
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)
    grad_query_: tp.Tensor | None = None
    grad_key_: tp.Tensor | None = None
    grad_value_: tp.Tensor | None = None
    accum_dtype = tp.float32 if _cp_options.convert_to_f32 else query.dtype
    grad_query = tp.zeros_like(query, dtype=accum_dtype)
    grad_key = tp.zeros_like(key, dtype=accum_dtype)
    grad_value = tp.zeros_like(value, dtype=accum_dtype)
    key = key.contiguous()
    value = value.contiguous()
    kv_rotater = _create_rotater(group, 2)
    dkv_rotater = _create_rotater(group, 2, method=_RotateMethod.ALL_TO_ALL)
    rest: list[Any] = []
    for i in range(size):
        if i > 0:
            buffer = kv_rotater.next_buffer()
            pointer = 0
            key = buffer[pointer : pointer + key.numel()].reshape(key.shape)
            pointer += key.numel()
            value = buffer[pointer : pointer + value.numel()].reshape(value.shape)
        if i != size - 1:
            kv_rotater.exchange_buffers(tp.cat([key.flatten(), value.flatten()]))
        causal_behavior = _is_causal_behavior(rank, size, i, is_causal)
        if causal_behavior != _CausalBehavior.SKIP:
            if i == 0 or not _cp_options.enable_load_balance or not is_causal:
                q, k, v, out_, dout, lse = query, key, value, out, grad_out, logsumexp
            elif i <= rank:
                q, k, v, out_, dout, lse = (
                    query,
                    key.chunk(2, dim=seq_dim)[0],
                    value.chunk(2, dim=seq_dim)[0],
                    out,
                    grad_out,
                    logsumexp,
                )
            else:
                q, k, v, out_, dout, lse = (
                    query.chunk(2, dim=seq_dim)[1],
                    key,
                    value,
                    out.chunk(2, dim=seq_dim)[1],
                    grad_out.chunk(2, dim=seq_dim)[1],
                    logsumexp.chunk(2, dim=seq_dim)[1].contiguous(),
                )
            kwargs[grad_out_name] = dout
            iter_kwargs = kwargs
            if _cp_options.enable_load_balance and i > 0:
                iter_kwargs = dict(kwargs)
                if "max_q" in iter_kwargs:
                    iter_kwargs["max_q"] = q.shape[seq_dim]
                if "max_k" in iter_kwargs:
                    iter_kwargs["max_k"] = k.shape[seq_dim]
            grad_query_, grad_key_, grad_value_, *rest = op(
                query=q,
                key=k,
                value=v,
                out=out_,
                logsumexp=lse,
                is_causal=causal_behavior.value,
                **iter_kwargs,
            )
        else:
            grad_query_ = tp.zeros_like(query, dtype=accum_dtype)
            grad_key_ = tp.zeros_like(key, dtype=accum_dtype)
            grad_value_ = tp.zeros_like(value, dtype=accum_dtype)
        if i == 0:
            grad_key += grad_key_
            grad_value += grad_value_
        else:
            next_grad_kv = dkv_rotater.next_buffer()
            pointer = 0
            grad_key = next_grad_kv[pointer : pointer + grad_key.numel()].reshape(grad_key.shape)
            pointer += grad_key.numel()
            grad_value = next_grad_kv[pointer : pointer + grad_value.numel()].reshape(grad_value.shape)
            if i <= rank and _cp_options.enable_load_balance:
                grad_key = _partial_update(grad_key, grad_key_, seq_dim, 2, 0, True)
                grad_value = _partial_update(grad_value, grad_value_, seq_dim, 2, 0, True)
            else:
                grad_key += grad_key_
                grad_value += grad_value_
        dkv_rotater.exchange_buffers(tp.cat([grad_key.flatten(), grad_value.flatten()]))
        if i <= rank or not _cp_options.enable_load_balance:
            grad_query += grad_query_
        else:
            grad_query = _partial_update(grad_query, grad_query_, seq_dim, 2, 1, True)
    if grad_key_ is None or grad_value_ is None:
        raise AssertionError
    grad_query = grad_query.to(query.dtype)
    next_grad_kv = dkv_rotater.next_buffer().to(key.dtype)
    grad_key = next_grad_kv[: grad_key.numel()].reshape(grad_key.shape)
    grad_value = next_grad_kv[grad_key.numel() :].reshape(grad_value.shape)
    return grad_query, grad_key, grad_value, *rest


def _scaled_dot_product_ring_flash_attention(
    mesh: DeviceMesh,
    query: tp.Tensor,
    key: tp.Tensor,
    value: tp.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
) -> tuple[tp.Tensor, ...]:
    if return_debug_mask:
        raise NotImplementedError("return_debug_mask is not supported yet")
    return _templated_ring_attention(
        mesh.get_group(), 2, tpF._scaled_dot_product_flash_attention,
        query, key, value, is_causal=is_causal,
        dropout_p=dropout_p, scale=scale,
    )


def _scaled_dot_product_ring_efficient_attention(
    mesh: DeviceMesh,
    query: tp.Tensor,
    key: tp.Tensor,
    value: tp.Tensor,
    attn_bias: tp.Tensor | None = None,
    compute_log_sumexp: bool = True,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
) -> tuple[tp.Tensor, ...]:
    if attn_bias is not None:
        raise NotImplementedError("attn_bias is not supported yet")
    return _templated_ring_attention(
        mesh.get_group(), 2, tpF._scaled_dot_product_efficient_attention,
        query, key, value, is_causal=is_causal,
        attn_bias=attn_bias, compute_log_sumexp=True,
        dropout_p=dropout_p, scale=scale,
    )


def _scaled_dot_product_ring_cudnn_attention(
    mesh: DeviceMesh,
    query: tp.Tensor,
    key: tp.Tensor,
    value: tp.Tensor,
    attn_bias: tp.Tensor | None = None,
    compute_log_sumexp: bool = True,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
) -> tuple[tp.Tensor, ...]:
    if attn_bias is not None:
        raise NotImplementedError("attn_bias is not supported yet")
    return _templated_ring_attention(
        mesh.get_group(), 2, tpF._scaled_dot_product_cudnn_attention,
        query, key, value, is_causal=is_causal,
        attn_bias=attn_bias, compute_log_sumexp=True,
        dropout_p=dropout_p, return_debug_mask=return_debug_mask,
        scale=scale,
    )


def _scaled_dot_product_ring_flash_attention_backward(
    mesh: DeviceMesh,
    grad_out: tp.Tensor,
    query: tp.Tensor,
    key: tp.Tensor,
    value: tp.Tensor,
    out: tp.Tensor,
    logsumexp: tp.Tensor,
    cum_seq_q: tp.Tensor,
    cum_seq_k: tp.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: tp.Tensor,
    philox_offset: tp.Tensor,
    *,
    scale: float | None = None,
) -> tuple[tp.Tensor, ...]:
    return _templated_ring_attention_backward(
        mesh.get_group(), 2, tpF._scaled_dot_product_flash_attention_backward,
        grad_out, "grad_out", query, key, value, out, logsumexp, is_causal,
        cum_seq_q=cum_seq_q, cum_seq_k=cum_seq_k, max_q=max_q, max_k=max_k,
        dropout_p=dropout_p, philox_seed=philox_seed,
        philox_offset=philox_offset, scale=scale,
    )


def _scaled_dot_product_ring_efficient_attention_backward(
    mesh: DeviceMesh,
    grad_out: tp.Tensor,
    query: tp.Tensor,
    key: tp.Tensor,
    value: tp.Tensor,
    bias: tp.Tensor,
    out: tp.Tensor,
    logsumexp: tp.Tensor,
    philox_seed: tp.Tensor,
    philox_offset: tp.Tensor,
    dropout_p: float,
    grad_input_mask: tuple[bool, ...],
    is_causal: bool = False,
    *,
    scale: float | None = None,
) -> tuple[tp.Tensor, ...]:
    return _templated_ring_attention_backward(
        mesh.get_group(), 2, tpF._scaled_dot_product_efficient_attention_backward,
        grad_out, "grad_out_", query, key, value, out, logsumexp, is_causal,
        attn_bias=bias, philox_seed=philox_seed, philox_offset=philox_offset,
        dropout_p=dropout_p, grad_input_mask=grad_input_mask, scale=scale,
    )


def _scaled_dot_product_ring_cudnn_attention_backward(
    mesh: DeviceMesh,
    grad_out: tp.Tensor,
    query: tp.Tensor,
    key: tp.Tensor,
    value: tp.Tensor,
    out: tp.Tensor,
    logsumexp: tp.Tensor,
    philox_seed: tp.Tensor,
    philox_offset: tp.Tensor,
    attn_bias: tp.Tensor,
    cum_seq_q: tp.Tensor,
    cum_seq_k: tp.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    *,
    scale: float | None = None,
) -> tuple[tp.Tensor, ...]:
    return _templated_ring_attention_backward(
        mesh.get_group(), 2, tpF._scaled_dot_product_cudnn_attention_backward,
        grad_out, "grad_out", query, key, value, out, logsumexp, is_causal,
        philox_seed=philox_seed, philox_offset=philox_offset,
        attn_bias=attn_bias, cum_seq_q=cum_seq_q, cum_seq_k=cum_seq_k,
        max_q=max_q, max_k=max_k, dropout_p=dropout_p, scale=scale,
    )


def _attention_output_wrap(
    operation: Any,
    results: Any,
    values: list[DTensor],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Any:
    if not values:
        return results
    template = values[0]
    kwargs = {} if kwargs is None else kwargs
    name = getattr(operation, "__name__", str(operation))
    name = str(name).rsplit(".", 1)[-1]
    if isinstance(results, tuple):
        return tuple(
            _attention_output_wrap_item(name, index, value, template, values, args, kwargs)
            for index, value in enumerate(results)
        )
    if isinstance(results, list):
        return [
            _attention_output_wrap_item(name, index, value, template, values, args, kwargs)
            for index, value in enumerate(results)
        ]
    return results


def _attention_output_wrap_item(
    name: str,
    index: int,
    value: Any,
    template: DTensor,
    values: list[DTensor],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    if not isinstance(value, tp.Tensor):
        return value
    if "backward" in name:
        distributed = index < 3
    else:
        distributed = index in (0, 1)
        debug_index = 7 if "cudnn" in name else 5
        if index == 8 and (len(args) > debug_index and args[debug_index]):
            distributed = True
        if index == 8 and kwargs.get("return_debug_mask", False):
            distributed = True
    if not distributed:
        return value
    placements = list(template.placements)
    if "efficient_attention_backward" in name and index == 3:
        distributed = any(isinstance(item, DTensor) for item in values[1:])
        if distributed:
            placements = [Shard(1) if isinstance(item, Shard) else item for item in placements]
    if not distributed:
        return value
    local_shape = tuple(int(item) for item in value.shape)
    source_local_shape = tuple(int(item) for item in template.to_local().shape)
    global_shape = list(local_shape)
    if len(local_shape) > 2 and len(source_local_shape) > 2:
        local_seq = source_local_shape[2]
        global_seq = int(template.shape[2])
        if local_seq and local_shape[2] == local_seq:
            global_shape[2] = global_seq
    return DTensor(
        value, template.device_mesh, placements,
        shape=tuple(global_shape), stride=tuple(int(item) for item in value.stride()),
    )


def _sdpa_handler(operation: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    values = _dtensors((args, kwargs))
    dispatcher = DTensor._op_dispatcher
    try:
        dispatcher._propagate(operation, args, kwargs)
    except (AssertionError, RuntimeError, TypeError, ValueError):
        logger.debug("attention sharding metadata is unavailable for %s", operation)
    call_maps: dict[Any, Callable[..., Any]] = {
        tpF._scaled_dot_product_flash_attention: _scaled_dot_product_ring_flash_attention,
        tpF._scaled_dot_product_efficient_attention: _scaled_dot_product_ring_efficient_attention,
        tpF._scaled_dot_product_cudnn_attention: _scaled_dot_product_ring_cudnn_attention,
        tpF._scaled_dot_product_flash_attention_backward: _scaled_dot_product_ring_flash_attention_backward,
        tpF._scaled_dot_product_efficient_attention_backward: _scaled_dot_product_ring_efficient_attention_backward,
        tpF._scaled_dot_product_cudnn_attention_backward: _scaled_dot_product_ring_cudnn_attention_backward,
    }
    handler = call_maps.get(operation)
    if handler is None:
        raise NotImplementedError("context parallel supports the registered attention operations only")
    local_results = handler(
        values[0].device_mesh,
        *unwrap_dtensor(args),
        **unwrap_dtensor(kwargs),
    )
    return _attention_output_wrap(operation, local_results, values, args, kwargs)


custom_ops = {
    tpF._scaled_dot_product_flash_attention: _sdpa_handler,
    tpF._scaled_dot_product_flash_attention_backward: _sdpa_handler,
    tpF._scaled_dot_product_efficient_attention: _sdpa_handler,
    tpF._scaled_dot_product_efficient_attention_backward: _sdpa_handler,
    tpF._scaled_dot_product_cudnn_attention: _sdpa_handler,
    tpF._scaled_dot_product_cudnn_attention_backward: _sdpa_handler,
}
existing_custom_ops = DTensor._op_dispatcher._custom_op_handlers

ArgsType = tuple[Any, ...]
KwargsType = dict[str, Any]
InputFnType = Callable[[Any, ArgsType, KwargsType, DeviceMesh], Any]
OutputFnType = Callable[[Any, Any, Any, DeviceMesh], Any]
_replaced_functions: dict[Callable, tuple[str, Callable]] = {}


def _distribute_function(
    fn: Callable,
    fn_module: types.ModuleType,
    device_mesh: DeviceMesh,
    input_fn: InputFnType,
    output_fn: OutputFnType,
) -> None:
    def wrapper(
        target_fn: Callable, input_callback: InputFnType, output_callback: OutputFnType
    ) -> Callable:
        def inner_fn(*args: ArgsType, **kwargs: KwargsType) -> Any:
            args, kwargs = input_callback(None, args, kwargs, device_mesh)
            outputs = target_fn(*args, **kwargs)
            return output_callback(None, (args, kwargs), outputs, device_mesh)

        return inner_fn

    if fn in _replaced_functions:
        return
    wrapper_fn = wrapper(fn, input_fn, output_fn)
    setattr(fn_module, fn.__name__, wrapper_fn)
    _replaced_functions[wrapper_fn] = (fn.__name__, fn)


def _restore_function(fn: Callable, fn_module: types.ModuleType) -> None:
    if fn not in _replaced_functions:
        return
    original_name, original_fn = _replaced_functions.pop(fn)
    setattr(fn_module, original_name, original_fn)


def _enable_cp_dtensor_dispatcher() -> None:
    DTensor._op_dispatcher._custom_op_handlers = {
        **existing_custom_ops,
        **custom_ops,
    }
    from ._sharding_rules import register_cp_sharding_rules

    register_cp_sharding_rules()


def _disable_cp_dtensor_dispatcher() -> None:
    DTensor._op_dispatcher._custom_op_handlers = dict(existing_custom_ops)
    from ._sharding_rules import unregister_cp_sharding_rules

    unregister_cp_sharding_rules(clear_the_cache=False)


def _enable_context_parallel_dispatcher_impl(seq_dim: int, mesh: DeviceMesh) -> None:
    sdpa_cp = _ContextParallel(seq_dim=seq_dim, attention_type=_ContextParallel.AttentionType.SDPA)
    if _dispatch_mode == _DispatchMode.MONKEY_PATCH:
        _distribute_function(
            nnF.scaled_dot_product_attention, nnF, mesh,
            sdpa_cp.sdpa_input_fn, sdpa_cp.sdpa_output_fn,
        )
        _enable_cp_dtensor_dispatcher()
    elif _dispatch_mode == _DispatchMode.MODULE_WRAPPER:
        _enable_cp_dtensor_dispatcher()
    else:
        raise ValueError(f"Unknown dispatch mode: {_dispatch_mode}")


def _disable_context_parallel_dispatcher_impl() -> None:
    if _dispatch_mode == _DispatchMode.MONKEY_PATCH:
        _restore_function(nnF.scaled_dot_product_attention, nnF)
    elif _dispatch_mode != _DispatchMode.MODULE_WRAPPER:
        raise NotImplementedError(f"Unknown dispatch mode: {_dispatch_mode}")
    _disable_cp_dtensor_dispatcher()


_compiled_create_block_mask: Any = None


def _context_parallel_buffers(
    mesh: DeviceMesh,
    buffers: list[Any],
    buffer_seq_dims: list[int],
    load_balancer: _LoadBalancer | None = None,
) -> list[Any]:
    load_balance_indices = load_balancer._generate_indices() if load_balancer else None
    if not (load_balance_indices is None or load_balance_indices.ndim == 2):
        raise AssertionError(
            "load balance index expects shape (1, seq_len) or (B, seq_len) "
            f"but got {load_balance_indices.shape}."
        )
    new_buffers = []
    for buffer, seq_dim in zip(buffers, buffer_seq_dims):
        if isinstance(buffer, tp.Tensor):
            if load_balance_indices is not None:
                idx_batch_size = load_balance_indices.size(0)
                data_batch_size = buffer.size(0) if seq_dim > 0 else 1
                if idx_batch_size != 1 and idx_batch_size != data_batch_size:
                    raise ValueError(
                        "Cannot rearrange buffer: "
                        f"load_balance_indices has shape {load_balance_indices.shape}, "
                        f"but buffer has shape {buffer.shape}."
                    )
                if seq_dim == 0:
                    buffer = tp.index_select(buffer, 0, load_balance_indices[0])
                else:
                    indices = load_balance_indices
                    if idx_batch_size == 1:
                        indices = indices.expand(data_batch_size, indices.size(1))
                    for i in range(1, seq_dim):
                        indices = indices.unsqueeze(i)
                    for _ in range(seq_dim + 1, buffer.ndim):
                        indices = indices.unsqueeze(-1)
                    indices = indices.expand(buffer.shape)
                    buffer = tp.gather(buffer, seq_dim, indices)
            sharded_buffer = distribute_tensor(
                buffer, mesh, [Shard(seq_dim)], src_data_rank=None
            ).to_local()
        elif isinstance(buffer, BlockMask):
            sharded_buffer = _create_cp_block_mask(
                buffer.mask_mod,
                int(buffer.kv_num_blocks.shape[0]),
                int(buffer.kv_num_blocks.shape[1]),
                int(buffer.seq_lengths[0]),
                int(buffer.seq_lengths[1]),
                mesh,
                load_balancer,
            )
        else:
            raise ValueError(f"Unknown buffer type: {type(buffer)}")
        new_buffers.append(sharded_buffer)
    return new_buffers


def _create_cp_block_mask(
    mask_mod: Callable[..., Any],
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    device_mesh: DeviceMesh,
    load_balancer: _LoadBalancer | None = None,
) -> BlockMask:
    if Q_LEN % (device_mesh.size() * _DEFAULT_SPARSE_BLOCK_SIZE) != 0:
        raise NotImplementedError(
            f"Q_LEN {Q_LEN} is not divisible by CP mesh world size {device_mesh.size()} "
            f"* BLOCK_SIZE {_DEFAULT_SPARSE_BLOCK_SIZE}"
        )

    global _compiled_create_block_mask
    if _compiled_create_block_mask is None:
        _compiled_create_block_mask = create_block_mask

    cp_rank = device_mesh.get_local_rank()
    cp_group_size = device_mesh.size()
    load_balancer = load_balancer or _create_default_load_balancer(
        Q_LEN, cp_group_size, device_mesh.device_type
    )
    Q_SHARD_LEN = Q_LEN // cp_group_size
    block_size = _DEFAULT_SPARSE_BLOCK_SIZE
    rearrange_indices = (
        load_balancer._generate_indices(restore=False) if load_balancer else None
    )

    def _rewrite_mask_mod(
        rank: int,
        local_q_size: int,
        qkv_rearrange_indices: tp.Tensor | None = None,
    ) -> Callable[..., Any]:
        if not (
            qkv_rearrange_indices is None or qkv_rearrange_indices.ndim == 2
        ):
            raise AssertionError(
                "load balance index expects shape (1, seq_len) or (B, seq_len) "
                f"but got {qkv_rearrange_indices.shape}."
            )

        def qkv_idx_restore(batch: tp.Tensor, index: tp.Tensor) -> tp.Tensor:
            if qkv_rearrange_indices is None:
                return index
            if qkv_rearrange_indices.size(0) == 1:
                return qkv_rearrange_indices.squeeze(0)[index]
            return qkv_rearrange_indices[batch][index]

        def local_q_idx_to_q_idx(index: tp.Tensor) -> tp.Tensor:
            local_block = index // block_size
            block_offset = index % block_size
            global_block = (local_q_size // block_size) * rank + local_block
            return global_block * block_size + block_offset

        def rewritten(
            batch: tp.Tensor,
            head: tp.Tensor,
            query: tp.Tensor,
            key: tp.Tensor,
        ) -> tp.Tensor:
            return mask_mod(
                batch,
                head,
                qkv_idx_restore(batch, local_q_idx_to_q_idx(query)),
                qkv_idx_restore(batch, key),
            )

        return rewritten

    return _compiled_create_block_mask(
        _rewrite_mask_mod(cp_rank, Q_SHARD_LEN, rearrange_indices),
        B,
        H,
        Q_SHARD_LEN,
        KV_LEN,
        device=device_mesh.device_type,
        BLOCK_SIZE=(block_size, block_size),
    )


class _ContextParallel(ParallelStyle):
    class AttentionType(Enum):
        FLEX = "omni_attention"
        SDPA = "scaled_dot_product_attention"

    def __init__(self, seq_dim: int, attention_type: AttentionType) -> None:
        super().__init__()
        self.seq_dim = seq_dim
        self.attention_type = attention_type

    def _apply(self, module: Any, mesh: DeviceMesh) -> Any:
        if self.attention_type == self.AttentionType.FLEX:
            module.register_forward_pre_hook(partial(self.omni_input_fn, mesh=mesh), with_kwargs=True)
        elif self.attention_type == self.AttentionType.SDPA:
            module.register_forward_pre_hook(partial(self.sdpa_input_fn, mesh=mesh), with_kwargs=True)
            module.register_forward_hook(partial(self.sdpa_output_fn, mesh=mesh))
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
        return module

    def omni_input_fn(
        self, module: Any, args: Any, kwargs: Any, mesh: DeviceMesh
    ) -> Any:
        del module
        expected_arg_names = ("query", "key", "value")
        args_list = list(args)
        for index, name in enumerate(expected_arg_names):
            if index >= len(args):
                args_list.append(kwargs.pop(name, None))
        query, key, value = args_list[:3]
        if not isinstance(query, tp.Tensor) or not isinstance(key, tp.Tensor) or not isinstance(value, tp.Tensor):
            raise AssertionError
        global_key, global_value = flex_cp_allgather(
            key, value, self.seq_dim, mesh.get_group().group_name
        )
        args_list[1], args_list[2] = global_key, global_value
        for index in range(len(args), len(expected_arg_names)):
            kwargs[expected_arg_names[index]] = args_list[index]
        return tuple(args_list[: len(args)]), kwargs

    def sdpa_input_fn(
        self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], mesh: DeviceMesh
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        del module
        placement = [Shard(self.seq_dim)]
        all_args = []
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, DTensor):
                if list(arg.placements) != placement:
                    raise AssertionError
            elif isinstance(arg, tp.Tensor):
                arg = DTensor.from_local(arg, mesh, placement, run_check=False)
            all_args.append(arg)
        return tuple(all_args[: len(args)]), dict(zip(kwargs.keys(), all_args[len(args) :]))

    def sdpa_output_fn(
        self, module: Any, inputs: Any, outputs: Any, mesh: DeviceMesh
    ) -> Any:
        del module, inputs, mesh
        if isinstance(outputs, DTensor):
            return outputs.to_local()
        if isinstance(outputs, tp.Tensor):
            return outputs
        return tuple(item.to_local() if isinstance(item, DTensor) else item for item in outputs)


CPBuffer: TypeAlias = Any
CPBufferContainer: TypeAlias = Sequence[CPBuffer] | Mapping[str, CPBuffer]
CPBufferSeqDims: TypeAlias = Sequence[int] | Mapping[str, int]


def _context_parallel_shard(
    mesh: DeviceMesh,
    buffers: CPBufferContainer,
    seq_dims: CPBufferSeqDims,
    load_balancer: _LoadBalancer | None = None,
) -> Any:
    global _dispatch_mode
    _dispatch_mode = _DispatchMode.MODULE_WRAPPER
    _cp_options.enable_load_balance = load_balancer is not None
    if len(buffers) != len(seq_dims):
        raise ValueError("seq_dims must have the same number of elements as buffers")
    flat_buffers, spec = tree_flatten(
        buffers, is_leaf=lambda value: not isinstance(value, (list, tuple, dict))
    )
    flat_seq_dims, _ = tree_flatten(seq_dims)
    if len(flat_buffers) != len(flat_seq_dims):
        raise ValueError("seq_dims must have the pytree structure as buffers")
    if not flat_buffers:
        return tree_unflatten([], spec)
    def buffer_device(buffer: Any) -> Any:
        if isinstance(buffer, tp.Tensor):
            return buffer.device
        if isinstance(buffer, BlockMask):
            return buffer.kv_num_blocks.device
        return None

    first_device = buffer_device(flat_buffers[0])
    for buffer in flat_buffers:
        if buffer_device(buffer) != first_device:
            raise AssertionError("All buffers must be on the same device")
    return tree_unflatten(
        _context_parallel_buffers(mesh, flat_buffers, flat_seq_dims, load_balancer), spec
    )


def _enable_context_parallel_dispatcher() -> None:
    _enable_cp_dtensor_dispatcher()


def _disable_context_parallel_dispatcher() -> None:
    _disable_cp_dtensor_dispatcher()


@contextlib.contextmanager
@tp.no_grad()
def context_parallel(
    mesh: DeviceMesh,
    *,
    buffers: list[tp.Tensor] | None = None,
    buffer_seq_dims: list[int] | None = None,
    no_restore_buffers: set[tp.Tensor] | None = None,
) -> Generator[None, None, None]:
    global _dispatch_mode
    _dispatch_mode = _DispatchMode.MONKEY_PATCH
    buffers = [] if buffers is None else buffers
    buffer_seq_dims = [] if buffer_seq_dims is None else buffer_seq_dims
    no_restore_buffers = set() if no_restore_buffers is None else no_restore_buffers
    if len(buffers) != len(buffer_seq_dims):
        raise ValueError("seq_dims must have the same number of elements as buffers")
    for buffer in no_restore_buffers:
        if not any(buffer is item for item in buffers):
            raise ValueError("no_restore_buffers must be a subset of buffers")
    original_buffers = [
        None if any(buffer is item for item in no_restore_buffers) else buffer.clone()
        for buffer in buffers
    ]
    if not buffers:
        raise ValueError("context_parallel requires at least one buffer")
    device = buffers[0].device
    seq_length = buffers[0].shape[buffer_seq_dims[0]]
    cp_world_size = mesh.size()
    old_enable_load_balance = _cp_options.enable_load_balance
    load_balancer = _create_default_load_balancer(seq_length, cp_world_size, device)
    _cp_options.enable_load_balance = load_balancer is not None
    shards = _context_parallel_buffers(mesh, buffers, buffer_seq_dims, load_balancer)
    for buffer, shard in zip(buffers, shards):
        if not isinstance(shard, tp.Tensor):
            raise AssertionError("ContextParallel only supports Tensor")
        shard = shard.clone()
        buffer.resize_(shard.shape)
        buffer.copy_(shard)
    _enable_context_parallel_dispatcher_impl(seq_dim=2, mesh=mesh)
    try:
        yield
    finally:
        _disable_context_parallel_dispatcher_impl()
        _cp_options.enable_load_balance = old_enable_load_balance
        for buffer, original_buffer in zip(buffers, original_buffers, strict=True):
            if original_buffer is not None:
                buffer.resize_(original_buffer.shape)
                buffer.copy_(original_buffer)


@tp.no_grad()
def context_parallel_unshard(
    mesh: DeviceMesh,
    buffers: list[tp.Tensor],
    seq_dims: list[int],
    load_balancer: _LoadBalancer | None = None,
) -> list[tp.Tensor]:
    if len(buffers) != len(seq_dims):
        raise ValueError("seq_dims must have the same number of elements as buffers")
    if not buffers:
        return []
    cp_world_size = mesh.size()
    seq_length = buffers[0].shape[seq_dims[0]] * cp_world_size
    load_balancer = load_balancer or _create_default_load_balancer(
        seq_length, cp_world_size, buffers[0].device
    )
    restore_indices = load_balancer._generate_indices(restore=True) if load_balancer else None
    if not (restore_indices is None or restore_indices.ndim == 2):
        raise AssertionError(
            "load balance restore index expects shape (1, seq_len) or (B, seq_len) "
            f"but got {restore_indices.shape}."
        )
    unsharded_buffers = []
    for buffer, dim in zip(buffers, seq_dims):
        unsharded = _maybe_wait(ft_c.all_gather_single(buffer.contiguous(), dim, mesh))
        if restore_indices is not None:
            idx_batch_size = restore_indices.size(0)
            data_batch_size = unsharded.size(0)
            if idx_batch_size != 1 and idx_batch_size != data_batch_size:
                raise ValueError(
                    "Cannot restore buffer: "
                    f"restore_indices has shape {restore_indices.shape}, "
                    f"but unsharded buffer has shape {unsharded.shape}."
                )
            for index in range(data_batch_size):
                restore_index = restore_indices[0] if idx_batch_size == 1 else restore_indices[index]
                unsharded[index] = tp.index_select(unsharded[index], dim - 1, restore_index)
        unsharded_buffers.append(unsharded)
    return unsharded_buffers


def set_rotate_method(rotate_method: str) -> None:
    logger.info("all-to-all rotation is intended for tensor attention paths")
    if rotate_method == "allgather":
        _cp_options.rotate_method = _RotateMethod.ALL_GATHER
    elif rotate_method == "alltoall":
        _cp_options.rotate_method = _RotateMethod.ALL_TO_ALL
    else:
        raise NotImplementedError(
            "Context Parallel does not support using "
            f"{rotate_method} for key-value shard rotation"
        )
