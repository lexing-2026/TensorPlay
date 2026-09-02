from __future__ import annotations

import contextlib
import math
import os
import threading
from collections.abc import Callable, Generator, Sequence
from datetime import timedelta
from enum import Enum
from typing import Any, Literal

import tensorplay as tp
from tensorplay.distributed import distributed_core as dist
from tensorplay.distributed import _functional_collectives as funcol

__all__ = [
    "empty",
    "is_symm_mem_tensor",
    "rendezvous",
    "is_nvshmem_available",
    "set_backend",
    "get_backend",
    "get",
    "set_signal_pad_size",
    "get_signal_pad_size",
    "get_mem_pool",
    "reduce_scatter_offset",
    "all_to_all_nd",
]

_group_enabled: set[str] = set()
_test_enabled = False
_test_groups: set[str] | None = None
_tensor_handles: dict[tuple[int, str], "_SymmetricMemory"] = {}
_symmetric_tensors: set[int] = set()
_workspace: dict[str, tp.Tensor] = {}
_backend: str | None = None
_signal_pad_size = 0
_mem_pools: dict[Any, object] = {}
_backend_streams: dict[int, Any] = {}
_lock = threading.RLock()


def _group_name(group: Any) -> str:
    if isinstance(group, str):
        return group
    name = getattr(group, "group_name", None)
    if name is not None:
        return str(name)
    raise TypeError(f"unsupported group type {type(group)!r}")


def _group_size(group: Any) -> int:
    if group is None:
        try:
            return max(1, int(dist.get_world_size()))
        except Exception:
            return 1
    if isinstance(group, str):
        try:
            return max(1, int(dist.get_world_size(group)))
        except RuntimeError:
            return 1
    size = getattr(group, "size", None)
    return max(1, int(size() if callable(size) else size or 1))


def _group_rank(group: Any) -> int:
    rank = getattr(group, "rank", None)
    if callable(rank):
        return int(rank())
    try:
        return int(dist.get_rank(group))
    except RuntimeError:
        return 0


def enable_symm_mem_for_group(group_name: str) -> None:
    _group_enabled.add(_group_name(group_name))


@contextlib.contextmanager
def _test_mode(group_names: set[str] | None = None) -> Generator[None, None, None]:
    global _test_enabled, _test_groups
    previous = (_test_enabled, _test_groups)
    _test_enabled, _test_groups = True, group_names
    try:
        yield
    finally:
        _test_enabled, _test_groups = previous


def is_symm_mem_enabled_for_group(group_name: str) -> bool:
    name = _group_name(group_name)
    if _test_enabled:
        return _test_groups is None or name in _test_groups
    return name in _group_enabled or any(group == name for _, group in _tensor_handles)


class _SymmetricMemory:
    signal_pad_size = 0

    def __init__(self, tensor: tp.Tensor, group_name: str) -> None:
        self.tensor = tensor
        self.group_name = group_name
        self.world_size = _group_size(group_name)
        self.rank = _group_rank(None)
        self.device = tensor.device
        self.offset = 0
        self.buffer_size = int(tensor.numel()) * int(tensor.element_size())
        self._signals: dict[int, int] = {}

    def barrier(self, channel: int = 0) -> None:
        del channel
        if self.world_size > 1:
            dist.barrier(group=self.group_name)

    def get_buffer(self, peer: int, shape: Sequence[int], dtype: Any, storage_offset: int = 0) -> tp.Tensor:
        if peer < 0 or peer >= self.world_size:
            raise ValueError("invalid peer")
        shape = tuple(int(value) for value in shape)
        if any(value < 0 for value in shape):
            raise ValueError("buffer shape must be non-negative")
        if dtype is not None and dtype != self.tensor.dtype:
            raise ValueError("buffer dtype must match the rendezvous tensor")
        size = math.prod(shape)
        if storage_offset < 0 or storage_offset + size > self.tensor.numel():
            raise ValueError("buffer range exceeds the rendezvous tensor")
        view = self.tensor.view(-1)[storage_offset:storage_offset + size]
        return view.view(shape) if shape else view.reshape(())

    def get_remote_tensor(self, peer: int, shape: Sequence[int], dtype: Any) -> tp.Tensor:
        return self.get_buffer(peer, shape, dtype)

    def get_backend(self) -> str | None:
        return get_backend(self.device)


def get_symm_mem_workspace(group_name: str, min_size: int) -> _SymmetricMemory:
    name = _group_name(group_name)
    tensor = _workspace.get(name)
    current = 0 if tensor is None else int(tensor.numel()) * int(tensor.element_size())
    if tensor is None or current < min_size:
        dtype = getattr(tp, "uint8", None)
        tensor = tp.empty((max(1, int(min_size)),), dtype=dtype, device=getattr(tp, "device", lambda x: "cpu")("cpu"))
        _workspace[name] = tensor
        _symmetric_tensors.add(id(tensor))
    return rendezvous(tensor, name)


def _get_backend_stream(priority: int = 0) -> Any:
    return _backend_streams.setdefault(priority, None)


def _copy_into(dst: tp.Tensor, src: tp.Tensor) -> None:
    if tuple(dst.shape) != tuple(src.shape):
        dst.copy_(src.reshape(dst.shape))
    else:
        dst.copy_(src)


def _all_gather(tensor: tp.Tensor, gather_dim: int, group_name: str) -> tp.Tensor:
    size = _group_size(group_name)
    if size == 1:
        return tensor.clone()
    return funcol.all_gather_tensor(tensor, gather_dim=gather_dim, group=group_name)


def _pipelined_multi_all_gather_and_consume(shard: list[tp.Tensor], shard_consumer: Callable[[list[tp.Tensor], int], None], ag_out: list[tp.Tensor], group_name: str, ag_out_needed: bool = True) -> None:
    for source_index, (local, output) in enumerate(zip(shard, ag_out)):
        gathered = _all_gather(local, 0, group_name)
        _copy_into(output, gathered)
        pieces = list(gathered.chunk(_group_size(group_name), dim=0))
        for rank, piece in enumerate(pieces):
            values = [piece] if len(shard) == 1 else [item.chunk(_group_size(group_name), dim=0)[rank] for item in ag_out]
            shard_consumer(values, rank)
        if not ag_out_needed:
            del source_index


def _pipelined_all_gather_and_consume(shard: tp.Tensor, shard_consumer: Callable[[tp.Tensor, int], None], ag_out: tp.Tensor, group_name: str, ag_out_needed: bool = True) -> None:
    def consume(values: list[tp.Tensor], rank: int) -> None:
        shard_consumer(values[0], rank)
    _pipelined_multi_all_gather_and_consume([shard], consume, [ag_out], group_name, ag_out_needed)


def _pipelined_produce_and_all2all(chunk_producer: Callable[[int, tp.Tensor], None], output: tp.Tensor, group_name: str, out_chunk_dim: int = 0) -> None:
    chunks = list(output.chunk(_group_size(group_name), dim=out_chunk_dim))
    for rank, chunk in enumerate(chunks):
        chunk_producer(rank, chunk)


def reduce_partials(partials: tp.Tensor, *, dim: int, reduce_op: str, output_dtype: Any, group_size: int) -> tp.Tensor:
    if reduce_op == "sum":
        return partials.sum(dim=dim).to(dtype=output_dtype)
    if reduce_op == "avg":
        return (partials.sum(dim=dim) / group_size).to(dtype=output_dtype)
    raise ValueError("reduce_op must be sum or avg")


def triton_reduce_partials_first_dim(partials: tp.Tensor, *, reduce_op: str, output_dtype: Any, group_size: int) -> tp.Tensor | None:
    if partials.dim() < 1 or int(partials.shape[0]) != group_size:
        return None
    return reduce_partials(partials, dim=0, reduce_op=reduce_op, output_dtype=output_dtype, group_size=group_size)


class _ScaleMode(Enum):
    UNSCALED = "unscaled"
    TENSOR_WISE = "tensor-wise"
    ROW_WISE_SHARDED = "row-wise-sharded"
    ROW_WISE_REPLICATED = "row-wise-replicated"


def _check_and_verify_fp8_all_gather_scale_mode(shard: tp.Tensor, scale: tp.Tensor | None, gather_dim: int, group_size: int) -> _ScaleMode:
    if scale is None:
        return _ScaleMode.UNSCALED
    if scale.numel() == 1:
        return _ScaleMode.TENSOR_WISE
    expected = list(shard.shape)
    expected[gather_dim] *= group_size
    if tuple(scale.shape[:-1]) == tuple(shard.shape[:-1]) and scale.shape[-1] == 1:
        return _ScaleMode.ROW_WISE_SHARDED
    if tuple(scale.shape[:-1]) == tuple(expected[:-1]):
        return _ScaleMode.ROW_WISE_REPLICATED
    raise ValueError("scale shape is incompatible with the gathered matrix")


def _fused_all_gather_matmul_impl(mm_out_op: Any, A_shard: tp.Tensor, Bs: list[tp.Tensor], A_scale: tp.Tensor | None, kwargs_list: list[dict[str, Any]], out_dtypes: list[Any], gather_dim: int, group_name: str, return_A: bool) -> tuple[tp.Tensor | None, list[tp.Tensor]]:
    del mm_out_op, kwargs_list, A_scale
    A = _all_gather(A_shard, gather_dim, group_name)
    result = [(A @ B).to(dtype=out_dtype or B.dtype) for B, out_dtype in zip(Bs, out_dtypes)]
    return (A if return_A else None), result


def _pipelined_all_gather_and_consume_last_dim(*args: Any, **kwargs: Any) -> None:
    return _pipelined_all_gather_and_consume(*args, **kwargs)


def _fused_all_gather_matmul_last_gather_dim_impl(*args: Any, **kwargs: Any) -> tuple[tp.Tensor | None, list[tp.Tensor]]:
    return _fused_all_gather_matmul_impl(*args, **kwargs)


def _fused_all_gather_matmul_fallback(A_shard: tp.Tensor, Bs: list[tp.Tensor], gather_dim: int, group_name: str, *, return_A: bool = True) -> tuple[tp.Tensor | None, list[tp.Tensor]]:
    return _fused_all_gather_matmul_impl(None, A_shard, Bs, None, [{} for _ in Bs], [B.dtype for B in Bs], gather_dim, group_name, return_A)


def _fused_all_gather_matmul(A_shard: tp.Tensor, Bs: list[tp.Tensor], gather_dim: int, group_name: str, *, return_A: bool = True) -> tuple[tp.Tensor | None, list[tp.Tensor]]:
    return _fused_all_gather_matmul_fallback(A_shard, Bs, gather_dim, group_name, return_A=return_A)


def _should_use_fused_all_gather_matmul_native(A_shard: tp.Tensor, Bs: list[tp.Tensor], gather_dim: int, group_name: str) -> bool:
    return False


def _fused_all_gather_matmul_native(A_shard: tp.Tensor, B: tp.Tensor, group_name: str) -> tuple[tp.Tensor, tp.Tensor]:
    A = _all_gather(A_shard, 0, group_name)
    return A, A @ B


def _should_use_multimem_all_gather_matmul(A_shard: tp.Tensor, gather_dim: int, group_name: str, return_A: bool) -> bool:
    return False


def _multimem_all_gather_matmul(A_shard: tp.Tensor, Bs: list[tp.Tensor], group_name: str) -> list[tp.Tensor]:
    A = _all_gather(A_shard, 0, group_name)
    return [A @ B for B in Bs]


def _scaled_mm(A: tp.Tensor, B: tp.Tensor, scale_a: tp.Tensor | None, scale_b: tp.Tensor | None, bias: tp.Tensor | None, result_scale: tp.Tensor | None, out_dtype: Any | None) -> tp.Tensor:
    if scale_a is not None:
        A = A * scale_a
    if scale_b is not None:
        B = B * scale_b
    result = A @ B
    if bias is not None:
        result = result + bias
    if result_scale is not None:
        result = result * result_scale
    return result.to(dtype=out_dtype) if out_dtype is not None else result


def _fused_all_gather_scaled_matmul_fallback(A_shard: tp.Tensor, Bs: list[tp.Tensor], A_scale: tp.Tensor, B_scales: list[tp.Tensor], gather_dim: int, group_name: str, biases: list[tp.Tensor | None], result_scales: list[tp.Tensor | None], out_dtypes: list[Any | None], use_fast_accum: list[bool]) -> tuple[tp.Tensor, list[tp.Tensor]]:
    del use_fast_accum
    A = _all_gather(A_shard, gather_dim, group_name)
    mode = _check_and_verify_fp8_all_gather_scale_mode(A_shard, A_scale, gather_dim, _group_size(group_name))
    if mode is _ScaleMode.ROW_WISE_SHARDED:
        A_scale = _all_gather(A_scale, gather_dim, group_name)
    result = [_scaled_mm(A, B, A_scale, b_scale, bias, result_scale, dtype) for B, b_scale, bias, result_scale, dtype in zip(Bs, B_scales, biases, result_scales, out_dtypes)]
    return A, result


def _fused_all_gather_scaled_matmul(*args: Any, **kwargs: Any) -> tuple[tp.Tensor, list[tp.Tensor]]:
    return _fused_all_gather_scaled_matmul_fallback(*args, **kwargs)


def make_contiguous_for_perm(t: tp.Tensor, perm: list[int]) -> tp.Tensor:
    inverse = [0] * len(perm)
    for index, value in enumerate(perm):
        inverse[value] = index
    return t.permute(perm).contiguous().permute(inverse)


def restride_A_shard_for_fused_all_gather_matmul(t: tp.Tensor, gather_dim: int) -> tp.Tensor:
    perm = list(range(t.dim()))
    perm.insert(0, perm.pop(gather_dim))
    return make_contiguous_for_perm(t, perm)


def _fused_matmul_reduce_scatter(A: tp.Tensor, B: tp.Tensor, reduce_op: str, scatter_dim: int, group_name: str) -> tp.Tensor:
    return _fused_matmul_reduce_scatter_fallback(A, B, reduce_op, scatter_dim, group_name)


def _fused_matmul_reduce_scatter_fallback(A: tp.Tensor, B: tp.Tensor, reduce_op: str, scatter_dim: int, group_name: str) -> tp.Tensor:
    result = A @ B
    if _group_size(group_name) == 1:
        return result
    return funcol.reduce_scatter_tensor(result, reduce_op, group=group_name, scatter_dim=scatter_dim)


def _fused_matmul_reduce_scatter_impl(mm_out_op: Any, A: tp.Tensor, B: tp.Tensor, kwargs: dict[str, Any], out_dtype: Any, reduce_op: str, scatter_dim: int, group_name: str) -> tp.Tensor:
    del mm_out_op, kwargs
    return _fused_matmul_reduce_scatter_fallback(A, B, reduce_op, scatter_dim, group_name).to(dtype=out_dtype) if out_dtype is not None else _fused_matmul_reduce_scatter_fallback(A, B, reduce_op, scatter_dim, group_name)


def _fused_scaled_matmul_reduce_scatter(A: tp.Tensor, B: tp.Tensor, A_scale: tp.Tensor, B_scale: tp.Tensor, reduce_op: str, orig_scatter_dim: int, scatter_dim_after_maybe_reshape: int, group_name: str, output_shape: list[int], bias: tp.Tensor | None = None, result_scale: tp.Tensor | None = None, out_dtype: Any | None = None, use_fast_accum: bool = False) -> tp.Tensor:
    del scatter_dim_after_maybe_reshape, use_fast_accum
    result = _scaled_mm(A, B, A_scale, B_scale, bias, result_scale, out_dtype).view(tuple(output_shape))
    if _group_size(group_name) == 1:
        return result
    return funcol.reduce_scatter_tensor(result, "sum" if reduce_op == "sum" else reduce_op, group=group_name, scatter_dim=orig_scatter_dim)


def _fused_scaled_matmul_reduce_scatter_fallback(*args: Any, **kwargs: Any) -> tp.Tensor:
    return _fused_scaled_matmul_reduce_scatter(*args, **kwargs)


def _fused_scaled_matmul_reduce_scatter_impl(mm_out_op: Any, A: tp.Tensor, B: tp.Tensor, A_scale: tp.Tensor, kwargs: dict[str, Any], out_dtype: Any, reduce_op: str, orig_scatter_dim: int, scatter_dim_after_maybe_reshape: int, group_name: str, output_shape: list[int]) -> tp.Tensor:
    del mm_out_op
    return _fused_scaled_matmul_reduce_scatter(A, B, A_scale, kwargs.get("scale_b"), reduce_op, orig_scatter_dim, scatter_dim_after_maybe_reshape, group_name, output_shape, kwargs.get("bias"), kwargs.get("scale_result"), out_dtype, kwargs.get("use_fast_accum", False))


def restride_A_for_fused_matmul_reduce_scatter(t: tp.Tensor, scatter_dim: int) -> tp.Tensor:
    perm = list(range(t.dim()))
    perm.insert(0, perm.pop(scatter_dim))
    return make_contiguous_for_perm(t, perm)


def _maybe_convert_scalar_types_to_dtypes(scalar_types: list[Any]) -> list[Any | None]:
    names = ["uint8", "int8", "int16", "int32", "int64", "float16", "float32", "float64", "complex32", "complex64", "complex128", "bool", "qint8", "quint8", "qint32", "bfloat16", "float8_e5m2", "float8_e4m3fn"]
    if any(not isinstance(value, (int, type(None))) for value in scalar_types):
        return scalar_types
    return [None if value is None else getattr(tp, names[value], None) if 0 <= value < len(names) else (_ for _ in ()).throw(ValueError(f"unrecognized scalar type {value}")) for value in scalar_types]


class Work:
    def __init__(self) -> None:
        self._complete = True

    def wait(self, timeout: timedelta = timedelta(seconds=0)) -> bool:
        del timeout
        return self._complete


def _low_contention_all_gather_meta(tensor: tp.Tensor, group_name: str) -> tuple[int, ...]:
    return (int(_group_size(group_name)), *tuple(tensor.shape))


def _low_contention_all_gather(tensor: tp.Tensor, group_name: str) -> tp.Tensor:
    return _all_gather(tensor, 0, group_name)


def _require_multicast(device_index: int, op_name: str) -> None:
    del device_index
    raise RuntimeError(f"{op_name} requires multicast support")


def _lc_ag_out_shape(tensor: tp.Tensor, world_size: int) -> tuple[int, ...]:
    return (world_size, *tuple(tensor.shape))


def _check_lc_ag_out(tensor: tp.Tensor, output: tp.Tensor, group_name: str) -> None:
    expected = _lc_ag_out_shape(tensor, _group_size(group_name))
    if tuple(output.shape) != expected:
        raise ValueError(f"output shape must be {expected}")


def _check_lc_signal_pad_capacity(symm_mem: _SymmetricMemory) -> None:
    if get_signal_pad_size() and get_signal_pad_size() < symm_mem.world_size * 4:
        raise ValueError("signal pad is too small for the group")


def _low_contention_all_gather_ce_multicast_meta(tensor: tp.Tensor, group_name: str) -> tuple[int, ...]:
    return _low_contention_all_gather_meta(tensor, group_name)


def _low_contention_all_gather_ce_multicast(tensor: tp.Tensor, group_name: str) -> tp.Tensor:
    return _low_contention_all_gather(tensor, group_name)


def _low_contention_all_gather_ce_multicast_out_meta(tensor: tp.Tensor, group_name: str, output: tp.Tensor) -> tuple[int, ...]:
    del output
    return _low_contention_all_gather_meta(tensor, group_name)


def _low_contention_all_gather_ce_multicast_out(tensor: tp.Tensor, group_name: str, output: tp.Tensor) -> tp.Tensor:
    _check_lc_ag_out(tensor, output, group_name)
    _copy_into(output, _low_contention_all_gather(tensor, group_name))
    return output


def _low_contention_all_gather_ce_multicast_impl(tensor: tp.Tensor, group_name: str) -> tp.Tensor:
    return _low_contention_all_gather_ce_multicast(tensor, group_name)


def _low_contention_reduce_scatter_meta(tensor: tp.Tensor, reduce_op: str, group_name: str) -> tuple[int, ...]:
    shape = list(tensor.shape)
    shape[0] //= _group_size(group_name)
    return tuple(shape)


def _low_contention_reduce_scatter_with_symm_mem_input(tensor: tp.Tensor, reduce_op: str, group_name: str) -> tp.Tensor:
    return _low_contention_reduce_scatter(tensor, reduce_op, group_name)


def _low_contention_reduce_scatter_with_workspace(tensor: tp.Tensor, reduce_op: str, group_name: str) -> tp.Tensor:
    return _low_contention_reduce_scatter(tensor, reduce_op, group_name)


def _low_contention_reduce_scatter(tensor: tp.Tensor, reduce_op: str, group_name: str) -> tp.Tensor:
    if _group_size(group_name) == 1:
        return tensor.clone()
    return funcol.reduce_scatter_tensor(tensor, reduce_op, group=group_name, scatter_dim=0)


def _all_to_all_vdev_2d_meta(input: tp.Tensor, out: tp.Tensor, scatter_dim: int, gather_dim: int, group_name: str) -> tuple[int, ...]:
    del input, scatter_dim, gather_dim, group_name
    return tuple(out.shape)


def _all_to_all_vdev_2d_offset_meta(input: tp.Tensor, out: tp.Tensor, scatter_dim: int, gather_dim: int, group_name: str, offset: int) -> tuple[int, ...]:
    del input, scatter_dim, gather_dim, group_name, offset
    return tuple(out.shape)


def _should_use_implicit_mempool() -> bool:
    return os.environ.get("TP_SYMM_MEM_IMPLICIT_POOL", "1") == "1"


def empty(*size: Any, dtype: Any | None = None, device: Any | None = None) -> tp.Tensor:
    shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], Sequence) else tuple(size)
    tensor = tp.empty(shape, dtype=dtype, device=device)
    _symmetric_tensors.add(id(tensor))
    return tensor


def rendezvous(tensor: tp.Tensor, group: Any) -> _SymmetricMemory:
    if not is_symm_mem_tensor(tensor):
        raise ValueError("tensor was not allocated by symmetric memory")
    name = _group_name(group)
    if _group_size(group) > 1:
        metadata = (tuple(tensor.shape), str(tensor.dtype), str(tensor.device))
        gathered = [None] * _group_size(group)
        dist.all_gather_object(gathered, metadata, group=group)
        if any(value != metadata for value in gathered):
            raise ValueError("all ranks must rendezvous with matching tensor metadata")
    with _lock:
        return _tensor_handles.setdefault((id(tensor), name), _SymmetricMemory(tensor, name))


def is_nvshmem_available() -> bool:
    return False


def set_backend(name: Literal["NVSHMEM", "CUDA", "NCCL"]) -> None:
    global _backend
    if name not in {"NVSHMEM", "CUDA", "NCCL"}:
        raise ValueError(f"unsupported symmetric-memory backend {name!r}")
    if _backend is not None and _backend != name and _symmetric_tensors:
        raise RuntimeError("backend cannot change after allocation")
    _backend = name


def get_backend(device: Any) -> str | None:
    del device
    return _backend


def get_mempool_allocator(device: Any) -> Any:
    return _mem_pools.setdefault(device, object())


def set_signal_pad_size(size: int) -> None:
    global _signal_pad_size
    if size < 0:
        raise ValueError("signal pad size must be non-negative")
    if _symmetric_tensors and size != _signal_pad_size:
        raise RuntimeError("signal pad size must be set before allocation")
    _signal_pad_size = int(size)
    _SymmetricMemory.signal_pad_size = int(size)


def get_signal_pad_size() -> int:
    return _signal_pad_size


def get_mem_pool(device: Any) -> object:
    return _mem_pools.setdefault(device, object())


def get(dst: tp.Tensor, hdl: _SymmetricMemory, peer: int, offset: int = 0) -> None:
    if peer < 0 or peer >= hdl.world_size or offset < 0:
        raise ValueError("invalid peer or offset")
    size = dst.numel()
    _copy_into(dst, hdl.tensor.view(-1)[offset:offset + size].view(dst.shape))


def put_signal(src: tp.Tensor, hdl: _SymmetricMemory, peer: int) -> None:
    if peer < 0 or peer >= hdl.world_size:
        raise ValueError("invalid peer")
    _copy_into(hdl.tensor.view(src.shape), src)
    hdl._signals[peer] = hdl._signals.get(peer, 0) + 1


def wait_signal(hdl: _SymmetricMemory, peer: int) -> None:
    if peer not in hdl._signals:
        return


def reduce_scatter_offset(input: tp.Tensor, out: list[tp.Tensor], group: Any, *, dim: int, offsets: list[int] | None = None, dst_ranks: list[int] | None = None, red_op: str = "sum") -> None:
    if dim not in (0, 1):
        raise ValueError("dim must be 0 or 1")
    if str(red_op).lower() not in {"sum", "avg", "average"}:
        raise ValueError("red_op must be sum or avg")
    count = _group_size(group)
    extent = int(input.shape[dim])
    if offsets is None:
        offsets = [round((index + 1) * extent / count) for index in range(count)]
    else:
        offsets = [int(value) for value in offsets]
    if len(offsets) != count or not offsets or offsets[-1] != extent:
        raise ValueError("offsets must end at the input extent and match group size")
    if any(stop <= start or stop > extent for start, stop in zip([0, *offsets[:-1]], offsets)):
        raise ValueError("offsets must be strictly increasing within the input extent")
    if dst_ranks is None:
        dst_ranks = list(range(count))
    else:
        dst_ranks = [int(value) for value in dst_ranks]
    if len(dst_ranks) != count or any(rank < 0 or rank >= count for rank in dst_ranks):
        raise ValueError("dst_ranks must contain one valid group rank per block")
    if len(out) != dst_ranks.count(_group_rank(group)):
        raise ValueError("out does not contain one tensor per owned block")

    partials = [tp.empty_like(input) for _ in range(count)]
    if count > 1:
        dist.all_gather(partials, input, group=group)
    else:
        partials[0].copy_(input)
    reduced = partials[0]
    for partial in partials[1:]:
        reduced = reduced + partial
    if str(red_op).lower() in {"avg", "average"} and count > 1:
        reduced = reduced / count

    rank = _group_rank(group)
    output_index = 0
    start = 0
    for index, stop in enumerate(offsets):
        if dst_ranks[index] == rank:
            block = reduced.narrow(dim, start, stop - start)
            _copy_into(out[output_index], block)
            output_index += 1
        start = stop


def is_symm_mem_tensor(tensor: tp.Tensor) -> bool:
    return id(tensor) in _symmetric_tensors


def all_to_all_nd(input: tp.Tensor, out: tp.Tensor, scatter_dim: int, gather_dim: int, *, group: Any) -> None:
    if (scatter_dim, gather_dim) not in {(0, 1), (1, 0)}:
        raise ValueError("only (0, 1) and (1, 0) dimension exchanges are supported")
    count = _group_size(group)
    if count == 1:
        _copy_into(out, input)
        return
    if input.dim() < 2:
        raise ValueError("all_to_all_nd requires at least two dimensions")
    if int(input.shape[scatter_dim]) % count != 0:
        raise ValueError("scatter dimension must be divisible by group size")
    chunks = list(input.chunk(count, dim=scatter_dim))
    received = [tp.empty_like(chunks[0]) for _ in range(count)]
    dist.all_to_all(received, chunks, group=group)
    result = tp.cat(received, dim=gather_dim)
    if tuple(out.shape) != tuple(result.shape):
        raise ValueError(f"out shape must be {tuple(result.shape)}")
    _copy_into(out, result)


def _get_remote_tensors_default(local: tp.Tensor, group_name: str) -> tuple[tp.Tensor, ...]:
    handle = rendezvous(local, group_name)
    return tuple(handle.get_remote_tensor(rank, local.shape, local.dtype) for rank in range(handle.world_size))


def _get_remote_tensors_meta(local: tp.Tensor, group_name: str) -> tuple[tp.Tensor, ...]:
    return tuple(local.clone() for _ in range(_group_size(group_name)))
