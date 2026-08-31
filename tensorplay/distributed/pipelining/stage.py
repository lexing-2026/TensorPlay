"""Pipeline stage execution and metadata management."""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable

import tensorplay as tp

from ._utils import (
    PipeliningMetadataError,
    _StageBackwardMeta,
    _StageForwardMeta,
    _StageMeta,
    extract_tensor_meta,
    extract_tensor_metas,
    validate_tensors_metadata,
)

__all__ = ["PipelineStage", "build_stage"]


def _normalize_model_output_as_tuple(output: Any) -> tuple[Any, ...]:
    if isinstance(output, list):
        return tuple(output)
    return output if isinstance(output, tuple) else (output,)


@dataclass
class _RecvInfo:
    input_name: str
    source: int | None
    buffer: Any
    tensor_meta: Any
    is_root_arg: bool = False

    def __repr__(self) -> str:
        return f"_RecvInfo(input={self.input_name!r}, source={self.source!r}, root_arg={self.is_root_arg})"


def _build_p2p_direction_groups(group: Any) -> tuple[Any, Any]:
    return group, group


class _PipelineStageBase(ABC):
    def __init__(self, submodule: Any, stage_index: int, num_stages: int, device: Any = None, group: Any = None, dw_builder: Callable[[], Callable[..., None]] | None = None) -> None:
        if stage_index < 0 or stage_index >= num_stages:
            raise ValueError("stage_index is outside the pipeline")
        self.submod = submodule
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.device = device
        self.group = group
        self.dw_builder = dw_builder
        self.group_rank = stage_index
        self.group_size = num_stages
        self._has_backward = False
        self.fwd_cache: dict[int, tuple[Any, tuple[Any, ...]]] = {}
        self.bwd_cache: dict[int, Any] = {}
        self.output_chunks: list[Any] = []
        self.args_recv_info: dict[int, tuple[_RecvInfo, ...]] = {}
        self.act_send_info: dict[int, list[Any]] = {}
        self.grad_recv_info: dict[int, tuple[_RecvInfo, ...]] = {}
        self.grad_send_info: list[Any] | None = None
        self.chunks: int | None = None
        self._stage_meta = _StageMeta()
        self._input_chunks: dict[int, tuple[Any, ...]] = {}

    @property
    def has_backward(self) -> bool:
        return self._has_backward

    @has_backward.setter
    def has_backward(self, value: bool) -> None:
        self._has_backward = bool(value)

    @property
    def is_first(self) -> bool:
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        return self.stage_index == self.num_stages - 1

    def _validate_stage_tensors(self, desc: str, expected: tuple[Any, ...] | None, actual: tuple[Any, ...]) -> None:
        if expected is None:
            raise PipeliningMetadataError(f"{desc}: metadata is unavailable")
        validate_tensors_metadata(desc, expected, actual)

    def _check_chunk_id(self, chunk_id: int) -> None:
        if self.chunks is None or chunk_id < 0 or chunk_id >= self.chunks:
            raise RuntimeError("chunk id is outside the configured range")

    def _create_grad_send_info(self, args_recv_info: tuple[_RecvInfo, ...]) -> list[Any]:
        return [item.tensor_meta for item in args_recv_info]

    def _prepare_forward_infra(self, num_microbatches: int, args: Any, kwargs: Any, has_backward: bool) -> None:
        self.chunks = num_microbatches
        self.has_backward = has_backward
        self._stage_meta.forward.input_metas = tuple(meta for meta in (extract_tensor_meta(value) for value in args) if meta is not None)
        self.args_recv_info = {index: tuple(_RecvInfo(str(pos), None, None, extract_tensor_meta(value), True) for pos, value in enumerate(args)) for index in range(num_microbatches)}

    def _prepare_backward_infra(self, num_microbatches: int, loss_fn: Any, target: Any, received_grad_meta: Any, loss_kwargs: Any) -> None:
        del loss_fn, target, loss_kwargs
        self.chunks = num_microbatches
        self.has_backward = True
        self._stage_meta.backward.output_grad_metas = tuple(received_grad_meta or ())

    def _setup_backward_recv_info(self, num_microbatches: int) -> None:
        self.grad_recv_info = {index: tuple() for index in range(num_microbatches)}

    def _create_grad_recv_info(self, act_send_info: Any) -> dict[int, tuple[_RecvInfo, ...]]:
        del act_send_info
        return self.grad_recv_info

    def _resolve_peer_global_rank(self, stage_idx: int) -> int:
        return int(stage_idx)

    def _get_recv_ops(self, recv_infos: Any, group: Any) -> list[Any]:
        del recv_infos, group
        return []

    def set_local_fwd_input(self, prev_stage_outputs: Any, mb_index: int) -> None:
        self._input_chunks[mb_index] = tuple(prev_stage_outputs) if isinstance(prev_stage_outputs, (tuple, list)) else (prev_stage_outputs,)

    def get_local_bwd_output(self, mb_index: int) -> Any:
        return self.bwd_cache.get(mb_index)

    def set_local_bwd_input(self, next_stage_bwd_outputs: Any, mb_index: int) -> None:
        self.bwd_cache[mb_index] = next_stage_bwd_outputs

    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> list[Any]:
        self._check_chunk_id(fwd_chunk_id)
        return []

    def get_bwd_recv_ops(self, bwd_chunk_id: int) -> list[Any]:
        self._check_chunk_id(bwd_chunk_id)
        return []

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> list[Any]:
        self._check_chunk_id(fwd_chunk_id)
        return []

    def _get_grad_send_meta(self, input_idx: int) -> Any:
        del input_idx
        return None

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> list[Any]:
        self._check_chunk_id(bwd_chunk_id)
        return []

    def clear_runtime_states(self) -> None:
        self.fwd_cache.clear()
        self.bwd_cache.clear()
        self.output_chunks.clear()
        self._input_chunks.clear()

    def _map_tensor_from_recv_info(self, recv_infos: Any) -> tuple[Any, ...]:
        return tuple(item.buffer for item in recv_infos)

    def _retrieve_recv_activations(self, fwd_chunk_id: int) -> tuple[Any, ...]:
        return self._map_tensor_from_recv_info(self.args_recv_info.get(fwd_chunk_id, ()))

    def _retrieve_recv_grads(self, bwd_chunk_id: int) -> tuple[Any, ...]:
        return self._map_tensor_from_recv_info(self.grad_recv_info.get(bwd_chunk_id, ()))

    def forward_maybe_with_nosync(self, *args: Any, **kwargs: Any) -> Any:
        return self.submod(*args, **kwargs)

    def scale_grads(self, grad_scale_factor: float) -> None:
        for param in self.submod.parameters():
            if getattr(param, "grad", None) is not None:
                param.grad.div_(grad_scale_factor)

    def backward_maybe_with_nosync(self, backward_type: Any, bwd_kwargs: dict[str, Any], last_backward: bool = False) -> Any:
        del backward_type, last_backward
        return self.backward_one_chunk(**bwd_kwargs)

    def forward_one_chunk(self, fwd_chunk_id: int, args: tuple[Any, ...], kwargs: dict[str, Any], save_forward_output: bool = True) -> Any:
        self._check_chunk_id(fwd_chunk_id)
        output = self.forward_maybe_with_nosync(*args, **kwargs)
        output_tuple = _normalize_model_output_as_tuple(output)
        self.fwd_cache[fwd_chunk_id] = (output, output_tuple)
        if save_forward_output:
            self.output_chunks.append(output)
        self._stage_meta.forward.output_metas = tuple(meta for meta in (extract_tensor_meta(value) for value in output_tuple) if meta is not None)
        return output

    def backward_one_chunk(self, bwd_chunk_id: int, loss: Any = None, full_backward: bool = True, last_backward: bool = False) -> Any:
        del full_backward, last_backward
        if loss is None:
            loss = self.fwd_cache[bwd_chunk_id][0]
        if hasattr(loss, "backward"):
            loss.backward()
        self.bwd_cache[bwd_chunk_id] = None
        return None

    def backward_weight_one_chunk(self, bwd_chunk_id: int, last_backward: bool = False) -> Any:
        return self.backward_one_chunk(bwd_chunk_id, last_backward=last_backward)

    def _get_init_p2p_neighbors_ops(self) -> list[Any]:
        return []

    def perform_reduce_grad(self, grad_scale_factor: float) -> None:
        self.scale_grads(grad_scale_factor)


class _PipelineStage(_PipelineStageBase):
    def __init__(self, stage_module: Any, stage_index: int, pipe_info: Any, device: Any = None, group: Any = None) -> None:
        super().__init__(stage_module, stage_index, pipe_info.num_stages, device, group)
        self.pipe_info = pipe_info

    def _move_submod_to_device(self) -> None:
        if self.device is not None and hasattr(self.submod, "to"):
            self.submod.to(self.device)

    def get_stage_index_of_submod(self, submod_name: str) -> int:
        del submod_name
        return self.stage_index

    def _create_act_recv_info(self) -> None:
        return None

    def find_dst_rank(self, user: Any) -> int:
        del user
        return self.stage_index + 1

    def _create_act_send_info(self) -> None:
        return None

    def _get_output_node(self) -> Any:
        return getattr(self.pipe_info, "graph", None)


def build_stage(stage_module: Any, stage_index: int, pipe_info: Any, device: Any = None, group: Any = None) -> _PipelineStage:
    return _PipelineStage(stage_module, stage_index, pipe_info, device, group)


class PipelineStage(_PipelineStageBase):
    def __init__(self, submodule: Any, stage_index: int, num_stages: int, device: Any = None, input_args: tuple[Any, ...] | None = None, output_args: Any = None, output_grads: Any = None, input_grads: Any = None, group: Any = None, dw_builder: Callable[[], Callable[..., None]] | None = None, get_mesh: Any = None) -> None:
        del output_args, output_grads, input_grads, get_mesh
        super().__init__(submodule, stage_index, num_stages, device, group, dw_builder)
        if input_args is not None:
            self._prepare_forward_infra(1, input_args, {}, False)

    def _recv_meta(self, src_stage: int) -> Any:
        return self.args_recv_info.get(src_stage)

    def _send_meta(self, meta: Any, dst_stage: int) -> None:
        self.act_send_info[dst_stage] = [meta]

    def _is_same_rank(self, other_stage: int) -> bool:
        return int(other_stage) == self.stage_index

    def _warmup_forward_vote(self, has_backward: bool, received_acc: Any) -> bool:
        return bool(has_backward and received_acc)

    def _warmup_backward_result(self, received_result: Any) -> Any:
        return received_result

    def _compute_outputs(self, module: Any) -> Any:
        return module()

    def _compute_input_grads(self, outputs: Any, all_fwd_inputs: Any, grad_outputs: Any) -> tuple[Any, ...]:
        grads = tp.autograd.grad(outputs, all_fwd_inputs, grad_outputs=grad_outputs, allow_unused=True)
        return tuple(grads)

    def _to_tensor(self, arg: Any) -> Any:
        return arg if isinstance(arg, tp.Tensor) else tp.tensor(arg)

    def _ones_from_metadata(self, meta: Any) -> Any:
        return tp.ones(meta.shape, dtype=meta.dtype)

    def _pre_metadata_inference_backup(self) -> None:
        self._metadata_backup = self._stage_meta

    def _forward_metadata_inference(self, args: Any, kwargs: Any, has_backward: bool) -> Any:
        return self.forward_one_chunk(0, tuple(args), dict(kwargs), save_forward_output=has_backward)

    def _backward_metadata_inference(self, loss_fn: Any, target: Any, received_grad_meta: Any, loss_kwargs: Any) -> None:
        del loss_fn, target, received_grad_meta, loss_kwargs

    def _post_metadata_inference_cleanup(self) -> None:
        self.clear_runtime_states()

    def _validate_inferred_metadata(self) -> None:
        if not self._stage_meta.forward.output_metas:
            raise PipeliningMetadataError("stage output metadata is empty")

    def _setup_forward_recv_info(self, num_microbatches: int, has_backward: bool) -> None:
        self._prepare_forward_infra(num_microbatches, (), {}, has_backward)

    def _setup_forward_send_info(self) -> None:
        self.act_send_info = {index: [] for index in range(self.chunks or 0)}
