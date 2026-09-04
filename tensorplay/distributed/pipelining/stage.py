"""Pipeline stage execution and metadata management."""

from abc import ABC
from dataclasses import dataclass
import operator
from typing import Any, Callable

import tensorplay as tp
from .. import config as dist_config
from .. import distributed_core as dist

from ._backward import (
    _autograd_grad_for_inputs,
    stage_backward,
    stage_backward_input,
    stage_backward_weight,
)
from ._utils import (
    _MeshCache,
    PipeliningMetadataError,
    _StageBackwardMeta,
    _StageForwardMeta,
    _StageMeta,
    _DTensorMeta,
    _TensorMeta,
    _derive_grad_metas,
    flatten_args,
    _make_tensor_from_meta,
    InferenceMode,
    extract_tensor_meta,
    extract_tensor_metas,
    to_local_if_dtensor,
    validate_static_arg_grad_correspondence,
    validate_tensors_metadata,
)
from ..tensor import DTensor

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

    def __init__(
        self,
        input_name: str,
        source: int | None,
        buffer: Any,
        tensor_meta: Any,
        is_root_arg: bool = False,
    ) -> None:
        self.input_name = input_name
        self.source = source
        self.buffer = buffer
        self.tensor_meta = tensor_meta
        self.is_root_arg = is_root_arg

    def __repr__(self) -> str:
        if self.is_root_arg:
            return f"_RecvInfo(input={self.input_name}, root_arg=True)"
        meta_type = type(self.tensor_meta).__name__ if self.tensor_meta else "None"
        buffer_shape = self.buffer.size() if self.buffer is not None else "None"
        return f"_RecvInfo(input={self.input_name}, source={self.source}, shape={buffer_shape}, meta={meta_type})"


def _build_p2p_direction_groups(group: Any) -> tuple[Any, Any]:
    if not dist.is_initialized():
        return group, group
    parent = group if group is not None else dist._get_default_group()
    if parent.size() <= 1:
        return group, group
    cache = getattr(_build_p2p_direction_groups, "_cache", None)
    if cache is None:
        cache = _build_p2p_direction_groups._cache = {}
    key = id(parent)
    cached = cache.get(key)
    if cached is not None and cached[0] is parent:
        return cached[1], cached[2]
    split_ranks = [list(range(parent.size()))]
    downstream = dist.split_group(
        parent_pg=parent,
        split_ranks=split_ranks,
        group_desc="pipeline_downstream",
    )
    upstream = dist.split_group(
        parent_pg=parent,
        split_ranks=split_ranks,
        group_desc="pipeline_upstream",
    )
    if downstream is dist.GroupMember.NON_GROUP_MEMBER or upstream is dist.GroupMember.NON_GROUP_MEMBER:
        raise RuntimeError("pipeline direction groups must contain the current rank")
    cache[key] = (parent, downstream, upstream)
    return downstream, upstream


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
        self.p2p_per_direction = bool(dist_config.pipeline_per_direction_p2p)
        if self.p2p_per_direction:
            self._downstream_group, self._upstream_group = _build_p2p_direction_groups(group)
        else:
            self._downstream_group = group
            self._upstream_group = group
        try:
            self.group_rank = int(dist.get_rank(group)) if dist.is_initialized() else stage_index
            self.group_size = int(dist.get_world_size(group)) if dist.is_initialized() else num_stages
        except (RuntimeError, ValueError):
            self.group_rank = stage_index
            self.group_size = num_stages
        if self.group_size > num_stages:
            raise ValueError("pipeline group cannot contain more ranks than stages")
        self.stage_index_to_group_rank = {
            index: index % self.group_size for index in range(num_stages)
        }
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
        self._mesh_cache = _MeshCache()
        self._input_chunks: dict[int, tuple[Any, ...]] = {}
        self._forward_inputs: dict[int, tuple[Any, ...]] = {}
        self.backward_state: dict[int, tuple[Any, Any, Any, Any]] = {}
        self.dw_runner: dict[int, Callable[[], Any]] = {}

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
        return [item.source if isinstance(item, _RecvInfo) else None for item in args_recv_info]

    def _prepare_forward_infra(self, num_microbatches: int, args: Any, kwargs: Any, has_backward: bool) -> Any:
        self.chunks = num_microbatches
        self.has_backward = has_backward
        self._stage_meta.forward.input_metas = tuple(meta for meta in (extract_tensor_meta(value) for value in args) if meta is not None)
        self.args_recv_info = {index: tuple(_RecvInfo(str(pos), None, None, extract_tensor_meta(value), True) for pos, value in enumerate(args)) for index in range(num_microbatches)}

    def _prepare_backward_infra(
        self,
        num_microbatches: int,
        loss_fn: Any = None,
        target: Any = None,
        received_grad_meta: Any = None,
        loss_kwargs: Any = None,
    ) -> None:
        del loss_fn, target, loss_kwargs
        self.chunks = num_microbatches
        self.has_backward = True
        self._stage_meta.backward.output_grad_metas = tuple(received_grad_meta or ())
        self.grad_recv_info = {
            index: self._create_grad_recv_info(self.act_send_info)
            for index in range(num_microbatches)
        }
        self.grad_send_info = self._create_grad_send_info(
            self.args_recv_info.get(0, ())
        )

    def _setup_backward_recv_info(self, num_microbatches: int) -> None:
        self.chunks = num_microbatches
        self.grad_recv_info = {
            index: self._create_grad_recv_info(self.act_send_info)
            for index in range(num_microbatches)
        }

    def _create_grad_recv_info(self, act_send_info: Any) -> tuple[_RecvInfo, ...]:
        del act_send_info
        return ()

    def _resolve_peer_global_rank(self, stage_idx: int) -> int:
        peer_group_rank = self.stage_index_to_group_rank[int(stage_idx)]
        if self.group is None:
            return int(peer_group_rank)
        return int(dist.get_global_rank(self.group, peer_group_rank))

    def _get_recv_ops(self, recv_infos: Any, group: Any) -> list[Any]:
        if not dist.is_initialized():
            return []
        process_group = self.group if group is None else group
        operations = []
        for info in recv_infos:
            if not isinstance(info, _RecvInfo) or info.source is None or info.buffer is None:
                continue
            peer_group_rank = self.stage_index_to_group_rank[int(info.source)]
            peer = (
                peer_group_rank
                if process_group is None
                else dist.get_global_rank(process_group, peer_group_rank)
            )
            operations.append(dist.P2POp(dist.irecv, info.buffer, peer, process_group))
        return operations

    def set_local_fwd_input(self, prev_stage_outputs: Any, mb_index: int) -> None:
        values = _normalize_model_output_as_tuple(prev_stage_outputs)
        recv_infos = self.args_recv_info[mb_index]
        if len(recv_infos) != len(values):
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: local forward input count does not match "
                f"the receive metadata ({len(values)} != {len(recv_infos)})"
            )
        if self.is_first:
            raise AssertionError("local forward input is only valid for a non-first stage")
        for info, value in zip(recv_infos, values, strict=True):
            if info.is_root_arg:
                raise AssertionError("local forward input cannot replace a root argument")
            local_value = to_local_if_dtensor(value)
            if isinstance(local_value, tp.Tensor):
                local_value = local_value.detach()
                if (
                    info.tensor_meta is not None
                    and info.tensor_meta.requires_grad
                    and (local_value.is_floating_point() or local_value.is_complex())
                ):
                    local_value.requires_grad_(True)
            info.buffer = local_value
        self._input_chunks[mb_index] = tuple(info.buffer for info in recv_infos)

    def get_local_bwd_output(self, mb_index: int) -> Any:
        if not self.has_backward:
            raise AssertionError("cannot get a backward output without backward enabled")
        if self.is_first:
            raise AssertionError("the first stage has no local backward output")
        self._check_chunk_id(mb_index)
        return self.bwd_cache.pop(mb_index)

    def set_local_bwd_input(self, next_stage_bwd_outputs: Any, mb_index: int) -> None:
        values = next_stage_bwd_outputs
        if not isinstance(values, tuple):
            raise AssertionError(f"expected a tuple of gradients, got {type(values)}")
        if not self.has_backward:
            raise AssertionError("cannot set a backward input without backward enabled")
        if self.is_last:
            raise AssertionError("the last stage has no local backward input")
        recv_infos = self.grad_recv_info[mb_index]
        if len(recv_infos) != len(values):
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: local backward input count does not match "
                f"the receive metadata ({len(values)} != {len(recv_infos)})"
            )
        for info, value in zip(recv_infos, values, strict=True):
            if value is None:
                if info.buffer is not None:
                    info.buffer.zero_()
                continue
            if info.is_root_arg:
                raise AssertionError("local backward input cannot target a root argument")
            info.buffer = to_local_if_dtensor(value)

    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> list[Any]:
        self._check_chunk_id(fwd_chunk_id)
        return self._get_recv_ops(
            self.args_recv_info.get(fwd_chunk_id, ()), self._downstream_group
        )

    def get_bwd_recv_ops(self, bwd_chunk_id: int) -> list[Any]:
        self._check_chunk_id(bwd_chunk_id)
        if not self.has_backward or self.is_last:
            return []
        return self._get_recv_ops(
            self.grad_recv_info.get(bwd_chunk_id, ()), self._upstream_group
        )

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> list[Any]:
        self._check_chunk_id(fwd_chunk_id)
        output = self.fwd_cache[fwd_chunk_id][0]
        values = _normalize_model_output_as_tuple(output)
        operations = []
        for index, value in enumerate(values):
            for destination in self.act_send_info.get(index, ()):
                if destination is None:
                    continue
                value = to_local_if_dtensor(value, detach=True)
                if not isinstance(value, tp.Tensor):
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: activation {index} is not a tensor"
                    )
                peer_group_rank = self.stage_index_to_group_rank[int(destination)]
                peer = (
                    peer_group_rank
                    if self._downstream_group is None
                    else dist.get_global_rank(self._downstream_group, peer_group_rank)
                )
                operations.append(
                    dist.P2POp(dist.isend, value, peer, self._downstream_group)
                )
        return operations

    def _get_grad_send_meta(self, input_idx: int) -> Any:
        input_grads = self._stage_meta.input_grads
        if input_grads is not None and input_idx < len(input_grads):
            return input_grads[input_idx]
        inputs = self._stage_meta.inputs
        if inputs is not None and input_idx < len(inputs):
            meta = inputs[input_idx]
            if meta is not None:
                return _derive_grad_metas((meta,))[0]
        raise PipeliningMetadataError(
            f"Stage {self.stage_index}: backward produced a gradient for input "
            f"{input_idx}, but no gradient metadata is available"
        )

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> list[Any]:
        self._check_chunk_id(bwd_chunk_id)
        if not self.has_backward or self.is_first:
            return []
        if self.grad_send_info is None:
            self.grad_send_info = self._create_grad_send_info(
                self.args_recv_info.get(bwd_chunk_id, ())
            )
        gradients = self.bwd_cache.pop(bwd_chunk_id, ())
        operations = []
        for index, (gradient, destination) in enumerate(
            zip(gradients or (), self.grad_send_info, strict=True)
        ):
            if destination is None:
                if gradient is not None:
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: input {index} has a gradient but "
                        "no previous stage receives it"
                    )
                continue
            grad_meta = self._get_grad_send_meta(index)
            if grad_meta is None:
                if gradient is not None:
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: input {index} produced a gradient "
                        "without gradient metadata"
                    )
                continue
            if gradient is None:
                send_tensor = _make_tensor_from_meta(grad_meta, self.device).zero_()
            else:
                send_tensor = to_local_if_dtensor(gradient, detach=True)
                if not isinstance(send_tensor, tp.Tensor):
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: input {index} gradient is not a tensor"
                    )
            peer_group_rank = self.stage_index_to_group_rank[int(destination)]
            peer = (
                peer_group_rank
                if self._upstream_group is None
                else dist.get_global_rank(self._upstream_group, peer_group_rank)
            )
            operations.append(
                dist.P2POp(dist.isend, send_tensor, peer, self._upstream_group)
            )
        return operations

    def clear_runtime_states(self) -> None:
        self.fwd_cache.clear()
        self.bwd_cache.clear()
        self.output_chunks.clear()
        self._input_chunks.clear()
        self._forward_inputs.clear()
        self.backward_state.clear()
        self.dw_runner.clear()
        for recv_infos in self.args_recv_info.values():
            for info in recv_infos:
                if not info.is_root_arg and isinstance(info.buffer, tp.Tensor):
                    info.buffer.grad = None

    def _map_tensor_from_recv_info(self, recv_infos: Any) -> tuple[Any, ...]:
        values = []
        for item in recv_infos:
            if item.is_root_arg:
                raise PipeliningMetadataError("root arguments are not received tensors")
            values.append(item.buffer)
        return tuple(values)

    def _retrieve_recv_activations(self, fwd_chunk_id: int) -> tuple[Any, ...]:
        recv_infos = self.args_recv_info.get(fwd_chunk_id, ())
        values = []
        for index, info in enumerate(recv_infos):
            if info.is_root_arg:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: root input cannot be received"
                )
            if info.buffer is None or info.tensor_meta is None:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: activation {index} has no receive buffer or metadata"
                )
            effective_requires_grad = bool(
                info.tensor_meta.requires_grad
                and self.has_backward
                and tp.is_grad_enabled()
            )
            if isinstance(info.tensor_meta, _DTensorMeta):
                local = info.buffer
                if not isinstance(local, tp.Tensor):
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: DTensor activation buffer is not a tensor"
                    )
                local = local.detach()
                if effective_requires_grad and (
                    local.is_floating_point() or local.is_complex()
                ):
                    local.requires_grad_(True)
                mesh = self._mesh_cache.get_mesh(info.tensor_meta.mesh_cache_key)
                values.append(
                    DTensor.from_local(
                        local,
                        device_mesh=mesh,
                        placements=info.tensor_meta.placements,
                        shape=info.tensor_meta.global_shape,
                        stride=info.tensor_meta.global_stride,
                        run_check=False,
                    )
                )
            else:
                value = info.buffer
                if not isinstance(value, tp.Tensor):
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: activation {index} is not a tensor"
                    )
                value.requires_grad_(
                    effective_requires_grad
                    and (value.is_floating_point() or value.is_complex())
                )
                values.append(value)
        return tuple(values)

    def _retrieve_recv_grads(self, bwd_chunk_id: int) -> tuple[Any, ...]:
        recv_infos = self.grad_recv_info.get(bwd_chunk_id, ())
        values = []
        for index, info in enumerate(recv_infos):
            if info.is_root_arg:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: root input cannot receive a gradient"
                )
            if info.buffer is None:
                if info.tensor_meta is not None:
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: gradient {index} has metadata but no buffer"
                    )
                values.append(None)
                continue
            if info.tensor_meta is None:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: gradient {index} has a buffer but no metadata"
                )
            if isinstance(info.tensor_meta, _DTensorMeta):
                mesh = self._mesh_cache.get_mesh(info.tensor_meta.mesh_cache_key)
                values.append(
                    DTensor.from_local(
                        info.buffer,
                        device_mesh=mesh,
                        placements=info.tensor_meta.placements,
                        shape=info.tensor_meta.global_shape,
                        stride=info.tensor_meta.global_stride,
                        run_check=False,
                    )
                )
            else:
                values.append(info.buffer)
        return tuple(values)

    def forward_maybe_with_nosync(self, *args: Any, **kwargs: Any) -> Any:
        from ...nn.parallel.distributed import DistributedDataParallel

        if isinstance(self.submod, DistributedDataParallel):
            with self.submod.no_sync():
                return self.submod(*args, **kwargs)
        return self.submod(*args, **kwargs)

    def scale_grads(self, grad_scale_factor: float) -> None:
        for param in self.submod.parameters():
            if getattr(param, "grad", None) is not None:
                param.grad.div_(grad_scale_factor)

    def backward_maybe_with_nosync(self, backward_type: Any, bwd_kwargs: dict[str, Any], last_backward: bool = False) -> Any:
        del last_backward

        fsdp_flags = (
            ("set_is_last_backward", False),
            ("set_reshard_after_backward", False),
            ("set_requires_gradient_sync", False),
        )
        for method_name, value in fsdp_flags:
            method = getattr(self.submod, method_name, None)
            if callable(method):
                method(value)
        if backward_type == "full":
            return stage_backward(
                bwd_kwargs["stage_output"],
                bwd_kwargs["output_grads"],
                bwd_kwargs["input_values"],
            ), None
        if backward_type == "input":
            return stage_backward_input(
                bwd_kwargs["stage_output"],
                bwd_kwargs["output_grads"],
                bwd_kwargs["input_values"],
                self.submod.parameters(),
            )
        if backward_type == "weight":
            return stage_backward_weight(
                self.submod.parameters(),
                bwd_kwargs["param_groups"] or [],
            ), None
        raise RuntimeError(f"unknown backward type {backward_type!r}")

    def forward_one_chunk(self, fwd_chunk_id: int, args: tuple[Any, ...], kwargs: dict[str, Any], save_forward_output: bool = True) -> Any:
        self._check_chunk_id(fwd_chunk_id)
        composite_args = args if self.is_first else self._retrieve_recv_activations(fwd_chunk_id)
        output = self.forward_maybe_with_nosync(*composite_args, **kwargs)
        self._forward_inputs[fwd_chunk_id] = tuple(
            value
            for value in flatten_args(composite_args)
            if isinstance(value, tp.Tensor) or value is not None
        ) + tuple(
            value
            for value in flatten_args(kwargs)
            if isinstance(value, tp.Tensor) or value is not None
        )
        output_tuple = _normalize_model_output_as_tuple(output)
        self.fwd_cache[fwd_chunk_id] = (output, output_tuple)
        if save_forward_output:
            while len(self.output_chunks) <= fwd_chunk_id:
                self.output_chunks.append(None)
            self.output_chunks[fwd_chunk_id] = output
        self._stage_meta.forward.output_metas = tuple(meta for meta in (extract_tensor_meta(value) for value in output_tuple) if meta is not None)
        return output

    def backward_one_chunk(self, bwd_chunk_id: int, loss: Any = None, full_backward: bool = True, last_backward: bool = False) -> Any:
        if not self.has_backward:
            return None
        self._check_chunk_id(bwd_chunk_id)
        output, output_values = self.fwd_cache.pop(bwd_chunk_id)
        if self.is_last:
            stage_output = output if loss is None else loss
            output_grads = None
        else:
            stage_output = output_values
            output_grads = self._retrieve_recv_grads(bwd_chunk_id)
        input_values = self._forward_inputs.pop(bwd_chunk_id, ())
        bwd_kwargs = {
            "stage_output": stage_output,
            "output_grads": output_grads,
            "input_values": input_values,
        }
        grads_input: tuple[Any, ...] = ()
        if self.dw_builder is not None:
            grads_input, _ = self.backward_maybe_with_nosync(
                "full", bwd_kwargs, last_backward=last_backward
            )
            if full_backward:
                self.dw_builder()()
            else:
                self.dw_runner[bwd_chunk_id] = self.dw_builder()
        elif full_backward:
            grads_input, _ = self.backward_maybe_with_nosync(
                "full", bwd_kwargs, last_backward=last_backward
            )
        else:
            param_groups = None
            if not self.is_first:
                grads_input, param_groups = self.backward_maybe_with_nosync(
                    "input", bwd_kwargs, last_backward=last_backward
                )
            self.backward_state[bwd_chunk_id] = (
                input_values,
                param_groups,
                stage_output,
                output_grads,
            )
            self.dw_runner[bwd_chunk_id] = lambda: None
        num_forward_inputs = len(self._stage_meta.inputs or ())
        self.bwd_cache[bwd_chunk_id] = tuple(grads_input[:num_forward_inputs])
        return self.bwd_cache[bwd_chunk_id]

    def backward_weight_one_chunk(self, bwd_chunk_id: int, last_backward: bool = False) -> Any:
        if not self.has_backward:
            return None
        runner = self.dw_runner.pop(bwd_chunk_id, None)
        if runner is None:
            raise AssertionError(
                f"backward weight requested for chunk {bwd_chunk_id} without input backward"
            )
        if self.dw_builder is not None:
            return runner()
        input_values, param_groups, stage_output, output_grads = self.backward_state.pop(
            bwd_chunk_id
        )
        if self.is_first:
            self.backward_maybe_with_nosync(
                "full",
                {
                    "stage_output": stage_output,
                    "output_grads": output_grads,
                    "input_values": input_values,
                },
                last_backward=last_backward,
            )
        else:
            self.backward_maybe_with_nosync(
                "weight",
                {"param_groups": param_groups},
                last_backward=last_backward,
            )
        return None

    def _get_init_p2p_neighbors_ops(self) -> list[Any]:
        operations: list[Any] = []
        next_stage_peer_rank = self.stage_index_to_group_rank.get(
            self.stage_index + 1
        )
        previous_stage_peer_rank = self.stage_index_to_group_rank.get(
            self.stage_index - 1
        )
        downstream_recv_tensor = tp.zeros(
            1, device=self.device, dtype=tp.float32
        )
        upstream_recv_tensor = tp.zeros(
            1, device=self.device, dtype=tp.float32
        )
        send_tensor = tp.tensor(
            self.stage_index, device=self.device, dtype=tp.float32
        )
        if not self.is_first:
            operations.append(
                dist.P2POp(
                    dist.irecv,
                    downstream_recv_tensor,
                    group_peer=previous_stage_peer_rank,
                    group=self._downstream_group,
                )
            )
        if not self.is_last:
            operations.append(
                dist.P2POp(
                    dist.isend,
                    send_tensor,
                    group_peer=next_stage_peer_rank,
                    group=self._downstream_group,
                )
            )
        if not self.is_first:
            operations.append(
                dist.P2POp(
                    dist.isend,
                    send_tensor,
                    group_peer=previous_stage_peer_rank,
                    group=self._upstream_group,
                )
            )
        if not self.is_last:
            operations.append(
                dist.P2POp(
                    dist.irecv,
                    upstream_recv_tensor,
                    group_peer=next_stage_peer_rank,
                    group=self._upstream_group,
                )
            )
        return operations

    def perform_reduce_grad(self, grad_scale_factor: float) -> None:
        state_getter = getattr(self.submod, "_get_fsdp_state", None)
        if not callable(state_getter):
            state_getter = getattr(self.submod, "_get_replicate_state", None)
        if callable(state_getter):
            for method_name, value in (
                ("set_is_last_backward", True),
                ("set_reshard_after_backward", True),
                ("set_requires_gradient_sync", True),
            ):
                method = getattr(self.submod, method_name, None)
                if callable(method):
                    method(value)
            state = state_getter()
            state_context = getattr(state, "_state_ctx", None)
            states = (
                getattr(state_context, "all_states", None)
                or getattr(state_context, "states", None)
                or [state]
            )
            for state_item in states:
                groups_getter = getattr(state_item, "_all_param_groups", None)
                if callable(groups_getter):
                    for param_group in groups_getter():
                        param_group.post_backward()
            callback = getattr(state, "_root_post_backward_final_callback", None)
            if callable(callback):
                callback()
        self.scale_grads(grad_scale_factor)


class _PipelineStage(_PipelineStageBase):
    def __init__(self, stage_module: Any, stage_index: int, pipe_info: Any, device: Any = None, group: Any = None) -> None:
        super().__init__(stage_module, stage_index, pipe_info.num_stages, device, group)
        self.pipe_info = pipe_info
        graph_owner = getattr(pipe_info, "graph", None)
        self.graph = getattr(graph_owner, "graph", graph_owner)
        submod_nodes = [
            node
            for node in getattr(self.graph, "nodes", ())
            if getattr(node, "op", None) == "call_module"
        ]
        if len(submod_nodes) != self.num_stages:
            raise PipeliningMetadataError(
                f"Number of submodules in pipe graph {len(submod_nodes)} does not match "
                f"number of stages {self.num_stages}"
            )
        self.node = submod_nodes[stage_index]
        self.name = self.node.name
        self.submod_to_stage_index = {
            getattr(node, "name", ""): index
            for index, node in enumerate(submod_nodes)
        }
        self._move_submod_to_device()

    def _move_submod_to_device(self) -> None:
        parameters = getattr(self.submod, "parameters", None)
        if callable(parameters) and any(
            bool(getattr(parameter, "is_meta", False))
            for parameter in parameters()
        ):
            return
        if self.device is not None and hasattr(self.submod, "to"):
            self.submod.to(self.device)

    def get_stage_index_of_submod(self, submod_name: str) -> int:
        try:
            return self.submod_to_stage_index[submod_name]
        except KeyError as exc:
            raise PipeliningMetadataError(
                f"stage {submod_name!r} is not present"
            ) from exc

    def _tensor_from_meta(self, meta: Any, value: Any = None) -> Any:
        if isinstance(value, tp.Tensor):
            result = value.detach().clone()
        elif meta is not None and hasattr(meta, "to_tensor"):
            result = meta.to_tensor(self.device)
        elif meta is not None and hasattr(meta, "shape"):
            result = tp.empty(tuple(meta.shape), dtype=meta.dtype, device=self.device)
        else:
            result = value
        if isinstance(result, tp.Tensor) and self.has_backward:
            if result.is_floating_point() or result.is_complex():
                result.requires_grad_(True)
        return result

    def _create_act_recv_info(self) -> tuple[_RecvInfo, ...]:
        if self.node is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: graph stage node is unavailable"
            )
        stage_graph = getattr(self.submod, "graph", None)
        placeholders = [
            node
            for node in getattr(stage_graph, "nodes", ())
            if getattr(node, "op", None) == "placeholder"
        ]
        outer_args = tuple(getattr(self.node, "args", ()))
        result: list[_RecvInfo] = []
        if len(placeholders) != len(outer_args):
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: graph placeholder and dependency counts differ"
            )
        for placeholder, arg_node in zip(placeholders, outer_args, strict=True):
            meta_value = getattr(placeholder, "meta", {}).get("val")
            if meta_value is None:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: placeholder metadata is unavailable"
                )
            if isinstance(meta_value, DTensor):
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: distributed tensor metadata is unsupported for graph stages"
                )
            if getattr(arg_node, "op", None) == "placeholder":
                result.append(
                    _RecvInfo(
                        f"root_input_{getattr(placeholder, 'name', 'input')}",
                        None,
                        None,
                        _TensorMeta.from_tensor(meta_value),
                        True,
                    )
                )
                continue
            while getattr(arg_node, "target", None) is operator.getitem:
                arg_node = arg_node.args[0]
            if getattr(arg_node, "op", None) != "call_module":
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: expected a stage dependency"
                )
            source = self.get_stage_index_of_submod(getattr(arg_node, "name", ""))
            meta = _TensorMeta(
                shape=tuple(meta_value.shape),
                stride=tuple(meta_value.stride()),
                dtype=meta_value.dtype,
                requires_grad=bool(
                    self.has_backward
                    and (
                        meta_value.is_floating_point()
                        or meta_value.is_complex()
                    )
                ),
            )
            result.append(
                _RecvInfo(
                    getattr(arg_node, "name", getattr(placeholder, "name", "input")),
                    source,
                    _make_tensor_from_meta(meta, self.device),
                    meta,
                )
            )
        return tuple(result)

    def _prepare_forward_infra(self, num_microbatches: int, args: Any, kwargs: Any, has_backward: bool) -> Any:
        del kwargs
        self.chunks = int(num_microbatches)
        self.has_backward = bool(has_backward)
        for index in range(self.chunks):
            self.args_recv_info[index] = self._create_act_recv_info()
        recv_infos = self.args_recv_info[0]
        if self.is_first:
            if not isinstance(args, tuple):
                raise AssertionError("first stage requires real tensor args")
            self._stage_meta.inputs = tuple(
                info.tensor_meta for info in recv_infos[: len(args)]
            )
        else:
            self._stage_meta.inputs = tuple(
                info.tensor_meta for info in recv_infos if not info.is_root_arg
            )
        self.act_send_info = self._create_act_send_info()

    def _prepare_backward_infra(
        self,
        num_microbatches: int,
        loss_fn: Any = None,
        target: Any = None,
        received_grad_meta: Any = None,
        loss_kwargs: Any = None,
    ) -> None:
        del loss_fn, target, received_grad_meta, loss_kwargs
        if self._stage_meta.inputs is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: inputs metadata required for backward inference."
            )
        self._stage_meta.input_grads = _derive_grad_metas(self._stage_meta.inputs)
        self._setup_backward_recv_info(num_microbatches)
        return None

    def find_dst_rank(self, user: Any) -> int:
        if getattr(user, "op", None) != "call_module":
            return None
        return self.get_stage_index_of_submod(getattr(user, "name", ""))

    def _create_act_send_info(self) -> dict[int, list[int]]:
        if self.node is None:
            return {0: [self.stage_index + 1]} if not self.is_last else {0: []}
        result: dict[int, list[int]] = {}
        for user in getattr(self.node, "users", ()):
            if getattr(user, "target", None) is operator.getitem:
                output_index = int(user.args[1])
                destinations = result.setdefault(output_index, [])
                for child in getattr(user, "users", ()):
                    destination = self.find_dst_rank(child)
                    if destination is not None and destination not in destinations:
                        destinations.append(destination)
            else:
                destination = self.find_dst_rank(user)
                if destination is not None:
                    destinations = result.setdefault(0, [])
                    if destination not in destinations:
                        destinations.append(destination)
        output_node = self._get_output_node()
        if output_node is not None:
            values = output_node.args[0] if getattr(output_node, "args", ()) else ()

            def flatten_graph_values(value: Any) -> list[Any]:
                if isinstance(value, (tuple, list)):
                    result_values: list[Any] = []
                    for item in value:
                        result_values.extend(flatten_graph_values(item))
                    return result_values
                if isinstance(value, dict):
                    result_values = []
                    for item in value.values():
                        result_values.extend(flatten_graph_values(item))
                    return result_values
                return [value]

            output_metas: list[_TensorMeta] = []
            for index, value in enumerate(flatten_graph_values(values)):
                example_value = getattr(value, "meta", {}).get("val")
                if example_value is None:
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: output metadata is unavailable at index {index}"
                    )
                if isinstance(example_value, DTensor):
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: distributed tensor metadata is unsupported for graph stages"
                    )
                if not isinstance(example_value, tp.Tensor):
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: output {index} is not a tensor"
                    )
                output_metas.append(
                    _TensorMeta(
                        shape=tuple(example_value.shape),
                        stride=tuple(example_value.stride()),
                        dtype=example_value.dtype,
                        requires_grad=bool(
                            self.has_backward
                            and (
                                example_value.is_floating_point()
                                or example_value.is_complex()
                            )
                        ),
                    )
                )
            self._stage_meta.outputs = tuple(output_metas)
        return result

    def _create_grad_recv_info(self, act_send_info: Any) -> tuple[_RecvInfo, ...]:
        if self._stage_meta.outputs is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: outputs metadata required for grad recv info."
            )

        outputs_meta = self._stage_meta.outputs
        output_grads_metas: list[Any] = []
        grad_recv_infos: list[_RecvInfo] = []
        for out_idx, out_meta in enumerate(outputs_meta):
            dst_list = act_send_info.get(out_idx, [])
            grad_src = dst_list[0] if dst_list else self.stage_index + 1
            if not dst_list or not out_meta.requires_grad:
                output_grads_metas.append(None)
                grad_recv_infos.append(
                    _RecvInfo(
                        f"recv_grad_for_{self.stage_index}_none_{out_idx}",
                        grad_src,
                        None,
                        None,
                    )
                )
                continue
            grad_meta = _TensorMeta(
                shape=out_meta.shape,
                stride=out_meta.stride,
                dtype=out_meta.dtype,
                requires_grad=False,
            )
            output_grads_metas.append(grad_meta)
            if len(dst_list) != 1:
                raise PipeliningMetadataError(
                    "Backward of skip connections not supported yet"
                )
            grad_recv_infos.append(
                _RecvInfo(
                    f"recv_grad_for_{self.stage_index}_from_{grad_src}",
                    grad_src,
                    _make_tensor_from_meta(grad_meta, self.device),
                    grad_meta,
                )
            )
        self._stage_meta.output_grads = tuple(output_grads_metas)
        if self._stage_meta.inputs is not None:
            self._stage_meta.input_grads = _derive_grad_metas(self._stage_meta.inputs)
        return tuple(grad_recv_infos)

    def backward_one_chunk(self, bwd_chunk_id: int, loss: Any = None, full_backward: bool = True, last_backward: bool = False) -> Any:
        self._check_chunk_id(bwd_chunk_id)
        return super().backward_one_chunk(
            bwd_chunk_id,
            loss=loss,
            full_backward=full_backward,
            last_backward=last_backward,
        )

    def _get_output_node(self) -> Any:
        for graph in (getattr(self.submod, "graph", None), self.graph):
            output_node = next(
                (
                    node
                    for node in getattr(graph, "nodes", ())
                    if getattr(node, "op", None) == "output"
                ),
                None,
            )
            if output_node is not None:
                return output_node
        return None


def build_stage(stage_module: Any, stage_index: int, pipe_info: Any, device: Any = None, group: Any = None) -> _PipelineStage:
    return _PipelineStage(stage_module, stage_index, pipe_info, device, group)


class PipelineStage(_PipelineStageBase):
    def __init__(self, submodule: Any, stage_index: int, num_stages: int, device: Any = None, input_args: tuple[Any, ...] | None = None, output_args: Any = None, output_grads: Any = None, input_grads: Any = None, group: Any = None, dw_builder: Callable[[], Callable[..., None]] | None = None, get_mesh: Any = None) -> None:
        super().__init__(submodule, stage_index, num_stages, device, group, dw_builder)
        self._mesh_cache = _MeshCache(get_mesh_cb=get_mesh)
        self._input_example = _normalize_model_output_as_tuple(input_args) if input_args is not None else ()
        self._output_example = _normalize_model_output_as_tuple(output_args) if output_args is not None else None
        input_grad_values = _normalize_model_output_as_tuple(input_grads) if input_grads is not None else None
        output_grad_values = _normalize_model_output_as_tuple(output_grads) if output_grads is not None else None
        self._user_meta = _StageMeta()
        self._user_meta.inputs = extract_tensor_metas(self._input_example) if self._input_example else None
        self._user_meta.outputs = extract_tensor_metas(self._output_example) if self._output_example is not None else None
        self._user_meta.input_grads = extract_tensor_metas(input_grad_values, allow_none=True) if input_grad_values is not None else None
        self._user_meta.output_grads = extract_tensor_metas(output_grad_values, allow_none=True) if output_grad_values is not None else None
        for values in (self._input_example, self._output_example, input_grad_values, output_grad_values):
            if values:
                self._mesh_cache.update_from_tensors(values)
        if self._user_meta.has_dtensors():
            if self._input_example and input_grad_values:
                validate_static_arg_grad_correspondence(
                    self.stage_index,
                    self._input_example,
                    input_grad_values,
                    is_input=True,
                )
            if self._output_example and output_grad_values:
                validate_static_arg_grad_correspondence(
                    self.stage_index,
                    self._output_example,
                    output_grad_values,
                    is_input=False,
                )
        self._inference_mode: InferenceMode | None = None
        self._fwd_outputs_for_bwd_meta: tuple[Any, ...] | None = None
        self._fwd_inputs_for_bwd_meta: tuple[Any, ...] | None = None
        self._fwd_kwargs_tensors_for_bwd_meta: tuple[Any, ...] | None = None
        self._metadata_inference_buffer_backup: list[tuple[Any, Any]] | None = None
        self._inference_mode = None

    def _prepare_forward_infra(self, num_microbatches: int, args: Any, kwargs: Any, has_backward: bool) -> Any:
        self.chunks = int(num_microbatches)
        self.has_backward = bool(has_backward)
        self._inference_mode = (
            InferenceMode.DYNAMIC
            if InferenceMode.needs_dynamic(self._user_meta, has_backward)
            else InferenceMode.STATIC
        )
        source_args = args
        if source_args is None or source_args == ():
            source_args = self._input_example
        fwd_meta_output = None
        if self._inference_mode == InferenceMode.DYNAMIC:
            fwd_meta_output = self._forward_metadata_inference(
                source_args, kwargs, has_backward
            )
        else:
            self._stage_meta.inputs = self._user_meta.inputs
            self._stage_meta.outputs = self._user_meta.outputs
        if self._stage_meta.inputs is None and source_args:
            self._stage_meta.inputs = extract_tensor_metas(tuple(source_args))
        if self._stage_meta.outputs is None and self._output_example is not None:
            self._stage_meta.outputs = extract_tensor_metas(self._output_example)
        self._setup_forward_recv_info(self.chunks, has_backward)
        self._setup_forward_send_info()
        return fwd_meta_output

    def _prepare_backward_infra(
        self,
        num_microbatches: int,
        loss_fn: Any = None,
        target: Any = None,
        received_grad_meta: Any = None,
        loss_kwargs: Any = None,
    ) -> Any:
        self.chunks = int(num_microbatches)
        self.has_backward = True
        if self._inference_mode == InferenceMode.DYNAMIC:
            result = self._backward_metadata_inference(
                loss_fn,
                target,
                received_grad_meta,
                loss_kwargs,
            )
            self._validate_inferred_metadata()
        else:
            result = None
            self._stage_meta.inputs = self._user_meta.inputs
            self._stage_meta.outputs = self._user_meta.outputs
            self._stage_meta.input_grads = self._user_meta.input_grads
            self._stage_meta.output_grads = self._user_meta.output_grads
        if isinstance(received_grad_meta, _StageBackwardMeta):
            self._stage_meta.output_grads = received_grad_meta.input_grad_metas
        if self._stage_meta.output_grads is None:
            if self._stage_meta.outputs is None:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: output metadata is required for backward inference."
                )
            self._stage_meta.output_grads = _derive_grad_metas(self._stage_meta.outputs)
        if self._stage_meta.input_grads is None:
            if self._stage_meta.inputs is None:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: input metadata is required for backward inference."
                )
            self._stage_meta.input_grads = _derive_grad_metas(self._stage_meta.inputs)
        self._setup_backward_recv_info(num_microbatches)
        self.grad_send_info = self._create_grad_send_info(self.args_recv_info.get(0, ()))
        return result

    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> list[Any]:
        self._check_chunk_id(fwd_chunk_id)
        if self.is_first:
            return []
        return self._get_recv_ops(
            self.args_recv_info[fwd_chunk_id], self._downstream_group
        )

    def _recv_meta(self, src_stage: int) -> Any:
        objects = [None]
        dist.recv_object_list(
            objects,
            src=self._resolve_peer_global_rank(src_stage),
            group=self.group,
            device=self.device,
        )
        if len(objects) != 1:
            raise PipeliningMetadataError(
                f"expected one metadata object, got {len(objects)}"
            )
        return objects[0]

    def _send_meta(self, meta: Any, dst_stage: int) -> None:
        dist.send_object_list(
            [meta],
            dst=self._resolve_peer_global_rank(dst_stage),
            group=self.group,
            device=self.device,
        )

    def _is_same_rank(self, other_stage: int) -> bool:
        return self.stage_index_to_group_rank[int(other_stage)] == self.group_rank

    def _warmup_forward_vote(self, has_backward: bool, received_acc: Any = None) -> Any:
        my_vote = 0 if InferenceMode.needs_dynamic(self._user_meta, has_backward) else 1
        vote = tp.tensor([my_vote], dtype=tp.int32, device=self.device)
        if self.is_first:
            accumulated = vote
        elif self._is_same_rank(self.stage_index - 1):
            if received_acc is None:
                raise AssertionError("forward vote is missing the accumulated value")
            accumulated = received_acc * vote
        else:
            accumulated = tp.zeros(1, dtype=tp.int32, device=self.device)
            dist.recv(
                accumulated,
                src=self._resolve_peer_global_rank(self.stage_index - 1),
                group=self.group,
            )
            accumulated = accumulated * vote
        if not self.is_last and not self._is_same_rank(self.stage_index + 1):
            dist.send(
                accumulated,
                dst=self._resolve_peer_global_rank(self.stage_index + 1),
                group=self.group,
            )
        return accumulated

    def _warmup_backward_result(self, received_result: Any = None) -> Any:
        if self.is_last or self._is_same_rank(self.stage_index + 1):
            if received_result is None:
                raise AssertionError("backward vote is missing the accumulated value")
            result = received_result
        else:
            result = tp.zeros(1, dtype=tp.int32, device=self.device)
            dist.recv(
                result,
                src=self._resolve_peer_global_rank(self.stage_index + 1),
                group=self.group,
            )
        if not self.is_first and not self._is_same_rank(self.stage_index - 1):
            dist.send(
                result,
                dst=self._resolve_peer_global_rank(self.stage_index - 1),
                group=self.group,
            )
        return result

    def _compute_outputs(self, *args: Any, module: Any = None, **kwargs: Any) -> Any:
        return (self.submod if module is None else module)(*args, **kwargs)

    def _compute_input_grads(
        self,
        outputs: Any,
        all_fwd_inputs: Any,
        grad_outputs: Any = None,
    ) -> tuple[Any, ...]:
        return _autograd_grad_for_inputs(
            tuple(outputs),
            tuple(all_fwd_inputs),
            None if grad_outputs is None else tuple(grad_outputs),
            allow_unused=True,
        )

    def backward_one_chunk(self, bwd_chunk_id: int, loss: Any = None, full_backward: bool = True, last_backward: bool = False) -> Any:
        return super().backward_one_chunk(
            bwd_chunk_id,
            loss=loss,
            full_backward=full_backward,
            last_backward=last_backward,
        )

    def _to_tensor(self, arg: Any) -> Any:
        if isinstance(arg, DTensor):
            local = arg.to_local().detach()
            if getattr(arg, "requires_grad", False) and (
                local.is_floating_point() or local.is_complex()
            ):
                local.requires_grad_(True)
            return DTensor.from_local(
                local,
                device_mesh=arg.device_mesh,
                placements=arg.placements,
                shape=arg.shape,
                stride=arg.stride(),
            )
        if isinstance(arg, tp.Tensor):
            result = arg.detach()
            if arg.requires_grad:
                result.requires_grad_(True)
            return result
        if isinstance(arg, _DTensorMeta):
            mesh = self._mesh_cache.get_mesh(arg.mesh_cache_key)
            local = _make_tensor_from_meta(arg, self.device)
            if arg.requires_grad and (
                local.is_floating_point() or local.is_complex()
            ):
                local.requires_grad_(True)
            return DTensor.from_local(
                local,
                device_mesh=mesh,
                placements=arg.placements,
                shape=arg.global_shape,
                stride=arg.global_stride,
            )
        if isinstance(arg, _TensorMeta):
            result = arg.to_tensor(self.device)
            if arg.requires_grad and (
                result.is_floating_point() or result.is_complex()
            ):
                result.requires_grad_(True)
            return result
        raise PipeliningMetadataError(
            f"unsupported metadata value {type(arg).__name__}"
        )

    def _ones_from_metadata(self, meta: Any) -> Any:
        local = tp.ones(meta.shape, dtype=meta.dtype, device=self.device)
        if isinstance(meta, _DTensorMeta):
            mesh = self._mesh_cache.get_mesh(meta.mesh_cache_key)
            return DTensor.from_local(
                local,
                device_mesh=mesh,
                placements=meta.placements,
                shape=meta.global_shape,
                stride=meta.global_stride,
            )
        return local

    def _pre_metadata_inference_backup(self) -> None:
        if self._inference_mode != InferenceMode.DYNAMIC:
            return
        if self._metadata_inference_buffer_backup is not None:
            raise RuntimeError("metadata inference backup is already active")
        named_buffers = getattr(self.submod, "named_buffers", None)
        if callable(named_buffers):
            self._metadata_inference_buffer_backup = [
                (buffer, buffer.detach().clone())
                for _, buffer in named_buffers(remove_duplicate=False)
            ]

    def _forward_metadata_inference(self, args: Any, kwargs: Any, has_backward: bool) -> Any:
        kwargs = kwargs or {}
        if self.is_first:
            if args is None or isinstance(args, _StageForwardMeta):
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: first stage requires tensor inputs"
                )
            values = tuple(args)
            self._stage_meta.inputs = extract_tensor_metas(values)
            inference_args = tuple(self._to_tensor(value) for value in values)
        elif self._is_same_rank(self.stage_index - 1) or isinstance(args, _StageForwardMeta):
            if not isinstance(args, _StageForwardMeta):
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: forward metadata from the previous stage is required"
                )
            input_metas = args.forward_metas
            self._stage_meta.inputs = tuple(input_metas)
            inference_args = tuple(self._to_tensor(meta) for meta in input_metas)
        else:
            recv_meta = self._recv_meta(self.stage_index - 1)
            if not isinstance(recv_meta, _StageForwardMeta):
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: invalid forward metadata received from the previous stage"
                )
            input_metas = recv_meta.forward_metas
            self._stage_meta.inputs = tuple(input_metas)
            inference_args = tuple(self._to_tensor(meta) for meta in input_metas)
        inference_kwargs = {
            key: self._to_tensor(value) if isinstance(value, tp.Tensor) else value
            for key, value in kwargs.items()
        }
        with (tp.enable_grad() if has_backward else tp.no_grad()):
            output = self._compute_outputs(
                *inference_args,
                module=self.submod,
                **inference_kwargs,
            )
        output_values = _normalize_model_output_as_tuple(output)
        self._stage_meta.outputs = tuple(
            meta
            for meta in (extract_tensor_meta(value) for value in output_values)
            if meta is not None
        )
        self._fwd_outputs_for_bwd_meta = output_values
        self._fwd_inputs_for_bwd_meta = inference_args
        self._fwd_kwargs_tensors_for_bwd_meta = tuple(
            value
            for value in flatten_args(inference_kwargs)
            if isinstance(value, tp.Tensor) or isinstance(value, DTensor)
        )
        fwd_meta = _StageForwardMeta(forward_metas=self._stage_meta.outputs)
        if self.is_last or self._is_same_rank(self.stage_index + 1):
            return fwd_meta
        self._send_meta(fwd_meta, self.stage_index + 1)
        return None

    def _backward_metadata_inference(self, loss_fn: Any, target: Any, received_grad_meta: Any, loss_kwargs: Any) -> Any:
        fwd_outputs = self._fwd_outputs_for_bwd_meta
        fwd_inputs = self._fwd_inputs_for_bwd_meta
        if fwd_outputs is None or fwd_inputs is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: forward metadata inference must run first"
            )
        all_inputs = list(fwd_inputs) + list(self._fwd_kwargs_tensors_for_bwd_meta or ())
        if self.is_last:
            if loss_fn is None or target is None:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: loss_fn and target are required for backward inference"
                )
            output_value = fwd_outputs[0] if len(fwd_outputs) == 1 else fwd_outputs
            loss = loss_fn(output_value, self._to_tensor(target), **(loss_kwargs or {}))
            input_grads = self._compute_input_grads((loss,), all_inputs)
            self._stage_meta.output_grads = None
        else:
            if self._is_same_rank(self.stage_index + 1) or (
                not dist.is_initialized() and received_grad_meta is not None
            ):
                if not isinstance(received_grad_meta, _StageBackwardMeta):
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: backward metadata from the next stage is required"
                    )
                output_grad_metas = received_grad_meta.backward_metas
            else:
                recv_meta = self._recv_meta(self.stage_index + 1)
                if not isinstance(recv_meta, _StageBackwardMeta):
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: invalid backward metadata received from the next stage"
                    )
                output_grad_metas = recv_meta.backward_metas
            self._stage_meta.output_grads = output_grad_metas
            if len(fwd_outputs) != len(output_grad_metas):
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: output and gradient metadata counts differ"
                )
            filtered_outputs = []
            filtered_grad_outputs = []
            for index, (output, grad_meta) in enumerate(
                zip(fwd_outputs, output_grad_metas, strict=True)
            ):
                if not isinstance(output, (tp.Tensor, DTensor)):
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: output {index} is not a tensor"
                    )
                if not output.requires_grad and getattr(output, "grad_fn", None) is None:
                    if grad_meta is not None:
                        raise PipeliningMetadataError(
                            f"Stage {self.stage_index}: output {index} has gradient metadata but does not require gradients"
                        )
                    continue
                filtered_outputs.append(output)
                filtered_grad_outputs.append(
                    self._ones_from_metadata(grad_meta) if grad_meta is not None else None
                )
            if filtered_outputs:
                input_grads = self._compute_input_grads(
                    filtered_outputs,
                    all_inputs,
                    filtered_grad_outputs,
                )
            else:
                input_grads = tuple(None for _ in all_inputs)
        input_metas = self._stage_meta.inputs or ()
        if len(input_grads) < len(input_metas):
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: backward returned too few input gradients"
            )
        self._stage_meta.input_grads = tuple(
            extract_tensor_meta(gradient)
            if isinstance(gradient, (tp.Tensor, DTensor))
            else (
                _derive_grad_metas((meta,))[0]
                if meta is not None and meta.requires_grad
                else None
            )
            for meta, gradient in zip(input_metas, input_grads)
        )
        bwd_meta = _StageBackwardMeta(backward_metas=self._stage_meta.input_grads)
        if self.is_first or self._is_same_rank(self.stage_index - 1):
            return bwd_meta
        self._send_meta(bwd_meta, self.stage_index - 1)
        return None

    def _post_metadata_inference_cleanup(self) -> None:
        if self._metadata_inference_buffer_backup is not None:
            with tp.no_grad():
                for buffer, saved in self._metadata_inference_buffer_backup:
                    buffer.copy_(saved)
            self._metadata_inference_buffer_backup = None
        self._fwd_outputs_for_bwd_meta = None
        self._fwd_inputs_for_bwd_meta = None
        self._fwd_kwargs_tensors_for_bwd_meta = None
        self.clear_runtime_states()

    def _validate_inferred_metadata(self) -> None:
        if not self._stage_meta.outputs:
            raise PipeliningMetadataError("stage output metadata is empty")
        for user_meta, inferred_meta, label in (
            (self._user_meta.inputs, self._stage_meta.inputs, "input"),
            (self._user_meta.outputs, self._stage_meta.outputs, "output"),
            (self._user_meta.input_grads, self._stage_meta.input_grads, "input_grad"),
            (self._user_meta.output_grads, self._stage_meta.output_grads, "output_grad"),
        ):
            if user_meta is not None and inferred_meta is not None:
                validate_tensors_metadata(
                    f"Stage {self.stage_index} {label}",
                    user_meta,
                    inferred_meta,
                    raise_on_mismatch=False,
                    warn_on_mismatch=True,
                )

    def _setup_forward_recv_info(self, num_microbatches: int, has_backward: bool) -> None:
        del has_backward
        if self._stage_meta.inputs is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: inputs metadata is required for receive setup."
            )
        self.args_recv_info = {}
        for chunk_id in range(num_microbatches):
            if self.is_first:
                infos = tuple(
                    _RecvInfo(
                        f"root_input_{index}",
                        None,
                        None,
                        meta,
                        True,
                    )
                    for index, meta in enumerate(self._stage_meta.inputs)
                )
            else:
                infos = tuple(
                    _RecvInfo(
                        f"recv_for_{self.stage_index}_from_{self.stage_index - 1}",
                        self.stage_index - 1,
                        self._to_tensor(meta),
                        meta,
                        False,
                    )
                    for meta in self._stage_meta.inputs
                )
            self.args_recv_info[chunk_id] = infos

    def _setup_forward_send_info(self) -> None:
        if self._stage_meta.outputs is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: outputs metadata is required for send setup."
            )
        self.act_send_info = {
            index: [self.stage_index + 1] if not self.is_last else []
            for index in range(len(self._stage_meta.outputs))
        }

    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple[_RecvInfo, ...]:
        grad_recv_infos: list[_RecvInfo] = []
        if not self.is_last:
            if self._stage_meta.output_grads is None:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: output_grads metadata is required for creating grad recv info."
                )
            output_grads = self._stage_meta.output_grads
            for index, destinations in act_send_info.items():
                if destinations is None or not destinations:
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: output {index} is not sent to any stage."
                    )
                source = destinations[0]
                grad_meta = output_grads[index]
                grad_recv_infos.append(
                    _RecvInfo(
                        f"recv_grad_for_{self.stage_index}_from_{source}",
                        source,
                        _make_tensor_from_meta(grad_meta, self.device)
                        if grad_meta is not None
                        else None,
                        grad_meta,
                    )
                )
        return tuple(grad_recv_infos)
