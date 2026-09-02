"""Pipeline stage execution and metadata management."""

from abc import ABC
from dataclasses import dataclass
import operator
from typing import Any, Callable

import tensorplay as tp
from .. import config as dist_config
from .. import distributed_core as dist

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
        self._input_chunks: dict[int, tuple[Any, ...]] = {}
        self._forward_inputs: dict[int, tuple[Any, ...]] = {}

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

    def _prepare_forward_infra(self, num_microbatches: int, args: Any, kwargs: Any, has_backward: bool) -> None:
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
        self.grad_recv_info = {index: tuple() for index in range(num_microbatches)}

    def _create_grad_recv_info(self, act_send_info: Any) -> tuple[_RecvInfo, ...]:
        del act_send_info
        return ()

    def _resolve_peer_global_rank(self, stage_idx: int) -> int:
        return int(stage_idx)

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
        values = tuple(prev_stage_outputs) if isinstance(prev_stage_outputs, (tuple, list)) else (prev_stage_outputs,)
        self._input_chunks[mb_index] = values
        recv_infos = self.args_recv_info.get(mb_index, ())
        for info, value in zip(recv_infos, values):
            if isinstance(info, _RecvInfo):
                info.buffer = value.detach().requires_grad_(True) if isinstance(value, tp.Tensor) else value

    def get_local_bwd_output(self, mb_index: int) -> Any:
        return self.bwd_cache.get(mb_index)

    def set_local_bwd_input(self, next_stage_bwd_outputs: Any, mb_index: int) -> None:
        values = tuple(next_stage_bwd_outputs) if isinstance(next_stage_bwd_outputs, (tuple, list)) else (next_stage_bwd_outputs,)
        self.bwd_cache[mb_index] = values
        recv_infos = self.grad_recv_info.get(mb_index, ())
        if not recv_infos and values:
            recv_infos = tuple(
                _RecvInfo(str(index), self.stage_index + 1, value, None, False)
                for index, value in enumerate(values)
            )
            self.grad_recv_info[mb_index] = recv_infos
        for info, value in zip(recv_infos, values):
            if isinstance(info, _RecvInfo):
                info.buffer = value

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
        if not self.output_chunks:
            return []
        output = self.output_chunks[fwd_chunk_id]
        values = _normalize_model_output_as_tuple(output)
        operations = []
        for index, value in enumerate(values):
            for destination in self.act_send_info.get(index, ()):
                if destination is None or not isinstance(value, tp.Tensor):
                    continue
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
        if self.grad_send_info is None:
            return None
        if input_idx < 0 or input_idx >= len(self.grad_send_info):
            return None
        return self.grad_send_info[input_idx]

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> list[Any]:
        self._check_chunk_id(bwd_chunk_id)
        if not self.has_backward or self.is_first:
            return []
        if self.grad_send_info is None:
            self.grad_send_info = self._create_grad_send_info(
                self.args_recv_info.get(bwd_chunk_id, ())
            )
        gradients = self.bwd_cache.get(bwd_chunk_id, ())
        operations = []
        for gradient, destination in zip(gradients or (), self.grad_send_info):
            if destination is None or gradient is None or not isinstance(gradient, tp.Tensor):
                continue
            peer_group_rank = self.stage_index_to_group_rank[int(destination)]
            peer = (
                peer_group_rank
                if self._upstream_group is None
                else dist.get_global_rank(self._upstream_group, peer_group_rank)
            )
            operations.append(
                dist.P2POp(dist.isend, gradient, peer, self._upstream_group)
            )
        return operations

    def clear_runtime_states(self) -> None:
        self.fwd_cache.clear()
        self.bwd_cache.clear()
        self.output_chunks.clear()
        self._input_chunks.clear()
        self._forward_inputs.clear()

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
        self._forward_inputs[fwd_chunk_id] = tuple(args)
        output_tuple = _normalize_model_output_as_tuple(output)
        self.fwd_cache[fwd_chunk_id] = (output, output_tuple)
        if save_forward_output:
            while len(self.output_chunks) <= fwd_chunk_id:
                self.output_chunks.append(None)
            self.output_chunks[fwd_chunk_id] = output
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
        graph_owner = getattr(pipe_info, "graph", None)
        self.graph = getattr(graph_owner, "graph", graph_owner)
        submod_nodes = [
            node
            for node in getattr(self.graph, "nodes", ())
            if getattr(node, "op", None) == "call_module"
        ]
        self.node = submod_nodes[stage_index] if len(submod_nodes) == self.num_stages else None
        self.name = getattr(self.node, "name", f"submod_{stage_index}")
        self.submod_to_stage_index = {
            getattr(node, "name", ""): index
            for index, node in enumerate(submod_nodes)
        }
        self._move_submod_to_device()

    def _move_submod_to_device(self) -> None:
        if self.device is not None and hasattr(self.submod, "to"):
            self.submod.to(self.device)

    def get_stage_index_of_submod(self, submod_name: str) -> int:
        try:
            return self.submod_to_stage_index[submod_name]
        except KeyError as exc:
            raise ValueError(f"stage {submod_name!r} is not present") from exc

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

    def _create_act_recv_info(self, args: Any = ()) -> tuple[_RecvInfo, ...]:
        if self.node is None:
            values = tuple(args)
            return tuple(
                _RecvInfo(
                    str(index),
                    None if self.is_first else self.stage_index - 1,
                    self._tensor_from_meta(extract_tensor_meta(value), value),
                    extract_tensor_meta(value),
                    self.is_first,
                )
                for index, value in enumerate(values)
            )
        stage_graph = getattr(self.submod, "graph", None)
        placeholders = [
            node
            for node in getattr(stage_graph, "nodes", ())
            if getattr(node, "op", None) == "placeholder"
        ]
        outer_args = tuple(getattr(self.node, "args", ()))
        result: list[_RecvInfo] = []
        for index, placeholder in enumerate(placeholders):
            arg_node = outer_args[index] if index < len(outer_args) else None
            value = None
            if index < len(args) and getattr(arg_node, "op", None) == "placeholder":
                value = args[index]
            meta_value = getattr(placeholder, "meta", {}).get("val")
            meta = extract_tensor_meta(meta_value)
            source = None
            source_name = str(getattr(placeholder, "name", index))
            while getattr(arg_node, "target", None) is operator.getitem:
                arg_node = arg_node.args[0]
            if getattr(arg_node, "op", None) == "call_module":
                source_name = getattr(arg_node, "name", source_name)
                source = self.get_stage_index_of_submod(source_name)
            result.append(
                _RecvInfo(
                    source_name,
                    source,
                    value if source is None else self._tensor_from_meta(meta, value),
                    meta,
                    source is None,
                )
            )
        return tuple(result)

    def _prepare_forward_infra(self, num_microbatches: int, args: Any, kwargs: Any, has_backward: bool) -> None:
        del kwargs
        self.chunks = int(num_microbatches)
        self.has_backward = bool(has_backward)
        self.args_recv_info = {
            index: self._create_act_recv_info(args)
            for index in range(self.chunks)
        }
        self.act_send_info = self._create_act_send_info()
        if self.has_backward:
            self._prepare_backward_infra(self.chunks)

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

            self._stage_meta.forward.output_metas = tuple(
                meta
                for meta in (
                    extract_tensor_meta(getattr(value, "meta", {}).get("val"))
                    for value in flatten_graph_values(values)
                )
                if meta is not None
            )
        return result

    def _create_grad_recv_info(self, act_send_info: Any) -> tuple[_RecvInfo, ...]:
        result: list[_RecvInfo] = []
        output_metas = self._stage_meta.forward.output_metas
        output_count = max(len(output_metas), max(act_send_info, default=-1) + 1)
        for output_index in range(output_count):
            destinations = act_send_info.get(output_index, ())
            if destinations:
                meta = output_metas[output_index] if output_index < len(output_metas) else None
                buffer = self._tensor_from_meta(meta)
                result.append(
                    _RecvInfo(str(output_index), int(destinations[0]), buffer, meta, False)
                )
            else:
                result.append(_RecvInfo(str(output_index), None, None, None, False))
        return tuple(result)

    def forward_one_chunk(self, fwd_chunk_id: int, args: tuple[Any, ...], kwargs: dict[str, Any], save_forward_output: bool = True) -> Any:
        if not self.is_first:
            args = self._retrieve_recv_activations(fwd_chunk_id)
        return super().forward_one_chunk(fwd_chunk_id, args, kwargs, save_forward_output)

    def backward_one_chunk(self, bwd_chunk_id: int, loss: Any = None, full_backward: bool = True, last_backward: bool = False) -> Any:
        del full_backward, last_backward
        self._check_chunk_id(bwd_chunk_id)
        output, output_values = self.fwd_cache.pop(bwd_chunk_id)
        if self.is_last:
            if loss is None:
                loss = output
            if not isinstance(loss, tp.Tensor):
                raise TypeError("the last pipeline stage loss must be a tensor")
            loss.backward()
        else:
            grad_values = tuple(
                info.buffer if isinstance(info, _RecvInfo) else None
                for info in self.grad_recv_info.get(bwd_chunk_id, ())
            )
            if grad_values and any(value is not None for value in grad_values):
                tp.autograd.backward(output_values, grad_tensors=grad_values)
        inputs = self._forward_inputs.pop(bwd_chunk_id, ())
        self.bwd_cache[bwd_chunk_id] = tuple(
            getattr(value, "grad", None) if isinstance(value, tp.Tensor) else None
            for value in inputs
        )
        return self.bwd_cache[bwd_chunk_id]

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
        del output_grads, input_grads, get_mesh
        super().__init__(submodule, stage_index, num_stages, device, group, dw_builder)
        self._input_example = tuple(input_args or ())
        self._output_example = _normalize_model_output_as_tuple(output_args) if output_args is not None else None
        self._prepare_forward_infra(1, self._input_example, {}, False)
        if self._output_example is not None:
            self._stage_meta.forward.output_metas = tuple(
                meta for meta in (extract_tensor_meta(value) for value in self._output_example)
                if meta is not None
            )

    def _prepare_forward_infra(self, num_microbatches: int, args: Any, kwargs: Any, has_backward: bool) -> None:
        del kwargs
        self.chunks = int(num_microbatches)
        self.has_backward = bool(has_backward)
        source_args = tuple(args) if args else self._input_example
        self._stage_meta.forward.input_metas = tuple(
            meta for meta in (extract_tensor_meta(value) for value in source_args)
            if meta is not None
        )
        infos = []
        for position, value in enumerate(source_args):
            meta = extract_tensor_meta(value)
            if self.is_first:
                infos.append(_RecvInfo(str(position), None, value, meta, True))
            else:
                if not isinstance(value, tp.Tensor):
                    raise TypeError("non-first pipeline stage inputs must be tensors")
                buffer = value.detach().clone()
                if getattr(buffer, "is_floating_point", lambda: False)() or getattr(buffer.dtype, "is_complex", False):
                    buffer.requires_grad_(True)
                infos.append(_RecvInfo(str(position), self.stage_index - 1, buffer, meta, False))
        self.args_recv_info = {index: tuple(infos) for index in range(self.chunks)}
        if self._output_example is not None:
            self._stage_meta.forward.output_metas = tuple(
                meta for meta in (extract_tensor_meta(value) for value in self._output_example)
                if meta is not None
            )

    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> list[Any]:
        self._check_chunk_id(fwd_chunk_id)
        if self.is_first:
            return []
        return self._get_recv_ops(
            self.args_recv_info[fwd_chunk_id], self._downstream_group
        )

    def forward_one_chunk(self, fwd_chunk_id: int, args: tuple[Any, ...], kwargs: dict[str, Any], save_forward_output: bool = True) -> Any:
        if not self.is_first:
            args = self._retrieve_recv_activations(fwd_chunk_id)
        return super().forward_one_chunk(fwd_chunk_id, args, kwargs, save_forward_output)

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> list[Any]:
        self._check_chunk_id(fwd_chunk_id)
        if self.is_last or not dist.is_initialized():
            return []
        output = self.output_chunks[fwd_chunk_id]
        values = _normalize_model_output_as_tuple(output)
        peer_group_rank = self.stage_index_to_group_rank[self.stage_index + 1]
        peer = (
            peer_group_rank
            if self._downstream_group is None
            else dist.get_global_rank(self._downstream_group, peer_group_rank)
        )
        return [
            dist.P2POp(dist.isend, value, peer, self._downstream_group)
            for value in values if isinstance(value, tp.Tensor)
        ]

    def get_bwd_recv_ops(self, bwd_chunk_id: int) -> list[Any]:
        self._check_chunk_id(bwd_chunk_id)
        if self.is_last or not dist.is_initialized():
            return []
        _, output_values = self.fwd_cache[bwd_chunk_id]
        infos = []
        for index, value in enumerate(output_values):
            if isinstance(value, tp.Tensor) and getattr(value, "requires_grad", False):
                infos.append(_RecvInfo(str(index), self.stage_index + 1, value.detach().new_empty(tuple(value.shape)), None, False))
            else:
                infos.append(_RecvInfo(str(index), None, None, None, False))
        self.grad_recv_info[bwd_chunk_id] = tuple(infos)
        return self._get_recv_ops(infos, self._upstream_group)

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> list[Any]:
        self._check_chunk_id(bwd_chunk_id)
        if self.is_first or not dist.is_initialized():
            return []
        gradients = self.bwd_cache.get(bwd_chunk_id, ())
        recv_infos = self.args_recv_info[bwd_chunk_id]
        peer_group_rank = self.stage_index_to_group_rank[self.stage_index - 1]
        peer = (
            peer_group_rank
            if self._upstream_group is None
            else dist.get_global_rank(self._upstream_group, peer_group_rank)
        )
        operations = []
        for info, gradient in zip(recv_infos, gradients):
            if isinstance(info, _RecvInfo) and info.source is not None and gradient is not None:
                operations.append(
                    dist.P2POp(dist.isend, gradient, peer, self._upstream_group)
                )
        return operations

    def backward_one_chunk(self, bwd_chunk_id: int, loss: Any = None, full_backward: bool = True, last_backward: bool = False) -> Any:
        del full_backward, last_backward
        self._check_chunk_id(bwd_chunk_id)
        output, output_values = self.fwd_cache.pop(bwd_chunk_id)
        if self.is_last:
            if loss is None:
                loss = output
            if not isinstance(loss, tp.Tensor):
                raise TypeError("the last pipeline stage loss must be a tensor")
            loss.backward()
        else:
            grad_values = tuple(
                info.buffer if isinstance(info, _RecvInfo) and info.source is not None else None
                for info in self.grad_recv_info.get(bwd_chunk_id, ())
            )
            tp.autograd.backward(output_values, grad_tensors=grad_values)
        inputs = self._forward_inputs.pop(bwd_chunk_id, ())
        self.bwd_cache[bwd_chunk_id] = tuple(
            getattr(value, "grad", None) if isinstance(value, tp.Tensor) else None
            for value in inputs
        )
        return self.bwd_cache[bwd_chunk_id]

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
