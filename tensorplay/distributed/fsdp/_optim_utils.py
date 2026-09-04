"""Optimizer-state conversion for sharded parameters."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Iterator, NamedTuple

import tensorplay as tp

from .. import distributed_core as dist
from ._common_utils import (
    _apply_to_modules,
    _named_parameters_with_duplicates,
    clean_tensor_name,
)

__all__ = [
    "FSDPParamInfo",
    "sorted_items",
    "StateInfo",
    "_ConsolidatedOptimState",
    "_PosDimTensorInfo",
    "_OptimStateKey",
]


@dataclass
class FSDPParamInfo:
    state: Any
    handle: Any
    param_indices: dict[str, int] = field(default_factory=dict)
    param_requires_grad: list[bool] = field(default_factory=list)


def sorted_items(dictionary: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    for key in sorted(dictionary):
        yield key, dictionary[key]


@dataclass
class _ConsolidatedOptimState:
    tensor_state: dict[str, Any] = field(default_factory=dict)
    zero_dim_tensor_state: dict[str, Any] = field(default_factory=dict)
    non_tensor_state: dict[str, Any] = field(default_factory=dict)


class _PosDimTensorInfo(NamedTuple):
    shape: tuple[int, ...]
    dtype: Any


class _OptimStateKey(NamedTuple):
    unflat_param_names: tuple[str, ...]
    is_fsdp_managed: bool


@dataclass
class StateInfo:
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _is_tensor(value: Any) -> bool:
    return isinstance(value, tp.Tensor)


def _is_zero_dim_tensor(value: Any) -> bool:
    return _is_tensor(value) and value.dim() == 0


def _clone_state(value: Any, cpu_offload: bool = False) -> Any:
    if _is_tensor(value):
        result = value.detach().clone()
        return result.cpu() if cpu_offload else result
    if isinstance(value, dict):
        return {key: _clone_state(item, cpu_offload) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item, cpu_offload) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item, cpu_offload) for item in value)
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _same_value(left: Any, right: Any) -> bool:
    if _is_tensor(left) or _is_tensor(right):
        if not (_is_tensor(left) and _is_tensor(right)):
            return False
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        try:
            result = tp.equal(left, right)
            return bool(result.item()) if _is_tensor(result) else bool(result)
        except Exception:
            try:
                return left.tolist() == right.tolist()
            except Exception:
                return False
    if type(left) is not type(right):
        return False
    try:
        result = left == right
        return bool(result) if not _is_tensor(result) else bool(result.item())
    except Exception:
        return left is right


def _group_for_info(fsdp_param_info: FSDPParamInfo) -> Any:
    state = fsdp_param_info.state
    if state is not None:
        group = getattr(state, "process_group", None)
        if group is not None:
            return group
        param_group = getattr(state, "_param_group", None)
        if param_group is None:
            getter = getattr(state, "_fsdp_param_group", None)
            if callable(getter):
                param_group = getter()
        if param_group is not None:
            getter = getattr(param_group, "_all_gather_process_group", None)
            if callable(getter):
                return getter()
        mesh = getattr(state, "_device_mesh", None) or getattr(state, "mesh", None)
        if mesh is not None:
            mesh_info = getattr(state, "mesh_info", None)
            mesh_dim = getattr(mesh_info, "shard_mesh_dim", 0)
            getter = getattr(mesh, "get_group", None)
            if callable(getter):
                return getter(mesh_dim)
    handle = fsdp_param_info.handle
    mesh_info = getattr(handle, "mesh_info", None)
    mesh = getattr(mesh_info, "mesh", None)
    if mesh is not None:
        getter = getattr(mesh, "get_group", None)
        if callable(getter):
            return getter(getattr(mesh_info, "shard_mesh_dim", 0))
    return None


def _world_size(group: Any) -> int:
    try:
        if dist.is_initialized():
            return int(dist.get_world_size(group))
    except Exception:
        pass
    return 1


def _rank(group: Any) -> int:
    try:
        if dist.is_initialized():
            return int(dist.get_rank(group))
    except Exception:
        pass
    return 0


def _records(fsdp_param_info: FSDPParamInfo) -> list[dict[str, Any]]:
    """Returns parameter metadata in flat-storage order."""
    handle = fsdp_param_info.handle
    records: list[dict[str, Any]] = []
    values = list(getattr(handle, "params", ())) if handle is not None else []
    if values and all(hasattr(value, "module_info") for value in values):
        for index, value in enumerate(values):
            name = str(value.module_info.fqn)
            records.append(
                {
                    "name": name,
                    "param": getattr(value, "param", value),
                    "owner": value,
                    "shape": tuple(int(size) for size in value.param.shape),
                    "requires_grad": bool(getattr(value.param, "requires_grad", False)),
                    "index": index,
                }
            )
        return records

    names: list[str] = []
    name_getter = getattr(handle, "param_module_names", None)
    if callable(name_getter):
        names = [str(name) for name in name_getter()]
    if not names and handle is not None:
        flat = getattr(handle, "flat_param", None)
        metadata = getattr(flat, "_param_metadata", None)
        names = [str(name) for name in getattr(metadata, "fqns", ())]
    if not names:
        names = [name for name, _ in sorted(fsdp_param_info.param_indices.items(), key=lambda item: item[1])]

    flat = getattr(handle, "flat_param", None) if handle is not None else None
    metadata = getattr(flat, "_param_metadata", None)
    shapes = [tuple(int(size) for size in shape) for shape in getattr(metadata, "shapes", ())]
    for index, name in enumerate(names):
        param = values[index] if index < len(values) else None
        if param is not None and hasattr(param, "param"):
            owner = param
            param = param.param
        else:
            owner = None
        if param is not None and hasattr(param, "shape"):
            shape = tuple(int(size) for size in param.shape)
            requires_grad = bool(getattr(param, "requires_grad", False))
        elif index < len(shapes):
            shape = shapes[index]
            requires_grad = True
        else:
            shape = ()
            requires_grad = True
        records.append(
            {
                "name": name,
                "param": param,
                "owner": owner,
                "shape": shape,
                "requires_grad": requires_grad,
                "index": index,
            }
        )
    return records


def _info_total_numel(fsdp_param_info: FSDPParamInfo) -> int:
    return sum(math.prod(record["shape"]) for record in _records(fsdp_param_info))


def _info_local_numel(fsdp_param_info: FSDPParamInfo) -> int | None:
    records = _records(fsdp_param_info)
    local = 0
    found = False
    for record in records:
        owner = record["owner"]
        getter = getattr(owner, "_sharded_local_tensor", None)
        if callable(getter):
            local += int(getter().numel())
            found = True
    if found:
        return local
    flat = getattr(fsdp_param_info.handle, "flat_param", None)
    if flat is not None and hasattr(flat, "numel"):
        return int(flat.numel())
    return None


def _record_for_name(fsdp_param_info: FSDPParamInfo, name: str) -> dict[str, Any] | None:
    records = _records(fsdp_param_info)
    for record in records:
        if record["name"] == name:
            return record
    if name in fsdp_param_info.param_indices:
        index = fsdp_param_info.param_indices[name]
        return next((record for record in records if record["index"] == index), None)
    suffix = name.rsplit(".", 1)[-1]
    matches = [record for record in records if record["name"].rsplit(".", 1)[-1] == suffix]
    return matches[0] if len(matches) == 1 else None


def _local_shard(value: Any, record: dict[str, Any], fsdp_param_info: FSDPParamInfo) -> Any:
    owner = record["owner"]
    if owner is None:
        return value.detach().clone()
    local_getter = getattr(owner, "_sharded_local_tensor", None)
    if not callable(local_getter):
        return value.detach().clone()
    local = local_getter()
    full_shape = tuple(int(size) for size in value.shape)
    local_shape = tuple(int(size) for size in local.shape)
    if full_shape == local_shape:
        return value.detach().clone()
    if int(value.numel()) != math.prod(record["shape"]):
        raise ValueError(
            f"optimizer state shape {tuple(value.shape)} does not match parameter {record['name']}"
        )
    placement = getattr(owner, "_placement", None)
    mesh_info = getattr(owner, "mesh_info", None)
    mesh = getattr(mesh_info, "mesh", None)
    mesh_dim = getattr(mesh_info, "shard_mesh_dim", 0)
    world = _world_size(_group_for_info(fsdp_param_info))
    if placement is None or mesh is None or world <= 1:
        return value.detach().clone()
    dim = int(getattr(placement, "dim", 0))
    if dim < 0:
        dim += value.dim()
    rank_getter = getattr(mesh, "get_local_rank", None)
    rank = int(rank_getter(mesh_dim)) if callable(rank_getter) else _rank(_group_for_info(fsdp_param_info))
    width = (full_shape[dim] + world - 1) // world
    start = min(rank * width, full_shape[dim])
    length = local_shape[dim]
    available = max(0, min(length, full_shape[dim] - start))
    part = value.narrow(dim, start, available) if available else value.narrow(dim, 0, 0)
    if available == length:
        return part.detach().clone()
    pad_shape = list(local_shape)
    pad_shape[dim] = length - available
    padding = value.new_zeros(tuple(pad_shape))
    return tp.cat((part, padding), dim=dim).detach().clone()


def _gather_param_value(value: Any, record: dict[str, Any], fsdp_param_info: FSDPParamInfo) -> Any:
    if not _is_tensor(value) or value.dim() == 0:
        return _clone_state(value)
    owner = record["owner"]
    local_getter = getattr(owner, "_sharded_local_tensor", None) if owner is not None else None
    if not callable(local_getter):
        return _clone_state(value)
    local_shape = tuple(int(size) for size in local_getter().shape)
    if tuple(int(size) for size in value.shape) != local_shape:
        return _clone_state(value)
    group = _group_for_info(fsdp_param_info)
    world = _world_size(group)
    if world <= 1:
        return _clone_state(value)
    mesh_info = getattr(owner, "mesh_info", None)
    mesh_dim = getattr(mesh_info, "shard_mesh_dim", 0)
    placement = getattr(owner, "_placement", None)
    dim = int(getattr(placement, "dim", 0)) if placement is not None else 0
    if dim < 0:
        dim += value.dim()
    outputs = [value.detach().new_empty(local_shape) for _ in range(world)]
    dist.all_gather(outputs, value.detach(), group=group)
    gathered = tp.cat(tuple(outputs), dim=dim)
    target_shape = tuple(int(size) for size in record["shape"])
    if int(gathered.shape[dim]) > target_shape[dim]:
        gathered = gathered.narrow(dim, 0, target_shape[dim])
    return gathered.reshape(target_shape).detach().clone()


def _unflatten_optim_state(
    fsdp_param_info: FSDPParamInfo,
    flat_param_state: dict[str, Any],
    to_save: bool,
    shard_state: bool,
    cpu_offload: bool,
) -> list[dict[str, Any]]:
    if shard_state and not to_save:
        raise AssertionError("shard_state requires to_save")
    if not to_save:
        _communicate_optim_state(fsdp_param_info, flat_param_state)
        return []
    consolidated = _communicate_optim_state(fsdp_param_info, flat_param_state)
    return _unflatten_communicated_optim_state(
        fsdp_param_info, consolidated, shard_state, cpu_offload
    )


def _communicate_optim_state(
    fsdp_param_info: FSDPParamInfo,
    flat_param_state: dict[str, Any],
) -> _ConsolidatedOptimState:
    result = _ConsolidatedOptimState()
    group = _group_for_info(fsdp_param_info)
    world = _world_size(group)
    total_numel = _info_total_numel(fsdp_param_info)
    local_numel = _info_local_numel(fsdp_param_info)
    for state_name, value in sorted_items(flat_param_state):
        if _is_tensor(value) and value.dim() > 0:
            if world <= 1 or local_numel is None or int(value.numel()) != local_numel:
                result.tensor_state[state_name] = _clone_state(value)
                continue
            gathered = value.detach().new_empty(int(value.numel()) * world)
            dist.all_gather_single(gathered, value.detach().reshape(-1), group=group)
            if total_numel and int(gathered.numel()) >= total_numel:
                gathered = gathered.narrow(0, 0, total_numel)
            result.tensor_state[state_name] = gathered
        elif _is_zero_dim_tensor(value):
            result.zero_dim_tensor_state[state_name] = _clone_state(value)
        else:
            result.non_tensor_state[state_name] = _clone_state(value)
    return result


def _unflatten_communicated_optim_state(
    fsdp_param_info: FSDPParamInfo,
    state: _ConsolidatedOptimState,
    shard_state: bool,
    cpu_offload: bool = False,
) -> list[dict[str, Any]]:
    records = _records(fsdp_param_info)
    output = [dict() for _ in records]
    for state_name, value in sorted_items(state.tensor_state):
        offset = 0
        for index, record in enumerate(records):
            size = math.prod(record["shape"])
            if int(value.numel()) < offset + size:
                raise ValueError(f"state {state_name} is shorter than the managed parameters")
            piece = value.reshape(-1).narrow(0, offset, size).reshape(record["shape"])
            if shard_state:
                piece = _local_shard(piece, record, fsdp_param_info)
            output[index][state_name] = _clone_state(piece, cpu_offload)
            offset += size
    for state_name, value in sorted_items(state.zero_dim_tensor_state):
        for target in output:
            target[state_name] = _clone_state(value, cpu_offload)
    for state_name, value in sorted_items(state.non_tensor_state):
        for target in output:
            target[state_name] = _clone_state(value, cpu_offload)
    return output


def _broadcast_processed_state(fsdp_state: Any, optim_state: Any, group: Any) -> Any:
    del fsdp_state
    if _world_size(group) <= 1:
        return optim_state
    objects = [optim_state if _rank(group) == 0 else None]
    dist.broadcast_object_list(objects, src=0, group=group)
    return objects[0]


def _broadcast_state(fsdp_state: Any, state: Any, group: Any) -> Any:
    if _world_size(group) <= 1:
        return state
    rank = _rank(group)
    if _is_tensor(state) and state.dim() > 0:
        if rank == 0:
            value = state.detach().clone()
        else:
            value = tp.zeros(tuple(state.shape), dtype=state.dtype, device=getattr(fsdp_state, "compute_device", None))
        dist.broadcast(value, src=0, group=group)
        return value
    objects = [state if rank == 0 else None]
    dist.broadcast_object_list(objects, src=0, group=group)
    return objects[0]


def _shard_orig_param_state(
    fsdp_param_info: FSDPParamInfo,
    fqn: str,
    optim_state: dict[str, Any],
) -> dict[str, Any]:
    if not optim_state:
        return {}
    record = _record_for_name(fsdp_param_info, fqn)
    if record is None:
        raise KeyError(fqn)
    result: dict[str, Any] = {}
    for state_name, value in optim_state.items():
        if _is_tensor(value) and value.dim() > 0:
            result[state_name] = _local_shard(value, record, fsdp_param_info)
        else:
            result[state_name] = _clone_state(value)
    return result


def _parameter_names(model: Any) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for name, param in model.named_parameters():
        result.setdefault(id(param), []).append(str(name))
    return result


def _fqn_aliases(model: Any, fqn_to_fsdp_param_info: dict[str, FSDPParamInfo]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    model_names = [str(name) for name, _ in model.named_parameters()]
    for name in model_names:
        if name in fqn_to_fsdp_param_info:
            aliases[name] = name
            continue
        matches = [candidate for candidate in fqn_to_fsdp_param_info if name.endswith(candidate)]
        if len(matches) == 1:
            aliases[name] = matches[0]
    return aliases


def _optimizer_key_maps(model: Any, optim: Any, optim_input: Any = None) -> tuple[dict[Any, Any], dict[Any, list[str]]]:
    key_to_param: dict[Any, Any] = {}
    if optim is not None:
        saved = optim.state_dict()
        for physical, packed in zip(getattr(optim, "param_groups", ()), saved.get("param_groups", ())):
            for param, key in zip(physical.get("params", ()), packed.get("params", ())):
                key_to_param[key] = param
    elif optim_input is not None:
        values: list[Any] = []
        source = list(optim_input)
        if source and isinstance(source[0], dict):
            for group in source:
                values.extend(group.get("params", ()))
        else:
            values = source
        for key, value in enumerate(values):
            key_to_param[key] = value[1] if isinstance(value, tuple) and len(value) == 2 else value
    else:
        for key, (_, param) in enumerate(model.named_parameters()):
            key_to_param[key] = param
    names = _parameter_names(model)
    return key_to_param, {key: names.get(id(param), []) for key, param in key_to_param.items()}


def _named_state(
    optim_state_dict: dict[str, Any],
    model: Any,
    optim: Any = None,
    optim_input: Any = None,
) -> tuple[dict[Any, dict[str, Any]], dict[Any, Any], dict[Any, list[str]]]:
    key_to_param, key_to_names = _optimizer_key_maps(model, optim, optim_input)
    aliases = _fqn_aliases(model, _get_fqn_to_fsdp_param_info(model))
    result: dict[Any, dict[str, Any]] = {}
    for key, value in optim_state_dict.get("state", {}).items():
        if isinstance(key, _OptimStateKey):
            result[key] = value
            continue
        elif isinstance(key, str):
            names = [aliases.get(key, key)]
        else:
            names = list(key_to_names.get(key, ()))
        if not names:
            continue
        for name in names:
            result[name] = value
    return result, key_to_param, key_to_names


def _public_names_for_info(model: Any, info: FSDPParamInfo) -> list[str]:
    model_names = [str(name) for name, _ in model.named_parameters()]
    names: list[str] = []
    for record in _records(info):
        matches = [name for name in model_names if name == record["name"] or name.endswith(record["name"])]
        names.append(matches[0] if matches else record["name"])
    return names


def _flatten_optim_state_dict(
    optim_state_dict: dict[str, Any],
    model: Any,
    use_orig_params: bool,
    optim: Any,
    rank0_only: bool,
    group: Any,
) -> dict[str, Any]:
    if not isinstance(optim_state_dict, dict) or "state" not in optim_state_dict:
        raise ValueError("optim_state_dict must contain a state mapping")
    if rank0_only and _world_size(group) > 1 and _rank(group) != 0:
        return {}
    named, _, _ = _named_state(optim_state_dict, model, optim)
    infos = _get_fqn_to_fsdp_param_info(model)
    result_state: dict[Any, Any] = {}
    consumed: set[Any] = set()
    for key, value in named.items():
        if not isinstance(key, _OptimStateKey):
            continue
        names = key.unflat_param_names
        if not names:
            continue
        info = infos.get(names[0])
        if info is None or not key.is_fsdp_managed:
            result_state[key] = _clone_state(value)
            consumed.add(key)
            continue
        if use_orig_params:
            states = _unflatten_optim_state(info, value, True, False, False)
            public_names = _public_names_for_info(model, info)
            for public_name, state in zip(public_names, states):
                result_state[public_name] = state
        else:
            result_state[key] = _clone_state(value)
        consumed.add(key)
        consumed.update(names)
    for info in {id(value): value for value in infos.values()}.values():
        names = _public_names_for_info(model, info)
        canonical = [record["name"] for record in _records(info)]
        state_by_name: dict[str, dict[str, Any]] = {}
        for public_name, local_name in zip(names, canonical):
            value = named.get(public_name, named.get(local_name))
            if value is not None:
                state_by_name[local_name] = value
                consumed.add(public_name)
                consumed.add(local_name)
        if not state_by_name:
            continue
        if use_orig_params:
            for public_name, local_name in zip(names, canonical):
                if local_name in state_by_name:
                    result_state[public_name] = _clone_state(state_by_name[local_name])
        else:
            flat_state = _flatten_optim_state(info, state_by_name, canonical)
            if flat_state:
                result_state[_OptimStateKey(tuple(names), True)] = flat_state
    for key, value in named.items():
        if key not in consumed and isinstance(key, str):
            result_state[_OptimStateKey((key,), False)] = _clone_state(value)
    result: dict[str, Any] = {"state": result_state}
    if "param_groups" in optim_state_dict:
        result["param_groups"] = copy.deepcopy(optim_state_dict["param_groups"])
    return result


def _flatten_optim_state(
    fsdp_param_info: FSDPParamInfo,
    unflat_osd_state: dict[str, dict[str, Any]],
    unflat_param_names: Iterable[str],
) -> dict[str, Any]:
    names = list(unflat_param_names)
    if not names:
        raise ValueError("at least one parameter is required")
    present = [unflat_osd_state.get(name) for name in names]
    if not any(state is not None for state in present):
        return {}
    state_names: set[str] | None = None
    for state in present:
        if state is None:
            continue
        current_names = set(state)
        if state_names is None:
            state_names = current_names
        elif state_names != current_names:
            raise ValueError(
                f"differing optimizer state names for parameters {tuple(names)}"
            )
    if state_names is None:
        raise AssertionError("optimizer state names are unavailable")
    result: dict[str, Any] = {}
    records = _records(fsdp_param_info)
    shapes = [record["shape"] for record in records]
    for state_name in sorted(state_names):
        values = [state.get(state_name) if state is not None else None for state in present]
        non_none = [value for value in values if value is not None]
        if all(_is_tensor(value) and value.dim() > 0 for value in non_none):
            result[state_name] = _flatten_tensor_optim_state(
                state_name, values, names, shapes, fsdp_param_info.handle
            )
        elif all(_is_zero_dim_tensor(value) for value in non_none):
            if any(value is None for value in values):
                raise ValueError(f"missing scalar state {state_name} for {names}")
            result[state_name] = _flatten_zero_dim_tensor_optim_state(state_name, values, names)
        elif all(not _is_tensor(value) for value in non_none):
            if any(value is None for value in values):
                raise ValueError(f"missing non-tensor state {state_name} for {names}")
            result[state_name] = _flatten_non_tensor_optim_state(state_name, values, names)
        else:
            raise ValueError(f"state {state_name} has incompatible value types")
    return result


def _flatten_tensor_optim_state(
    state_name: str,
    pos_dim_tensors: Sequence[Any],
    unflat_param_names: Sequence[str],
    unflat_param_shapes: Sequence[tuple[int, ...]],
    handle: Any,
) -> Any:
    tensors = [value for value in pos_dim_tensors if value is not None]
    if not tensors:
        return None
    dtype = tensors[0].dtype
    if any(value.dtype != dtype for value in tensors):
        raise ValueError(f"state {state_name} uses different dtypes")
    values: list[Any] = []
    for value, shape, name in zip(pos_dim_tensors, unflat_param_shapes, unflat_param_names):
        if value is None:
            values.append(tp.zeros(shape, dtype=dtype, device=tensors[0].device))
            continue
        if tuple(int(size) for size in value.shape) != tuple(shape):
            raise ValueError(f"state {state_name} for {name} has shape {value.shape}, expected {shape}")
        values.append(value.detach().reshape(-1))
    flatten = getattr(handle, "flatten_tensors", None)
    if callable(flatten):
        return flatten(values)
    return tp.cat(tuple(values), dim=0)


def _flatten_zero_dim_tensor_optim_state(
    state_name: str,
    zero_dim_tensors: Sequence[Any],
    unflat_param_names: Sequence[str],
) -> Any:
    if any(value is None or not _is_zero_dim_tensor(value) for value in zero_dim_tensors):
        raise ValueError(f"state {state_name} must be present for every parameter")
    first = zero_dim_tensors[0]
    if any(not _same_value(first, value) for value in zero_dim_tensors[1:]):
        raise ValueError(f"state {state_name} differs across {tuple(unflat_param_names)}")
    return first.detach().clone().cpu()


def _flatten_non_tensor_optim_state(
    state_name: str,
    non_tensors: Sequence[Any],
    unflat_param_names: Sequence[str],
) -> Any:
    if any(value is None for value in non_tensors):
        raise ValueError(f"state {state_name} must be present for every parameter")
    first = non_tensors[0]
    if any(not _same_value(first, value) for value in non_tensors[1:]):
        raise ValueError(f"state {state_name} differs across {tuple(unflat_param_names)}")
    return _clone_state(first)


def _rekey_sharded_optim_state_dict(
    sharded_osd: dict[str, Any],
    model: Any,
    optim: Any,
    optim_input: Any,
    using_optim_input: bool,
    is_named_optimizer: bool,
) -> dict[str, Any]:
    if "state" not in sharded_osd:
        raise ValueError("sharded optimizer state must contain state")
    key_to_param, key_to_names = _optimizer_key_maps(
        model,
        None if using_optim_input else optim,
        optim_input if using_optim_input else None,
    )
    del key_to_param
    names_to_key: dict[str, Any] = {}
    for key, names in key_to_names.items():
        for name in names:
            names_to_key[name] = key
    aliases = _fqn_aliases(model, _get_fqn_to_fsdp_param_info(model))
    for external, canonical in aliases.items():
        if canonical in names_to_key:
            names_to_key[external] = names_to_key[canonical]
    result_state: dict[Any, Any] = {}
    for key, value in sharded_osd["state"].items():
        if isinstance(key, _OptimStateKey):
            candidates = [names_to_key.get(name) for name in key.unflat_param_names]
            candidates = [candidate for candidate in candidates if candidate is not None]
            target = candidates[0] if candidates else key.unflat_param_names[0]
        elif isinstance(key, str):
            target = key if is_named_optimizer else names_to_key.get(key, key)
        else:
            target = key
        result_state[target] = value
    result: dict[str, Any] = {"state": result_state}
    if "param_groups" in sharded_osd:
        groups = []
        for source in sharded_osd["param_groups"]:
            group = copy.deepcopy(source)
            params = []
            for key in source.get("params", ()):
                name = aliases.get(key, key) if isinstance(key, str) else key
                params.append(name if is_named_optimizer else names_to_key.get(name, name))
            group["params"] = sorted(set(params), key=lambda item: (isinstance(item, str), item))
            groups.append(group)
        result["param_groups"] = groups
    return result


def _get_param_id_to_param_from_optim_input(model: Any, optim_input: Any) -> dict[Any, Any]:
    del model
    if optim_input is None:
        return {}
    values: list[Any] = []
    source = list(optim_input)
    if source and isinstance(source[0], dict):
        for group in source:
            values.extend(group.get("params", ()))
    else:
        values = source
    return {
        index: value[1] if isinstance(value, tuple) and len(value) == 2 else value
        for index, value in enumerate(values)
    }


def _get_flat_param_to_fqn(model: Any) -> dict[Any, str]:
    from ._flat_param import FlatParameter

    def module_fn(
        module: Any,
        prefix: str,
        tree_level: int,
        flat_param_to_fqn: dict[Any, str],
    ) -> None:
        del tree_level
        for param_name, param in _named_parameters_with_duplicates(
            module, recurse=False
        ):
            if isinstance(param, FlatParameter):
                flat_param_to_fqn[param] = clean_tensor_name(prefix + param_name)

    def return_fn(flat_param_to_fqn: dict[Any, str]) -> dict[Any, str]:
        return flat_param_to_fqn

    result: dict[Any, str] = {}
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        [name for name, _ in _named_parameters_with_duplicates(model)],
        result,
    )


def _get_param_key_to_param(
    optim: Any,
    model: Any,
    is_named_optimizer: bool,
    param_to_fqns: Any,
    flat_param_to_fqn: Any,
) -> dict[Any, Any]:
    del is_named_optimizer, param_to_fqns, flat_param_to_fqn
    result, _ = _optimizer_key_maps(model, optim)
    return result


def _get_param_to_param_key(
    optim: Any,
    model: Any,
    is_named_optimizer: bool,
    param_to_fqns: Any,
    flat_param_to_fqn: Any,
) -> dict[Any, Any]:
    return {
        value: key
        for key, value in _get_param_key_to_param(
            optim, model, is_named_optimizer, param_to_fqns, flat_param_to_fqn
        ).items()
    }


def _get_param_to_param_id_from_optim_input(model: Any, optim_input: Any) -> dict[Any, int]:
    return {value: key for key, value in _get_param_id_to_param_from_optim_input(model, optim_input).items()}


def _check_missing_keys_on_rank(
    state: Mapping[Any, Any],
    expected_keys: Iterable[Any],
    rank: int = 0,
) -> None:
    missing = [key for key in expected_keys if key not in state]
    if missing:
        raise RuntimeError(f"rank {rank} is missing optimizer states for {missing}")


def _map_param_key_to_optim_keys(
    optim_state_dict: Any,
    group: Any,
    param_key_to_param: Any,
    param_to_fqns: Any,
    fqn_to_fsdp_param_info: Any,
    merge_keys: bool,
) -> tuple[list[_OptimStateKey], dict[_OptimStateKey, Any]]:
    del group, merge_keys
    keys: list[_OptimStateKey] = []
    mapping: dict[_OptimStateKey, Any] = {}
    for param_key in optim_state_dict.get("state", {}):
        if isinstance(param_key, _OptimStateKey):
            key = param_key
        elif isinstance(param_key, str):
            key = _OptimStateKey((param_key,), param_key in fqn_to_fsdp_param_info)
        else:
            param = param_key_to_param.get(param_key)
            fqns = list(param_to_fqns.get(param, ()))
            if not fqns:
                continue
            key = _OptimStateKey(tuple(fqns), fqns[0] in fqn_to_fsdp_param_info)
        keys.append(key)
        mapping[key] = param_key
    return keys, mapping


def _unflatten_param_groups(
    state_dict: dict[str, Any],
    param_key_to_param: Any,
    param_to_fqns: Any,
) -> dict[str, Any]:
    if "param_groups" not in state_dict:
        return {}
    result = copy.deepcopy(state_dict)
    groups = []
    for source in state_dict["param_groups"]:
        group = copy.deepcopy(source)
        names: list[Any] = []
        for key in source.get("params", ()):
            if isinstance(key, str):
                names.append(key)
                continue
            param = param_key_to_param.get(key)
            names.extend(param_to_fqns.get(param, ()))
        group["params"] = names
        groups.append(group)
    result["param_groups"] = groups
    return result


def _is_named_optimizer(optim_state_dict: dict[str, Any]) -> bool:
    return bool(
        optim_state_dict.get("param_groups")
        and isinstance(optim_state_dict["param_groups"][0].get("params", [None])[0], str)
    )


def _allgather_state_info(fsdp_state: Any, input_states: Any) -> list[StateInfo]:
    local = StateInfo()
    for fqn, state in input_states.items():
        local.state[fqn] = {}
        local.metadata[fqn] = {}
        for state_name, value in state.items():
            if _is_tensor(value) and value.dim() > 0:
                local.metadata[fqn][state_name] = _PosDimTensorInfo(tuple(value.shape), value.dtype)
            else:
                local.state[fqn][state_name] = _clone_state(value, cpu_offload=True)
    info = FSDPParamInfo(fsdp_state, getattr(fsdp_state, "_param_group", None))
    group = _group_for_info(info)
    if _world_size(group) <= 1:
        return [local]
    gathered = [None for _ in range(_world_size(group))]
    dist.all_gather_object(gathered, local, group=group)
    return [value for value in gathered if isinstance(value, StateInfo)]


def _convert_all_state_info(
    fsdp_param_info: Any,
    gathered_state_info: Any,
    input_states: Any,
    output_states: Any,
) -> tuple[Any, dict[str, list[Any]]]:
    del fsdp_param_info
    state_buffers: dict[str, list[Any]] = {}
    dtype = None
    for info in gathered_state_info:
        for fqn, metadata in info.metadata.items():
            output_states.setdefault(fqn, {})
            for state_name, description in metadata.items():
                if not isinstance(description, _PosDimTensorInfo):
                    continue
                if dtype is None:
                    dtype = description.dtype
                elif dtype != description.dtype:
                    raise ValueError(f"state {state_name} has different dtypes across ranks")
                state_buffers.setdefault(state_name, []).append(description)
        for fqn, state in info.state.items():
            output_states.setdefault(fqn, {}).update(_clone_state(state))
    for fqn, state in input_states.items():
        output_states.setdefault(fqn, {})
        for state_name, value in state.items():
            if not (_is_tensor(value) and value.dim() > 0):
                output_states[fqn].setdefault(state_name, _clone_state(value))
    return dtype, state_buffers


def _unflatten_orig_param_states(
    fsdp_param_info: FSDPParamInfo,
    output_states: dict[str, dict[str, Any]],
    state_name: str,
    shard_state: bool,
    to_save: bool,
    cpu_offload: bool,
) -> None:
    if not to_save:
        return
    records = _records(fsdp_param_info)
    for record in records:
        state = output_states.get(record["name"])
        if state is None or state_name not in state:
            continue
        value = state[state_name]
        if _is_tensor(value) and value.dim() > 0:
            if int(value.numel()) != math.prod(record["shape"]):
                raise ValueError(f"state {state_name} has an invalid size")
            value = value.reshape(record["shape"])
            if shard_state:
                value = _local_shard(value, record, fsdp_param_info)
        state[state_name] = _clone_state(value, cpu_offload)


def _allgather_orig_param_states(
    fsdp_param_info: FSDPParamInfo,
    gathered_state_info: list[StateInfo],
    input_states: dict[str, Any],
    shard_state: bool,
    to_save: bool,
    cpu_offload: bool,
) -> dict[str, dict[str, Any]]:
    if not to_save:
        return {}
    records = _records(fsdp_param_info)
    records_by_name = {record["name"]: record for record in records}
    state_names: dict[str, set[str]] = {
        name: set(values) for name, values in input_states.items()
    }
    for info in gathered_state_info:
        for name, values in getattr(info, "state", {}).items():
            state_names.setdefault(name, set()).update(values)
        for name, values in getattr(info, "metadata", {}).items():
            state_names.setdefault(name, set()).update(values)
    ordered_names = [record["name"] for record in records]
    ordered_names.extend(name for name in state_names if name not in records_by_name)
    result: dict[str, dict[str, Any]] = {
        name: {} for name in ordered_names if name in state_names
    }
    group = _group_for_info(fsdp_param_info)
    world = _world_size(group)
    for name in ordered_names:
        if name not in state_names:
            continue
        record = records_by_name.get(name)
        local_states = input_states.get(name, {})
        remote_states = [
            getattr(info, "state", {}).get(name, {}) for info in gathered_state_info
        ]
        remote_metadata = [
            getattr(info, "metadata", {}).get(name, {})
            for info in gathered_state_info
        ]
        for state_name in sorted(state_names[name]):
            value = local_states.get(state_name)
            descriptions = [
                metadata.get(state_name)
                for metadata in remote_metadata
                if state_name in metadata
            ]
            is_tensor_state = _is_tensor(value) and value.dim() > 0
            is_tensor_state = is_tensor_state or bool(descriptions)
            if record is None or not is_tensor_state:
                candidates = [
                    states[state_name]
                    for states in remote_states
                    if state_name in states
                ]
                if value is not None:
                    candidates.insert(0, value)
                if not candidates:
                    continue
                first = candidates[0]
                if any(not _same_value(first, candidate) for candidate in candidates[1:]):
                    raise ValueError(
                        f"optimizer state {state_name} differs across ranks"
                    )
                result[name][state_name] = _clone_state(first, cpu_offload)
                continue

            owner = record["owner"]
            local_getter = getattr(owner, "_sharded_local_tensor", None)
            if not callable(local_getter):
                if value is not None:
                    result[name][state_name] = _clone_state(value, cpu_offload)
                continue
            local_param = local_getter()
            local_shape = tuple(int(size) for size in local_param.shape)
            dtype = getattr(value, "dtype", None)
            if dtype is None and descriptions:
                dtype = descriptions[0].dtype
            for description in descriptions[1:]:
                if description.dtype != dtype:
                    raise ValueError(
                        f"optimizer state {state_name} uses different dtypes across ranks"
                    )
            if value is None or not (_is_tensor(value) and value.dim() > 0):
                value = tp.zeros(local_shape, dtype=dtype, device=local_param.device)
            elif tuple(int(size) for size in value.shape) != local_shape:
                value = _local_shard(value, record, fsdp_param_info)
            elif world <= 1:
                value = value.detach().clone()
            full = _gather_param_value(value, record, fsdp_param_info)
            if shard_state:
                full = _local_shard(full, record, fsdp_param_info)
            result[name][state_name] = _clone_state(full, cpu_offload)
    return result


def _gather_all_orig_param_state(
    fsdp_param_info: FSDPParamInfo,
    input_states: dict[str, Any],
    shard_state: bool,
    to_save: bool,
    cpu_offload: bool,
) -> dict[str, Any]:
    return _allgather_orig_param_states(
        fsdp_param_info, [], input_states, shard_state, to_save, cpu_offload
    )


def _convert_state_with_orig_params(
    all_optim_state_keys: list[_OptimStateKey],
    optim_state_key_to_param_key: dict[_OptimStateKey, Any],
    fqn_to_fsdp_param_info: dict[str, FSDPParamInfo],
    optim_state_dict: dict[Any, Any],
    to_save: bool,
    shard_state: bool,
    cpu_offload: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in all_optim_state_keys:
        param_key = optim_state_key_to_param_key.get(key)
        if param_key not in optim_state_dict:
            continue
        state = optim_state_dict[param_key]
        if key.is_fsdp_managed:
            info = fqn_to_fsdp_param_info.get(key.unflat_param_names[0])
            if info is None:
                continue
            for name in key.unflat_param_names:
                result[name] = _shard_orig_param_state(info, name, state) if shard_state else _clone_state(state, cpu_offload)
        elif to_save:
            result[key.unflat_param_names[0]] = _clone_state(state, cpu_offload)
    return result


def _convert_state_with_flat_params(
    all_optim_state_keys: list[_OptimStateKey],
    optim_state_key_to_param_key: dict[_OptimStateKey, Any],
    fqn_to_fsdp_param_info: dict[str, FSDPParamInfo],
    optim_state_dict: dict[Any, Any],
    to_save: bool,
    shard_state: bool,
    cpu_offload: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in all_optim_state_keys:
        param_key = optim_state_key_to_param_key.get(key)
        if param_key not in optim_state_dict or not to_save:
            continue
        if key.is_fsdp_managed:
            info = fqn_to_fsdp_param_info.get(key.unflat_param_names[0])
            if info is None:
                continue
            states = _unflatten_optim_state(
                info, optim_state_dict[param_key], True, shard_state, cpu_offload
            )
            result.update(zip(key.unflat_param_names, states))
        else:
            result[key.unflat_param_names[0]] = _clone_state(optim_state_dict[param_key], cpu_offload)
    return result


def _optim_state_dict(
    model: Any,
    optim: Any,
    optim_state_dict: Any,
    optim_input: Any,
    rank0_only: bool,
    shard_state: bool,
    group: Any,
    using_optim_input: bool,
    use_orig_params: bool,
    cpu_offload: bool,
) -> dict[str, Any]:
    source = optim_state_dict if optim_state_dict is not None else optim.state_dict()
    if not isinstance(source, dict) or "state" not in source:
        raise ValueError("optimizer state must contain a state mapping")
    if rank0_only and not shard_state and _world_size(group) > 1 and _rank(group) != 0:
        return {}
    infos = _get_fqn_to_fsdp_param_info(model)
    aliases = _fqn_aliases(model, infos)
    _, key_to_names = _optimizer_key_maps(
        model,
        None if using_optim_input else optim,
        optim_input if using_optim_input else None,
    )
    result_state: dict[Any, Any] = {}
    for key, value in source["state"].items():
        if isinstance(key, _OptimStateKey):
            names = list(key.unflat_param_names)
            info = infos.get(names[0]) if names else None
            if info is None:
                result_state[names[0] if names else key] = _clone_state(value, cpu_offload)
                continue
            states = _unflatten_optim_state(info, value, True, shard_state, cpu_offload)
            result_state.update(zip(_public_names_for_info(model, info), states))
            continue
        names = [aliases.get(key, key)] if isinstance(key, str) else list(key_to_names.get(key, ()))
        if not names:
            result_state[key] = _clone_state(value, cpu_offload)
            continue
        for name in names:
            canonical = aliases.get(name, name)
            info = infos.get(canonical)
            if info is None:
                result_state[name] = _clone_state(value, cpu_offload)
                continue
            record = _record_for_name(info, canonical)
            if record is None:
                result_state[name] = _clone_state(value, cpu_offload)
                continue
            converted = value
            if _is_tensor(value) and value.dim() > 0 and not shard_state:
                converted = _gather_param_value(value, record, info)
            result_state[name] = _clone_state(converted, cpu_offload)
    result: dict[str, Any] = {"state": result_state}
    if "param_groups" in source:
        groups = copy.deepcopy(source["param_groups"])
        for group_data in groups:
            params: list[Any] = []
            for key in group_data.get("params", ()):
                params.extend(key_to_names.get(key, (key,)))
            group_data["params"] = list(dict.fromkeys(params))
        result["param_groups"] = groups
    del use_orig_params
    return result


def _get_fqn_to_fsdp_param_info(model: Any) -> dict[str, FSDPParamInfo]:
    def module_fn(
        module: Any,
        prefix: str,
        tree_level: int,
        fqn_to_param_info: dict[str, FSDPParamInfo],
    ) -> None:
        del tree_level
        state = getattr(module, "_fsdp_state", None)
        if state is None:
            return
        groups_getter = getattr(state, "_all_param_groups", None)
        if callable(groups_getter):
            param_groups = list(groups_getter())
        else:
            getter = getattr(state, "_fsdp_param_group", None)
            param_group = (
                getter() if callable(getter) else getattr(state, "_param_group", None)
            )
            param_groups = [param_group] if param_group is not None else []
        for param_group in param_groups:
            params = list(getattr(param_group, "params", ()))
            if not params:
                continue
            info = FSDPParamInfo(state, param_group)
            for index, owner in enumerate(params):
                local_fqn = str(
                    getattr(getattr(owner, "module_info", None), "fqn", index)
                )
                if local_fqn.isdigit() or (
                    prefix and not local_fqn.startswith(prefix.rstrip("."))
                ):
                    fqn = clean_tensor_name(prefix + local_fqn)
                else:
                    fqn = clean_tensor_name(local_fqn)
                info.param_indices[fqn] = index
                info.param_requires_grad.append(
                    bool(getattr(getattr(owner, "param", owner), "requires_grad", False))
                )
            for fqn in info.param_indices:
                fqn_to_param_info[fqn] = info

    def return_fn(
        fqn_to_param_info: dict[str, FSDPParamInfo],
    ) -> dict[str, FSDPParamInfo]:
        return fqn_to_param_info

    result: dict[str, FSDPParamInfo] = {}
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        [name for name, _ in _named_parameters_with_duplicates(model)],
        result,
    )


def _set_optim_use_dtensor(fsdp_state: Any, state_dict_settings: Any) -> None:
    config = getattr(state_dict_settings, "state_dict_config", None)
    fsdp_state._use_dtensor = bool(getattr(config, "_use_dtensor", False))
