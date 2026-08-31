from __future__ import annotations

import copy
import io
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any, NamedTuple

import tensorplay as tp

try:
    from tensorplay.distributed.tensor import DTensor
except Exception:
    DTensor = ()
try:
    from tensorplay.distributed._shard.sharded_tensor import ShardedTensor
except Exception:
    ShardedTensor = ()

from . import distributed_core as dist

__all__ = [
    "CompanionMismatch",
    "_identity_func",
    "_all_gather_sharded_tensor",
    "_iterate_state_dict",
    "_gather_state_dict",
    "_offload_state_dict_to_cpu",
    "_copy_state_dict",
    "_create_cpu_state_dict",
    "_check_state_dict_similarity",
    "_TensorInfo",
    "_broadcast_tensors",
    "_distribute_tensors",
    "_broadcast_state_dict",
    "_distribute_state_dict",
    "_traverse_state_dict",
    "_flatten_state_dict",
    "_set_element",
    "_unflatten_state_dict",
]


def _identity_func(obj: Any, pg: Any = None, device: Any = None, companion_obj: Any = None) -> Any:
    del pg, device, companion_obj
    return obj


def _all_gather_sharded_tensor(sharded_tensor: Any, pg: Any = None, device: Any = None) -> Any:
    del pg, device
    gather = getattr(sharded_tensor, "gather", None)
    if gather is None:
        raise TypeError("object does not provide gather")
    return gather()


class CompanionMismatch(Exception):
    pass


def _is_tensor(value: Any) -> bool:
    return isinstance(value, tp.Tensor)


def _is_dtensor(value: Any) -> bool:
    return DTensor != () and isinstance(value, DTensor)


def _is_sharded_tensor(value: Any) -> bool:
    return ShardedTensor != () and isinstance(value, ShardedTensor)


def _iterate_state_dict(
    iter_object: Any,
    sharded_tensor_func: Callable[..., Any],
    dtensor_func: Callable[..., Any],
    tensor_func: Callable[..., Any],
    *,
    pg: Any = None,
    device: Any = None,
    cpu_offload: bool = False,
    companion_obj: Any = None,
    ranks_only: tuple[int, ...] = (),
    type_check: bool = True,
    non_blocking: bool = True,
) -> Any:
    if _is_sharded_tensor(iter_object):
        result = sharded_tensor_func(iter_object, pg, device, companion_obj)
    elif _is_dtensor(iter_object):
        result = dtensor_func(iter_object, pg, device, companion_obj)
    elif _is_tensor(iter_object):
        result = tensor_func(iter_object, pg, device, companion_obj)
    elif iter_object is None or isinstance(iter_object, (bool, int, float, complex, str, bytes, io.BytesIO)):
        result = iter_object
    elif isinstance(iter_object, Mapping):
        if companion_obj is not None and (not isinstance(companion_obj, Mapping) or set(companion_obj) != set(iter_object)):
            raise CompanionMismatch("mapping structures differ")
        result = {
            key: _iterate_state_dict(
                value,
                sharded_tensor_func,
                dtensor_func,
                tensor_func,
                pg=pg,
                device=device,
                cpu_offload=cpu_offload,
                companion_obj=companion_obj[key] if companion_obj is not None else None,
                ranks_only=ranks_only,
                type_check=type_check,
                non_blocking=non_blocking,
            )
            for key, value in iter_object.items()
        }
    elif isinstance(iter_object, (list, tuple)):
        if companion_obj is not None and (not isinstance(companion_obj, (list, tuple)) or len(companion_obj) != len(iter_object)):
            raise CompanionMismatch("sequence structures differ")
        result_items = [
            _iterate_state_dict(
                value,
                sharded_tensor_func,
                dtensor_func,
                tensor_func,
                pg=pg,
                device=device,
                cpu_offload=cpu_offload,
                companion_obj=companion_obj[index] if companion_obj is not None else None,
                ranks_only=ranks_only,
                type_check=type_check,
                non_blocking=non_blocking,
            )
            for index, value in enumerate(iter_object)
        ]
        result = tuple(result_items) if isinstance(iter_object, tuple) else result_items
    elif not type_check:
        result = copy.deepcopy(iter_object)
    else:
        raise ValueError(f"unsupported state-dict value type: {type(iter_object)!r}")

    if ranks_only:
        try:
            active = dist.get_rank(pg) in ranks_only
        except Exception:
            active = 0 in ranks_only
        if not active:
            return {} if isinstance(result, dict) else None

    if cpu_offload:
        if _is_tensor(result):
            result = result.cpu()
        elif _is_dtensor(result) or _is_sharded_tensor(result):
            result = _to_cpu(result)

    if companion_obj is not None and _is_tensor(result):
        if _is_tensor(companion_obj):
            companion_obj.copy_(result, non_blocking=non_blocking)
            return companion_obj
        raise CompanionMismatch("tensor companion has a different type")
    return result


def _to_cpu(value: Any) -> Any:
    if _is_dtensor(value):
        return value.full_tensor().cpu()
    if _is_sharded_tensor(value):
        return value.gather().cpu()
    return value.cpu() if hasattr(value, "cpu") else value


def _gather_state_dict(state_dict: dict[str, Any], *, pg: Any = None, device: Any = None, cpu_offload: bool = False, ranks_only: tuple[int, ...] = (), type_check: bool = True) -> dict[str, Any]:
    def sharded(value: Any, group: Any, target_device: Any, companion: Any) -> Any:
        del group, target_device, companion
        return _all_gather_sharded_tensor(value, pg, device)

    def distributed(value: Any, group: Any, target_device: Any, companion: Any) -> Any:
        del group, target_device, companion
        return value.full_tensor() if hasattr(value, "full_tensor") else value.to_local()

    return _iterate_state_dict(
        state_dict, sharded, distributed, _identity_func,
        pg=pg, device=device, cpu_offload=cpu_offload,
        ranks_only=ranks_only, type_check=type_check,
    )


def _offload_state_dict_to_cpu(state_dict: dict[str, Any], *, ranks_only: tuple[int, ...] = (), type_check: bool = True) -> dict[str, Any]:
    return _iterate_state_dict(
        state_dict, lambda value, *_: _to_cpu(value), lambda value, *_: _to_cpu(value),
        lambda value, *_: value.cpu(), ranks_only=ranks_only, type_check=type_check,
    )


def _copy_state_dict(state_dict: dict[str, Any], copy_state_dict: dict[str, Any], non_blocking: bool = False, type_check: bool = True) -> dict[str, Any]:
    return _iterate_state_dict(
        state_dict, _identity_func, _identity_func, _identity_func,
        companion_obj=copy_state_dict, type_check=type_check, non_blocking=non_blocking,
    )


def _create_cpu_state_dict(state_dict: dict[str, Any], pin_memory: bool = False, share_memory: bool = False) -> dict[str, Any]:
    def copy_tensor(value: Any, *_: Any) -> Any:
        result = value.detach().cpu().clone()
        if pin_memory and hasattr(result, "pin_memory"):
            result = result.pin_memory()
        if share_memory and hasattr(result, "share_memory_"):
            result = result.share_memory_()
        return result

    return _iterate_state_dict(
        state_dict,
        lambda value, *_: copy_tensor(value),
        lambda value, *_: copy_tensor(value.full_tensor() if hasattr(value, "full_tensor") else value.to_local()),
        copy_tensor,
        type_check=False,
    )


def _check_state_dict_similarity(state_dict: dict[str, Any], compared_state_dict: dict[str, Any]) -> bool:
    def check(value: Any, _pg: Any, _device: Any, companion: Any) -> Any:
        if not _is_tensor(companion) or value.shape != companion.shape or value.dtype != companion.dtype:
            raise CompanionMismatch
        return value
    try:
        _iterate_state_dict(state_dict, _identity_func, _identity_func, check, companion_obj=compared_state_dict, type_check=False)
    except (CompanionMismatch, KeyError, TypeError, ValueError):
        return False
    return True


class _TensorInfo(NamedTuple):
    size: tuple[int, ...]
    dtype: Any


def _broadcast_tensors(full_state_dict: dict[str, Any], local_state_dict: dict[str, Any], keys: list[str], device: Any, pg: Any = None) -> None:
    if not keys:
        return
    if not dist.is_initialized():
        for key in keys:
            if key in full_state_dict and key in local_state_dict and _is_tensor(local_state_dict[key]):
                local_state_dict[key].copy_(full_state_dict[key])
        return
    for key in keys:
        value = full_state_dict[key] if dist.get_rank(pg) == 0 else local_state_dict[key]
        if not _is_tensor(value):
            continue
        value = value.to(device) if device is not None else value
        dist.broadcast(value, src=0, group=pg)
        local_state_dict[key] = value


def _distribute_tensors(local_state_dict: dict[str, Any], keys: list[str], device: Any, pg: Any = None) -> None:
    del device, pg
    for key in keys:
        value = local_state_dict.get(key)
        if isinstance(value, tuple) and len(value) == 2:
            local, full = value
            if hasattr(local, "from_local"):
                local_state_dict[key] = local.from_local(full)
            elif hasattr(local, "to_local"):
                local.to_local().copy_(full)
                local_state_dict[key] = local
            else:
                local_state_dict[key] = full


def _broadcast_state_dict(full_state_dict: dict[str, Any], local_state_dict: dict[str, Any], device: Any, pg: Any = None, strict: bool = False, cpu_offload: bool = False) -> None:
    if not dist.is_initialized():
        local_state_dict.update(copy.deepcopy(full_state_dict))
        return
    if dist.get_rank(pg) == 0:
        payload = {key: (_TensorInfo(tuple(value.shape), value.dtype) if _is_tensor(value) else value) for key, value in full_state_dict.items()}
    else:
        payload = {}
    objects = [payload]
    dist.broadcast_object_list(objects, src=0, group=pg)
    payload = objects[0]
    keys = [key for key, value in payload.items() if isinstance(value, _TensorInfo)]
    for key, value in payload.items():
        if not isinstance(value, _TensorInfo):
            local_state_dict[key] = value
        elif dist.get_rank(pg) == 0:
            local_state_dict[key] = full_state_dict[key]
        elif key not in local_state_dict:
            local_state_dict[key] = tp.empty(value.size, dtype=value.dtype, device=device)
    _broadcast_tensors(full_state_dict, local_state_dict, keys, device, pg)
    if cpu_offload:
        for key in keys:
            local_state_dict[key] = local_state_dict[key].cpu()
    if strict:
        for key in set(local_state_dict) - set(payload):
            del local_state_dict[key]


def _distribute_state_dict(full_state_dict: dict[str, Any], local_state_dict: dict[str, Any], device: Any, pg: Any = None) -> None:
    del pg
    for key, value in full_state_dict.items():
        if not _is_tensor(value):
            local_state_dict[key] = value
        elif key not in local_state_dict:
            local_state_dict[key] = value.to(device) if device is not None else value
        elif _is_dtensor(local_state_dict[key]):
            local_state_dict[key] = local_state_dict[key].from_local(value.to(device) if device is not None else value)
        else:
            local_state_dict[key].copy_(value)


PATH_ITEM = str | int
OBJ_PATH = tuple[PATH_ITEM, ...]
FLATTEN_MAPPING = dict[str, OBJ_PATH]
STATE_DICT_TYPE = dict[str, Any]
CONTAINER_TYPE = MutableMapping[PATH_ITEM, Any]


def _traverse_state_dict(state_dict: STATE_DICT_TYPE, visitor: Callable[[OBJ_PATH, Any], None]) -> None:
    def visit(path: OBJ_PATH, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(path + (str(key),), child)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(path + (index,), child)
        else:
            visitor(path, value)
    for key, value in state_dict.items():
        visit((str(key),), value)


def _flatten_state_dict(state_dict: STATE_DICT_TYPE) -> tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
    flattened: STATE_DICT_TYPE = {}
    mapping: FLATTEN_MAPPING = {}
    def add(path: OBJ_PATH, value: Any) -> None:
        key = ".".join(str(item) for item in path)
        if key in flattened:
            raise ValueError(f"duplicated flattened key {key}")
        flattened[key] = value
        mapping[key] = path
    _traverse_state_dict(state_dict, add)
    return flattened, mapping


def _set_element(root_dict: STATE_DICT_TYPE, path: OBJ_PATH, value: Any) -> None:
    if not path:
        raise ValueError("object path cannot be empty")
    current: Any = root_dict
    for index, key in enumerate(path[:-1]):
        next_key = path[index + 1]
        default: Any = {} if isinstance(next_key, str) else []
        if isinstance(current, Mapping):
            if key not in current:
                current[key] = default
            current = current[key]
        else:
            if not isinstance(key, int):
                raise TypeError("list path components must be integers")
            while len(current) <= key:
                current.append(None)
            if current[key] is None:
                current[key] = default
            current = current[key]
    last = path[-1]
    if isinstance(current, Mapping):
        current[last] = value
    else:
        if not isinstance(last, int):
            raise TypeError("list path components must be integers")
        while len(current) <= last:
            current.append(None)
        current[last] = value


def _unflatten_state_dict(state_dict: STATE_DICT_TYPE, mapping: FLATTEN_MAPPING) -> STATE_DICT_TYPE:
    result: STATE_DICT_TYPE = {}
    for key, value in state_dict.items():
        _set_element(result, mapping[key], value)
    return result
