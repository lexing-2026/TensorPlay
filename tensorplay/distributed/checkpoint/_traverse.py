from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, MutableMapping, Sequence
from typing import Any, TypeVar, cast

import tensorplay as tp

PATH_ITEM = str | int
OBJ_PATH = tuple[PATH_ITEM, ...]
T = TypeVar("T")
STATE_DICT_ITEM = object
CONTAINER_TYPE = MutableMapping[PATH_ITEM, STATE_DICT_ITEM]

__all__ = ["traverse_state_dict", "set_element", "get_element", "print_tensor"]


def _keep_visiting_tensors(value: Any) -> bool:
    return isinstance(value, tp.Tensor) or callable(getattr(value, "__create_write_items__", None))


def traverse_state_dict(
    state_dict: Any,
    visitor: Callable[[OBJ_PATH, STATE_DICT_ITEM], None],
    keep_traversing: Callable[[STATE_DICT_ITEM], bool] = _keep_visiting_tensors,
) -> None:
    def _is_terminal(value: STATE_DICT_ITEM) -> bool:
        values: Collection[STATE_DICT_ITEM]
        if isinstance(value, Mapping):
            return False
        if isinstance(value, list):
            values = value
        else:
            return True
        for entry in values:
            if isinstance(entry, (Mapping, list)) and not _is_terminal(entry):
                return False
            if keep_traversing is not None and keep_traversing(entry):
                return False
        return True

    def _traverse_obj(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                _traverse_obj(path + (str(key),), child)
        elif _is_terminal(value):
            visitor(path, value)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                _traverse_obj(path + (index,), child)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)
    del _traverse_obj, _is_terminal


def set_element(root_dict: Any, path: Sequence[Any], value: Any) -> None:
    cur_container = cast(CONTAINER_TYPE, root_dict)

    def extend_list(values: list[STATE_DICT_ITEM], index: int) -> None:
        while len(values) <= index:
            values.append(None)

    for index in range(1, len(path)):
        previous = path[index - 1]
        key = path[index]
        default: STATE_DICT_ITEM = {} if type(key) is str else []
        if isinstance(cur_container, Mapping):
            cur_container = cast(CONTAINER_TYPE, cur_container.setdefault(previous, default))
        else:
            extend_list(cast(list[STATE_DICT_ITEM], cur_container), int(previous))
            if cur_container[int(previous)] is None:
                cur_container[int(previous)] = default
            cur_container = cast(CONTAINER_TYPE, cur_container[int(previous)])

    key = path[-1]
    if type(key) is int:
        extend_list(cast(list[STATE_DICT_ITEM], cur_container), key)
    cur_container[key] = value


def get_element(root_dict: Any, path: Sequence[Any], default_value: T | None = None) -> T | None:
    current = cast(CONTAINER_TYPE, root_dict)
    for part in path:
        if type(part) is int:
            if not isinstance(current, list) or len(current) <= part:
                return default_value
        elif not isinstance(current, Mapping) or part not in current:
            return default_value
        current = cast(CONTAINER_TYPE, current[part])
    return cast(T | None, current)


def _print_nested(value: Any, prefix: str = "", print_fun: Callable[[str], None] = print) -> None:
    if isinstance(value, tp.Tensor):
        print_fun(f"{prefix} Tensor size: {tuple(value.shape)}")
        return
    if callable(getattr(value, "local_shards", None)):
        print_fun(f"{prefix} ShardedTensor size: {value.size()}")
        for shard in value.local_shards():
            _print_nested(shard.tensor, f"{shard.metadata.shard_offsets} ", print_fun)
        return
    if callable(getattr(value, "to_local", None)) and hasattr(value, "device_mesh"):
        print_fun(f"{prefix} DistributedTensor size: {value.size()}")
        _print_nested(value.to_local(), print_fun=print_fun)
        return
    print_fun(f"{prefix} Type: {type(value)}")


def print_tensor(path: OBJ_PATH, value: STATE_DICT_ITEM, print_fun: Callable[[str], None] = print) -> None:
    _print_nested(value, prefix=str(path), print_fun=print_fun)
