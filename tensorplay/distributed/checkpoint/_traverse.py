from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import tensorplay as tp

__all__ = ["traverse_state_dict", "set_element", "get_element", "print_tensor"]


def _keep_visiting_tensors(value: Any) -> bool:
    return isinstance(value, (Mapping, list, tuple)) and not isinstance(value, tp.Tensor)


def traverse_state_dict(state_dict: Any, visitor: Callable[[tuple[Any, ...], Any], None]) -> None:
    def visit(path: tuple[Any, ...], value: Any) -> None:
        if _keep_visiting_tensors(value):
            items = value.items() if isinstance(value, Mapping) else enumerate(value)
            for key, child in items:
                visit(path + (key,), child)
        else:
            visitor(path, value)
    visit((), state_dict)


def traverse_state_dict_v_2_3(state_dict: Any, visitor: Callable[[tuple[Any, ...], Any], None]) -> None:
    traverse_state_dict(state_dict, visitor)


def set_element(root: Any, path: Sequence[Any], value: Any) -> None:
    current = root
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def get_element(root: Any, path: Sequence[Any]) -> Any:
    current = root
    for part in path:
        current = current[part]
    return current


def print_tensor(path: Sequence[Any], value: Any) -> None:
    if isinstance(value, tp.Tensor):
        print(".".join(map(str, path)), tuple(value.shape), value.dtype)
    else:
        print(".".join(map(str, path)), type(value).__name__)
