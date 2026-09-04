from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from ._traverse import traverse_state_dict

FLATTEN_MAPPING = dict[str, tuple[Any, ...]]

__all__ = ["flatten_state_dict", "unflatten_state_dict"]


def flatten_state_dict(state_dict: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[Any, ...]]]:
    flat: dict[str, Any] = {}
    mapping: dict[str, tuple[Any, ...]] = {}

    def flat_copy(path: tuple[Any, ...], value: Any) -> None:
        key = ".".join(str(part) for part in path)
        if key in flat:
            raise ValueError(f"duplicated flattened key {key}")
        flat[key] = value
        mapping[key] = path

    traverse_state_dict(state_dict, flat_copy)
    return flat, mapping


def _set_element(root: MutableMapping[Any, Any], path: tuple[Any, ...], value: Any) -> None:
    if not path:
        raise ValueError("object path cannot be empty")
    current: Any = root
    for index, key in enumerate(path[:-1]):
        next_key = path[index + 1]
        default: Any = [] if isinstance(next_key, int) else {}
        if isinstance(current, MutableMapping):
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
    if isinstance(current, MutableMapping):
        current[last] = value
    else:
        if not isinstance(last, int):
            raise TypeError("list path components must be integers")
        while len(current) <= last:
            current.append(None)
        current[last] = value


def unflatten_state_dict(
    state_dict: Mapping[str, Any],
    mapping: Mapping[str, tuple[Any, ...]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for flat_key, value in state_dict.items():
        path = (
            tuple(mapping[flat_key])
            if mapping is not None and flat_key in mapping
            else tuple(flat_key.split("."))
        )
        _set_element(result, path, value)
    return result
