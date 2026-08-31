from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["flatten_state_dict", "unflatten_state_dict"]


def flatten_state_dict(state_dict: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[Any, ...]]]:
    flat: dict[str, Any] = {}
    mapping: dict[str, tuple[Any, ...]] = {}

    def visit(value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, path + (key,))
            return
        key = ".".join(str(part) for part in path)
        flat[key] = value
        mapping[key] = path

    visit(state_dict, ())
    return flat, mapping


def unflatten_state_dict(state_dict: Mapping[str, Any], mappings: Mapping[str, tuple[Any, ...]] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for flat_key, value in state_dict.items():
        path = mappings.get(flat_key) if mappings is not None else tuple(flat_key.split("."))
        current: dict[str, Any] = result
        for part in path[:-1]:
            current = current.setdefault(str(part), {})
        if path:
            current[str(path[-1])] = value
    return result
