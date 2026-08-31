"""Metadata for values that are stored outside the graph data flow."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

__all__ = [
    "ScriptObjectMeta",
    "register_custom_object",
    "register_dataclass",
    "resolve_custom_object",
]


@dataclass(frozen=True)
class ScriptObjectMeta:
    constant_name: str
    class_fqn: str


_CUSTOM_OBJECTS: dict[str, Any] = {}


def register_custom_object(name: str, value: Any) -> ScriptObjectMeta:
    if not isinstance(name, str) or not name:
        raise ValueError("custom object name must be a non-empty string")
    _CUSTOM_OBJECTS[name] = value
    return ScriptObjectMeta(name, f"{type(value).__module__}.{type(value).__qualname__}")


def resolve_custom_object(meta: ScriptObjectMeta | str) -> Any:
    name = meta.constant_name if isinstance(meta, ScriptObjectMeta) else meta
    try:
        return _CUSTOM_OBJECTS[name]
    except KeyError as exc:
        raise KeyError(f"custom object {name!r} is not registered") from exc


def register_dataclass(
    cls: type[Any],
    *,
    serialized_type_name: str | None = None,
    return_none_fields: bool = True,
) -> type[Any]:
    """Register a dataclass as a flattenable graph value."""

    del serialized_type_name, return_none_fields
    if not isinstance(cls, type) or not is_dataclass(cls):
        raise TypeError("register_dataclass expects a dataclass type")
    from ..graph._pytree import register_pytree_node

    field_names = tuple(field.name for field in fields(cls))

    def flatten(value: Any) -> tuple[list[Any], tuple[str, ...]]:
        return [getattr(value, name) for name in field_names], field_names

    def unflatten(values: list[Any], context: tuple[str, ...]) -> Any:
        return cls(**dict(zip(context, values)))

    register_pytree_node(cls, flatten, unflatten)
    return cls
