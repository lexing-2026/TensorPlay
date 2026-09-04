"""Metadata for values that are stored outside the graph data flow."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

__all__ = [
    "ScriptObjectMeta",
    "register_custom_object",
    "register_dataclass",
    "registered_dataclass_name",
    "resolve_custom_object",
]


@dataclass(frozen=True)
class ScriptObjectMeta:
    constant_name: str
    class_fqn: str


_CUSTOM_OBJECTS: dict[str, Any] = {}
_DATACLASS_NAMES: dict[type, str] = {}


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


def registered_dataclass_name(cls: type[Any]) -> str | None:
    """The stable name a dataclass was registered under, if any."""

    return _DATACLASS_NAMES.get(cls)


def register_dataclass(
    cls: type[Any],
    *,
    serialized_type_name: str | None = None,
    return_none_fields: bool = True,
) -> type[Any]:
    """Register a dataclass as a flattenable graph value.

    ``serialized_type_name`` pins the qualified name recorded in serialized
    artifacts; it must resolve back to ``cls`` when the program is loaded.
    ``return_none_fields`` keeps ``None``-valued fields as single ``None``
    leaves instead of flattening them by field.
    """

    if not isinstance(cls, type) or not is_dataclass(cls):
        raise TypeError("register_dataclass expects a dataclass type")
    name = serialized_type_name or f"{cls.__module__}.{cls.__qualname__}"
    if "<locals>" in name:
        # locally-scoped classes cannot be re-imported by a loader; only
        # reject them when no stable serialized name was supplied
        if serialized_type_name is None:
            name = f"{cls.__module__}.{cls.__name__}"
    if name in _DATACLASS_NAMES and _DATACLASS_NAMES[name] is not cls:
        raise ValueError(f"serialized type name {name!r} is already registered")
    _DATACLASS_NAMES[cls] = name
    from ..graph._pytree import register_pytree_node

    field_names = tuple(field.name for field in fields(cls))

    def flatten(value: Any) -> tuple[list[Any], tuple[str, ...]]:
        if return_none_fields:
            return [getattr(value, name) for name in field_names], field_names
        values: list[Any] = []
        kept: list[str] = []
        for name in field_names:
            item = getattr(value, name)
            if item is None:
                values.append(None)
            else:
                values.append(item)
                kept.append(name)
        return values, tuple(kept)

    def unflatten(values: list[Any], context: tuple[str, ...]) -> Any:
        bound = dict(zip(context, values))
        return cls(**{name: bound.get(name) for name in field_names})

    register_pytree_node(cls, flatten, unflatten)
    return cls
