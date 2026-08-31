"""State and attribute restoration helpers for captured modules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

__all__ = ["_clear_traced_params_buffers", "_get_underlying_module", "_restore_state_dict"]


def _get_underlying_module(module_or_method: Any) -> Any:
    if callable(getattr(module_or_method, "named_modules", None)):
        return module_or_method
    owner = getattr(module_or_method, "__self__", None)
    if owner is not None and callable(getattr(owner, "named_modules", None)):
        return owner
    raise TypeError(f"expected a module or bound module method, got {type(module_or_method).__name__}")


def _clear_traced_params_buffers(traced_module: Any, const_keys: Sequence[str]) -> None:
    buffers = getattr(traced_module, "_buffers", None)
    if not isinstance(buffers, dict):
        buffers = getattr(getattr(traced_module, "root", None), "_buffers", None)
    if not isinstance(buffers, dict):
        raise TypeError("traced module does not expose a buffer mapping")
    for key in const_keys:
        if key not in buffers:
            raise KeyError(f"buffer {key!r} was not found")
        value = buffers.pop(key)
        setattr(traced_module, key, value)


def _get_attr(root: Any, name: str) -> Any:
    value = root
    for atom in name.split("."):
        value = getattr(value, atom)
    return value


def _restore_state_dict(
    original_module: Any | Callable[..., Any],
    traced_module: Any,
) -> None:
    """Restore graph attribute targets using object identity and qualified names."""

    original = _get_underlying_module(original_module)
    by_id: dict[int, str] = {}
    named_parameters = getattr(original, "named_parameters", None)
    if callable(named_parameters):
        for name, value in named_parameters():
            by_id[id(value)] = name
    named_buffers = getattr(original, "named_buffers", None)
    if callable(named_buffers):
        for name, value in named_buffers():
            by_id[id(value)] = name

    root = getattr(traced_module, "root", traced_module)
    for node in getattr(getattr(traced_module, "graph", None), "nodes", ()):
        if node.op != "get_attr" or not isinstance(node.target, str):
            continue
        try:
            value = _get_attr(root, node.target)
        except AttributeError:
            continue
        replacement = by_id.get(id(value))
        if replacement is not None:
            node.target = replacement
