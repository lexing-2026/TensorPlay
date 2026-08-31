"""Module traversal helpers for composable sharding."""

from typing import Any

from ._common_utils import _get_module_fsdp_state

__all__ = ["_get_fsdp_states_with_modules", "_get_fsdp_states", "_get_fsdp_handles"]


def _get_fsdp_states_with_modules(root: Any) -> list[tuple[Any, Any]]:
    return [
        (module, state)
        for module in root.modules()
        if (state := _get_module_fsdp_state(module)) is not None
    ]


def _get_fsdp_states(root: Any) -> list[Any]:
    return [state for _, state in _get_fsdp_states_with_modules(root)]


def _get_fsdp_handles(root: Any) -> list[Any]:
    handles = []
    for state in _get_fsdp_states(root):
        handle = getattr(state, "_handle", None)
        if handle is not None:
            handles.append(handle)
    return handles
