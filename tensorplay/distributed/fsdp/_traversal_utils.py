"""Module traversal helpers for composable sharding."""

import collections
from typing import Any

from ._common_utils import _get_module_fsdp_state

__all__ = ["_get_fsdp_states_with_modules", "_get_fsdp_states", "_get_fsdp_handles"]


def _composable(module: Any) -> bool:
    if _get_module_fsdp_state(module) is not None:
        return True
    try:
        from ..._composable.contract import _get_registry

        registry = _get_registry(module)
    except (ImportError, AttributeError):
        return True
    return not any(
        key in registry
        for key in ("replicate", "__replicate_state_key__", "__replicate_with_fsdp_state__")
    )


def _get_fsdp_states_with_modules(root: Any) -> tuple[list[Any], list[Any]]:
    states: list[Any] = []
    modules: list[Any] = []
    visited_states: set[int] = set()
    visited_modules: set[int] = set()
    queue: collections.deque[Any] = collections.deque([root])
    while queue:
        module = queue.popleft()
        if id(module) in visited_modules:
            continue
        visited_modules.add(id(module))
        if not _composable(module):
            continue
        for child in reversed(list(module.children())):
            queue.appendleft(child)
        state = _get_module_fsdp_state(module)
        if state is not None and id(state) not in visited_states:
            visited_states.add(id(state))
            states.append(state)
            modules.append(module)
    return states, modules


def _get_fsdp_states(root: Any) -> list[Any]:
    states, _ = _get_fsdp_states_with_modules(root)
    return states


def _get_fsdp_handles(root: Any) -> list[Any]:
    handles = []
    for state in _get_fsdp_states(root):
        handle = getattr(state, "_handle", None)
        if handle is not None:
            handles.append(handle)
    return handles
