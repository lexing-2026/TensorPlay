from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def _group_membership_management(store: Any, name: str, is_join: bool):
    token_key = "RpcGroupManagementToken"
    action = "join" if is_join else "leave"
    token = f"Token_for_{name}_{action}"
    while True:
        returned = store.compare_set(token_key, "", token)
        if isinstance(returned, bytes):
            returned = returned.decode()
        if returned in {"", token}:
            try:
                yield
            finally:
                store.set(token_key, "")
                store.set(token, "Done")
            return
        try:
            store.wait([returned])
        except Exception:
            logger.exception("group membership token wait failed")
            raise


def _update_group_membership(worker_info: Any, my_devices: list[Any], reverse_device_map: dict[Any, Any], is_join: bool) -> Any:
    from .api import _get_current_rpc_agent

    agent = _get_current_rpc_agent()
    update = getattr(agent, "_update_group_membership", None)
    if update is not None:
        return update(worker_info, my_devices, reverse_device_map, is_join)
    state = getattr(agent, "group_membership", None)
    if state is None:
        state = agent.group_membership = {}
    key = getattr(worker_info, "name", str(worker_info))
    if is_join:
        state[key] = {"devices": list(my_devices), "reverse_device_map": dict(reverse_device_map)}
    else:
        state.pop(key, None)
    return None
