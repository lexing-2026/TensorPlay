from __future__ import annotations

from functools import partial
from typing import Any

from . import functions
from .api import RRef, rpc_async, rpc_sync
from .constants import UNSET_RPC_TIMEOUT

__all__ = ["RRefProxy"]


def _local_invoke(rref: RRef[Any], func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return getattr(rref.local_value(), func_name)(*args, **kwargs)


@functions.async_execution
def _local_invoke_async_execution(rref: RRef[Any], func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return getattr(rref.local_value(), func_name)(*args, **kwargs)


def _invoke_rpc(rref: RRef[Any], rpc_api: Any, func_name: str, timeout: float, *args: Any, **kwargs: Any) -> Any:
    invoke = _local_invoke
    method = getattr(type(rref.local_value()), func_name, None)
    if method is not None and hasattr(method, "_wrapped_async_rpc_function"):
        invoke = _local_invoke_async_execution
    return rpc_api(
        rref.owner(),
        invoke,
        args=(rref, func_name, args, kwargs),
        timeout=timeout,
    )


class RRefProxy:
    def __init__(self, rref: RRef[Any], rpc_api: Any, timeout: float = UNSET_RPC_TIMEOUT) -> None:
        self.rref = rref
        self.rpc_api = rpc_api
        self.rpc_timeout = timeout

    def __getattr__(self, func_name: str) -> Any:
        return partial(_invoke_rpc, self.rref, self.rpc_api, func_name, self.rpc_timeout)
