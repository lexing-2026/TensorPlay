from __future__ import annotations

import collections
import enum
from typing import Any

from . import constants as rpc_constants

__all__ = [
    "backend_registered",
    "register_backend",
    "construct_rpc_backend_options",
    "init_backend",
    "BackendValue",
    "BackendType",
]

BackendValue = collections.namedtuple(
    "BackendValue", ["construct_rpc_backend_options_handler", "init_backend_handler"]
)


def _backend_type_repr(self: enum.Enum) -> str:
    return "BackendType." + self.name


def _construct_tensorpipe(rpc_timeout: float, init_method: str, **kwargs: Any) -> Any:
    from .options import TensorPipeRpcBackendOptions

    return TensorPipeRpcBackendOptions(
        rpc_timeout=rpc_timeout,
        init_method=init_method,
        **{key: value for key, value in kwargs.items() if key in {"num_worker_threads", "device_maps", "devices", "_transports", "_channels"}},
    )


def _init_tensorpipe(store: Any, name: str, rank: int, world_size: int, rpc_backend_options: Any, native: Any = None) -> Any:
    from . import api

    return api._create_native_agent(store, name, rank, world_size, rpc_backend_options, native)


BackendType = enum.Enum(
    "BackendType",
    {"TENSORPIPE": BackendValue(_construct_tensorpipe, _init_tensorpipe)},
)
BackendType.__repr__ = _backend_type_repr  # type: ignore[assignment]


def _coerce_backend(backend: Any) -> enum.Enum:
    if isinstance(backend, str):
        try:
            return BackendType[backend]
        except KeyError as exc:
            raise ValueError(f"unknown RPC backend {backend!r}") from exc
    if isinstance(backend, BackendType):
        return backend
    raise TypeError(f"backend must be a BackendType or string, got {type(backend)!r}")


def backend_registered(backend_name: str) -> bool:
    return str(backend_name) in BackendType.__members__


def register_backend(backend_name: str, construct_rpc_backend_options_handler: Any, init_backend_handler: Any) -> enum.Enum:
    global BackendType
    name = str(backend_name)
    if backend_registered(name):
        raise RuntimeError(f"RPC backend {name} is already registered")
    members = {member.name: member.value for member in BackendType}
    members[name] = BackendValue(construct_rpc_backend_options_handler, init_backend_handler)
    BackendType = enum.Enum("BackendType", members)
    BackendType.__repr__ = _backend_type_repr  # type: ignore[assignment]
    return BackendType[name]


def construct_rpc_backend_options(
    backend: Any,
    rpc_timeout: float = rpc_constants.DEFAULT_RPC_TIMEOUT_SEC,
    init_method: str = rpc_constants.DEFAULT_INIT_METHOD,
    **kwargs: Any,
) -> Any:
    value = _coerce_backend(backend).value
    return value.construct_rpc_backend_options_handler(rpc_timeout, init_method, **kwargs)


def init_backend(backend: Any, *args: Any, **kwargs: Any) -> Any:
    return _coerce_backend(backend).value.init_backend_handler(*args, **kwargs)
