from __future__ import annotations

import collections
import enum
from typing import Any

from . import constants as rpc_constants
from .options import _parse_device

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


def _init_process_group(store: Any, rank: int, world_size: int) -> Any:
    from tensorplay._C import _distributed as distributed

    if store is None:
        raise TypeError("store must be provided for process-group initialization")
    options = distributed.GlooOptions()
    options.timeout_ms = int(
        rpc_constants.DEFAULT_PROCESS_GROUP_TIMEOUT.total_seconds() * 1000
    )
    options.add_device(distributed.ProcessGroupGloo.create_default_device(False))
    group = distributed.ProcessGroupGloo(store, int(rank), int(world_size), options)
    if group is None:
        raise RuntimeError("failed to initialize the process group")
    if rank != -1 and int(group.rank()) != int(rank):
        raise RuntimeError("rank does not match the process group")
    if world_size != -1 and int(group.size()) != int(world_size):
        raise RuntimeError("world size does not match the process group")
    return group


def _tensorpipe_construct_rpc_backend_options_handler(
    rpc_timeout: float,
    init_method: str,
    num_worker_threads: int = rpc_constants.DEFAULT_NUM_WORKER_THREADS,
    _transports: Any = None,
    _channels: Any = None,
    **kwargs: Any,
) -> Any:
    return _construct_tensorpipe(
        rpc_timeout,
        init_method,
        num_worker_threads=num_worker_threads,
        _transports=_transports,
        _channels=_channels,
        **kwargs,
    )


def _tensorpipe_validate_devices(devices: Any, device_count: int) -> bool:
    for device in devices:
        parsed = device if hasattr(device, "type") else _parse_device(device)
        if parsed.type == "cpu":
            continue
        if parsed.type != "cuda" or parsed.index is None:
            return False
        if parsed.index < 0 or parsed.index >= int(device_count):
            return False
    return True


def _tensorpipe_exchange_and_check_all_device_maps(
    my_name: str,
    my_device_count: int,
    my_device_maps: dict[str, dict[Any, Any]],
    my_devices: list[Any],
    group: Any,
) -> tuple[dict[str, dict[Any, Any]], list[Any]]:
    gathered: list[Any] = [None for _ in range(int(group.size()))]
    value = (my_name, int(my_device_count), my_device_maps, my_devices)
    if hasattr(group, "all_gather_object"):
        group.all_gather_object(gathered, value)
    elif hasattr(group, "all_gather"):
        group.all_gather(gathered, value)
    else:
        raise TypeError("process group does not support object gathering")
    all_names = [item[0] for item in gathered]
    all_device_counts = {item[0]: item[1] for item in gathered}
    all_device_maps = {item[0]: item[2] for item in gathered}
    all_devices = {item[0]: item[3] for item in gathered}
    _validate_device_maps(
        all_names, all_device_counts, all_device_maps, all_devices
    )
    reverse_device_maps = _create_reverse_mapping(
        my_name, all_names, all_device_maps
    )
    return reverse_device_maps, _create_device_list(
        my_devices, my_device_maps, reverse_device_maps
    )


def _validate_device_maps(
    all_names: list[str],
    all_device_counts: dict[str, int],
    all_device_maps: dict[str, dict[str, dict[Any, Any]]],
    all_devices: dict[str, list[Any]],
    is_static_group: bool = True,
) -> None:
    names = set(all_names)
    for node in all_names:
        devices = all_devices.get(node, [])
        if len(set(devices)) != len(devices):
            raise ValueError(f"worker {node} has duplicated devices")
        if not _tensorpipe_validate_devices(devices, all_device_counts[node]):
            raise ValueError(f"worker {node} has invalid devices")

    for source_node in all_names:
        source_maps = all_device_maps.get(source_node, {})
        if is_static_group and not set(source_maps).issubset(names):
            raise ValueError(f"worker {source_node} has an unknown map target")
        for target_node, mapping in source_maps.items():
            if len(set(mapping.values())) != len(mapping):
                raise ValueError(
                    f"worker {source_node} maps multiple devices to one target"
                )
            source_devices = all_devices.get(source_node, [])
            if source_devices:
                if not set(mapping).issubset(source_devices):
                    raise ValueError(
                        f"worker {source_node} maps an unlisted source device"
                    )
            elif not _tensorpipe_validate_devices(
                mapping.keys(), all_device_counts[source_node]
            ):
                raise ValueError(f"worker {source_node} has invalid source devices")
            target_devices = all_devices.get(target_node, [])
            if target_devices:
                if not set(mapping.values()).issubset(target_devices):
                    raise ValueError(
                        f"worker {source_node} maps to an unlisted target device"
                    )
            elif target_node in all_device_counts and not _tensorpipe_validate_devices(
                mapping.values(), all_device_counts[target_node]
            ):
                raise ValueError(f"worker {source_node} has invalid target devices")


def _create_device_list(
    my_devices: list[Any],
    my_device_maps: dict[str, dict[Any, Any]],
    reverse_device_maps: dict[str, dict[Any, Any]],
) -> list[Any]:
    if not my_devices:
        devices: set[Any] = set()
        for mapping in my_device_maps.values():
            devices.update(mapping)
        for mapping in reverse_device_maps.values():
            devices.update(mapping)
        devices = {device for device in devices if getattr(device, "type", None) != "cpu"}
        my_devices = list(devices)
    return sorted(
        my_devices,
        key=lambda device: (
            getattr(device, "index", None) is None,
            getattr(device, "index", -1),
            str(device),
        ),
    )


def _create_reverse_mapping(
    my_name: str,
    all_names: list[str],
    all_device_maps: dict[str, dict[str, dict[Any, Any]]],
) -> dict[str, dict[Any, Any]]:
    reverse: dict[str, dict[Any, Any]] = {}
    for node in all_names:
        mapping = all_device_maps.get(node, {}).get(my_name)
        if mapping:
            reverse[node] = {target: source for source, target in mapping.items()}
    return reverse


def _get_device_infos() -> tuple[int, dict[str, dict[Any, Any]], list[Any]]:
    from . import api
    from .options import _to_device_list

    agent = api._get_current_rpc_agent()
    options = agent.get_backend_options()
    try:
        from tensorplay import cuda

        device_count = int(cuda.device_count())
    except (AttributeError, ImportError):
        device_count = 0
    devices = list(getattr(options, "devices", []))
    return device_count, dict(getattr(options, "device_maps", {})), _to_device_list(devices)


def _set_devices_and_reverse_device_map(agent: Any) -> None:
    from . import api
    from ._utils import _update_group_membership

    worker = agent.get_worker_info()
    workers = agent.get_worker_infos()
    counts: dict[str, int] = {}
    maps: dict[str, dict[str, dict[Any, Any]]] = {}
    devices: dict[str, list[Any]] = {}
    names: list[str] = []
    for info in workers:
        if info.name == worker.name:
            count, mapping, listed = _get_device_infos()
        else:
            count, mapping, listed = api.rpc_sync(info.name, _get_device_infos)
        counts[info.name] = count
        maps[info.name] = mapping
        devices[info.name] = listed
        names.append(info.name)
    _validate_device_maps(names, counts, maps, devices, is_static_group=False)
    reverse = _create_reverse_mapping(worker.name, names, maps)
    for name in names:
        devices[name] = _create_device_list(devices[name], maps[name], reverse)
        _update_group_membership(worker, devices[name], reverse, True)


def _tensorpipe_init_backend_handler(
    store: Any,
    name: str,
    rank: int,
    world_size: int,
    rpc_backend_options: Any,
) -> Any:
    from . import api
    from .options import TensorPipeRpcBackendOptions

    if not isinstance(rpc_backend_options, TensorPipeRpcBackendOptions):
        raise TypeError("rpc_backend_options must be TensorPipeRpcBackendOptions")
    try:
        from tensorplay import cuda

        device_count = int(cuda.device_count())
    except (AttributeError, ImportError):
        device_count = 0
    _validate_device_maps(
        [str(name)],
        {str(name): device_count},
        {str(name): rpc_backend_options.device_maps},
        {str(name): list(rpc_backend_options.devices)},
        is_static_group=False,
    )
    agent = _init_tensorpipe(
        store, name, rank, world_size, rpc_backend_options
    )
    api._init_rpc_states(agent)
    return agent


def _init_tensorpipe(store: Any, name: str, rank: int, world_size: int, rpc_backend_options: Any, native: Any = None) -> Any:
    from . import api

    return api._create_native_agent(store, name, rank, world_size, rpc_backend_options, native)


BackendType = enum.Enum(
    "BackendType",
    {
        "TENSORPIPE": BackendValue(
            _tensorpipe_construct_rpc_backend_options_handler,
            _tensorpipe_init_backend_handler,
        )
    },
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
