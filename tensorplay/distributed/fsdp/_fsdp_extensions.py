"""Extension points for storage transformations."""

from typing import Any

__all__ = [
    "FSDPExtensions",
    "_set_fsdp_extensions",
    "_ext_pre_flatten_transform",
    "_ext_post_unflatten_transform",
    "_ext_chunk_tensor",
    "_ext_chunk_dtensor",
    "_ext_pre_load_state_dict_transform",
    "_ext_all_gather_dtensor",
]


class FSDPExtensions:
    def pre_flatten_transform(self, tensor: Any) -> tuple[Any, Any]:
        return tensor, None

    def post_unflatten_transform(self, tensor: Any, param_extension: Any) -> Any:
        del param_extension
        return tensor

    def chunk_tensor(self, tensor: Any, rank: int, world_size: int, num_devices_per_node: int, pg: Any, device: Any = None) -> Any:
        del num_devices_per_node, pg, device
        return tensor.chunk(world_size, dim=0)[rank]

    def chunk_dtensor(self, tensor: Any, rank: int, device_mesh: Any) -> Any:
        del rank
        return tensor.redistribute(device_mesh=device_mesh).to_local()

    def pre_load_state_dict_transform(self, tensor: Any) -> tuple[Any, list[Any]]:
        return tensor, []

    def all_gather_dtensor(self, tensor: Any, parent_mesh: Any = None) -> Any:
        del parent_mesh
        return tensor.full_tensor()


_extensions: FSDPExtensions | None = None


def _set_fsdp_extensions(extensions: FSDPExtensions | None) -> None:
    global _extensions
    _extensions = extensions


def _get_extensions() -> FSDPExtensions:
    global _extensions
    if _extensions is None:
        _extensions = FSDPExtensions()
    return _extensions


def _ext_pre_flatten_transform(
    tensor: Any, fsdp_extension: FSDPExtensions | None = None
) -> tuple[Any, Any]:
    if fsdp_extension is not None:
        return fsdp_extension.pre_flatten_transform(tensor)
    return tensor, None


def _ext_post_unflatten_transform(
    tensor: Any, param_extension: Any, fsdp_extension: FSDPExtensions | None = None
) -> Any:
    if fsdp_extension is not None and param_extension is not None:
        return fsdp_extension.post_unflatten_transform(tensor, param_extension)
    return tensor


def _ext_chunk_tensor(
    tensor: Any,
    rank: int,
    world_size: int,
    num_devices_per_node: int,
    pg: Any,
    fsdp_extension: FSDPExtensions | None = None,
    device: Any = None,
) -> Any:
    if fsdp_extension is not None:
        return fsdp_extension.chunk_tensor(
            tensor, rank, world_size, num_devices_per_node, pg, device
        )
    from ._shard_utils import _create_chunk_sharded_tensor

    return _create_chunk_sharded_tensor(
        tensor, rank, world_size, num_devices_per_node, pg, device
    )


def _ext_chunk_dtensor(
    tensor: Any, rank: int, device_mesh: Any, fsdp_extension: FSDPExtensions | None = None
) -> Any:
    if fsdp_extension is not None:
        return fsdp_extension.chunk_dtensor(tensor, rank, device_mesh)
    from ._shard_utils import _create_chunk_dtensor

    return _create_chunk_dtensor(tensor, rank, device_mesh)


def _ext_pre_load_state_dict_transform(
    tensor: Any, fsdp_extension: FSDPExtensions | None = None
) -> tuple[Any, list[Any]]:
    if fsdp_extension is not None:
        return fsdp_extension.pre_load_state_dict_transform(tensor)
    local_shards = getattr(tensor, "local_shards", None)
    return tensor, list(local_shards()) if callable(local_shards) else []


def _ext_all_gather_dtensor(
    tensor: Any, parent_mesh: Any = None, fsdp_extension: FSDPExtensions | None = None
) -> Any:
    if fsdp_extension is not None:
        return fsdp_extension.all_gather_dtensor(tensor, parent_mesh)
    from ._shard_utils import _all_gather_dtensor

    return _all_gather_dtensor(tensor, parent_mesh)
