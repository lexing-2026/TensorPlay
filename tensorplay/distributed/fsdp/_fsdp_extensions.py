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


def _ext_pre_flatten_transform(tensor: Any) -> tuple[Any, Any]:
    return _get_extensions().pre_flatten_transform(tensor)


def _ext_post_unflatten_transform(tensor: Any, param_extension: Any) -> Any:
    return _get_extensions().post_unflatten_transform(tensor, param_extension)


def _ext_chunk_tensor(*args: Any, **kwargs: Any) -> Any:
    return _get_extensions().chunk_tensor(*args, **kwargs)


def _ext_chunk_dtensor(*args: Any, **kwargs: Any) -> Any:
    return _get_extensions().chunk_dtensor(*args, **kwargs)


def _ext_pre_load_state_dict_transform(tensor: Any) -> tuple[Any, list[Any]]:
    return _get_extensions().pre_load_state_dict_transform(tensor)


def _ext_all_gather_dtensor(tensor: Any, parent_mesh: Any = None) -> Any:
    return _get_extensions().all_gather_dtensor(tensor, parent_mesh)
