from __future__ import annotations

from typing import Any

from ..device_mesh import DeviceMesh

__all__ = [
    "_runtime_compute_coordinate_on_dim_fake",
    "_runtime_compute_coordinate_on_dim_impl",
    "_get_flattened_submesh_impl",
    "_get_flattened_submesh",
    "_get_flattened_submesh_fake",
    "_get_submesh_impl",
    "_get_submesh",
    "_get_submesh_fake",
]


def _runtime_compute_coordinate_on_dim_fake(full_mesh: Any, index: int) -> int:
    del full_mesh, index
    return 0


def _runtime_compute_coordinate_on_dim_impl(full_mesh: Any, index: int) -> int:
    mesh = DeviceMesh._get_mesh_tensor_from_full_mesh(full_mesh) if hasattr(DeviceMesh, "_get_mesh_tensor_from_full_mesh") else full_mesh
    rank = 0
    try:
        from .. import distributed_core as dist
        rank = dist.get_rank()
    except Exception:
        pass
    coords = DeviceMesh._compute_coordinates_from_mesh(mesh, rank) if hasattr(DeviceMesh, "_compute_coordinates_from_mesh") else None
    if coords is None:
        raise ValueError("rank is not present in the mesh")
    return int(coords[index])


def _get_flattened_submesh_impl(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    if not mesh_dims:
        raise ValueError("mesh_dims cannot be empty")
    names = mesh.mesh_dim_names
    if names is None:
        raise ValueError("mesh dimension names are required")
    selected = tuple(names[index] for index in mesh_dims)
    return mesh[selected[0] if len(selected) == 1 else selected]


def _get_flattened_submesh(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    return _get_flattened_submesh_impl(mesh, mesh_dims)


def _get_flattened_submesh_fake(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    return _get_flattened_submesh_impl(mesh, mesh_dims)


def _get_submesh_impl(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    return _get_flattened_submesh_impl(mesh, mesh_dims)


def _get_submesh(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    return _get_submesh_impl(mesh, mesh_dims)


def _get_submesh_fake(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    return _get_submesh_impl(mesh, mesh_dims)
