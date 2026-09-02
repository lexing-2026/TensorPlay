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
    coords = DeviceMesh._compute_coordinates_from_mesh(mesh, rank) if hasattr(DeviceMesh, "_compute_coordinates_from_mesh") else _find_rank_coordinate(mesh, rank)
    if coords is None:
        raise ValueError("rank is not present in the mesh")
    if index < 0 or index >= len(coords):
        raise IndexError("mesh dimension is out of range")
    return int(coords[index])


def _get_flattened_submesh_impl(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    if not mesh_dims:
        raise ValueError("mesh_dims cannot be empty")
    names = mesh.mesh_dim_names
    if names is None:
        raise ValueError("mesh dimension names are required")
    ndim = int(mesh.ndim) if not callable(getattr(mesh, "ndim", None)) else int(mesh.ndim())
    dims = tuple(int(index) for index in mesh_dims)
    if len(set(dims)) != len(dims):
        raise ValueError("mesh dimensions must be unique")
    if any(index < 0 or index >= ndim for index in dims):
        raise IndexError("mesh dimension is out of range")
    selected = tuple(names[index] for index in dims)
    submesh = mesh[selected[0] if len(selected) == 1 else selected]
    flatten = getattr(submesh, "_flatten", None)
    if not callable(flatten):
        raise ValueError(f"mesh cannot flatten dimensions {mesh_dims!r}")
    return flatten("_".join(selected))


def _get_flattened_submesh(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    return _get_flattened_submesh_impl(mesh, mesh_dims)


def _get_flattened_submesh_fake(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    return _get_flattened_submesh_impl(mesh, mesh_dims)


def _get_submesh_impl(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    if not mesh_dims:
        raise ValueError("mesh_dims cannot be empty")
    names = mesh.mesh_dim_names
    if names is None:
        raise ValueError("mesh dimension names are required")
    ndim = int(mesh.ndim) if not callable(getattr(mesh, "ndim", None)) else int(mesh.ndim())
    dims = tuple(int(index) for index in mesh_dims)
    if len(set(dims)) != len(dims):
        raise ValueError("mesh dimensions must be unique")
    if any(index < 0 or index >= ndim for index in dims):
        raise IndexError("mesh dimension is out of range")
    selected = tuple(names[index] for index in dims)
    return mesh[selected[0] if len(selected) == 1 else selected]


def _find_rank_coordinate(mesh: Any, rank: int) -> tuple[int, ...] | None:
    value = mesh
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()

    def visit(current: Any, prefix: tuple[int, ...]) -> tuple[int, ...] | None:
        if isinstance(current, (list, tuple)):
            for index, child in enumerate(current):
                found = visit(child, prefix + (index,))
                if found is not None:
                    return found
            return None
        try:
            return prefix if int(current) == rank else None
        except (TypeError, ValueError):
            return None

    return visit(value, ())


def _get_submesh(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    return _get_submesh_impl(mesh, mesh_dims)


def _get_submesh_fake(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    return _get_submesh_impl(mesh, mesh_dims)
