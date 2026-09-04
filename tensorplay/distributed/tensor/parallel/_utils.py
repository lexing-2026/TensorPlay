"""Validation helpers for tensor-parallel mesh selection."""

from __future__ import annotations

from ...device_mesh import DeviceMesh

__all__ = ["_validate_tp_mesh_dim"]


def _validate_tp_mesh_dim(device_mesh: DeviceMesh) -> None:
    mesh_ndim = int(device_mesh.ndim)
    if mesh_ndim > 1:
        raise ValueError(
            "Tensor Parallel only accepts a 1D DeviceMesh, "
            f"but found {mesh_ndim}D. Select one mesh dimension first."
        )

    root_getter = getattr(device_mesh, "_get_root_mesh", None)
    root_mesh = root_getter() if callable(root_getter) else device_mesh
    if root_mesh is None or root_mesh == device_mesh:
        return

    axis_getter = getattr(device_mesh, "_get_axis_root_dims", None)
    axes = axis_getter() if callable(axis_getter) else ()
    root_ndim = int(root_mesh.ndim)
    root_dim = axes[0][0] if len(axes) == 1 and len(axes[0]) == 1 else None
    if root_dim != root_ndim - 1:
        raise RuntimeError(
            "Tensor Parallel mesh must use the innermost dimension of its "
            f"parent; found parent dimension {root_dim}."
        )
