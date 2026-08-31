"""Mesh helpers used by distributed tensor APIs."""

from __future__ import annotations

from typing import Any

from ..device_mesh import DeviceMesh, init_device_mesh

__all__ = ["DeviceMesh", "get_current_device_mesh", "init_device_mesh", "mesh_dim_size"]


def get_current_device_mesh() -> DeviceMesh:
    from ..device_mesh import _MeshEnv

    stack = _MeshEnv.get().mesh_stack
    if not stack:
        raise RuntimeError("no DeviceMesh is active")
    return stack[-1]


def mesh_dim_size(mesh: DeviceMesh, mesh_dim: int | str) -> int:
    return mesh.size(mesh_dim)
