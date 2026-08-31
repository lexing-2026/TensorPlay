"""Distributed tensor layout and construction APIs."""

from ..device_mesh import DeviceMesh, init_device_mesh
from ._api import (
    DTensor,
    distribute_module,
    distribute_tensor,
    empty,
    from_local,
    full,
    linspace,
    logspace,
    ones,
    rand,
    randn,
    zeros,
)
from .placement_types import Partial, Placement, Replicate, Shard

__all__ = [
    "DTensor",
    "distribute_tensor",
    "distribute_module",
    "Shard",
    "Replicate",
    "Partial",
    "Placement",
    "from_local",
    "ones",
    "empty",
    "full",
    "linspace",
    "logspace",
    "rand",
    "randn",
    "zeros",
    "DeviceMesh",
    "init_device_mesh",
]
