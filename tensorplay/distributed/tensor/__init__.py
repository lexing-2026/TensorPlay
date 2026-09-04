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

from . import _ops as _ops
from ._ops.utils import _install_builtin_rules

_install_builtin_rules()

from tensorplay.serialization import add_safe_globals
from tensorplay.optim import optimizer as _optimizer
from tensorplay.utils import _foreach_utils

DTensor.__module__ = "tensorplay.distributed.tensor"
add_safe_globals([DTensor, DeviceMesh, Partial, Placement, Replicate, Shard])
if DTensor not in _optimizer._foreach_supported_types:
    _optimizer._foreach_supported_types.append(DTensor)
if DTensor not in _foreach_utils._foreach_supported_types:
    _foreach_utils._foreach_supported_types.append(DTensor)

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
