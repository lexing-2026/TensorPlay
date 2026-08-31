"""Random distributed tensor constructors."""

from __future__ import annotations

from typing import Any, Sequence

import tensorplay

from ._api import DTensor, distribute_tensor
from .placement_types import Placement

__all__ = ["rand", "randn", "manual_seed"]


def manual_seed(seed: int) -> Any:
    return tensorplay.manual_seed(seed)


def rand(*size: Any, device_mesh: Any = None, placements: Sequence[Placement] | None = None, **kwargs: Any) -> DTensor:
    return distribute_tensor(tensorplay.rand(*size, **kwargs), device_mesh, placements)


def randn(*size: Any, device_mesh: Any = None, placements: Sequence[Placement] | None = None, **kwargs: Any) -> DTensor:
    return distribute_tensor(tensorplay.randn(*size, **kwargs), device_mesh, placements)
