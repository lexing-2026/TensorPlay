"""Placement rules for random and in-place random operations."""

from __future__ import annotations

from typing import Any

from .._api import DTensor
from .._dtensor_spec import DTensorSpec

__all__ = ["multinomial_single_dim_strategy", "random_inplace_single_dim_strategy"]


def random_inplace_single_dim_strategy(value: DTensor, *args: Any, **kwargs: Any) -> DTensorSpec:
    del args, kwargs
    return DTensorSpec(value.device_mesh, value.placements, None)


def multinomial_single_dim_strategy(value: DTensor, *args: Any, **kwargs: Any) -> DTensorSpec:
    return random_inplace_single_dim_strategy(value, *args, **kwargs)
