"""Placement rules for embedding lookups."""

from __future__ import annotations

from typing import Any

from .._api import DTensor
from .._dtensor_spec import DTensorSpec

__all__ = ["embedding_dense_backward_strategy", "embedding_strategy"]


def embedding_strategy(weight: DTensor, *args: Any, **kwargs: Any) -> DTensorSpec:
    del args, kwargs
    return DTensorSpec(weight.device_mesh, weight.placements, None)


def embedding_dense_backward_strategy(weight: DTensor, *args: Any, **kwargs: Any) -> DTensorSpec:
    return embedding_strategy(weight, *args, **kwargs)
