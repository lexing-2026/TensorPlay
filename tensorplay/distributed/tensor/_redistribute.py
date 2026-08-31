"""Conversion of local values between distributed placements."""

from __future__ import annotations

from typing import Any

from ._api import DTensor, distribute_tensor
from ._dtensor_spec import DTensorSpec
from ._utils import ExplicitRedistributionContext

__all__ = ["Redistribute", "redistribute_local_tensor"]


class Redistribute:
    """Callable operation object for a single placement conversion."""

    def __init__(self, current_spec: DTensorSpec, target_spec: DTensorSpec) -> None:
        self.current_spec = current_spec
        self.target_spec = target_spec

    def __call__(self, local_tensor: Any) -> Any:
        return redistribute_local_tensor(local_tensor, self.current_spec, self.target_spec)


def redistribute_local_tensor(
    local_tensor: Any, current_spec: DTensorSpec, target_spec: DTensorSpec
) -> Any:
    if current_spec == target_spec:
        return local_tensor
    ExplicitRedistributionContext.observe_redistribution(current_spec, target_spec)
    value = DTensor.from_local(
        local_tensor,
        current_spec.mesh,
        current_spec.placements,
        shape=current_spec.shape,
        stride=current_spec.tensor_meta.stride if current_spec.tensor_meta else None,
    )
    return distribute_tensor(
        value.full_tensor(),
        target_spec.mesh,
        target_spec.placements,
        src_data_rank=0,
    ).to_local()
