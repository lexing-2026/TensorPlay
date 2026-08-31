from __future__ import annotations

from typing import Any

import tensorplay as tp

__all__ = ["reduce_partials_first_dim_kernel"]


def reduce_partials_first_dim_kernel(partials: tp.Tensor, output: tp.Tensor, group_size: int, average: bool = False) -> tp.Tensor:
    """Reduce the leading peer dimension into a preallocated output tensor."""
    reduced = partials.sum(dim=0)
    if average:
        reduced = reduced / group_size
    output.copy_(reduced)
    return output
