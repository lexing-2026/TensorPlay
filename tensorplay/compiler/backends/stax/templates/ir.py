"""Two index helpers the templates address operands with."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence


def contiguous_stride(size: Sequence[int]) -> tuple:
    """The strides of a shape stored as one unbroken run.

    A template is handed the extents it was written against and has to work
    out the strides itself.  The answer is not a second opinion about what a
    contiguous layout is -- it is the same answer the layout itself would give,
    so it is asked of the layout rather than worked out again here.
    """

    from ..ir import FlexibleLayout

    return tuple(FlexibleLayout.contiguous_strides(tuple(size)))


def next_power_of_2(value: int) -> int:
    """The smallest power of two that is at least ``value``.

    A tile is measured in powers of two because that is what the hardware
    measures in, so a tile that is not one is rounded up to the next.  The
    rounding is the runtime's, since a tile that two callers rounded
    differently would be two different tiles.
    """

    from ..runtime.runtime_utils import next_power_of_2 as _next_power_of_2

    return _next_power_of_2(value)
