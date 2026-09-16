"""Scalar, dtype, and device aliases shared by the ported modules."""

from collections.abc import Sequence
from typing import Union

import tensorplay

Number = Union[bool, int, float]
_Number = Number
_dtype = tensorplay.dtype
Device = Union[tensorplay.device, str, int]
SymInt = int
_size = Union[tensorplay.Size, Sequence[int]]

__all__ = ["Number", "_Number", "_dtype", "Device", "SymInt", "_size"]
