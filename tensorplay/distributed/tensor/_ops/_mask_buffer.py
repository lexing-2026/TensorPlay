"""Reusable masks for partial and uneven shard operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorplay

__all__ = ["MaskBuffer"]


@dataclass
class MaskBuffer:
    data: Any = None
    refcount: int = 0

    def materialize_mask(self, mask: Any) -> None:
        if self.refcount == 0:
            self.data = mask
        else:
            if self.data is None:
                raise AssertionError("mask buffer data is missing")
            if not bool(tensorplay.equal(self.data, mask)):
                raise RuntimeError("mask buffer received conflicting data")
        self.refcount += 1

    def release_mask(self) -> None:
        if self.refcount == 0 or self.data is None:
            raise RuntimeError("mask buffer has not been materialized")
        self.refcount -= 1
        if self.refcount == 0:
            self.data = None

    def apply_mask(self, tensor: Any) -> None:
        if self.refcount == 0 or self.data is None:
            raise RuntimeError("mask buffer has not been materialized")
        if int(tensor.ndim) == int(self.data.ndim):
            tensor[self.data] = 0.0
        else:
            tensor[self.data, :] = 0.0
