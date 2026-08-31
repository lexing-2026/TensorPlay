"""Execution order tracking for parameter communication."""

from dataclasses import dataclass, field
from typing import Any

__all__ = ["_ExecOrderWarnStatus", "_ExecOrderData"]


@dataclass
class _ExecOrderWarnStatus:
    warned: bool = False


@dataclass
class _ExecOrderData:
    handles: list[Any] = field(default_factory=list)
    forward_order: list[Any] = field(default_factory=list)
    backward_order: list[Any] = field(default_factory=list)

    def record_forward(self, handle: Any) -> None:
        self.forward_order.append(handle)

    def record_backward(self, handle: Any) -> None:
        self.backward_order.append(handle)
