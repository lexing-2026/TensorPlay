from __future__ import annotations

from typing import Any

from ..proxy import Proxy

__all__ = ["BackwardState"]


class BackwardState:
    """Container for values shared between forward and reverse graph stages."""

    proxy: Proxy | None

    def __init__(self, proxy: Proxy | None = None, **attributes: Any) -> None:
        self.proxy = proxy
        self.__dict__.update(attributes)
