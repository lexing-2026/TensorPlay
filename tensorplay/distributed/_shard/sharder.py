"""Interface for module-specific sharders."""

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["Sharder"]


class Sharder(ABC):
    @abstractmethod
    def shard(self, module: Any, params: Any, fqn: str) -> Any:
        raise NotImplementedError
