# Python. tp keeps its config in-process; the C++ side has no counterpart.
from dataclasses import dataclass

__all__: list[str] = []


@dataclass
class DistributedConfig:
    pass
