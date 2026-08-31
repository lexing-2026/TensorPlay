from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

STATE_DICT: TypeAlias = dict[str, Any]


@dataclass
class RankInfo:
    global_rank: int
    global_world_size: int
