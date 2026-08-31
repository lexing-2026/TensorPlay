from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorplay as tp

from .types import RankInfo, STATE_DICT


class CheckpointReader:
    def __init__(self, rank_info: RankInfo) -> None:
        self._rank_info = rank_info

    def read(self, path: str, state_dict: STATE_DICT | None = None, *, map_location: Any = None, **kwargs: Any) -> tuple[STATE_DICT, list[str]]:
        del map_location, kwargs
        file_path = Path(path) / f"checkpoint_{self._rank_info.global_rank}.tp"
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        loaded = tp.load(file_path)
        if state_dict is None:
            return loaded, []
        missing = []
        for key, value in loaded.items():
            if key not in state_dict:
                missing.append(key)
            elif isinstance(state_dict[key], tp.Tensor) and isinstance(value, tp.Tensor):
                state_dict[key].copy_(value)
            else:
                state_dict[key] = value
        return state_dict, missing

    def _partial_read(self, file_path: Path, state_dict: STATE_DICT, *, map_location: Any = None, **kwargs: Any) -> tuple[STATE_DICT, list[str]]:
        del map_location, kwargs
        return self.read(str(file_path.parent), state_dict)
