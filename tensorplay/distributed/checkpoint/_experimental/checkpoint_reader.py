from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorplay as tp

from .types import RankInfo, STATE_DICT


class CheckpointReader:
    def __init__(self, rank_info: RankInfo) -> None:
        self._rank_info = rank_info

    def read(self, path: str, state_dict: STATE_DICT | None = None, *, map_location: Any = None, **kwargs: Any) -> tuple[STATE_DICT, list[str]]:
        file_path = Path(path) / f"checkpoint_{self._rank_info.global_rank}.pt"
        if not file_path.exists():
            legacy_path = Path(path) / f"checkpoint_{self._rank_info.global_rank}.tp"
            if legacy_path.exists():
                file_path = legacy_path
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        loaded = tp.load(file_path, map_location=map_location, **kwargs)
        if not isinstance(loaded, dict):
            raise TypeError("checkpoint contents must be a dictionary")
        if state_dict is None:
            return loaded, []
        missing: list[str] = []
        return self._copy_into_target(state_dict, loaded, "", missing), missing

    def _partial_read(self, file_path: Path, state_dict: STATE_DICT, *, map_location: Any = None, **kwargs: Any) -> tuple[STATE_DICT, list[str]]:
        loaded = tp.load(file_path, map_location=map_location, **kwargs)
        if not isinstance(loaded, dict):
            raise TypeError("checkpoint contents must be a dictionary")
        missing: list[str] = []
        return self._copy_into_target(state_dict, loaded, "", missing), missing

    @staticmethod
    def _path(path: str, key: Any) -> str:
        value = str(key)
        return f"{path}.{value}" if path else value

    @classmethod
    def _copy_into_target(
        cls,
        target: Any,
        source: Any,
        path: str,
        missing: list[str],
    ) -> Any:
        if isinstance(source, tp.Tensor):
            if target is None:
                return source
            if not isinstance(target, tp.Tensor):
                raise TypeError(
                    f"checkpoint value {path or '<root>'} has incompatible type"
                )
            if target.shape != source.shape or target.dtype != source.dtype:
                raise ValueError(
                    f"checkpoint tensor {path or '<root>'} has shape/dtype "
                    f"{tuple(source.shape)}/{source.dtype}, expected "
                    f"{tuple(target.shape)}/{target.dtype}"
                )
            target.copy_(source)
            return target
        if isinstance(source, dict):
            if target is None:
                target = {}
            if not isinstance(target, dict):
                raise TypeError(
                    f"checkpoint value {path or '<root>'} has incompatible type"
                )
            for key, target_value in list(target.items()):
                current = cls._path(path, key)
                if key not in source:
                    missing.append(current)
                    continue
                target[key] = cls._copy_into_target(
                    target_value, source[key], current, missing
                )
            return target
        if isinstance(source, (list, tuple)):
            if target is None:
                target = [None] * len(source) if isinstance(source, list) else (None,) * len(source)
            if not isinstance(target, (list, tuple)):
                raise TypeError(
                    f"checkpoint value {path or '<root>'} has incompatible type"
                )
            result: list[Any] = []
            for index, target_value in enumerate(target):
                current = f"{path}[{index}]" if path else f"[{index}]"
                if index >= len(source):
                    missing.append(current)
                    result.append(target_value)
                else:
                    result.append(
                        cls._copy_into_target(target_value, source[index], current, missing)
                    )
            return tuple(result) if isinstance(target, tuple) else result
        if target is None:
            return source
        if type(target) is not type(source) and target != source:
            raise TypeError(
                f"checkpoint value {path or '<root>'} has incompatible type"
            )
        return source
