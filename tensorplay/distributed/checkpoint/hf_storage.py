from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .filesystem import FileSystemReader, FileSystemWriter

__all__ = ["HuggingFaceStorageWriter", "HuggingFaceStorageReader"]


class HuggingFaceStorageWriter(FileSystemWriter):
    def __init__(self, path: str, fqn_to_index_mapping: dict[str, int] | None = None, **kwargs: Any) -> None:
        super().__init__(path, **kwargs)
        self.fqn_to_index_mapping = fqn_to_index_mapping or {}
        self.weight_map: dict[str, str] = {}

    def prepare_global_plan(self, plans: list[Any]) -> list[Any]:
        return plans

    def write_data(self, plan: Any, planner: Any = None) -> Any:
        state = plan if isinstance(plan, dict) else {item.index.fqn: planner.resolve_data(item) for item in plan.items}
        path = Path(self.path)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "model.bin"
        import tensorplay as tp
        tp.save(state, str(target))
        self.weight_map = {key: target.name for key in state}
        return []

    def finish(self, metadata: Any = None, results: Any = None) -> None:
        del results
        Path(self.path).mkdir(parents=True, exist_ok=True)
        (Path(self.path) / "model.safetensors.index.json").write_text(json.dumps({"metadata": metadata or {}, "weight_map": self.weight_map}, indent=2))

    def _split_by_storage_plan(self, *args: Any, **kwargs: Any) -> Any:
        del kwargs
        return args[0] if args else None

    def metadata_path(self) -> str:
        return str(Path(self.path) / "model.safetensors.index.json")


class HuggingFaceStorageReader(FileSystemReader):
    def __init__(self, path: str, thread_count: int = 1) -> None:
        del thread_count
        super().__init__(path)

    def _process_read_request(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def _read_files_from_queue(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def read_data(self, plan: Any, planner: Any = None) -> Any:
        del plan, planner
        import tensorplay as tp
        return tp.load(str(Path(self.path) / "model.bin"))

    def read_metadata(self) -> Any:
        path = Path(self.path) / "model.safetensors.index.json"
        return json.loads(path.read_text()) if path.exists() else {}
