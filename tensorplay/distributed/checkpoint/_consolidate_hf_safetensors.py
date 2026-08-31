from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorplay as tp


@dataclass
class _FqnData:
    fqn: str
    file_name: str
    data_offsets: tuple[int, int] = (0, 0)


@dataclass
class _OutputFileData:
    file_name: str
    tensors: list[_FqnData]


@dataclass
class _InputFileData:
    file_name: str
    metadata: dict[str, Any]


def _parse_input_metadata(path: str | Path) -> _InputFileData:
    return _InputFileData(str(path), json.loads(Path(path).read_text()))


def _write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2))


def _read_tensor_data(path: str | Path, fqn: str) -> Any:
    values = tp.load(path)
    return values[fqn]


def _process_output_file(*args: Any, **kwargs: Any) -> None:
    del args, kwargs


def _write_data(path: str | Path, values: dict[str, Any]) -> None:
    tp.save(values, path)


def _write_sub_tensor_to_file_optimized(*args: Any, **kwargs: Any) -> None:
    del args, kwargs


def _calculate_max_contiguous_elements(*args: Any, **kwargs: Any) -> int:
    del kwargs
    return int(args[0].numel()) if args and hasattr(args[0], "numel") else 0


def _write_overall_metadata_file(path: str | Path, weight_map: dict[str, str], total_size: int = 0) -> None:
    _write_metadata(path, {"metadata": {"total_size": total_size}, "weight_map": weight_map})


def _consolidate_safetensors_files(input_dir: str | Path, output_dir: str | Path, **kwargs: Any) -> dict[str, str]:
    del kwargs
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    merged: dict[str, Any] = {}
    for file in sorted(input_path.glob("*.mega")):
        loaded = tp.load(file)
        if isinstance(loaded, dict):
            merged.update(loaded)
    target = output_path / "model.mega"
    tp.save(merged, target)
    mapping = {key: target.name for key in merged}
    _write_overall_metadata_file(output_path / "model.mega.index.json", mapping)
    return mapping


def consolidate_safetensors_files(input_dir: str | Path, output_dir: str | Path, **kwargs: Any) -> dict[str, str]:
    return _consolidate_safetensors_files(input_dir, output_dir, **kwargs)


def consolidate_safetensors_files_on_every_rank(input_dir: str | Path, output_dir: str | Path, **kwargs: Any) -> dict[str, str]:
    return consolidate_safetensors_files(input_dir, output_dir, **kwargs)
