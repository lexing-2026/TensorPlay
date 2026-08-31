"""Reader and writer for the TensorPlay model archive."""

from __future__ import annotations

import json
import os
import pickle
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import (
    AOTINDUCTOR_DIR,
    ARCHIVE_FORMAT_PATH,
    ARCHIVE_FORMAT_VALUE,
    ARCHIVE_VERSION_PATH,
    ARCHIVE_VERSION_VALUE,
    CONSTANTS_CONFIG_FILENAME_FORMAT,
    CONSTANTS_DIR,
    CUSTOM_OBJ_FILENAME_PREFIX,
    EXTRA_DIR,
    MODELS_DIR,
    MODELS_FILENAME_FORMAT,
    SAMPLE_INPUTS_FILENAME_FORMAT,
    WEIGHT_FILENAME_PREFIX,
    WEIGHTS_CONFIG_FILENAME_FORMAT,
    WEIGHTS_DIR,
)

__all__ = [
    "AOTICompiledModel",
    "PT2ArchiveContents",
    "PT2ArchiveReader",
    "PT2ArchiveWriter",
    "is_pt2_package",
    "load_pt2",
    "load_weights_to_pt2_contents",
    "package_pt2",
]

DEFAULT_PICKLE_PROTOCOL = 4


def _open_source(value: Any, mode: str) -> tuple[Any, bool]:
    if isinstance(value, (str, os.PathLike)):
        return open(value, mode), True
    return value, False


def is_pt2_package(serialized_model: bytes | str | os.PathLike[str]) -> bool:
    """Return whether a file or byte buffer contains a model archive."""

    source: Any = io.BytesIO(serialized_model) if isinstance(serialized_model, bytes) else serialized_model
    handle, owned = _open_source(source, "rb")
    try:
        with zipfile.ZipFile(handle) as archive:
            return (
                ARCHIVE_FORMAT_PATH in archive.namelist()
                and archive.read(ARCHIVE_FORMAT_PATH).decode() == ARCHIVE_FORMAT_VALUE
            )
    except (OSError, zipfile.BadZipFile, KeyError, UnicodeError):
        return False
    finally:
        if owned:
            handle.close()


class PT2ArchiveWriter:
    def __init__(self, archive_path_or_buffer: Any) -> None:
        self.archive_file, self._owned = _open_source(archive_path_or_buffer, "w+b")
        self._archive = zipfile.ZipFile(self.archive_file, "w", compression=zipfile.ZIP_DEFLATED)

    def __enter__(self) -> "PT2ArchiveWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        del args
        if not self.has_record(ARCHIVE_FORMAT_PATH):
            self.write_string(ARCHIVE_FORMAT_PATH, ARCHIVE_FORMAT_VALUE)
        if not self.has_record(ARCHIVE_VERSION_PATH):
            self.write_string(ARCHIVE_VERSION_PATH, ARCHIVE_VERSION_VALUE)
        self.close()

    def has_record(self, name: str) -> bool:
        return name in self._archive.namelist()

    def count_prefix(self, prefix: str) -> int:
        return sum(name.startswith(prefix) for name in self._archive.namelist())

    def write_bytes(self, name: str, data: bytes) -> None:
        self._archive.writestr(name, data)

    def write_string(self, name: str, data: str) -> None:
        self.write_bytes(name, data.encode("utf-8"))

    def write_file(self, name: str, file_path: str | os.PathLike[str]) -> None:
        self._archive.write(file_path, arcname=name)

    def write_folder(self, archive_dir: str, folder_dir: str | os.PathLike[str]) -> None:
        root = Path(folder_dir)
        for file_path in root.rglob("*"):
            if file_path.is_file():
                self.write_file(f"{archive_dir.rstrip('/')}/{file_path.relative_to(root)}", file_path)

    def close(self) -> None:
        if self._archive.fp is None:
            return
        self._archive.close()
        if hasattr(self.archive_file, "flush"):
            self.archive_file.flush()
        if self._owned:
            self.archive_file.close()


class PT2ArchiveReader:
    def __init__(self, archive_path_or_buffer: Any) -> None:
        self.archive_file, self._owned = _open_source(archive_path_or_buffer, "rb")
        self._archive = zipfile.ZipFile(self.archive_file, "r")

    def __enter__(self) -> "PT2ArchiveReader":
        return self

    def __exit__(self, *args: Any) -> None:
        del args
        self._archive.close()
        if self._owned:
            self.archive_file.close()

    def read_bytes(self, name: str) -> bytes:
        return self._archive.read(name)

    def read_string(self, name: str) -> str:
        return self.read_bytes(name).decode("utf-8")

    def archive_version(self) -> int:
        return int(self.read_string(ARCHIVE_VERSION_PATH))

    def get_file_names(self) -> list[str]:
        return self._archive.namelist()


@dataclass
class PT2ArchiveContents:
    exported_programs: dict[str, Any] = field(default_factory=dict)
    aoti_runners: dict[str, Any] = field(default_factory=dict)
    extra_files: dict[str, Any] = field(default_factory=dict)


class AOTICompiledModel:
    def __init__(self, model_name: str, files: dict[str, bytes] | None = None) -> None:
        self.model_name = model_name
        self.files = dict(files or {})

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"compiled runner {self.model_name!r} is not executable in this process")

    def get_metadata(self) -> dict[str, str]:
        return {"model_name": self.model_name, "file_count": str(len(self.files))}


def _programs_mapping(programs: Any) -> dict[str, Any]:
    if programs is None:
        return {}
    if isinstance(programs, dict):
        return dict(programs)
    return {"model": programs}


def _tensor_meta(tensor: Any) -> dict[str, Any]:
    from ..._serialization_torch import _dtype_name_of, _contiguous_stride

    shape = [int(dim) for dim in tuple(tensor.shape)]
    stride_fn = getattr(tensor, "stride", None)
    stride = (
        [int(step) for step in stride_fn()]
        if callable(stride_fn)
        else _contiguous_stride(shape)
    )
    return {
        "dtype": _dtype_name_of(tensor),
        "sizes": shape,
        "strides": stride,
        "storage_offset": int(getattr(tensor, "storage_offset", lambda: 0)()),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
    }


def _tensor_from_meta(data: bytes, meta: dict[str, Any]) -> Any:
    from ..._serialization_torch import (
        _ITEMSIZE,
        _tensor_from_flat_bytes,
        _reshape_or_view,
    )
    import tensorplay as tp

    dtype_name = meta["dtype"]
    numel = len(data) // _ITEMSIZE[dtype_name]
    flat = _tensor_from_flat_bytes(data, dtype_name) if numel > 0 else None
    if flat is None:
        total = 1
        for dim in meta["sizes"]:
            total *= dim
        flat = tp.empty([max(total, 1)])
    return _reshape_or_view(
        flat, list(meta["sizes"]), list(meta["strides"]), meta["storage_offset"]
    )


def _dtype_from_name(name: str) -> Any:
    from ..._serialization_torch import _tp_dtype, _NUMPY_DTYPES

    return _tp_dtype(name)


def _package_weights(
    writer: "PT2ArchiveWriter",
    directory: str,
    values: dict[str, Any],
    is_param_flags: dict[str, bool] | None = None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> dict[str, Any]:
    """Write tensors as raw payloads plus a JSON index; returns the index."""

    from ..._serialization_torch import _tensor_bytes

    is_param_flags = is_param_flags or {}
    config: dict[str, Any] = {}
    for index, (name, tensor) in enumerate(values.items()):
        if not hasattr(tensor, "shape"):
            path_name = f"{CUSTOM_OBJ_FILENAME_PREFIX}{index}"
            writer.write_bytes(
                f"{directory}{path_name}",
                pickle.dumps(tensor, protocol=pickle_protocol),
            )
            config[name] = {"path_name": path_name, "use_pickle": True}
            continue
        payload = tensor.detach() if hasattr(tensor, "detach") else tensor
        path_name = f"{WEIGHT_FILENAME_PREFIX}{index}"
        writer.write_bytes(f"{directory}{path_name}", _tensor_bytes(payload))
        meta = _tensor_meta(payload)
        meta["path_name"] = path_name
        meta["use_pickle"] = False
        meta["is_param"] = bool(is_param_flags.get(name, False))
        config[name] = meta
    return config


def _load_weights(reader: "PT2ArchiveReader", directory: str, config: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    import tensorplay as tp

    for name, meta in config.items():
        path_name = meta["path_name"]
        data = reader.read_bytes(f"{directory}{path_name}")
        if meta.get("use_pickle"):
            values[name] = pickle.loads(data)
            continue
        tensor = _tensor_from_meta(data, meta)
        if meta.get("is_param"):
            tensor = tp.nn.Parameter(tensor, requires_grad=meta.get("requires_grad", False))
        elif meta.get("requires_grad"):
            tensor.requires_grad_(True)
        values[name] = tensor
    return values


def _export_state_values(program: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    """Split program state into parameters, buffers, and constant values."""

    signature = program.graph_signature
    root = program.graph_module.root
    state: dict[str, Any] = {}
    flags: dict[str, bool] = {}

    def fetch(target: str) -> Any:
        value = root
        for atom in target.split("."):
            value = getattr(value, atom)
        return value

    for target in signature.parameters:
        state[target] = fetch(target)
        flags[target] = True
    for target in signature.buffers:
        state[target] = fetch(target)
        flags[target] = False
    constants: dict[str, Any] = dict(getattr(program.graph_module, "meta", {}).get("constants", {}))
    for target in signature.lifted_tensor_constants:
        if target not in constants:
            constants[target] = fetch(target)
    return state, constants, flags


def package_pt2(
    f: Any,
    *,
    exported_programs: Any = None,
    aoti_files: Any = None,
    extra_files: dict[str, Any] | None = None,
    opset_version: dict[str, int] | None = None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
    executorch_files: dict[str, bytes] | None = None,
) -> Any:
    from ..exported_program import ExportedProgram
    from ..serde import serialize

    if exported_programs is None and aoti_files is None and extra_files is None and executorch_files is None:
        raise ValueError("at least one archive artifact is required")
    programs = _programs_mapping(exported_programs)
    for name, program in programs.items():
        if not isinstance(program, ExportedProgram):
            raise TypeError(
                f"exported program {name!r} must be an ExportedProgram, "
                f"got {type(program).__name__}"
            )
    with PT2ArchiveWriter(f) as writer:
        for name, program in programs.items():
            state, constants, flags = _export_state_values(program)
            weights_config = _package_weights(
                writer,
                WEIGHTS_DIR,
                state,
                is_param_flags=flags,
                pickle_protocol=pickle_protocol,
            )
            writer.write_string(
                WEIGHTS_CONFIG_FILENAME_FORMAT.format(name),
                json.dumps(weights_config),
            )
            constants_config = _package_weights(
                writer,
                CONSTANTS_DIR,
                constants,
                pickle_protocol=pickle_protocol,
            )
            writer.write_string(
                CONSTANTS_CONFIG_FILENAME_FORMAT.format(name),
                json.dumps(constants_config),
            )
            artifact = serialize(program, opset_version, pickle_protocol)
            writer.write_bytes(
                MODELS_FILENAME_FORMAT.format(name),
                artifact.exported_program,
            )
            writer.write_bytes(
                SAMPLE_INPUTS_FILENAME_FORMAT.format(name),
                artifact.example_inputs,
            )
        if aoti_files is not None:
            files = aoti_files if isinstance(aoti_files, dict) else {"model": aoti_files}
            for model_name, model_files in files.items():
                for path in model_files:
                    path_obj = Path(path)
                    writer.write_file(f"{AOTINDUCTOR_DIR}{model_name}/{path_obj.name}", path_obj)
        for name, content in (extra_files or {}).items():
            data = content if isinstance(content, bytes) else str(content).encode("utf-8")
            writer.write_bytes(f"{EXTRA_DIR}{name}", data)
        for name, content in (executorch_files or {}).items():
            writer.write_bytes(f"data/executorch/{name}", content)
    if hasattr(f, "seek"):
        f.seek(0)
    return f


def load_pt2(
    f: Any,
    *,
    expected_opset_version: dict[str, int] | None = None,
    run_single_threaded: bool = False,
    num_runners: int = 1,
    device_index: int = -1,
    load_weights_from_disk: bool = False,
) -> PT2ArchiveContents:
    from ..serde import deserialize

    del expected_opset_version, run_single_threaded, num_runners, device_index
    programs: dict[str, Any] = {}
    aoti_runners: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    with PT2ArchiveReader(f) as reader:
        if reader.read_string(ARCHIVE_FORMAT_PATH) != ARCHIVE_FORMAT_VALUE:
            raise ValueError("archive format marker is invalid")
        if reader.read_string(ARCHIVE_VERSION_PATH) != ARCHIVE_VERSION_VALUE:
            raise ValueError("archive version is unsupported")
        file_names = reader.get_file_names()
        model_files = [
            name for name in file_names
            if name.startswith(MODELS_DIR) and name.endswith(".json")
        ]
        for model_file in model_files:
            model_name = Path(model_file).stem
            state = _load_weights(
                reader,
                WEIGHTS_DIR,
                json.loads(reader.read_string(WEIGHTS_CONFIG_FILENAME_FORMAT.format(model_name))),
            )
            constants = _load_weights(
                reader,
                CONSTANTS_DIR,
                json.loads(reader.read_string(CONSTANTS_CONFIG_FILENAME_FORMAT.format(model_name))),
            )
            example_inputs = pickle.loads(
                reader.read_bytes(SAMPLE_INPUTS_FILENAME_FORMAT.format(model_name))
            )
            programs[model_name] = deserialize(
                reader.read_bytes(model_file),
                state_dict=state,
                constants=constants,
                example_inputs=example_inputs,
            )
        for name in file_names:
            if name.startswith(AOTINDUCTOR_DIR):
                aoti_runners.setdefault(name[len(AOTINDUCTOR_DIR):].split("/")[0], AOTICompiledModel(name))
            elif name.startswith(EXTRA_DIR):
                extra[name[len(EXTRA_DIR):]] = reader.read_string(name)
    return PT2ArchiveContents(programs, aoti_runners, extra)


def load_weights_to_pt2_contents(pt2_contents: PT2ArchiveContents, weights_map: dict[str, Any]) -> None:
    for model_name, weights in weights_map.items():
        runner = pt2_contents.aoti_runners.get(model_name)
        if runner is None or not hasattr(runner, "load_constants"):
            raise KeyError(f"model {model_name!r} has no loadable runner")
        runner.load_constants(weights, check_full_update=True, user_managed=True)
