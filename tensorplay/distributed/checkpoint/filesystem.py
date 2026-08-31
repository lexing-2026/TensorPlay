#
# Adaptation: tp implements a consolidated single-file layout
# sharded-planner file matrix; the FileSystemReader/FileSystemWriter public
# API is preserved.
import dataclasses
from dataclasses import dataclass
import os
import pickle as _pickle
from abc import ABC, abstractmethod

__all__ = ["FileSystemWriter", "FileSystemReader", "StorageReader", "StorageWriter"]

_METADATA_FILE = ".metadata"
_DATA_FILE_FORMAT = "__{0}_0.distcp"


@dataclass(frozen=True)
class _StoragePrefix:
    name: str = "default"


class StorageWriter(ABC):

    @abstractmethod
    def set_up_storage_writer(self, is_coordinator: bool) -> None: ...

    @abstractmethod
    def write_data(self, state_dict) -> None: ...

    @abstractmethod
    def finish(self, metadata) -> None: ...

    def reset(self) -> None: ...


class StorageReader(ABC):

    @abstractmethod
    def read_metadata(self): ...

    @abstractmethod
    def read_data(self, plan, state_dict) -> None: ...

    def reset(self, checkpoint_id=None) -> None: ...


class FileSystemWriter(StorageWriter):
    """Writes consolidated checkpoint files under a directory."""

    def __init__(self, path: str, single_file_per_rank: bool = True,
                 thread_count: int = 1, overwrite: bool = True):
        self.path = path
        self.overwrite = overwrite

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        os.makedirs(self.path, exist_ok=True)

    def write_data(self, state_dict) -> None:
        import pickle

        self._data_path = os.path.join(
            self.path, _DATA_FILE_FORMAT.format(0))
        with open(self._data_path, "wb") as f:
            _pickle.dump(state_dict, f)

    def finish(self, metadata) -> None:
        import pickle

        with open(os.path.join(self.path, _METADATA_FILE), "wb") as f:
            _pickle.dump({"version": "tp-1"}, f)


class FileSystemReader(StorageReader):
    """Reads checkpoints produced by :class:`FileSystemWriter`."""

    def __init__(self, path: str):
        self.path = path
        self._data_path = os.path.join(path, _DATA_FILE_FORMAT.format(0))

    def read_metadata(self):
        import pickle

        meta_path = os.path.join(self.path, _METADATA_FILE)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Checkpoint metadata not found at {meta_path}"
            )
        with open(meta_path, "rb") as f:
            return _pickle.load(f)

    def read_data(self, plan, state_dict) -> dict:
        import pickle

        with open(self._data_path, "rb") as f:
            return _pickle.load(f)
