from __future__ import annotations

import contextlib
import io
import os
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

from ._extension import StreamTransformExtension
from .filesystem import FileSystemReader, FileSystemWriter

__all__ = ["FsspecWriter", "FsspecReader"]


class FileSystem:
    def __init__(self) -> None:
        self.fs: Any = None

    def init_path(self, path: str | os.PathLike[str], **kwargs: Any) -> str:
        del kwargs
        try:
            import fsspec
            self.fs, target = fsspec.core.url_to_fs(path)
            return target
        except ImportError:
            return os.fspath(path)

    @contextlib.contextmanager
    def create_stream(self, path: str | os.PathLike[str], mode: str) -> Generator[io.IOBase, None, None]:
        if self.fs is None:
            stream = open(path, mode)
        else:
            stream = self.fs.open(path, mode)
        try:
            yield stream
        except BaseException:
            stream.close()
            raise
        finally:
            stream.close()

    def concat_path(self, path: str | os.PathLike[str], suffix: str) -> str:
        return os.path.join(os.fspath(path), suffix)
    def rename(self, path: str | os.PathLike[str], new_path: str | os.PathLike[str]) -> None:
        if self.fs is None:
            os.replace(path, new_path)
        else:
            self.fs.rename(path, new_path)
    def mkdir(self, path: str | os.PathLike[str]) -> None:
        if self.fs is None:
            Path(path).mkdir(parents=True, exist_ok=True)
        else:
            self.fs.makedirs(path, exist_ok=True)
    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike[str]) -> bool:
        return isinstance(checkpoint_id, (str, os.PathLike))
    def exists(self, path: str | os.PathLike[str]) -> bool:
        return os.path.exists(path) if self.fs is None else bool(self.fs.exists(path))
    def rm_file(self, path: str | os.PathLike[str]) -> None:
        if self.fs is None:
            os.unlink(path)
        else:
            self.fs.rm(path)
    def ls(self, path: str | os.PathLike[str]) -> list[str]:
        return [str(item) for item in (os.listdir(path) if self.fs is None else self.fs.ls(path, detail=False))]


class FsspecWriter(FileSystemWriter):
    def __init__(self, path: str | os.PathLike[str], single_file_per_rank: bool = True, sync_files: bool = True, thread_count: int = 1, per_thread_copy_ahead: int = 10_000_000, overwrite: bool = True, _extensions: Sequence[StreamTransformExtension] | None = None, serialization_format: Any = None, **kwargs: Any) -> None:
        del single_file_per_rank, sync_files, thread_count, per_thread_copy_ahead, _extensions, serialization_format
        super().__init__(os.fspath(path), overwrite=overwrite)
        self.fs = FileSystem()
        self.path = self.fs.init_path(path, **kwargs)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike[str]) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)


class FsspecReader(FileSystemReader):
    def __init__(self, path: str | os.PathLike[str], **kwargs: Any) -> None:
        super().__init__(os.fspath(path))
        self.fs = FileSystem()
        self.path = self.fs.init_path(path, **kwargs)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike[str]) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)
