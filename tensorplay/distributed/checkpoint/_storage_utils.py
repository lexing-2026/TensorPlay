from __future__ import annotations

import os
from typing import Any

from .filesystem import FileSystemReader, FileSystemWriter
from .storage import StorageReader, StorageWriter


def _storage_setup(
    storage: StorageReader | StorageWriter | None,
    checkpoint_id: str | os.PathLike[str] | None,
    reader: bool = False,
) -> StorageReader | StorageWriter:
    if storage is not None:
        if checkpoint_id is not None:
            storage.reset(checkpoint_id)
        return storage
    if checkpoint_id is None or not str(checkpoint_id):
        raise RuntimeError(
            "checkpoint_id must be specified if storage_reader/storage_writer is None"
        )
    targets: list[type[StorageReader | StorageWriter]] = [
        FileSystemReader if reader else FileSystemWriter
    ]
    try:
        from ._fsspec_filesystem import FsspecReader, FsspecWriter

        targets.append(FsspecReader if reader else FsspecWriter)
    except (ImportError, TypeError):
        pass
    for target in targets:
        if target.validate_checkpoint_id(checkpoint_id):
            value = target(checkpoint_id)
            value.reset(checkpoint_id)
            return value
    raise RuntimeError("cannot detect a storage backend for checkpoint_id")
