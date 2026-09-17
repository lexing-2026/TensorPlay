# mypy: allow-untyped-defs
"""Archive (zip) primitives for reading and writing package files.

The package format is an uncompressed zip archive whose members are addressed
by record name (e.g. ``my_package/module.py`` or ``.data/extern_modules``).
These classes provide the small record-oriented interface the exporter and
importer operate on, on top of the standard library's ``zipfile`` module.
"""

import os
import zipfile
from typing import Any, IO


__all__ = ["PackageFileWriter", "PackageFileReader"]


class PackageFileWriter:
    """Writes records into an in-progress package archive.

    ``f`` may be a filename/path or a binary write-mode file object. The
    central directory of the zip archive is only emitted once the writer is
    closed, either explicitly through :meth:`close` or when the writer is
    garbage collected.
    """

    def __init__(self, f: str | os.PathLike | IO[bytes]) -> None:
        target = os.fspath(f) if isinstance(f, (os.PathLike, str)) else f
        self._zf: zipfile.ZipFile | None = zipfile.ZipFile(
            target, mode="w", compression=zipfile.ZIP_STORED
        )

    def set_min_version(self, version: int) -> None:
        """Accepted for interface compatibility. The underlying archive format
        version is managed by the zip writer itself."""

    def write_record(self, name: str, data: bytes, length: int) -> None:
        """Write a single record. ``length`` bytes of ``data`` are stored
        verbatim under the record name ``name``."""
        if self._zf is None:
            raise RuntimeError("PackageFileWriter is already closed")
        payload = bytes(data[:length])
        self._zf.writestr(name, payload)

    def close(self) -> None:
        """Finalize the archive by writing the zip central directory."""
        zf, self._zf = self._zf, None
        if zf is not None:
            zf.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class PackageFileReader:
    """Reads records from an existing package archive.

    ``f`` may be a filename/path or a binary read/seek-mode file object.
    """

    def __init__(self, f: str | os.PathLike | IO[bytes]) -> None:
        target = os.fspath(f) if isinstance(f, (os.PathLike, str)) else f
        self._zf = zipfile.ZipFile(target, mode="r")
        self._names = frozenset(self._zf.namelist())

    def get_record(self, name: str) -> bytes:
        """Return the raw bytes of the record ``name``."""
        return self._zf.read(name)

    def has_record(self, name: str) -> bool:
        return name in self._names

    def get_all_records(self) -> list[str]:
        """Return the names of every record in the archive."""
        return list(self._zf.namelist())

    def serialization_id(self) -> str:
        """Stable identity of the archive contents. The zip-based package
        format does not currently embed such an identifier."""
        return ""


def _make_reader(file_or_buffer: Any) -> PackageFileReader:
    """Open a package archive from a path or a binary file-like object."""
    return PackageFileReader(file_or_buffer)
