from __future__ import annotations

import pickle
import struct
from dataclasses import dataclass
from io import BufferedIOBase
from typing import Any

import tensorplay as tp

__all__ = ["_Entry", "_PseudoZipFile", "_streaming_save", "_streaming_load"]


@dataclass
class _Entry:
    key: str
    is_storage: bool
    length: int


class _PseudoZipFile:
    def __init__(self) -> None:
        self.records: dict[str, tuple[object, int]] = {}

    def write_record(self, key: str, data: object, length: int) -> None:
        self.records[key] = (data, int(length))

    def write_to(self, stream: BufferedIOBase) -> None:
        entries: list[_Entry] = []
        encoded: list[bytes] = []
        for key, (value, declared_length) in self.records.items():
            payload = value if isinstance(value, bytes) else pickle.dumps(value, protocol=5)
            if declared_length and declared_length != len(payload):
                raise ValueError(f"record {key!r} length does not match its payload")
            entries.append(_Entry(key, isinstance(value, tp.Tensor), len(payload)))
            encoded.append(payload)
        header = pickle.dumps(entries, protocol=5)
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        for payload in encoded:
            stream.write(payload)

    def read_from(self, stream: BufferedIOBase) -> None:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError("truncated streaming header")
        (header_size,) = struct.unpack("<Q", raw_size)
        entries = pickle.loads(stream.read(header_size))
        if not isinstance(entries, list):
            raise ValueError("invalid streaming record table")
        for entry in entries:
            if not isinstance(entry, _Entry):
                raise ValueError("invalid streaming record")
            payload = stream.read(entry.length)
            if len(payload) != entry.length:
                raise ValueError(f"truncated record {entry.key!r}")
            if entry.is_storage:
                self.records[entry.key] = (pickle.loads(payload), entry.length)
            else:
                try:
                    value = pickle.loads(payload)
                except Exception:
                    value = payload
                self.records[entry.key] = (value, entry.length)

    def has_record(self, key: str) -> bool:
        return key in self.records

    def get_record(self, key: str) -> object:
        return self.records[key][0]

    def get_storage_from_record(self, key: str, _length: int, _type: int) -> Any:
        value = self.records[key][0]
        if isinstance(value, tp.Tensor):
            return value
        if isinstance(value, bytes):
            return tp.frombuffer(memoryview(value), dtype=getattr(tp, "uint8", None))
        return value

    def serialization_id(self) -> str:
        return "tensorplay-stream"


def _streaming_save(
    obj: object,
    stream: BufferedIOBase,
    pickle_module: Any = pickle,
    pickle_protocol: int = 5,
) -> None:
    del pickle_module
    pickle.dump(obj, stream, protocol=pickle_protocol)


def _streaming_load(
    stream: BufferedIOBase,
    map_location: Any = None,
    pickle_module: Any = None,
    *,
    weights_only: bool = True,
    **pickle_load_args: Any,
) -> object:
    del map_location, weights_only
    loader = pickle_module or pickle
    return loader.load(stream, **pickle_load_args)
