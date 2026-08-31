from __future__ import annotations

import abc
import io
import zlib
from collections.abc import Sequence
from typing import IO, Any

__all__ = ["Extension", "StreamTransformExtension", "ZStandard", "ExtensionRegistry"]


class Extension(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def registry_name() -> str: ...
    @staticmethod
    @abc.abstractmethod
    def from_descriptor(version: str) -> "Extension": ...
    @abc.abstractmethod
    def get_descriptor(self) -> str: ...


class StreamTransformExtension(Extension):
    @abc.abstractmethod
    def transform_to(self, output: IO[bytes]) -> IO[bytes]: ...
    @abc.abstractmethod
    def transform_from(self, input: IO[bytes]) -> IO[bytes]: ...


class ZStandard(StreamTransformExtension):
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def registry_name() -> str:
        return "stream.zstd"

    @staticmethod
    def from_descriptor(version: str) -> "ZStandard":
        if version.partition(".")[0] != "1":
            raise ValueError(f"unknown extension version {version!r}")
        return ZStandard()

    def get_descriptor(self) -> str:
        return f"{self.registry_name()}/1"

    def transform_to(self, output: IO[bytes]) -> IO[bytes]:
        class Writer(io.RawIOBase):
            def write(self, data: bytes) -> int:
                encoded = zlib.compress(bytes(data))
                output.write(len(encoded).to_bytes(8, "little"))
                output.write(encoded)
                return len(data)
            def flush(self) -> None:
                output.flush()
        return Writer()

    def transform_from(self, input: IO[bytes]) -> IO[bytes]:
        encoded = bytearray()
        while True:
            size = input.read(8)
            if not size:
                break
            if len(size) != 8:
                raise ValueError("truncated compressed stream")
            encoded.extend(input.read(int.from_bytes(size, "little")))
        return io.BytesIO(zlib.decompress(bytes(encoded)))


class ExtensionRegistry:
    def __init__(self) -> None:
        self.extensions: dict[str, type[Extension]] = {ZStandard.registry_name(): ZStandard}

    def register(self, cls: type[Extension]) -> None:
        self.extensions[cls.registry_name()] = cls

    def from_descriptor_list(self, descriptors: Sequence[str]) -> Sequence[Extension]:
        result = []
        for descriptor in descriptors:
            name, _, version = descriptor.partition("/")
            cls = self.extensions.get(name)
            if cls is None:
                raise ValueError(f"unknown extension {name!r}")
            result.append(cls.from_descriptor(version or "0"))
        return result
