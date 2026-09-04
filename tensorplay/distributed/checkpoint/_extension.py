from __future__ import annotations

import abc
import io
import importlib
from collections.abc import Sequence
from typing import IO, Any


pyzstd_module_name = "pyzstd"
pyzstd = None
try:
    pyzstd = importlib.import_module(pyzstd_module_name)
except ImportError:
    pass

zstandard_module_name = "zstandard"
zstandard = None
try:
    zstandard = importlib.import_module(zstandard_module_name)
except ImportError:
    pass

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
        return zstandard is not None or pyzstd is not None

    @staticmethod
    def registry_name() -> str:
        return "stream.zstd"

    @staticmethod
    def from_descriptor(version: str) -> "ZStandard":
        if version.partition(".")[0] != "1":
            raise ValueError(f"Unknown extension version {version!r}")
        return ZStandard()

    def __init__(self) -> None:
        super().__init__()
        if not self.is_available():
            raise ValueError(
                f"ZStandard extension is unavailable because no module named "
                f"'{zstandard_module_name}' or '{pyzstd_module_name}'"
            )

    def get_descriptor(self) -> str:
        return f"{self.registry_name()}/1"

    def transform_to(self, output: IO[bytes]) -> IO[bytes]:
        if zstandard is not None:
            compressor = zstandard.ZstdCompressor()
            return compressor.stream_writer(output)

        class Writer(io.RawIOBase):
            def __init__(self, target: IO[bytes]) -> None:
                self.output = target
                self.compressor = pyzstd.ZstdCompressor()

            def writeable(self) -> bool:
                return True

            def write(self, data: bytes) -> int:
                encoded = self.compressor.compress(data)
                if encoded:
                    self.output.write(encoded)
                return len(memoryview(data))

            def flush(self) -> None:
                encoded = self.compressor.flush()
                if encoded:
                    self.output.write(encoded)
                self.output.flush()
        return Writer(output)

    def transform_from(self, input: IO[bytes]) -> IO[bytes]:
        if zstandard is not None:
            decompressor = zstandard.ZstdDecompressor()
            return decompressor.stream_reader(input)

        class Reader(io.RawIOBase):
            def __init__(self, stream: IO[bytes]) -> None:
                self.input = stream
                self.decompressor = pyzstd.EndlessZstdDecompressor()

            def readable(self) -> bool:
                return True

            def readinto(self, buffer: bytearray | memoryview) -> int | None:
                if self.decompressor.needs_input:
                    data = self.input.read((128 + 6) * 1024)
                else:
                    data = b""
                output = self.decompressor.decompress(data, len(memoryview(buffer)))
                if output is None:
                    return None
                count = len(output)
                memoryview(buffer)[:count] = output
                return count

            def seekable(self) -> bool:
                return False

        return Reader(input)


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
            result.append(cls.from_descriptor(version))
        return result
