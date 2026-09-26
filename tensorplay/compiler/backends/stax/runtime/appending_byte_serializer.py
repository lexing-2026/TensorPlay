"""Writing a list of things out as bytes, and reading the list back.

A cache that outlives the process has to be bytes, and a list of values is not
bytes: each value has to be written in a way that can be found again from the
one after it, which is what the length before each piece buys.  The length is
a whole number of bytes rather than something self-delimiting, so that reading
is arithmetic and not a search.

Two things are added to what the caller asked to be written.  A version, so
that bytes written by one build and read by another is refused rather than
misread.  And a checksum over everything but itself, so that bytes that were
truncated or altered on the way are refused rather than acted on -- a cache
that is quietly wrong costs more than one that is missing.
"""

from __future__ import annotations

import base64
import zlib
from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

T = TypeVar("T")

#: Bumped when the layout of what is written changes, so that bytes written by
#: an older build are refused instead of read as something they are not.
_ENCODING_VERSION: int = 1

#: The room left at the front of the buffer for the digest, and the width of
#: that digest.
CHECKSUM_DIGEST_SIZE = 4


class BytesWriter:
    """Bytes being built up, one piece at a time."""

    def __init__(self) -> None:
        # The digest is computed over everything written, so its room is
        # reserved first and filled in when the rest is known.
        self._data = bytearray(CHECKSUM_DIGEST_SIZE)

    def write_uint64(self, i: int) -> None:
        self._data.extend(i.to_bytes(8, byteorder="big", signed=False))

    def write_str(self, s: str) -> None:
        self.write_bytes(base64.b64encode(s.encode("utf-8")))

    def write_bytes(self, b: bytes) -> None:
        self.write_uint64(len(b))
        self._data.extend(b)

    def to_bytes(self) -> bytes:
        digest = zlib.crc32(self._data[CHECKSUM_DIGEST_SIZE:]).to_bytes(
            4, byteorder="big", signed=False
        )
        if len(digest) != CHECKSUM_DIGEST_SIZE:
            raise AssertionError("the computed digest has an unexpected size")
        self._data[0:CHECKSUM_DIGEST_SIZE] = digest
        return bytes(self._data)


class BytesReader:
    """Bytes being taken apart again, one piece at a time.

    Whether the pieces are the ones that were written is settled before any of
    them is read, because a length read out of bytes that were altered is a
    length that points somewhere else.
    """

    def __init__(self, data: bytes) -> None:
        if len(data) < CHECKSUM_DIGEST_SIZE:
            raise AssertionError("input data is too short to contain a checksum")
        digest = zlib.crc32(data[CHECKSUM_DIGEST_SIZE:]).to_bytes(
            4, byteorder="big", signed=False
        )
        if len(digest) != CHECKSUM_DIGEST_SIZE:
            raise AssertionError("the computed digest has an unexpected size")
        if data[0:CHECKSUM_DIGEST_SIZE] != digest:
            raise RuntimeError(
                "the bytes are corrupted: the checksum does not match "
                f"(expected {data[0:CHECKSUM_DIGEST_SIZE]!r}, got {digest!r})"
            )
        self._data = data
        self._i = CHECKSUM_DIGEST_SIZE

    def is_finished(self) -> bool:
        return len(self._data) == self._i

    def read_uint64(self) -> int:
        result = int.from_bytes(
            self._data[self._i : self._i + 8], byteorder="big", signed=False
        )
        self._i += 8
        return result

    def read_str(self) -> str:
        return base64.b64decode(self.read_bytes()).decode("utf-8")

    def read_bytes(self) -> bytes:
        size = self.read_uint64()
        result = self._data[self._i : self._i + size]
        self._i += size
        return result


class AppendingByteSerializer(Generic[T]):
    """A list of values, written out and read back in the order given.

    The order the bytes come out in is the order they went in, and nothing is
    promised about the order of the bytes themselves.
    """

    def __init__(self, *, serialize_fn: Callable[[BytesWriter, T], None]) -> None:
        self._serialize_fn = serialize_fn
        self.clear()

    def clear(self) -> None:
        self._writer = BytesWriter()
        self._writer.write_uint64(_ENCODING_VERSION)

    def append(self, data: T) -> None:
        self._serialize_fn(self._writer, data)

    def extend(self, elems: Iterable[T]) -> None:
        for elem in elems:
            self.append(elem)

    def to_bytes(self) -> bytes:
        return self._writer.to_bytes()

    @staticmethod
    def to_list(data: bytes, *, deserialize_fn: Callable[[BytesReader], T]) -> list[T]:
        reader = BytesReader(data)
        version = reader.read_uint64()
        if version != _ENCODING_VERSION:
            raise AssertionError(
                f"the bytes were written in encoding version {version}, "
                f"which is not the version read here ({_ENCODING_VERSION})"
            )
        result: list[T] = []
        while not reader.is_finished():
            result.append(deserialize_fn(reader))
        return result
