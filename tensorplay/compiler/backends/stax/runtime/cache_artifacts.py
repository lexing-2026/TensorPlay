"""The pieces a cache is made of, and the list of what went into it.

A cache that survives the process has to be written down before the process
ends, and what is written down is not the cache itself but the pieces that
would rebuild it -- a compiled kernel, a set of measurements, a warmed module.
Each piece is an artifact: bytes, a name to find them by, and the act of
putting them back where the compiler will look.

The pieces are gathered as they are produced rather than collected at the end,
so that the work of walking them is only paid by a caller that actually writes
a cache.  What each kind of piece is called is registered once, and every
registration adds a way to ask for the pieces of that kind by name -- so a
caller that wants to log them does not have to know the kinds in advance.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from itertools import chain
from typing import Any

from .appending_byte_serializer import (
    AppendingByteSerializer,
    BytesReader,
    BytesWriter,
)

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class CacheArtifact(ABC):
    """One piece of a cache: what it is called and what it holds.

    The content is kept out of the printed form because it is usually binary,
    and a list of artifacts that printed itself as a screen of bytes would be a
    list nobody reads.
    """

    key: str
    content: bytes = dataclasses.field(repr=False)

    @staticmethod
    def serialize(writer: BytesWriter, cls: "CacheArtifact") -> None:
        writer.write_str(cls.key)
        writer.write_bytes(cls.content)

    @staticmethod
    def deserialize(artifact_type: str, reader: BytesReader) -> "CacheArtifact":
        key = reader.read_str()
        content = reader.read_bytes()
        return CacheArtifactFactory.create(artifact_type, key, content)

    @staticmethod
    def encode(content: Any) -> bytes:
        """The bytes this piece is, refusing anything that is not bytes already."""

        if not isinstance(content, bytes):
            raise AssertionError(f"expected bytes, got {type(content)}")
        return content

    @abstractmethod
    def populate_cache(self) -> None:
        """Put this piece back where the compiler will find it."""

    @staticmethod
    def type() -> str:
        """What kind of piece this is, which names it across the whole cache.

        The name has to be unique across every kind of piece, because it is
        what a reader uses to decide what it is holding, and two kinds sharing
        a name would leave that undecidable.
        """

        raise RuntimeError(
            "this is the base of the artifact kinds; use one of the kinds that "
            "says which it is"
        )


class CacheArtifactFactory:
    """The kinds of piece that exist, and how to make one of a given kind."""

    _artifact_types: dict[str, type[CacheArtifact]] = {}

    @classmethod
    def register(cls, artifact_cls: type[CacheArtifact]) -> type[CacheArtifact]:
        artifact_type_key = artifact_cls.type()
        if artifact_type_key in cls._artifact_types:
            raise AssertionError(
                f"a piece of kind {artifact_type_key} is already registered"
            )
        cls._artifact_types[artifact_type_key] = artifact_cls
        setattr(
            CacheInfo,
            f"{artifact_type_key}_artifacts",
            property(lambda self: self.artifacts[artifact_type_key]),
        )
        return artifact_cls

    @classmethod
    def _get_artifact_type(cls, artifact_type_key: str) -> type[CacheArtifact]:
        if artifact_type_key not in cls._artifact_types:
            raise AssertionError(
                f"no piece of kind {artifact_type_key} is registered"
            )
        return cls._artifact_types[artifact_type_key]

    @classmethod
    def create(
        cls, artifact_type_key: str, key: str, content: bytes
    ) -> CacheArtifact:
        return cls._get_artifact_type(artifact_type_key)(key, content)

    @classmethod
    def encode_create(
        cls, artifact_type_key: str, key: str, content: Any
    ) -> CacheArtifact:
        artifact_cls = cls._get_artifact_type(artifact_type_key)
        return artifact_cls(key, artifact_cls.encode(content))


@dataclasses.dataclass
class CacheInfo:
    """What went into a cache, by kind -- the record an instrumented run keeps."""

    artifacts: defaultdict[str, list[str]] = dataclasses.field(
        default_factory=lambda: defaultdict(list)
    )

    #: One of these per kind, added by :meth:`CacheArtifactFactory.register`.
    #: A kind nobody registered has no property, and asking for its pieces
    #: then is an attribute error rather than an empty answer: a caller asking
    #: for pieces of a kind that does not exist is asking the wrong question.
    @property
    def tp_artifacts(self) -> list[str]:
        ...

    @property
    def autotune_artifacts(self) -> list[str]:
        ...

    @property
    def aot_autograd_artifacts(self) -> list[str]:
        ...

    @property
    def pgo_artifacts(self) -> list[str]:
        ...

    @property
    def precompile_artifacts(self) -> list[str]:
        ...

    def add(self, artifact: CacheArtifact) -> None:
        self.artifacts[artifact.type()].append(artifact.key)

    def clear(self) -> None:
        self.artifacts.clear()

    def empty(self) -> bool:
        return not self.artifacts


def _serialize_single_cache(
    writer: BytesWriter, cls: "tuple[str, list[CacheArtifact]]"
) -> None:
    writer.write_str(cls[0])
    writer.write_uint64(len(cls[1]))
    for artifact in cls[1]:
        CacheArtifact.serialize(writer, artifact)


def _deserialize_single_cache(
    reader: BytesReader,
) -> "tuple[str, list[CacheArtifact]]":
    artifacts = []
    artifact_type_key = reader.read_str()
    num_artifacts = reader.read_uint64()
    for _ in range(num_artifacts):
        artifacts.append(CacheArtifact.deserialize(artifact_type_key, reader))
    return artifact_type_key, artifacts


CacheArtifactsResult = dict[str, list[CacheArtifact]]


@dataclasses.dataclass(frozen=True)
class CacheArtifactRecorder:
    """Where one caller records the pieces it produced.

    Cache implementations share this so that a caller holding one of these can
    record without knowing which implementation is underneath.
    """

    artifact_type: str
    key: str

    def record(self, content: Any) -> None:
        CacheArtifactManager.record_artifact(self.artifact_type, self.key, content)

    def record_if_present(self, content: Any | None) -> None:
        if content is not None:
            self.record(content)


class CacheArtifactManager:
    """The pieces produced so far, and the act of writing them out.

    The intended order of use is: produce pieces while compiling, recording
    each as it is produced; then serialize to write them down; then, in a
    process that did not produce them, deserialize and populate.  What the
    pieces are worth is not guaranteed across builds -- they are only used
    when the code that produced them matches, so a mismatch costs a rebuild
    rather than a wrong answer.
    """

    #: Guarded by whatever lock the compilation is already holding.
    _new_cache_artifacts: CacheArtifactsResult = defaultdict(list)
    #: Kept apart from the list above so that a piece recorded twice is written
    #: once, and so that this list survives a serialize that emptied the other.
    _seen_artifacts: set = set()
    #: Written to only when serialize is called, so a run that never writes a
    #: cache never pays for walking the pieces.
    _serializer: AppendingByteSerializer = AppendingByteSerializer(
        serialize_fn=_serialize_single_cache
    )
    _cache_info: CacheInfo = CacheInfo()

    @classmethod
    def clear(cls) -> None:
        cls._new_cache_artifacts.clear()
        cls._seen_artifacts.clear()
        cls._serializer.clear()
        cls._cache_info.clear()

    @classmethod
    @contextmanager
    def with_fresh_cache(cls):
        """Collect a separate set of pieces for the length of one block.

        What a block records has to be recordable on its own -- a caller
        compiling something and then rolling it back should not find the
        pieces of the attempt it rolled back -- so the four places the pieces
        are kept are set aside and put back afterwards.
        """

        original_new_cache_artifacts = cls._new_cache_artifacts
        original_seen_artifacts = cls._seen_artifacts
        original_serializer = cls._serializer
        original_cache_info = cls._cache_info

        cls._new_cache_artifacts = defaultdict(list)
        cls._seen_artifacts = set()
        cls._serializer = AppendingByteSerializer(serialize_fn=_serialize_single_cache)
        cls._cache_info = CacheInfo()
        try:
            yield
        finally:
            cls._new_cache_artifacts = original_new_cache_artifacts
            cls._seen_artifacts = original_seen_artifacts
            cls._serializer = original_serializer
            cls._cache_info = original_cache_info

    @classmethod
    def record_artifact(cls, artifact_type: str, key: str, content: Any) -> None:
        """Record one piece, unless an identical one was already recorded."""

        artifact = CacheArtifactFactory.encode_create(artifact_type, key, content)
        if artifact in cls._seen_artifacts:
            return
        log.debug("recording %s", artifact)
        cls._new_cache_artifacts[artifact_type].append(artifact)
        cls._seen_artifacts.add(artifact)

    @classmethod
    def need_serialize(cls) -> bool:
        """Whether anything has been recorded since the last write."""

        return len(cls._new_cache_artifacts) != 0

    @classmethod
    def serialize(cls) -> tuple[bytes, CacheInfo] | None:
        """Write the pieces down, or answer that there is nothing to write.

        Nothing to write is answered with nothing rather than with an empty
        payload: an empty payload and a payload carrying no pieces are the same
        thing, and only one of them is worth keeping.
        """

        for artifact in chain(*cls._new_cache_artifacts.values()):
            log.debug("saving: %s", artifact)
            cls._cache_info.add(artifact)

        if cls._cache_info.empty():
            return None

        try:
            # Copied because compiling more can keep adding to the record, and
            # what is returned should describe what was written.
            info = copy.deepcopy(cls._cache_info)
            cls._serializer.extend(cls._new_cache_artifacts.items())
            artifact_bytes = cls._serializer.to_bytes()
            cls._new_cache_artifacts.clear()
            return artifact_bytes, info
        except Exception:
            log.warning("could not write the cache pieces down", exc_info=True)
        return None

    @staticmethod
    def deserialize(serialized_artifacts: bytes) -> CacheArtifactsResult | None:
        """Read pieces back, or answer that these bytes are not readable.

        Bytes that cannot be read are not a reason to fail: a cache that
        cannot be read is a cache that gets rebuilt, which is what happens
        anyway when there is no cache.
        """

        try:
            return dict(
                AppendingByteSerializer.to_list(
                    serialized_artifacts,
                    deserialize_fn=_deserialize_single_cache,
                )
            )
        except Exception:
            log.warning("could not read the cache pieces back", exc_info=True)
            return None

    @staticmethod
    def populate_caches(artifacts: CacheArtifactsResult) -> CacheInfo:
        """Put every piece back where the compiler will look for it."""

        info = CacheInfo()
        for artifact in chain(*artifacts.values()):
            log.debug("writing: %s", artifact)
            info.add(artifact)
            artifact.populate_cache()
        return info

    @classmethod
    def _ensure_cache_artifacts_registered(cls) -> None:
        """Bring every kind of piece into the registry.

        A process that reads a cache never ran the code that registered the
        kinds, so a kind it has not heard of would be refused.  Importing the
        modules that register them is what makes them known; the imports are
        here for that effect and nothing else.
        """

        from .. import codecache  # noqa: F401
        from .autotune_cache import AutotuneCacheArtifact  # noqa: F401
