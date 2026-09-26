"""Gathering compiled kernels into one blob, and putting them back on disk.

A kernel that has been compiled lives in a directory of its own, named after
the hash of what it was compiled from, holding a binary its driver loads and
some text describing it.  That is fine for one machine compiling once, and not
fine for a cache: a cache entry has to carry the kernels it needs, or reading
it on another machine produces a cache entry that cannot run.

So each compilation is noted as it happens -- its hash, its device, where it
went -- and when a cache entry is being written those notes are turned into
one blob.  Reading an entry turns the blob back into the directories the
runtime expects.  Paths inside those files are where the kernel happened to
land on the machine that wrote it, so they are replaced with a marker and put
back on read; a marker that survives into a bundle means something wrote a
path that looked like the marker, which is refused rather than bundled.

Only the configurations that won are bundled.  A kernel that lost a
measurement will not be launched, and a cache entry carrying it is a cache
entry carrying a file nobody asked for.  Where nothing was measured -- one
configuration, or a template that was not measured -- everything is bundled,
since there is nothing to have lost.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import os
import shutil
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from tensorplay.utils._filelock import FileLock

from .config import force_disable_caches
from .runtime.cache_dir_utils import triton_cache_dir
from .utils import _IS_WINDOWS, GPU_KERNEL_BIN_EXTS

if TYPE_CHECKING:
    from .runtime.triton_heuristics import CachingAutotuner

log = logging.getLogger(__name__)

#: What a compilation was noted as, so that it can be found again on disk: the
#: hash its directory is named after, the device it was compiled for, and that
#: device's cache root.
TritonBundleEntry = dataclasses.make_dataclass(
    "TritonBundleEntry",
    [("kernel_hash", str), ("device", int), ("directory", str)],
)


@dataclasses.dataclass(frozen=True)
class TritonKernelArtifact:
    """One file belonging to a kernel, held as bytes.

    Which file it is depends on the device: a binary its driver loads, the
    intermediate the compiler produced, or a description of where things came
    from.  All of them travel together, because a driver that cannot load its
    binary has nothing to run whatever else arrived.
    """

    filename: str
    payload: bytes = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True)
class StaticallyLaunchedAutotuner:
    """An autotuner whose kernels can be launched without being compiled first.

    The autotuner is kept whole, because which configuration it settled on is
    part of what it is; the binaries it refers to travel separately, as the
    kernel artifacts of the entries that recorded them.
    """

    cache_key: str
    kernel_name: str
    kernel: "CachingAutotuner"


@dataclasses.dataclass(frozen=True)
class TritonKernelArtifacts:
    """Every file belonging to one kernel on one device."""

    kernel_hash: str
    device: int
    artifacts: list[TritonKernelArtifact]


@dataclasses.dataclass(frozen=True)
class TritonBundlerMetadata:
    """What a gather or an emit produced, for whoever is keeping a record."""

    cached_kernel_names: list[str]
    statically_launched_kernel_names: list[str]


@dataclasses.dataclass(frozen=True)
class TritonBundle:
    """A bundle, in the form it is stored in a cache entry."""

    kernel_artifacts: list[TritonKernelArtifacts]
    static_autotuners: list[StaticallyLaunchedAutotuner]


class TritonBundler:
    """Notes each compilation, and turns the notes into a bundle on demand.

    The intended order of use is: begin before compiling anything, put for
    each kernel as it is compiled, collect when a cache entry is written, end
    when compiling is finished -- and read_and_emit when an entry is read.
    Nothing is gathered outside that order, which is what keeps a kernel that
    was never written from ending up in a bundle.
    """

    _entries: list | None = None
    _static_autotuners: list | None = None
    _winners: "OrderedDict[str, bool]" | None = None

    #: Stands in for a path inside a bundled file.  A payload carrying it is
    #: one whose path was not replaced, which means the replacement can be
    #: undone wrongly, so it is refused.
    _REPLACE_BYTES: bytes = b"[REPLACE]"

    @staticmethod
    def is_enabled() -> bool:
        """Whether kernels should be bundled into cache entries at all.

        Off unless asked for: a bundle makes an entry carry every kernel it
        needs, which is worth it when the entry will travel and not worth it
        when the cache is only ever read on the machine that wrote it.
        """

        if force_disable_caches:
            return False
        from . import config

        if (b := config.bundle_triton_into_fx_graph_cache) is not None:
            return b
        return False

    @classmethod
    def begin_compile(cls) -> None:
        """Start a fresh set of notes.

        Refuses to start while notes are open: a second compilation writing
        into the same set would have its kernels attributed to whichever
        compilation finished last.
        """

        if not TritonBundler.is_enabled():
            return
        log.debug("beginning a kernel bundle")
        if cls._entries is not None:
            raise AssertionError(
                "beginning a bundle while one is open; the open one has to be "
                "collected or ended first"
            )
        cls._entries = []
        cls._static_autotuners = []
        cls._winners = OrderedDict()

    @classmethod
    def end_compile(cls) -> None:
        """Stop taking notes, throwing away anything not yet collected."""

        log.debug("ending a kernel bundle")
        cls._entries = None
        cls._static_autotuners = None
        cls._winners = None

    @classmethod
    def put(cls, kernel_hash: str, device: int) -> None:
        """Note that a kernel was compiled, for a bundle to pick up later."""

        if (entries := cls._entries) is not None:
            entries.append(
                TritonBundleEntry(kernel_hash, device, triton_cache_dir(device))
            )

    @classmethod
    def put_winner(cls, kernel_hash: str) -> None:
        """Note that this kernel is the one that was chosen.

        Only chosen kernels are bundled.  Where no choice was made -- a single
        configuration, or a template that was not measured -- nothing is noted
        as a winner and everything is bundled, because nothing was lost.
        """

        if cls._winners is not None:
            cls._winners[kernel_hash] = True

    @classmethod
    def put_static_autotuner(cls, key: str, kernel: "CachingAutotuner") -> None:
        """Note an autotuner that can be launched without compiling first."""

        from . import config

        if not config.use_static_triton_launcher:
            raise AssertionError(
                "a statically launched autotuner was noted while static "
                "launching is switched off"
            )
        if (entries := cls._static_autotuners) is not None:
            # What cannot be written out is cleared first, and put back
            # afterwards: the autotuner is still needed as it is.
            old_values = kernel.prepare_for_pickle()
            new_kernel = copy.deepcopy(kernel)
            new_kernel.prepare_for_caching()
            new_kernel._reload_kernel = None
            entries.append(
                StaticallyLaunchedAutotuner(
                    key,
                    new_kernel.inductor_meta.get("kernel_name", "unknown_kernel"),
                    new_kernel,
                )
            )
            kernel.restore_after_unpickle(old_values)

    @classmethod
    def collect_static_autotuners(
        cls,
    ) -> tuple[list, list[str]]:
        if not cls._static_autotuners:
            return [], []
        log.info(
            "saving %d statically launchable autotuners",
            len(cls._static_autotuners),
        )
        return cls._static_autotuners, [i.kernel_name for i in cls._static_autotuners]

    @classmethod
    def collect(cls) -> tuple[TritonBundle, TritonBundlerMetadata | None]:
        """Turn the notes into a bundle, and stop taking notes.

        This is the step a cache write depends on.  A kernel whose directory is
        gone contributes nothing rather than failing the write: the cache entry
        is then written without it, and the next run recompiles that kernel,
        which is the same outcome as not having had a cache.
        """

        if not TritonBundler.is_enabled():
            cls.end_compile()
            return TritonBundle([], []), None

        entries = cls._entries
        if entries is None:
            return TritonBundle([], []), None

        from . import config

        winners = cls._winners
        result: list[TritonKernelArtifacts] = []
        kernel_names: list[str] = []
        for entry in entries:
            if winners and entry.kernel_hash not in winners:
                log.debug("skipping a kernel that was not chosen: %s",
                          entry.kernel_hash)
                continue
            artifacts: list[TritonKernelArtifact] = []
            path = os.path.join(entry.directory, entry.kernel_hash)
            if not os.path.exists(path):
                continue
            for filename in os.listdir(path):
                filepath = os.path.join(path, filename)
                try:
                    if not os.path.isfile(filepath):
                        raise AssertionError(
                            f"expected a regular file, got {filepath}"
                        )
                    with open(filepath, "rb") as file:
                        payload = file.read()
                        if filepath.endswith(".json"):
                            # A description holding the marker would mean a
                            # path was written that looked like the marker,
                            # and reading the bundle back would rewrite it.
                            if TritonBundler._REPLACE_BYTES in payload:
                                log.warning(
                                    "the bundle holds a path that looks like "
                                    "the marker: %s", TritonBundler._REPLACE_BYTES
                                )
                                raise AssertionError(
                                    "the bundle holds bytes it cannot replace"
                                )
                            payload = payload.replace(
                                str.encode(path), TritonBundler._REPLACE_BYTES
                            )
                        artifacts.append(TritonKernelArtifact(filename, payload))
                except Exception:
                    log.debug("could not collect a kernel file", exc_info=True)
                    continue
                if os.path.splitext(filename)[1] in GPU_KERNEL_BIN_EXTS.values():
                    # The binary is the one file that names the kernel; the
                    # rest describe it.  The name is the binary's without its
                    # extension.
                    kernel_names.append(Path(filename).stem)
            if artifacts:
                result.append(
                    TritonKernelArtifacts(entry.kernel_hash, entry.device, artifacts)
                )

        if config.use_static_triton_launcher:
            static_autotuners, static_kernel_names = cls.collect_static_autotuners()
        else:
            static_autotuners = []
            static_kernel_names = []
        cls.end_compile()
        return TritonBundle(result, static_autotuners), TritonBundlerMetadata(
            kernel_names, static_kernel_names
        )

    @staticmethod
    def read_and_emit(bundle: TritonBundle) -> TritonBundlerMetadata | None:
        """Put a bundle's kernels back as the directories the runtime wants.

        Written under a name nobody else is using and moved into place, so that
        a reader never sees half a directory.  A directory that is already there
        and not empty is left alone: something on this machine has already
        compiled that kernel, and replacing it would throw that away.
        """

        if not TritonBundler.is_enabled():
            return None

        kernel_names: list[str] = []
        for artifacts in bundle.kernel_artifacts:
            basedir = triton_cache_dir(artifacts.device)
            directory = os.path.join(basedir, artifacts.kernel_hash)

            if os.path.exists(directory) and len(os.listdir(directory)) != 0:
                log.debug(
                    "leaving a kernel's directory alone: %s is not empty",
                    directory,
                )
                continue

            Path(basedir).mkdir(parents=True, exist_ok=True)
            rnd_id = str(uuid.uuid4())
            tmp_dir = os.path.join(basedir, f"tmp.{rnd_id}")
            os.makedirs(tmp_dir)

            for artifact in artifacts.artifacts:
                filepath = os.path.join(tmp_dir, artifact.filename)
                with open(filepath, "wb") as file:
                    payload = artifact.payload
                    if artifact.filename.endswith(".json"):
                        payload = payload.replace(
                            TritonBundler._REPLACE_BYTES, str.encode(directory)
                        )
                    file.write(payload)
                if os.path.splitext(artifact.filename)[1] in (
                    GPU_KERNEL_BIN_EXTS.values()
                ):
                    kernel_names.append(Path(artifact.filename).stem)

            if _IS_WINDOWS:
                with FileLock(directory + ".lock"):
                    if os.path.exists(directory):
                        shutil.rmtree(directory)
                    os.replace(tmp_dir, directory)
            else:
                try:
                    os.replace(tmp_dir, directory)
                except OSError:
                    log.warning(
                        "the destination is not empty, leaving the kernel to be "
                        "compiled here: %s", tmp_dir
                    )

        static_kernel_names: list[str] = []
        return TritonBundlerMetadata(kernel_names, static_kernel_names)
