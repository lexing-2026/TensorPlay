"""What the machine this is running on is, as far as a kernel writer cares.

Three questions get asked over and over while a kernel is written: which
architecture it is for, which toolkit built it, and whether a driver is around
to build it at all. None of them change while a program runs, so each is asked
once and remembered; and a program that says which architecture it wants is
believed, because a kernel is often built on one machine for another.
"""

from __future__ import annotations

import functools
import logging
import shutil

import tensorplay as tp

from ... import config
from ...utils import clear_on_fresh_cache

log = logging.getLogger(__name__)


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_cuda_arch() -> str | None:
    """Which architecture a kernel is for, as the two digits the device reports.

    What the program said if it said anything, and otherwise what the first
    device present reports -- the first, because a machine with more than one
    device of different kinds has no single answer and the first is the one that
    decides where the program starts.

    Nothing rather than a guess when the device cannot be asked: an architecture
    that was guessed would compile a kernel that does not run, which is worse
    than not knowing.
    """

    try:
        cuda_arch = config.cuda.arch
        if cuda_arch is None:
            major, minor = tp.cuda.get_device_capability(0)
            return str(major * 10 + minor)
        return str(cuda_arch)
    except Exception:
        log.exception("Could not read which architecture to compile for")
        return None


@clear_on_fresh_cache
@functools.lru_cache(1)
def is_datacenter_blackwell_arch() -> bool:
    """Whether this is the generation of device that a datacenter kernel is for.

    A range rather than one number, because the generation spans more than one
    numbered architecture and a kernel written for the first of them runs on the
    rest. An architecture that could not be read is not this one: a kernel
    written for hardware nobody has will not run on the hardware that is here.
    """

    arch = get_cuda_arch()
    if arch is None:
        return False
    arch_number = int(arch)
    return 100 <= arch_number < 110


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_cuda_version() -> str | None:
    """Which toolkit is building, as its version.

    What the program said if it said anything, and otherwise what is installed.
    Nothing rather than a guess when neither can be read, for the same reason as
    the architecture: a version that was guessed would build against an
    interface that may not be the one there.
    """

    try:
        cuda_version = config.cuda.version
        if cuda_version is None:
            cuda_version = tp.version.cuda
        return cuda_version
    except Exception:
        log.exception("Could not read which toolkit version is installed")
        return None


@functools.cache
def nvcc_exist(nvcc_path: str | None = "nvcc") -> bool:
    """Whether a driver is on the path to build with.

    Asked with a path because the driver is not always the one on the path: a
    build that names its own driver is looking for that one, and answering about
    the one that happens to be there would be answering about a different thing.
    """

    return nvcc_path is not None and shutil.which(nvcc_path) is not None
