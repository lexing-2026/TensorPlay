"""Where this route keeps what it built, and whose it is when it cannot write.

An artifact is named after what it was built from, so the same source compiles
once however many times, and from wherever, it is asked for.  The directory
those artifacts live under is settled once per process and then agreed on by
everything that writes there, because two writers that disagreed about the
directory would each keep their own copy of the same thing.

A cache directory written by another user is the case worth handling: it is
unwritable, and an unwritable directory fails silently by never being written,
so a process would measure the same source compiling over and over and blame
the compiler.  So the directory is moved to one this user owns, once, before
anything is written to it.
"""

from __future__ import annotations

import getpass
import os
import re
import tempfile
from collections.abc import Generator
from contextlib import contextmanager

#: The environment variable that says where artifacts go.  A relative path is
#: read as relative to the working directory, so a run started somewhere else
#: does not silently write into a different tree than the one it read.
CACHE_DIR_ENV = "TP_CACHE_DIR"

#: What the directory is called when nothing says otherwise.
DEFAULT_CACHE_DIRNAME = ".tp_cache"

#: Characters a path may not carry on the platforms a cache directory is
#: shared across.  A user name is the one thing in the default path that can
#: contain them, so it is the one thing that has to be scrubbed.
_UNSAFE_IN_NAME = re.compile(r'[\\/:*?"<>|]')


def _username() -> str:
    """Who is running, in a form that can be part of a path."""

    try:
        name = getpass.getuser()
    except (KeyError, ModuleNotFoundError, OSError):
        # A process with no passwd entry still has to land somewhere writable,
        # and the numeric id is as good a discriminator as a name would be.
        getuid = getattr(os, "getuid", None)
        name = f"uid_{getuid()}" if callable(getuid) else "unknown_user"
    return _UNSAFE_IN_NAME.sub("_", name)


def default_cache_dir() -> str:
    """Where artifacts go when nothing has said otherwise.

    Under the working directory, so a tree that is built and then thrown away
    takes its artifacts with it, and a tree that is shared has one set between
    everyone who works in it.
    """

    return os.path.join(os.getcwd(), DEFAULT_CACHE_DIRNAME)


def cache_dir() -> str:
    """The one directory this process writes artifacts under.

    Settled on the first call and remembered in the environment, so a caller
    that reads the environment directly and a caller that calls this agree
    without either having to know about the other.
    """

    configured = os.environ.get(CACHE_DIR_ENV)
    if configured is None:
        configured = default_cache_dir()
    resolved = os.path.abspath(configured)
    os.environ[CACHE_DIR_ENV] = resolved
    os.makedirs(resolved, exist_ok=True)
    return resolved


def writable_cache_dir() -> str:
    """A directory under the cache root that this process can write to.

    A directory another user created leaves this one unable to write, and an
    unwritable cache is worse than no cache: the artifact is rebuilt every time
    and the cost is invisible.  So the root is moved aside once, to a place
    named for who is running, rather than discovered to be unwritable once per
    artifact.
    """

    base = cache_dir()
    if os.access(base, os.W_OK):
        return base
    fallback = os.path.join(tempfile.gettempdir(), f"tp-cache-{os.getuid()}")
    os.makedirs(fallback, exist_ok=True)
    os.environ[CACHE_DIR_ENV] = fallback
    return fallback


def triton_cache_dir(device: int) -> str:
    """Where one device's generated device code is kept.

    Keyed by device, because generated code is specialized to the device it was
    compiled for and is not interchangeable between them.
    """

    configured = os.environ.get("TRITON_CACHE_DIR")
    if configured is not None:
        return configured
    return os.path.join(cache_dir(), "triton", str(device))


@contextmanager
def temporary_cache_dir(directory: str) -> Generator[None, None, None]:
    """Point the cache at ``directory`` for the length of the block.

    Restored on the way out, including when the block raises, so a failed
    attempt does not leave every later write going somewhere unexpected.
    """

    from ..kernel_cache import clear_caches

    original = os.environ.get(CACHE_DIR_ENV)
    os.environ[CACHE_DIR_ENV] = directory
    try:
        clear_caches()
        yield
    finally:
        clear_caches()
        if original is None:
            os.environ.pop(CACHE_DIR_ENV, None)
        else:
            os.environ[CACHE_DIR_ENV] = original


__all__ = [
    "CACHE_DIR_ENV",
    "DEFAULT_CACHE_DIRNAME",
    "cache_dir",
    "default_cache_dir",
    "temporary_cache_dir",
    "triton_cache_dir",
    "writable_cache_dir",
]
