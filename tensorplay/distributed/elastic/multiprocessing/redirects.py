"""Std stream redirection descriptors for worker processes."""
import ctypes
import logging
import os
import sys
from contextlib import contextmanager
from functools import partial
from dataclasses import dataclass
from enum import IntFlag
from typing import Union


class Std(IntFlag):
    """Which standard streams a worker's output should go to."""

    NONE = 0
    OUT = 1
    ERR = 2
    ALL = 3

    @classmethod
    def from_str(cls, vm: str) -> Union["Std", dict[int, "Std"]]:
        """Parse ``to_map``-style strings: ``0``/``1``/``2``/``3`` or ``i:j:k``.

        The per-rank form maps rank ``i`` (or ``default``) to a mode; both
        ``0:1`` (shorthand for ``0:1:1``) and full triples are accepted.
        """
        text = vm.strip().lower()

        if "," in text:
            result: dict[int, Std] = {}
            for item in text.split(","):
                rank, value = item.split(":", 1)
                result[int(rank)] = cls(int(value))
            return result

        def _to_std(v: str) -> Std:
            if v.isdigit():
                return Std(int(v))
            raise ValueError(f"Invalid std redirection value: {v!r}")

        if ":" not in text:
            return _to_std(text)
        parts = text.split(":")
        out: dict[int, Std] = {}
        if len(parts) == 2:
            parts.append(parts[1])
        if len(parts) != 3:
            raise ValueError(f"Invalid std redirection map: {vm!r}")
        for i, part in enumerate(parts):
            if part == "default":
                continue
            out[i] = _to_std(part)
        return out


def to_map(val_or_map: Union[Std, dict[int, Std]], local_world_size: int) -> dict[int, Std]:
    """Expand a per-rank redirection spec into one entry per local rank."""
    if isinstance(val_or_map, Std):
        return {i: val_or_map for i in range(local_world_size)}
    return {
        rank: val_or_map.get(rank, Std.NONE)
        for rank in range(local_world_size)
    }


logger = logging.getLogger(__name__)


def get_libc():
    if sys.platform == "darwin":
        return None
    if sys.platform == "win32":
        for name in ("ucrtbase", "msvcrt", "msvcr110", "msvcr100"):
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        raise RuntimeError("Could not load the C runtime library")
    return ctypes.CDLL("libc.so.6")


libc = get_libc()


def _c_std(stream: str):
    if stream not in {"stdout", "stderr"}:
        raise ValueError(f"unknown standard stream: {stream}")
    if libc is None:
        return None
    if sys.platform == "win32":
        stream_index = 2 if stream == "stderr" else 1
        iob_func = getattr(libc, "__acrt_iob_func", None)
        if iob_func is not None:
            iob_func.restype = ctypes.POINTER(ctypes.c_void_p)
            iob_func.argtypes = [ctypes.c_uint]
            return iob_func(stream_index)
        iob = (ctypes.POINTER(ctypes.c_void_p) * 3).in_dll(libc, "_iob")
        return iob[stream_index]
    return ctypes.c_void_p.in_dll(libc, stream)


def _python_std(stream: str):
    return {"stdout": sys.stdout, "stderr": sys.stderr}[stream]


@contextmanager
def redirect(std: str, to_file: str):
    """Redirect one standard stream at the file-descriptor level."""
    python_std = _python_std(std)
    fd = python_std.fileno()
    with os.fdopen(os.dup(fd)) as original, open(to_file, "w+b") as destination:
        python_std.flush()
        os.dup2(destination.fileno(), fd)
        try:
            yield
        finally:
            python_std.flush()
            os.dup2(original.fileno(), fd)


redirect_stdout = partial(redirect, "stdout")
redirect_stderr = partial(redirect, "stderr")


@dataclass
class Redirects:
    """Per-stream redirection modes for a worker group."""

    stdouts: Union[Std, dict[int, Std]] = Std.NONE
    stderrs: Union[Std, dict[int, Std]] = Std.NONE

    @classmethod
    def from_default(cls, std: Std) -> "Redirects":
        return cls(stdouts=std, stderrs=std)

    @classmethod
    def from_str(cls, vm: str) -> "Redirects":
        if "->" in vm:
            outs, errs = vm.split("->", 1)
            return cls(stdouts=Std.from_str(outs), stderrs=Std.from_str(errs))
        value = Std.from_str(vm)
        return cls(stdouts=value, stderrs=value)
