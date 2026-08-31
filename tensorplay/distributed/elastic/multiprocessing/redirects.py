"""Std stream redirection descriptors for worker processes."""
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
    out: dict[int, Std] = {}
    for rank, value in val_or_map.items():
        out[rank] = value
    if 0 in out:
        default = out[0]
    else:
        default = Std.NONE
    for i in range(local_world_size):
        if i not in out:
            out[i] = default
    return out


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
