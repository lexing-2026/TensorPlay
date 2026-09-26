"""Compiling one graph once and running it on every rank.

The feature: a graph is traced and compiled once, on a single rank, and the
resulting code is made independent of which rank runs it, so that the same
binary serves every rank of a job.  What that costs is that anything naming a
device index in the generated text has to be turned into a variable the runtime
fills in, because a literal index would tie the text to one rank.

The switches below say whether that rewriting is on.  Where a build has no way
to compile on one rank, the answer is no, and the checks that would refuse a
device have nothing to refuse: a graph naming an accelerator is compiled
normally, with its device as it was given.
"""

from __future__ import annotations

import tensorplay as tp


def _coor_enabled() -> bool:
    """Whether the device-index rewriting runs at all.

    Read from a flag rather than decided here, so that a caller which turns the
    feature on for one scope does not have to reach into the emitters.
    """

    from . import config

    return bool(getattr(config, "compile_on_one_rank", False))


def _coor_current_accelerator() -> tp.device | None:
    """The accelerator in use, with its index, or None where there is none.

    A machine whose build supports an accelerator but shows none is treated as
    having none: a graph that is entirely on the host still has to compile
    there, so the absence has to be an answer rather than an error.
    """

    try:
        kind = tp.accelerator.current_accelerator(check_available=True)
    except Exception:
        return None
    if kind is None:
        return None
    try:
        index = tp.accelerator.current_device_index()
    except Exception:
        return None
    # What comes back is already a device; only the index has to be filled in.
    kind = getattr(kind, "type", kind)
    if index is None:
        return tp.device(kind)
    return tp.device(kind, index)


def _coor_check_current_accelerator(
    device: tp.device, cur: tp.device | None
) -> None:
    """Refuse a device that could not be made independent of the rank.

    A different kind of accelerator, or a different index of the same one, is
    something the compiled text would have to name.  Rather than emit text that
    only happens to be right on the rank that produced it, the graph is refused
    here.  The host and the abstract device are exempt: neither names a piece of
    hardware, so neither ties the text to a rank.
    """

    if device.type in ("cpu", "meta"):
        return
    if cur is None or device.type != cur.type:
        raise AssertionError(
            f"device {device} is not the current accelerator {cur}, so the "
            f"generated text would name one rank's hardware"
        )
    if device.index is not None and cur.index is not None and device.index != cur.index:
        raise AssertionError(
            f"device {device} is not the current accelerator {cur}, so the "
            f"generated text would name one rank's hardware"
        )
