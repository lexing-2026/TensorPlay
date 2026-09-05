# mypy: allow-untyped-defs
"""Accelerator helpers.

``current_accelerator`` reports the device type the runtime selects for
tensor work: the GPU accelerator when one is available, otherwise CPU.
"""

import tensorplay

__all__ = ["current_accelerator"]


def current_accelerator() -> tensorplay.device:
    """Return the current accelerator device, or ``None`` without a GPU."""
    if tensorplay.cuda.is_available():
        return tensorplay.device("cuda", tensorplay.cuda.current_device())
    return None
