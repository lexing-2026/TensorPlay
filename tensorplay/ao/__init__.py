"""Model optimization toolkit.

Hosts the quantization stack (post-training calibration, quantization-aware
training, native quantized modules under :mod:`tensorplay.ao.nn`), the
eager quantization workflow (:mod:`tensorplay.ao.quantization`), weight
pruning (:mod:`tensorplay.ao.pruning`) and numerical comparison utilities for
float versus quantized models (:mod:`tensorplay.ao.ns`).

The subpackages are heavily interdependent, so they load lazily through
module-level ``__getattr__`` instead of eager imports.
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from types import ModuleType

    from tensorplay.ao import (
        nn as nn,
        ns as ns,
        pruning as pruning,
        quantization as quantization,
    )

__all__ = [
    "nn",
    "ns",
    "pruning",
    "quantization",
]


def __getattr__(name: str) -> "ModuleType":
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
