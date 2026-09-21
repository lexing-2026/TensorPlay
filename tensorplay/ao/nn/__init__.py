"""Neural-network building blocks for quantized inference and training.

``quantized`` hosts modules whose inputs and outputs are native quantized
tensors, ``quantized.dynamic`` hosts modules that quantize activations at
runtime, and ``qat`` hosts float modules carrying simulated quantization for
training. ``intrinsic`` provides fused building blocks (for example linear
followed by relu) used by the fusion workflow.

The subpackages are mutually interdependent, so they load lazily through
module-level ``__getattr__`` instead of eager imports.
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from types import ModuleType

    from tensorplay.ao.nn import (
        intrinsic as intrinsic,
        qat as qat,
        quantized as quantized,
    )

__all__ = [
    "intrinsic",
    "qat",
    "quantized",
]


def __getattr__(name: str) -> "ModuleType":
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
