"""Linear module for dynamic quantization-aware training.

The float weight is fake-quantized on every forward like the static QAT
linear, but the activation grid is collected by a memoryless (moving-average
with averaging constant one) observer, so activations can be quantized at
runtime with no calibration pass.
"""

from __future__ import annotations

from tensorplay.ao.nn.qat.modules.linear import Linear as QATLinear

__all__ = ["Linear"]


class Linear(QATLinear):
    """A QAT linear module that requires a memoryless activation observer."""

    def __init__(self, in_features, out_features, bias=True, qconfig=None,
                 device=None, dtype=None):
        super().__init__(in_features, out_features, bias, qconfig, device, dtype)
        from .....quantization.qconfig import _activation_is_memoryless

        if not _activation_is_memoryless(qconfig):
            raise ValueError(
                "Dynamic QAT requires a memoryless observer. "
                "This means a MovingAverage observer with averaging constant equal to 1"
            )