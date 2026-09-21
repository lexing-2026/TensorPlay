"""Quantization-aware linear module.

The float weight is passed through a straight-through fake-quantize module on
every forward, so training optimizes the network against the quantized grid
while the parameters themselves stay float.
"""

from __future__ import annotations

import tensorplay
from tensorplay import nn

__all__ = ["Linear"]


class Linear(nn.Linear):
    """A linear module with simulated per-output-channel weight quantization."""

    def __init__(self, in_features, out_features, bias=True, qconfig=None):
        super().__init__(in_features, out_features, bias)
        if qconfig is None:
            # Resolved lazily: the quantization package imports this package
            # at module scope, so an eager import here would be circular.
            from ....quantization.fake_quantize import PerChannelFakeQuantize

            self.weight_fake_quant = PerChannelFakeQuantize(ch_axis=0)
        else:
            self.weight_fake_quant = qconfig.weight()
        self.qconfig = qconfig

    def forward(self, input):
        import tensorplay.nn.functional as F

        return F.linear(input, self.weight_fake_quant(self.weight), self.bias)

    def _get_name(self):
        return "QATLinear"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        if not isinstance(mod, nn.Linear):
            raise TypeError("from_float(): expected a Linear module")
        qconfig = getattr(mod, "qconfig", None)
        qat_linear = cls(mod.in_features, mod.out_features,
                         bias=mod.bias is not None, qconfig=qconfig)
        qat_linear.weight = mod.weight
        if mod.bias is not None:
            qat_linear.bias = mod.bias
        return qat_linear

    def to_float(self):
        float_linear = nn.Linear(self.in_features, self.out_features,
                                 self.bias is not None)
        float_linear.weight = nn.Parameter(self.weight.detach())
        if self.bias is not None:
            float_linear.bias = nn.Parameter(self.bias.detach())
        float_linear.train(self.training)
        return float_linear
