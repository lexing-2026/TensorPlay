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

    def __init__(self, in_features, out_features, bias=True, qconfig=None, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.qconfig, self.weight_fake_quant = _weight_fake_quant(qconfig, self.weight.device)

    def forward(self, input):
        import tensorplay.nn.functional as F

        return F.linear(input, self.weight_fake_quant(self.weight), self.bias)

    def _get_name(self):
        return "QATLinear"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        source = getattr(mod, "op", mod)
        if not isinstance(source, nn.Linear):
            raise TypeError("from_float(): expected a Linear module")
        qconfig = getattr(mod, "qconfig", None)
        training = mod.training
        mod = source
        qat_linear = cls(mod.in_features, mod.out_features,
                         bias=mod.bias is not None, qconfig=qconfig,
                         device=mod.weight.device, dtype=mod.weight.dtype)
        qat_linear.weight = mod.weight
        if mod.bias is not None:
            qat_linear.bias = mod.bias
        qat_linear.train(training)
        return qat_linear

    def to_float(self):
        float_linear = nn.Linear(self.in_features, self.out_features,
                                 self.bias is not None, device=self.weight.device, dtype=self.weight.dtype)
        float_linear.weight = nn.Parameter(self.weight.detach())
        if self.bias is not None:
            float_linear.bias = nn.Parameter(self.bias.detach())
        float_linear.train(self.training)
        return float_linear


def _weight_fake_quant(qconfig, device):
    from ....quantization.qconfig import default_qat_qconfig
    from ....quantization.fake_quantize import FakeQuantize, FakeQuantizeBase
    from ....quantization.observer import ObserverBase
    qconfig = default_qat_qconfig if qconfig is None else qconfig
    module = qconfig.weight()
    if isinstance(module, ObserverBase):
        module = FakeQuantize(observer=module)
    if not isinstance(module, FakeQuantizeBase):
        raise TypeError('QAT weights require a simulated quantization module')
    return qconfig, module.to(device)
