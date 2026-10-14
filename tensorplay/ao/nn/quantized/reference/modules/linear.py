"""Reference quantized linear module with float weights."""

from typing import Any

import tensorplay as tp
from tensorplay import nn
from tensorplay.nn import functional as F

from .utils import ReferenceQuantizedModule

__all__ = ["Linear"]


class Linear(nn.Linear, ReferenceQuantizedModule):
    """A reference quantized linear module.

    Keeps a floating point weight and fake-quantizes it in forward, so the
    graph can later lower the quant-dequant pattern to a native quantized op.
    """

    _IS_REFERENCE = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias_: bool = True,
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias_, device, dtype)
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self) -> str:
        return "QuantizedLinear(Reference)"

    def forward(self, x: tp.Tensor) -> tp.Tensor:
        weight_quant_dequant = self.get_weight()
        return F.linear(x, weight_quant_dequant, self.bias)

    @classmethod
    def from_float(cls, float_linear: nn.Linear, weight_qparams: dict[str, Any]):
        qref_linear = Linear(
            float_linear.in_features,
            float_linear.out_features,
            float_linear.bias is not None,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
            weight_qparams=weight_qparams,
        )
        qref_linear.weight = nn.Parameter(float_linear.weight.detach())
        if float_linear.bias is not None:
            qref_linear.bias = nn.Parameter(float_linear.bias.detach())
        return qref_linear