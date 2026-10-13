"""Activation-aware Weight Quantization (AWQ) on native quantized modules.

AWQ searches a per-channel equalization scale for each linear weight so the
quantized ``weight * scale`` preserves the channels that matter most to the
calibration activations.  At runtime the input is divided by the same scale
before the quantized matmul, which the quantized module applies through an
``act_pre_scale`` attribute.

The workflow follows a three-step flow::

    quantize_(model, awq(int4_weight_only(), step="prepare"))
    model(calibration_data)          # collect activations
    quantize_(model, awq(int4_weight_only(), step="convert"))
"""

from dataclasses import dataclass
from typing import Optional

import tensorplay as tp
from tensorplay import Tensor, nn
from tensorplay.nn import functional as F

from tensorplay.ao.quantization.int4_weight_only import (
    WeightOnlyInt4Linear,
    _pack_int4_weight,
    _quantize_int4_groupwise,
)
from tensorplay.ao.quantization.quantize_api import register_quantize_module_handler

__all__ = [
    "AWQConfig",
    "awq",
    "AWQObserver",
    "AWQObservedLinear",
]


@tp.no_grad()
def _get_act_scale(x: tp.Tensor) -> tp.Tensor:
    """Mean absolute value per input channel over the calibration batch."""
    return x.abs().reshape(-1, x.shape[-1]).mean(0)


class AWQObserver(nn.Module):
    """Collects calibration activations and searches the equalization scale."""

    def __init__(self, weight, bias, scale_search_space_size: int = 20, groupsize: int = 128):
        super().__init__()
        self.weight = weight.detach()
        self.bias = bias
        self.inputs = []
        self.scale_options = scale_search_space_size
        self.groupsize = groupsize

    def forward(self, input: tp.Tensor, output: tp.Tensor):
        del output
        self.inputs.append(input.detach().cpu())

    @tp.no_grad()
    def calculate_qparams(self):
        acc = tp.cat(self.inputs, dim=-2).to(self.weight.device)
        x_max = _get_act_scale(acc)

        best_loss = float("inf")
        best_scales = None
        for i in range(self.scale_options):
            ratio = i * 1.0 / self.scale_options
            scales = x_max.pow(ratio).to(self.weight.dtype).clamp(min=1e-4)
            if best_scales is None:
                best_scales = tp.ones_like(scales)
            scales = scales / (scales.max() * scales.min()).sqrt()

            # Quantized weight for this candidate scale.
            wq = _quantize_dequant_int4(self.weight * scales, self.groupsize)
            orig_out = F.linear(acc, self.weight, self.bias)
            q_out = F.linear(acc / scales, wq, self.bias)
            loss = ((orig_out - q_out) ** 2).mean().item()
            if loss < best_loss:
                best_scales = scales
                best_loss = loss
        return best_scales.detach()


class AWQObservedLinear(nn.Linear):
    """Linear that records its input activations for AWQ calibration."""

    def __init__(self, in_features, out_features, act_obs, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_obs = act_obs

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        output = F.linear(input, self.weight, self.bias)
        self.act_obs(input, output)
        return output

    @classmethod
    def from_float(cls, float_linear: nn.Linear, act_obs) -> "AWQObservedLinear":
        mod = cls(
            float_linear.in_features,
            float_linear.out_features,
            act_obs,
            bias=False,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
        )
        mod.weight = float_linear.weight
        mod.bias = float_linear.bias
        return mod


def _quantize_dequant_int4(weight: tp.Tensor, groupsize: int = 128) -> tp.Tensor:
    """Symmetric int4 quantize + dequantize back to float (zero point 0)."""
    codes, scale, zero = _quantize_int4_groupwise(weight, groupsize)
    N, K = weight.shape
    groups = K // groupsize
    wq = (codes.reshape(N, groups, groupsize).to(scale.dtype) - 8.0 - zero.unsqueeze(-1))
    wq = wq * scale.unsqueeze(-1)
    return wq.reshape(N, K)


@dataclass
class AWQConfig:
    """AWQ quantization configuration.

    Attributes:
        base_config: the underlying weight-only config, e.g.
            ``int4_weight_only(groupsize=128)``.
        step: ``"prepare"`` (insert observers), ``"convert"`` (quantize using
            the collected activations), or ``"prepare_for_loading"``.
        scale_search_space_size: number of candidate ratios to search.
    """

    base_config: object
    step: str = "prepare"
    scale_search_space_size: int = 20


# Backwards-compatible alias matching the workflow config name.
awq = AWQConfig


@register_quantize_module_handler(AWQConfig)
def _awq_transform(module, config):
    step = config.step
    if step == "prepare" or step == "prepare_for_loading":
        groupsize = getattr(config.base_config, "groupsize", 128)
        observer = AWQObserver(
            module.weight,
            module.bias,
            config.scale_search_space_size,
            groupsize,
        )
        observed = AWQObservedLinear.from_float(module, observer)
        if step == "prepare_for_loading":
            example = tp.randn(
                (1, module.weight.shape[1]),
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            observed(example)
        return observed
    if step == "convert":
        if not isinstance(module, AWQObservedLinear):
            return module
        groupsize = getattr(config.base_config, "groupsize", 128)
        eq_scale = module.act_obs.calculate_qparams()
        packed, scale_zero = _pack_int4_weight(module.weight * eq_scale, groupsize)
        linear = WeightOnlyInt4Linear(
            module.in_features,
            module.out_features,
            groupsize=groupsize,
            bias=module.bias is not None,
        )
        linear.weight = packed
        linear.scales_and_zeros = scale_zero
        linear.act_pre_scale = 1.0 / eq_scale
        if module.bias is not None:
            linear.bias = module.bias
        return linear
    raise ValueError(f"awq: unknown step {step!r}")