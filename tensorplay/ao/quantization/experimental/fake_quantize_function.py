"""Autograd function for APoT fake quantization."""

import tensorplay as tp
from tensorplay.autograd.function import Function

from .apot_utils import float_to_apot


class _APoTFakeQuantize(Function):
    @staticmethod
    def forward(ctx, x, alpha, gamma, quantization_levels, level_indices):
        flat = x.detach().clone().flatten()
        indices = [
            float_to_apot(v, quantization_levels, level_indices, alpha)
            for v in flat.tolist()
        ]
        return tp.tensor(indices, dtype=x.dtype, device=x.device).reshape(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass the gradient through unchanged.
        return grad_output, None, None, None, None


fake_quantize_function = _APoTFakeQuantize.apply