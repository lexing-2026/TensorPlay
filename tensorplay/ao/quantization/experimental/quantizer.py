"""APoT quantizer and global conversion helpers."""

import tensorplay as tp

from .apot_utils import apot_to_float, float_to_apot, quant_dequant_util


class APoTQuantizer:
    """Quantizer for additive power-of-two non-uniform quantization."""

    def __init__(self, alpha, gamma, quantization_levels, level_indices):
        self.alpha = alpha
        self.gamma = gamma
        self.quantization_levels = quantization_levels
        self.level_indices = level_indices

    def quantize(self, tensor2quantize):
        """Quantize a float tensor to APoT integer representation."""
        data = tensor2quantize.detach().clone()
        flat = data.flatten()
        indices = []
        for x in flat.tolist():
            indices.append(float_to_apot(
                x, self.quantization_levels, self.level_indices, self.alpha
            ))
        result = tp.tensor(indices, dtype=tp.int64, device=tensor2quantize.device).reshape(data.shape)
        from .APoT_tensor import TensorAPoT

        return TensorAPoT(self, result)

    def dequantize(self, apot_tensor):
        """Dequantize an APoT tensor back to float."""
        flat = apot_tensor.data.flatten()
        values = [
            apot_to_float(int(x), self.quantization_levels, self.level_indices)
            for x in flat.tolist()
        ]
        return tp.tensor(values, dtype=tp.float32, device=apot_tensor.data.device).reshape(apot_tensor.data.shape)

    def quant_dequant(self, tensor2quantize):
        """Return the float nearest-level approximation of the input."""
        levels = self.quantization_levels.tolist()
        flat = tensor2quantize.detach().clone().flatten()
        values = [quant_dequant_util(x, levels) for x in flat.tolist()]
        return tp.tensor(values, dtype=tp.float32, device=tensor2quantize.device).reshape(tensor2quantize.shape)

    def q_apot_alpha(self) -> float:
        return float(self.alpha)


def quantize_APoT(tensor2quantize, alpha, gamma, quantization_levels, level_indices):
    """Create a quantizer and quantize the input tensor."""
    quantizer = APoTQuantizer(alpha, gamma, quantization_levels, level_indices)
    return quantizer.quantize(tensor2quantize)


def dequantize_APoT(apot_tensor):
    """Dequantize an APoT tensor using its stored quantizer."""
    return apot_tensor.quantizer.dequantize(apot_tensor)


def quant_dequant_APoT(tensor2quantize, alpha, gamma, quantization_levels, level_indices):
    """Create a quantizer and return the float nearest-level approximation."""
    quantizer = APoTQuantizer(alpha, gamma, quantization_levels, level_indices)
    return quantizer.quant_dequant(tensor2quantize)