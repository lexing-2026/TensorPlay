"""APoT quantized linear module."""

import tensorplay as tp
from tensorplay import nn

from .observer import APoTObserver
from .quantizer import quantize_APoT


class LinearAPoT(nn.Module):
    """A linear module whose weight is stored in APoT representation."""

    def __init__(self, weight2quantize: tp.Tensor, b: int, k: int):
        if weight2quantize.dim() != 2:
            raise AssertionError(f"weight2quantize must be a 2-D tensor, got dim={weight2quantize.dim()}")
        if b % k != 0:
            raise AssertionError(f"b must be divisible by k, got b={b}, k={k}")
        super().__init__()
        self.b = b
        self.k = k
        self.n = self.b // self.k

        observer = APoTObserver(b=self.b, k=self.k)
        observer(weight2quantize)
        (
            self.alpha,
            self.gamma,
            self.quantization_levels,
            self.level_indices,
        ) = observer.calculate_qparams(signed=False)

        quantized_weight = quantize_APoT(
            weight2quantize,
            self.alpha,
            self.gamma,
            self.quantization_levels,
            self.level_indices,
        )
        self.weight = quantized_weight.data
        self.weight_transposed = tp.transpose(self.weight, 0, 1)

    def forward(self, activation: tp.Tensor) -> tp.Tensor:
        """Multiply APoT weight with uniformly quantized activation."""
        if activation.dim() != 2:
            raise AssertionError(f"activation must be a 2-D tensor, got dim={activation.dim()}")

        levels = self.quantization_levels.tolist()
        indices = self.level_indices.tolist()
        weight_rows = self.weight_transposed.size(0)
        weight_cols = self.weight_transposed.size(1)
        result = tp.zeros(activation.size(0), weight_cols, dtype=tp.float32)
        for i in range(activation.size(0)):
            for j in range(weight_cols):
                acc = 0.0
                for k in range(weight_rows):
                    apot_idx = int(self.weight_transposed[k][j])
                    weight_val = levels[indices.index(apot_idx)]
                    acc += weight_val * float(activation[i][k])
                result[i][j] = acc
        return result

    @classmethod
    def from_reference(cls, ref_qlinear, alpha, gamma, quantization_levels, level_indices):
        raise NotImplementedError