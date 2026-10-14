"""Non-uniform (APoT) observer."""

import itertools

import tensorplay as tp
from tensorplay.ao.quantization.observer import ObserverBase

from .apot_utils import apot_to_float, float_to_apot


class APoTObserver(ObserverBase):
    """Observer that records min/max and computes APoT quantization levels."""

    def __init__(self, b, k, dtype=tp.quint8) -> None:
        super().__init__(dtype)
        self.b = b
        self.k = k
        self.min_val = tp.tensor([])
        self.max_val = tp.tensor([])

    def calculate_qparams(self, signed):  # type: ignore[override]
        return self._calculate_qparams(signed, self.min_val, self.max_val)

    def _calculate_qparams(self, signed: bool, min_val=None, max_val=None):
        if min_val is not None:
            self.min_val = min_val
        if max_val is not None:
            self.max_val = max_val

        alpha = tp.max(-self.min_val, self.max_val)

        if not self.k or self.k == 0:
            raise AssertionError(f"k must be a non-zero integer, got k={self.k}")
        if self.b % self.k != 0:
            raise AssertionError(f"b must be divisible by k, got b={self.b}, k={self.k}")

        self.n = self.b // self.k
        p_all = []

        for i in range(self.n):
            p_curr = tp.tensor([0])
            for j in range((2**self.k - 2) + 1):
                curr_ele = 2 ** (-(i + j * self.n))
                p_curr = tp.cat((p_curr, tp.tensor([curr_ele])))
                if signed:
                    p_curr = tp.cat((p_curr, tp.tensor([-curr_ele])))
            if signed:
                sorted, _indices = tp.sort(p_curr, descending=True)
                p_all.append(sorted)
            else:
                p_all.append(p_curr)

        p_sum = 0.0
        for tens in p_all:
            if signed:
                p_sum += float(tens[0])
            else:
                p_sum += float(tens[1])
        gamma = alpha / p_sum

        cartesian_product = list(itertools.product(*p_all))
        quantization_levels_list = []
        for row in cartesian_product:
            quantization_levels_list.append(float(sum(row)))
        quantization_levels_gamma = [float(gamma) * ele for ele in quantization_levels_list]
        quantization_levels = tp.tensor(quantization_levels_gamma)
        quantization_levels, level_indices = quantization_levels.sort()
        return (alpha, gamma, quantization_levels, level_indices)

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        min_val, max_val = tp.aminmax(x)
        if self.min_val.numel():
            min_val = tp.min(min_val, self.min_val)
        if self.max_val.numel():
            max_val = tp.max(max_val, self.max_val)
        self.min_val = min_val
        self.max_val = max_val
        return x_orig

    def quant_levels_visualization(self, signed=False):
        alpha, _gamma, quantization_levels, level_indices = self.calculate_qparams(signed)
        xs = [float(x) / 1000.0 for x in range(1000)]
        ys = [
            apot_to_float(
                float_to_apot(x, quantization_levels, level_indices, alpha),
                quantization_levels,
                level_indices,
            )
            for x in xs
        ]
        return xs, ys