"""APoT fake quantize module."""

from collections.abc import Callable
from typing import Any

import tensorplay as tp

from .fake_quantize_function import fake_quantize_function
from .observer import APoTObserver
from ..fake_quantize import FakeQuantizeBase


class APoTFakeQuantize(FakeQuantizeBase):
    """Fake quantize using additive power-of-two non-uniform levels."""

    def __init__(self, observer: Callable = APoTObserver, **observer_kwargs: Any):
        super().__init__()
        self.activation_post_process = observer(**observer_kwargs)
        self.dtype = self.activation_post_process.dtype
        self.alpha = None
        self.gamma = None
        self.quantization_levels = None
        self.level_indices = None

    def calculate_qparams(self, signed: bool = False):
        return self.activation_post_process.calculate_qparams(signed=signed)

    def forward(self, X):  # type: ignore[override]
        if int(self.observer_enabled[0]) == 1:
            self.activation_post_process.forward(X)
            result = self.activation_post_process.calculate_qparams(signed=False)
            self.alpha = result[0]
            self.gamma = result[1]
            self.quantization_levels = result[2]
            self.level_indices = result[3]

        if int(self.fake_quant_enabled[0]) == 1:
            if self.alpha is None or self.gamma is None or self.quantization_levels is None or self.level_indices is None:
                raise AssertionError("Must set qparams for fake quant")
            X = fake_quantize_function(
                X, self.alpha, self.gamma, self.quantization_levels, self.level_indices
            )
        return X