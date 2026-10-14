"""Adaptive rounding fake quantizer."""

from typing import Any

import tensorplay as tp
from tensorplay import nn

from ..fake_quantize import FakeQuantize
from ..observer import MinMaxObserver
from ..utils import is_per_tensor


def _is_symmetric_quant(qscheme):
    return qscheme in (tp.per_tensor_symmetric, tp.per_channel_symmetric)


class AdaroundFakeQuantizer(FakeQuantize):
    """Fake quantizer with a learnable soft-rounding parameter per weight.

    Implements the adaptive rounding technique for weight quantization.
    Only symmetric quantization is supported.
    """

    def __init__(
        self,
        observer: type = MinMaxObserver,
        qscheme=tp.per_tensor_symmetric,
        quant_min: int = -128,
        quant_max: int = 127,
        ch_axis: int = 0,
        **observer_kwargs: Any,
    ) -> None:
        super().__init__(
            observer=observer,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_dynamic=False,
            **observer_kwargs,
        )
        if quant_min is not None and quant_max is not None and quant_min > quant_max:
            raise AssertionError(
                "quant_min must be less than or equal to quant_max, "
                f"got quant_min:{quant_min}, quant_max:{quant_max}"
            )
        self.qscheme = qscheme
        self.is_per_tensor = is_per_tensor(qscheme)
        self.is_symmetric = _is_symmetric_quant(qscheme)
        if not self.is_symmetric:
            raise AssertionError("Only symmetric quantization is supported")
        self.ch_axis = ch_axis

        self.scale = tp.tensor([], requires_grad=False)
        self.zero_point = tp.tensor([], requires_grad=False)
        self.V = nn.Parameter(tp.tensor([]), requires_grad=True)
        self.zeta = tp.tensor(1.1, requires_grad=False)
        self.gamma = tp.tensor(-0.1, requires_grad=False)
        self.sigmoid = nn.Sigmoid()
        self.use_soft_rounding = True

    def calculate_qparams(self):  # type: ignore[override]
        return self.scale, self.zero_point

    def extra_repr(self) -> str:
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, "
            f"dtype={self.dtype}, qscheme={self.qscheme}, ch_axis={self.ch_axis}, "
            f"scale={self.scale}, zero_point={self.zero_point}"
        )

    def enable_weight_fake_quant(self) -> None:
        self.fake_quant_enabled[0] = 1

    def get_rectified_sigmoid_func(self) -> tp.Tensor:
        if self.use_soft_rounding:
            return tp.clamp(
                self.sigmoid(self.V) * (self.zeta - self.gamma) + self.gamma,
                min=0,
                max=1,
            )
        return (self.V >= 0).int()

    def update_scale(self, X, _scale, _zero_point) -> None:
        if self.scale.numel() == 0:
            self.scale.data = _scale.to(X.device)
            self.zero_point = _zero_point.to(X.device)
        else:
            self.scale.data = _scale
            if not self.is_symmetric:
                self.zero_point = _zero_point
            else:
                self.zero_point = tp.zeros_like(_zero_point)
            for i in range(X.dim()):
                if i == self.ch_axis:
                    continue
                self.zero_point = self.zero_point.unsqueeze(i)
        X_q = X / self.scale
        X_q_floor = tp.floor(X_q)
        residual = X_q - X_q_floor
        if not bool(tp.all(tp.ge(residual, 0))):
            raise AssertionError("residual should be non-negative in [0, 1)")
        V_init = -tp.log((self.zeta - self.gamma) / (residual - self.gamma) - 1)
        self.V.data = V_init

    def forward(self, X):
        if int(self.observer_enabled[0]) == 1:
            X_detached = X.detach()
            self.activation_post_process(X_detached)
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)
            if not self.is_per_tensor:
                for i in range(X.dim()):
                    if i == self.ch_axis:
                        continue
                    _scale = _scale.unsqueeze(i)
                    _zero_point = _zero_point.unsqueeze(i)
            self.update_scale(X_detached, _scale, _zero_point)

        if int(self.fake_quant_enabled[0]) == 1:
            h_v = self.get_rectified_sigmoid_func()
            X_q = X / self.scale
            X_q_floor = tp.floor(X_q) + self.zero_point
            X_q_dq = (
                tp.clamp(X_q_floor + h_v, min=self.quant_min, max=self.quant_max)
                - self.zero_point
            ) * self.scale
            return X_q_dq
        return X