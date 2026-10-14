"""Adaptive rounding loss functions."""

import tensorplay as tp
from tensorplay import nn
from tensorplay.nn import functional as F

ADAROUND_ZETA: float = 1.1
ADAROUND_GAMMA: float = -0.1


class AdaptiveRoundingLoss(nn.Module):
    """Adaptive rounding loss: rounding regularization plus reconstruction."""

    def __init__(
        self,
        max_iter: int,
        warm_start: float = 0.2,
        beta_range: tuple[int, int] = (20, 2),
        reg_param: float = 0.001,
    ) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.beta_range = beta_range
        self.reg_param = reg_param

    def rounding_regularization(self, h_v):
        """Equation (24) from the adaptive rounding paper."""
        return (
            self.reg_param
            * (1 - tp.abs(2 * h_v - 1)).pow(ADAROUND_GAMMA).sum()
        )

    def reconstruction_loss(self, x, x_hat):
        """Mean squared reconstruction error between input and approximation."""
        return F.mse_loss(x_hat, x)

    def get_beta(self, iter_count):
        """Warm-start beta schedule for the rectified sigmoid."""
        if iter_count / self.max_iter <= self.warm_start:
            return self.beta_range[0]
        step = (self.beta_range[0] - self.beta_range[1]) / (
            self.max_iter * (1 - self.warm_start)
        )
        return max(self.beta_range[1], self.beta_range[0] - (iter_count - self.warm_start * self.max_iter) * step)

    def forward(self, x, x_hat, h_v, iter_count):
        round_loss = self.rounding_regularization(h_v)
        recon_loss = self.reconstruction_loss(x, x_hat)
        beta = self.get_beta(iter_count)
        return recon_loss + beta * round_loss