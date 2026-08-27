# Ported from upstream vision ops/stochastic_depth.py; torch -> tensorplay.

import tensorplay
from tensorplay import nn, Tensor

__all__ = ["StochasticDepth", "stochastic_depth"]


def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    """
    Implements the Stochastic Depth from ``"Deep Networks with Stochastic Depth"``

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary shapes with a first
            dimension for batch.
        p (float): probability of the input to be zeroed.
        mode (str): ``"row"``
            randomly zeroes the entire row, ``"dot"``
            randomly zeroes zeroes individual elements.
        training (bool): apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]): The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got a value of {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0], 1]
    else:
        size = [input.shape[0]]
    noise = tensorplay.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    input.div(survival_rate).mul(noise)
    return input


class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}, mode={self.mode}"
