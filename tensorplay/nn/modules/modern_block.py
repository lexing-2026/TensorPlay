# mypy: allow-untyped-defs
"""Building blocks of modern decoder stacks.

``SwiGLUMLP`` is the gated feed-forward used by current architectures: two
parallel projections (gate/up) with a SiLU nonlinearity between them, then a
down projection. It buys per-channel selectivity at the same parameter count
as a plain ``2x`` MLP by splitting the hidden matrix.

``TransformerBlock`` is the pre-norm residual cell: ``x + mixer(norm(x))``
followed by ``x + mlp(norm(x))``. The mixer is injected, so attention and
linear-attention layers are interchangeable inside the same block.
"""

from __future__ import annotations

from typing import Optional

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import Tensor

from .linear import Linear
from .module import Module
from .normalization import RMSNorm

__all__ = ["SwiGLUMLP", "TransformerBlock"]


class SwiGLUMLP(Module):
    """Gated feed-forward: ``down(silu(gate(x)) * up(x))``.

    Args:
        hidden_size: model width of the input and output.
        intermediate_size: width of the gate/up projections. Defaults to
            ``4 * hidden_size * 2 // 3``, the SwiGLU parameter-matched width.
        bias: whether the projections carry a bias.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size * 2 // 3
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=bias, **factory_kwargs)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=bias, **factory_kwargs)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=bias, **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(Module):
    """Pre-norm residual block around an injected mixer.

    Args:
        hidden_size: model width.
        mixer: token mixer with the signature
            ``mixer(x, position_ids=None, past_key_values=None)``.
        mlp_intermediate_size: feed-forward width; see :class:`SwiGLUMLP`.
        rms_norm_eps: epsilon of both pre-norms.
    """

    def __init__(
        self,
        hidden_size: int,
        mixer: Module,
        mlp_intermediate_size: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.mixer = mixer
        self.mlp = SwiGLUMLP(
            hidden_size, intermediate_size=mlp_intermediate_size, **factory_kwargs
        )
        self.input_norm = RMSNorm(hidden_size, eps=rms_norm_eps, **factory_kwargs)
        self.post_norm = RMSNorm(hidden_size, eps=rms_norm_eps, **factory_kwargs)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Optional[Tensor] = None,
        past_key_values=None,
    ) -> Tensor:
        hidden_states = hidden_states + self.mixer(
            self.input_norm(hidden_states),
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        hidden_states = hidden_states + self.mlp(self.post_norm(hidden_states))
        return hidden_states
