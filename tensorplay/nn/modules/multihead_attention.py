"""Multi-head attention, mirroring ``torch.nn.MultiheadAttention``.

The parameter/state-dict layout follows torch exactly so torchvision
checkpoints (e.g. ViT weights) load without key translation:

* ``in_proj_weight`` / ``q_proj_weight``, ``k_proj_weight``, ``v_proj_weight``
* ``in_proj_bias``
* ``out_proj`` — a Linear registered under the name torch uses
  (torch calls it NonDynamicallyQuantizableLinear)
* ``bias_k`` / ``bias_v`` when add_bias_kv=True

The functional path delegates to
``tensorplay.nn.functional.multi_head_attention_forward`` which composes the
same math as ATen's native implementation.
"""

from typing import Optional

from tensorplay import Tensor
from tensorplay.nn import functional as F
from tensorplay.nn import init
from tensorplay.nn.parameter import Parameter

from .linear import Linear
from .module import Module


__all__ = ["MultiheadAttention", "NonDynamicallyQuantizableLinear"]


# torch defines this alias so state dict keys stay stable across versions.
class NonDynamicallyQuantizableLinear(Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information from different
    representation subspaces, as described in the paper *Attention Is All You
    Need*.

    Args mirror torch.nn.MultiheadAttention; ``batch_first=True`` is required
    by the torchvision transformer models and is supported here.
    """

    bias_k: Optional[Parameter]
    bias_v: Optional[Parameter]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = True
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn

        factory_kwargs = {}
        if self._qkv_same_embed_dim:
            self.in_proj_weight = Parameter(
                tensorplay.empty(3 * embed_dim, embed_dim))
        else:
            self.q_proj_weight = Parameter(tensorplay.empty(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(tensorplay.empty(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(tensorplay.empty(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(tensorplay.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        # torch registers out_proj as NonDynamicallyQuantizableLinear so that
        # the state dict keys ("out_proj.weight"/"out_proj.bias") are stable.
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(tensorplay.empty(1, 1, embed_dim))
            self.bias_v = Parameter(tensorplay.empty(1, 1, embed_dim))
        else:
            self.bias_k = None
            self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Mirrors torch.nn.MultiheadAttention._reset_parameters
        if self._qkv_same_embed_dim:
            init.xavier_uniform_(self.in_proj_weight)
        else:
            init.xavier_uniform_(self.q_proj_weight)
            init.xavier_uniform_(self.k_proj_weight)
            init.xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            init.constant_(self.in_proj_bias, 0.0)
            init.constant_(self.out_proj.bias, 0.0)
        if self.add_bias_kv:
            init.constant_(self.bias_k, 0.0)
            init.constant_(self.bias_v, 0.0)

    def __setstate__(self, state) -> None:
        # Support loading older torch checkpoints that lack batch_first.
        state.setdefault("batch_first", True)
        super().__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        # batch_first=True inputs are (N, L, E); convert to (L, N, E).
        if query.dim() == 3 and self.batch_first:
            query = query.transpose(1, 0)
            key = key.transpose(1, 0)
            value = value.transpose(1, 0)

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            getattr(self, "in_proj_weight", None),
            self.in_proj_bias if self.in_proj_bias is not None else None,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            self.training,
            key_padding_mask,
            need_weights,
            attn_mask,
            not self._qkv_same_embed_dim,
            getattr(self, "q_proj_weight", None),
            getattr(self, "k_proj_weight", None),
            getattr(self, "v_proj_weight", None),
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if self.batch_first and attn_output.dim() == 3:
            attn_output = attn_output.transpose(1, 0)
        return attn_output, attn_output_weights
