"""Quantizable attention module."""

import tensorplay as tp
from tensorplay import nn

__all__ = ["MultiheadAttention"]


class MultiheadAttention(nn.MultiheadAttention):
    """A float multihead attention that participates in quantization.

    The projection layers are kept as explicit submodules so observers can
    be attached to each of them during calibration.
    """

    _FLOAT_MODULE = nn.MultiheadAttention

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 batch_first=False, device=None, dtype=None):
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            device=device,
            dtype=dtype,
        )
        factory = {"device": device, "dtype": dtype}
        self.linear_Q = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory)
        self.linear_K = nn.Linear(self.kdim, self.embed_dim, bias=bias, **factory)
        self.linear_V = nn.Linear(self.vdim, self.embed_dim, bias=bias, **factory)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory)

    def _get_name(self):
        return "QuantizableMultiheadAttention"

    @classmethod
    def from_float(cls, other):
        if type(other) is not cls._FLOAT_MODULE:
            raise AssertionError(f"Expected type {cls._FLOAT_MODULE}, got {type(other)}")
        if not hasattr(other, "qconfig"):
            raise AssertionError("The float module must have 'qconfig'")
        observed = cls(
            other.embed_dim,
            other.num_heads,
            other.dropout,
            other.in_proj_bias is not None,
            other.bias_k is not None,
            other.add_zero_attn,
            other.kdim,
            other.vdim,
            other.batch_first,
        )
        observed.qconfig = other.qconfig
        observed.out_proj.weight = other.out_proj.weight
        observed.out_proj.bias = other.out_proj.bias
        if other._qkv_same_embed_dim:
            bias = other.in_proj_bias
            w = other.in_proj_weight
            observed.linear_Q.weight = nn.Parameter(w[: other.embed_dim].detach().clone())
            observed.linear_K.weight = nn.Parameter(w[other.embed_dim : 2 * other.embed_dim].detach().clone())
            observed.linear_V.weight = nn.Parameter(w[2 * other.embed_dim:].detach().clone())
            if bias is not None:
                observed.linear_Q.bias = nn.Parameter(bias[: other.embed_dim].detach().clone())
                observed.linear_K.bias = nn.Parameter(bias[other.embed_dim : 2 * other.embed_dim].detach().clone())
                observed.linear_V.bias = nn.Parameter(bias[2 * other.embed_dim:].detach().clone())
        else:
            observed.linear_Q.weight = nn.Parameter(other.q_proj_weight.detach().clone())
            observed.linear_K.weight = nn.Parameter(other.k_proj_weight.detach().clone())
            observed.linear_V.weight = nn.Parameter(other.v_proj_weight.detach().clone())
            if other.in_proj_bias is not None:
                observed.linear_Q.bias = nn.Parameter(other.in_proj_bias[: other.embed_dim].detach().clone())
                observed.linear_K.bias = nn.Parameter(other.in_proj_bias[other.embed_dim : 2 * other.embed_dim].detach().clone())
                observed.linear_V.bias = nn.Parameter(other.in_proj_bias[2 * other.embed_dim:].detach().clone())
        observed.eval()
        return observed
