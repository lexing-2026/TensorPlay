"""Reference quantized embedding modules with float weights."""

from typing import Any

import tensorplay as tp
from tensorplay import nn
from tensorplay.nn import functional as F

from .utils import ReferenceQuantizedModule

__all__ = ["Embedding", "EmbeddingBag"]


class Embedding(nn.Embedding, ReferenceQuantizedModule):
    """A reference quantized embedding that fake-quantizes its weight."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: tp.Tensor | None = None,
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self):
        return "QuantizedEmbedding(Reference)"

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        weight_quant_dequant = self.get_weight()
        return F.embedding(
            input,
            weight_quant_dequant,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    @classmethod
    def from_float(cls, mod, weight_qparams):
        return cls(
            mod.num_embeddings,
            mod.embedding_dim,
            mod.padding_idx,
            mod.max_norm,
            mod.norm_type,
            mod.scale_grad_by_freq,
            mod.sparse,
            mod.weight,
            mod.weight.device,
            mod.weight.dtype,
            weight_qparams,
        )


class EmbeddingBag(nn.EmbeddingBag, ReferenceQuantizedModule):
    """A reference quantized embedding bag that fake-quantizes its weight."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        _weight: tp.Tensor | None = None,
        include_last_offset: bool = False,
        padding_idx: int | None = None,
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            mode,
            sparse,
            _weight,
            include_last_offset,
            padding_idx,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self):
        return "QuantizedEmbeddingBag(Reference)"

    def forward(self, input: tp.Tensor, offsets=None, per_sample_weights=None) -> tp.Tensor:
        weight_quant_dequant = self.get_weight()
        return F.embedding_bag(
            input,
            weight_quant_dequant,
            offsets,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            per_sample_weights,
            self.include_last_offset,
            self.padding_idx,
        )

    @classmethod
    def from_float(cls, mod, weight_qparams):
        return cls(
            mod.num_embeddings,
            mod.embedding_dim,
            mod.max_norm,
            mod.norm_type,
            mod.scale_grad_by_freq,
            mod.mode,
            mod.sparse,
            mod.weight,
            mod.include_last_offset,
            mod.padding_idx,
            mod.weight.device,
            mod.weight.dtype,
            weight_qparams,
        )