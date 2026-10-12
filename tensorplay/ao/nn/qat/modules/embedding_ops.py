"""Embedding training with simulated rowwise affine weight grids."""

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F
from tensorplay import Tensor
__all__ = ['Embedding', 'EmbeddingBag']

class Embedding(nn.Embedding):
    _FLOAT_MODULE = nn.Embedding

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None, qconfig=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, **factory_kwargs)
        if not qconfig:
            raise AssertionError('qconfig must be provided for QAT module')
        weight_qscheme = qconfig.weight().qscheme
        if weight_qscheme != tensorplay.per_channel_affine_float_qparams:
            raise AssertionError('Embedding weights requires a qscheme of per_channel_affine_float_qparams Got ' + str(weight_qscheme))
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input):
        return F.embedding(input, self.weight_fake_quant(self.weight), self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError(' qat.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__)
        if not hasattr(mod, 'qconfig'):
            raise AssertionError('Input float module must have qconfig defined')
        if not mod.qconfig:
            raise AssertionError('Input float module must have a valid qconfig')
        weight_qscheme = mod.qconfig.weight().qscheme
        if weight_qscheme != tensorplay.per_channel_affine_float_qparams:
            raise AssertionError('Embedding weights requires a qscheme of per_channel_affine_float_qparams Got ' + str(weight_qscheme))
        qconfig = mod.qconfig
        qat_embedding_bag = cls(mod.num_embeddings, mod.embedding_dim, mod.padding_idx, mod.max_norm, mod.norm_type, mod.scale_grad_by_freq, mod.sparse, mod.weight, qconfig=qconfig, device=mod.weight.device, dtype=mod.weight.dtype)
        qat_embedding_bag.train(mod.training)
        return qat_embedding_bag

    def to_float(self):
        embedding_bag = tensorplay.nn.Embedding(self.num_embeddings, self.embedding_dim, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse, None, device=self.weight.device, dtype=self.weight.dtype)
        embedding_bag.weight = tensorplay.nn.Parameter(self.weight.detach())
        embedding_bag.train(self.training)
        return embedding_bag

class EmbeddingBag(nn.EmbeddingBag):
    _FLOAT_MODULE = nn.EmbeddingBag

    def __init__(self, num_embeddings, embedding_dim, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, mode='mean', sparse=False, _weight=None, include_last_offset=False, padding_idx=None, qconfig=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_embeddings, embedding_dim, max_norm, norm_type, scale_grad_by_freq, mode, sparse, _weight, include_last_offset, padding_idx, **factory_kwargs)
        if not qconfig:
            raise AssertionError('qconfig must be provided for QAT module')
        weight_qscheme = qconfig.weight().qscheme
        if weight_qscheme != tensorplay.per_channel_affine_float_qparams:
            raise AssertionError('Embedding Bag weights requires a qscheme of per_channel_affine_float_qparams Got ' + str(weight_qscheme))
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input, offsets=None, per_sample_weights=None):
        return F.embedding_bag(input, self.weight_fake_quant(self.weight), offsets, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse, per_sample_weights, self.include_last_offset, self.padding_idx)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError(' qat.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__)
        if not hasattr(mod, 'qconfig'):
            raise AssertionError('Input float module must have qconfig defined')
        if not mod.qconfig:
            raise AssertionError('Input float module must have a valid qconfig')
        weight_qscheme = mod.qconfig.weight().qscheme
        if weight_qscheme != tensorplay.per_channel_affine_float_qparams:
            raise AssertionError('Embedding Bag weights requires a qscheme of per_channel_affine_float_qparams Got ' + str(weight_qscheme))
        qconfig = mod.qconfig
        qat_embedding_bag = cls(mod.num_embeddings, mod.embedding_dim, mod.max_norm, mod.norm_type, mod.scale_grad_by_freq, mod.mode, mod.sparse, mod.weight, mod.include_last_offset, mod.padding_idx, qconfig=qconfig, device=mod.weight.device, dtype=mod.weight.dtype)
        qat_embedding_bag.train(mod.training)
        return qat_embedding_bag

    def to_float(self):
        embedding_bag = tensorplay.nn.EmbeddingBag(self.num_embeddings, self.embedding_dim, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse, None, self.include_last_offset, self.padding_idx, device=self.weight.device, dtype=self.weight.dtype)
        embedding_bag.weight = tensorplay.nn.Parameter(self.weight.detach())
        embedding_bag.train(self.training)
        return embedding_bag
