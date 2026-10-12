"""Rowwise integer embedding tables with persistent logical weights."""
import tensorplay as tp
from tensorplay import nn, _C


class EmbeddingPackedParams(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer('logical_weight', weight)
        self.register_buffer('packed', None, persistent=False)
        self.set_weight(weight)

    def set_weight(self, weight):
        self.logical_weight = weight.contiguous()
        self.packed = _C._quantized_embedding_weight_pack(self.logical_weight)

    def _weight(self):
        return self.logical_weight

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.set_weight(self.logical_weight)
        return self

    def __getstate__(self):
        state = super().__getstate__()
        state['_buffers'] = state['_buffers'].copy()
        state['_buffers']['packed'] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.set_weight(self.logical_weight)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + 'logical_weight'
        if key in state_dict:
            self.set_weight(state_dict[key].to(self.logical_weight.device))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        self.set_weight(self.logical_weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                 norm_type=2., scale_grad_by_freq=False, sparse=False, _weight=None, dtype=tp.quint8):
        super().__init__()
        if dtype not in (tp.quint8,tp.quint4x2):
            raise ValueError('Embedding logical weights require unsigned 8-bit or 4-bit storage')
        self.num_embeddings, self.embedding_dim = int(num_embeddings), int(embedding_dim)
        self.dtype = dtype
        if _weight is None:
            _weight = _C.quantize_per_channel(
                tp.zeros((self.num_embeddings, self.embedding_dim)),
                tp.ones(self.num_embeddings), tp.zeros(self.num_embeddings), 0,dtype)
        if tuple(_weight.shape) != (self.num_embeddings, self.embedding_dim):
            raise ValueError('Embedding weight dimensions do not match the module')
        if _weight.dtype!=dtype:
            raise ValueError('Embedding weight dtype must match the module dtype')
        self._packed_params = EmbeddingPackedParams(_weight)

    def weight(self):
        return self._packed_params.logical_weight

    def set_weight(self, weight):
        if tuple(weight.shape) != (self.num_embeddings, self.embedding_dim):
            raise ValueError('Embedding weight dimensions do not match the module')
        self._packed_params.set_weight(weight)
        self.dtype=weight.dtype

    def forward(self, indices):
        return _C._quantized_embedding(self._packed_params.packed, indices,
                                      4 if self.weight().dtype==tp.quint4x2 else 8)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        from tensorplay.ao.quantization.qconfig import float_qparams_weight_only_qconfig
        from tensorplay.ao.quantization.fake_quantize import FakeQuantizeBase
        observer = getattr(mod, 'weight_fake_quant', None)
        if observer is None:
            config = getattr(mod, 'qconfig', None) or float_qparams_weight_only_qconfig
            observer = config.weight()
        if observer.dtype not in (tp.quint8,tp.quint4x2) or observer.qscheme != tp.per_channel_affine_float_qparams or observer.ch_axis != 0:
            raise ValueError('Embedding conversion requires unsigned row weights with floating affine parameters on axis zero')
        observer.to(mod.weight.device)
        if not use_precomputed_fake_quant:
            observer(mod.weight)
        if isinstance(observer, FakeQuantizeBase):
            scales, zps = observer.scale, observer.zero_point
        else:
            scales, zps = observer.calculate_qparams()
        weight = _C.quantize_per_channel(mod.weight.detach().float(), scales.float(), zps.float(), 0, observer.dtype)
        if issubclass(cls, EmbeddingBag):
            result = cls(mod.num_embeddings, mod.embedding_dim, mode=mod.mode,
                         include_last_offset=mod.include_last_offset, _weight=weight,dtype=observer.dtype)
        else:
            result = cls(mod.num_embeddings, mod.embedding_dim, _weight=weight,dtype=observer.dtype)
        return result

    def extra_repr(self):
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, dtype={self.dtype}'


class EmbeddingBag(Embedding):
    def __init__(self, num_embeddings, embedding_dim, max_norm=None, norm_type=2.,
                 scale_grad_by_freq=False, mode='sum', sparse=False, _weight=None,
                 include_last_offset=False, dtype=tp.quint8, padding_idx=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, sparse, _weight, dtype)
        self.mode = mode
        self.include_last_offset = bool(include_last_offset)
        self.pruned_weights = False
        self.register_buffer('compressed_indices_mapping', None)

    def forward(self, indices, offsets=None, per_sample_weights=None, compressed_indices_mapping=None):
        mapping = self.compressed_indices_mapping if compressed_indices_mapping is None else compressed_indices_mapping
        return _C._quantized_embedding_bag(self._packed_params.packed, indices,
            4 if self.weight().dtype==tp.quint4x2 else 8, offsets,
            per_sample_weights, mapping, self.include_last_offset, self.pruned_weights)
