"""Dynamically quantized recurrent modules using native sequence execution."""
import tensorplay as tp
from tensorplay import nn, _C
from tensorplay.nn.utils.rnn import PackedSequence
from ...packed import quantize_module_weight


def _initial_weight(shape, dtype):
    weight = tp.zeros(shape)
    if dtype == tp.float16:
        return _C._quantized_fp16_prepack(weight)
    if dtype != tp.qint8:
        raise ValueError('Recurrent weights require signed integer or half storage')
    return tp.quantize_per_tensor(weight, 1., 0, tp.qint8)


def _converted_weight(module, name, dtype, precomputed):
    weight = getattr(module, name).detach()
    return (_C._quantized_fp16_prepack(weight) if dtype == tp.float16 else
            quantize_module_weight(module, precomputed, weight=weight))


class RNNBase(nn.Module):
    _MODE = 1

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0., bidirectional=False, dtype=tp.qint8):
        super().__init__()
        self.input_size, self.hidden_size = int(input_size), int(hidden_size)
        self.num_layers, self.bias = int(num_layers), bool(bias)
        self.batch_first, self.bidirectional = bool(batch_first), bool(bidirectional)
        self.dropout, self.dtype = float(dropout), dtype
        self.reduce_range = True
        if self.input_size <= 0 or self.hidden_size <= 0 or self.num_layers <= 0 or not 0 <= self.dropout <= 1:
            raise ValueError('Invalid recurrent dimensions or dropout probability')
        self._weight_names, self._bias_names = [], []
        directions = 2 if bidirectional else 1
        gates = 4 if self._MODE == 2 else 3 if self._MODE == 3 else 1
        for layer in range(self.num_layers):
            for direction in range(directions):
                suffix = f'_l{layer}' + ('_reverse' if direction else '')
                size = self.input_size if layer == 0 else self.hidden_size * directions
                for prefix, width in (('weight_ih', size), ('weight_hh', self.hidden_size)):
                    name = prefix + suffix
                    self.register_buffer(name, _initial_weight((gates * self.hidden_size, width), dtype))
                    self._weight_names.append(name)
                if bias:
                    for prefix in ('bias_ih', 'bias_hh'):
                        name = prefix + suffix
                        self.register_buffer(name, tp.zeros(gates * self.hidden_size))
                        self._bias_names.append(name)

    def forward(self, input, hx=None):
        packed = isinstance(input, PackedSequence)
        data = input.data if packed else input
        unbatched = not packed and data.dim() == 2
        batch_first = self.batch_first and not unbatched
        if unbatched:
            data = data.unsqueeze(1)
        if data.dim() != (2 if packed else 3) or data.size(-1) != self.input_size:
            raise ValueError('Recurrent input dimensions do not match the module')
        batch = int(input.batch_sizes[0]) if packed else data.size(0 if batch_first else 1)
        shape = (self.num_layers * (2 if self.bidirectional else 1), batch, self.hidden_size)
        if hx is None:
            h = tp.zeros(shape, dtype=tp.float32, device=data.device)
            c = tp.zeros_like(h) if self._MODE == 2 else None
        else:
            h, c = hx if self._MODE == 2 else (hx, None)
            if unbatched:
                h = h.unsqueeze(1)
                if c is not None:
                    c = c.unsqueeze(1)
            if packed and input.sorted_indices is not None:
                h = h.index_select(1, input.sorted_indices)
                if c is not None:
                    c = c.index_select(1, input.sorted_indices)
        output, h, c = _C._quantized_recurrent(data, h, c,
            [getattr(self, name) for name in self._weight_names],
            [getattr(self, name) for name in self._bias_names], self.num_layers, self.dropout,
            self.training, self.bidirectional, batch_first, self._MODE, self.reduce_range,
            input.batch_sizes if packed else None)
        if packed:
            output = PackedSequence(output, input.batch_sizes, input.sorted_indices, input.unsorted_indices)
            if input.unsorted_indices is not None:
                h = h.index_select(1, input.unsorted_indices)
                if self._MODE == 2:
                    c = c.index_select(1, input.unsorted_indices)
        elif unbatched:
            output, h = output.squeeze(1), h.squeeze(1)
            if self._MODE == 2:
                c = c.squeeze(1)
        return output, (h, c) if self._MODE == 2 else h

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        config = getattr(mod, 'qconfig', None)
        dtype = config.weight().dtype if config is not None else tp.qint8
        options = dict(num_layers=mod.num_layers, bias=mod.bias, batch_first=mod.batch_first,
                       dropout=mod.dropout, bidirectional=mod.bidirectional, dtype=dtype)
        if cls is RNN:
            options['nonlinearity'] = mod.nonlinearity
        result = cls(mod.input_size, mod.hidden_size, **options)
        for name in result._weight_names:
            setattr(result, name, _converted_weight(mod, name, dtype, use_precomputed_fake_quant))
        for name in result._bias_names:
            setattr(result, name, getattr(mod, name).detach().float())
        result.train(mod.training)
        return result

    def extra_repr(self):
        return f'{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, bidirectional={self.bidirectional}, dtype={self.dtype}'


class LSTM(RNNBase):
    _MODE = 2


class GRU(RNNBase):
    _MODE = 3


class RNN(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', **kwargs):
        if nonlinearity not in ('tanh', 'relu'):
            raise ValueError('Recurrent nonlinearity must be tanh or relu')
        self._MODE = 1 if nonlinearity == 'tanh' else 0
        self.nonlinearity = nonlinearity
        super().__init__(input_size, hidden_size, num_layers, **kwargs)


class RNNCellBase(nn.Module):
    _MODE = 1

    def __init__(self, input_size, hidden_size, bias=True, dtype=tp.qint8):
        super().__init__()
        self.input_size, self.hidden_size = int(input_size), int(hidden_size)
        self.bias, self.dtype = bool(bias), dtype
        self.reduce_range = True
        gates = 4 if self._MODE == 2 else 3 if self._MODE == 3 else 1
        self.register_buffer('weight_ih', _initial_weight((gates * hidden_size, input_size), dtype))
        self.register_buffer('weight_hh', _initial_weight((gates * hidden_size, hidden_size), dtype))
        self.register_buffer('bias_ih', tp.zeros(gates * hidden_size) if bias else None)
        self.register_buffer('bias_hh', tp.zeros(gates * hidden_size) if bias else None)

    def forward(self, input, hx=None):
        unbatched = input.dim() == 1
        data = input.unsqueeze(0) if unbatched else input
        if data.dim() != 2 or data.size(1) != self.input_size:
            raise ValueError('Recurrent cell input dimensions do not match the module')
        if hx is None:
            h = tp.zeros((data.size(0), self.hidden_size), dtype=tp.float32, device=data.device)
            c = tp.zeros_like(h) if self._MODE == 2 else None
        else:
            h, c = hx if self._MODE == 2 else (hx, None)
            if unbatched:
                h = h.unsqueeze(0)
                if c is not None:
                    c = c.unsqueeze(0)
        h, c = _C._quantized_recurrent_cell(data, h, c, self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh, self._MODE, self.reduce_range)
        if unbatched:
            h = h.squeeze(0)
            if self._MODE == 2:
                c = c.squeeze(0)
        return (h, c) if self._MODE == 2 else h

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        config = getattr(mod, 'qconfig', None)
        dtype = config.weight().dtype if config is not None else tp.qint8
        options = dict(bias=mod.bias, dtype=dtype)
        if cls is RNNCell:
            options['nonlinearity'] = mod.nonlinearity
        result = cls(mod.input_size, mod.hidden_size, **options)
        for name in ('weight_ih', 'weight_hh'):
            setattr(result, name, _converted_weight(mod, name, dtype, use_precomputed_fake_quant))
        for name in ('bias_ih', 'bias_hh'):
            bias = getattr(mod, name, None)
            setattr(result, name, None if bias is None else bias.detach().float())
        result.train(mod.training)
        return result


class LSTMCell(RNNCellBase):
    _MODE = 2


class GRUCell(RNNCellBase):
    _MODE = 3


class RNNCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', dtype=tp.qint8):
        if nonlinearity not in ('tanh', 'relu'):
            raise ValueError('Recurrent nonlinearity must be tanh or relu')
        self._MODE = 1 if nonlinearity == 'tanh' else 0
        self.nonlinearity = nonlinearity
        super().__init__(input_size, hidden_size, bias, dtype)
