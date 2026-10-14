"""Statically quantized recurrent modules.

The weights are stored on the affine grid and the input/output tensors are
native quantized tensors; the recurrent compute runs through the native
sequence kernels with the input dequantized to float inside the kernel path.
"""

import tensorplay as tp
from tensorplay import nn, _C
from tensorplay.ao.nn.quantized.dynamic.modules.rnn import (
    RNNBase,
    RNNCellBase,
    LSTM as _DynamicLSTM,
    GRU as _DynamicGRU,
    RNN as _DynamicRNN,
    LSTMCell as _DynamicLSTMCell,
    GRUCell as _DynamicGRUCell,
    RNNCell as _DynamicRNNCell,
)

__all__ = ["LSTM", "GRU", "RNN", "LSTMCell", "GRUCell", "RNNCell"]


class LSTM(RNNBase):
    """Static quantized LSTM with quantized input and output tensors."""

    _MODE = 2

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False,
                 dtype=tp.qint8, scale=1.0, zero_point=0):
        super().__init__(input_size, hidden_size, num_layers, bias, batch_first,
                         dropout, bidirectional, dtype)
        self.scale = float(scale)
        self.zero_point = int(zero_point)

    def forward(self, input, hx=None):
        quantized_input = input.is_quantized()
        data = input if not quantized_input else input.dequantize()
        output, h = super().forward(data, hx)
        if quantized_input:
            output = _C.quantize_per_tensor(output, self.scale, self.zero_point, tp.qint8)
        return output, h

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        result = super().from_float(mod, use_precomputed_fake_quant)
        post_process = getattr(mod, "activation_post_process", None)
        if post_process is not None:
            out_scale, out_zero_point = post_process.calculate_qparams()
            result.scale = float(out_scale)
            result.zero_point = int(out_zero_point)
        return result


class GRU(_DynamicGRU):
    """Static quantized GRU with quantized input and output tensors."""

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False,
                 dtype=tp.qint8, scale=1.0, zero_point=0):
        super().__init__(input_size, hidden_size, num_layers, bias, batch_first,
                         dropout, bidirectional, dtype)
        self.scale = float(scale)
        self.zero_point = int(zero_point)

    def forward(self, input, hx=None):
        quantized_input = input.is_quantized()
        data = input if not quantized_input else input.dequantize()
        output, h = super().forward(data, hx)
        if quantized_input:
            output = _C.quantize_per_tensor(output, self.scale, self.zero_point, tp.qint8)
        return output, h

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        result = super().from_float(mod, use_precomputed_fake_quant)
        post_process = getattr(mod, "activation_post_process", None)
        if post_process is not None:
            out_scale, out_zero_point = post_process.calculate_qparams()
            result.scale = float(out_scale)
            result.zero_point = int(out_zero_point)
        return result


class RNN(_DynamicRNN):
    """Static quantized RNN with quantized input and output tensors."""

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False,
                 dtype=tp.qint8, scale=1.0, zero_point=0, nonlinearity="tanh"):
        super().__init__(input_size, hidden_size, num_layers, nonlinearity=nonlinearity,
                         bias=bias, batch_first=batch_first, dropout=dropout,
                         bidirectional=bidirectional, dtype=dtype)
        self.scale = float(scale)
        self.zero_point = int(zero_point)

    def forward(self, input, hx=None):
        quantized_input = input.is_quantized()
        data = input if not quantized_input else input.dequantize()
        output, h = super().forward(data, hx)
        if quantized_input:
            output = _C.quantize_per_tensor(output, self.scale, self.zero_point, tp.qint8)
        return output, h

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        result = super().from_float(mod, use_precomputed_fake_quant)
        post_process = getattr(mod, "activation_post_process", None)
        if post_process is not None:
            out_scale, out_zero_point = post_process.calculate_qparams()
            result.scale = float(out_scale)
            result.zero_point = int(out_zero_point)
        return result


class LSTMCell(_DynamicLSTMCell):
    """Static quantized LSTM cell."""

    def __init__(self, input_size, hidden_size, bias=True, dtype=tp.qint8,
                 scale=1.0, zero_point=0):
        super().__init__(input_size, hidden_size, bias, dtype)
        self.scale = float(scale)
        self.zero_point = int(zero_point)

    def forward(self, input, hx=None):
        quantized_input = input.is_quantized()
        data = input if not quantized_input else input.dequantize()
        result = super().forward(data, hx)
        if quantized_input:
            if isinstance(result, tuple):
                result = (_C.quantize_per_tensor(result[0], self.scale, self.zero_point, tp.qint8), result[1])
            else:
                result = _C.quantize_per_tensor(result, self.scale, self.zero_point, tp.qint8)
        return result


class GRUCell(_DynamicGRUCell):
    """Static quantized GRU cell."""

    def __init__(self, input_size, hidden_size, bias=True, dtype=tp.qint8,
                 scale=1.0, zero_point=0):
        super().__init__(input_size, hidden_size, bias, dtype)
        self.scale = float(scale)
        self.zero_point = int(zero_point)

    def forward(self, input, hx=None):
        quantized_input = input.is_quantized()
        data = input if not quantized_input else input.dequantize()
        result = super().forward(data, hx)
        if quantized_input:
            result = _C.quantize_per_tensor(result, self.scale, self.zero_point, tp.qint8)
        return result


class RNNCell(_DynamicRNNCell):
    """Static quantized RNN cell."""

    def __init__(self, input_size, hidden_size, bias=True, dtype=tp.qint8,
                 scale=1.0, zero_point=0, nonlinearity="tanh"):
        super().__init__(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity,
                         dtype=dtype)
        self.scale = float(scale)
        self.zero_point = int(zero_point)

    def forward(self, input, hx=None):
        quantized_input = input.is_quantized()
        data = input if not quantized_input else input.dequantize()
        result = super().forward(data, hx)
        if quantized_input:
            result = _C.quantize_per_tensor(result, self.scale, self.zero_point, tp.qint8)
        return result