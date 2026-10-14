"""Reference quantized recurrent modules with float weights."""

import tensorplay as tp
from tensorplay import nn
from tensorplay.nn import functional as F

from .utils import ReferenceQuantizedModule, _quantize_and_dequantize_weight

__all__ = ["RNNCell", "LSTMCell", "GRUCell", "LSTM", "GRU"]


class RNNCellBase(ReferenceQuantizedModule):
    _MODE = 1  # 1: rnn_tanh, 2: lstm, 3: gru

    def _init_rnn_cell(self, input_size, hidden_size, bias=True, device=None,
                       dtype=None, weight_qparams=None):
        nn.Module.__init__(self)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.bias = bool(bias)
        gates = 4 if self._MODE == 2 else 3 if self._MODE == 3 else 1
        self.weight_ih = nn.Parameter(tp.empty(gates * hidden_size, input_size, device=device, dtype=dtype))
        self.weight_hh = nn.Parameter(tp.empty(gates * hidden_size, hidden_size, device=device, dtype=dtype))
        if bias:
            self.bias_ih = nn.Parameter(tp.empty(gates * hidden_size, device=device, dtype=dtype))
            self.bias_hh = nn.Parameter(tp.empty(gates * hidden_size, device=device, dtype=dtype))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()
        self._init_weight_qparams(weight_qparams, device)

    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for w in (self.weight_ih, self.weight_hh):
            nn.init.uniform_(w, -stdv, stdv)
        for b in (self.bias_ih, self.bias_hh):
            if b is not None:
                nn.init.uniform_(b, -stdv, stdv)

    def _get_name(self):
        return f"Quantized{type(self).__name__}(Reference)"

    def _quantized_weights(self):
        w_ih = _quantize_and_dequantize_weight(
            self.weight_ih, self.weight_qscheme, self.weight_dtype,
            self.weight_scale, self.weight_zero_point, self.weight_axis_int)
        w_hh = _quantize_and_dequantize_weight(
            self.weight_hh, self.weight_qscheme, self.weight_dtype,
            self.weight_scale, self.weight_zero_point, self.weight_axis_int)
        return w_ih, w_hh

    def forward(self, input, hx=None):
        w_ih, w_hh = self._quantized_weights()
        if self._MODE == 2:
            if hx is None:
                hx = (tp.zeros(input.size(0), self.hidden_size, device=input.device),
                      tp.zeros(input.size(0), self.hidden_size, device=input.device))
            h, c = hx
            return F.lstm_cell(input, h, c, w_ih, w_hh, self.bias_ih, self.bias_hh)
        if self._MODE == 3:
            if hx is None:
                hx = tp.zeros(input.size(0), self.hidden_size, device=input.device)
            return F.gru_cell(input, hx, w_ih, w_hh, self.bias_ih, self.bias_hh)
        raise NotImplementedError("Reference RNN tanh/relu cells are not implemented")


class RNNCell(RNNCellBase):
    _MODE = 1

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh",
                 device=None, dtype=None, weight_qparams=None):
        self.nonlinearity = nonlinearity
        self._init_rnn_cell(input_size, hidden_size, bias, device, dtype, weight_qparams)


class LSTMCell(RNNCellBase):
    _MODE = 2

    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=None,
                 weight_qparams=None):
        self._init_rnn_cell(input_size, hidden_size, bias, device, dtype, weight_qparams)


class GRUCell(RNNCellBase):
    _MODE = 3

    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=None,
                 weight_qparams=None):
        self._init_rnn_cell(input_size, hidden_size, bias, device, dtype, weight_qparams)


class RNNBase(ReferenceQuantizedModule):
    _MODE = 1

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False, device=None,
                 dtype=None, weight_qparams=None):
        nn.Module.__init__(self)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bias = bool(bias)
        self.batch_first = bool(batch_first)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self._init_weight_qparams(weight_qparams, device)
        num_directions = 2 if bidirectional else 1
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = f"_l{layer}" + ("_reverse" if direction else "")
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                cell = self._make_cell(layer_input_size, hidden_size, bias, device, dtype, weight_qparams)
                self.add_module("cell" + suffix, cell)

    def _make_cell(self, input_size, hidden_size, bias, device, dtype, weight_qparams):
        raise NotImplementedError

    def _get_name(self):
        return f"Quantized{type(self).__name__}(Reference)"

    def forward(self, input, hx=None):
        if self.batch_first:
            input = input.transpose(0, 1)
        seq_len, batch, _ = input.size()
        num_directions = 2 if self.bidirectional else 1
        if hx is None:
            h = tp.zeros(self.num_layers * num_directions, batch, self.hidden_size, device=input.device)
            c = tp.zeros_like(h) if self._MODE == 2 else None
        else:
            h, c = hx if self._MODE == 2 else (hx, None)
        output = input
        for layer in range(self.num_layers):
            layer_outputs = []
            for direction in range(num_directions):
                cell = getattr(self, f"cell_l{layer}" + ("_reverse" if direction else ""))
                layer_input = output if direction == 0 else tp.flip(output, (0,))
                idx = layer * num_directions + direction
                hh = h[idx]
                cc = c[idx] if self._MODE == 2 else None
                hidden = (hh, cc) if self._MODE == 2 else hh
                outs = []
                for t in range(seq_len):
                    hidden = cell(layer_input[t], hidden)
                    if self._MODE == 2:
                        outs.append(hidden[0])
                    else:
                        outs.append(hidden)
                layer_out = tp.stack(outs, 0)
                if direction == 1:
                    layer_out = tp.flip(layer_out, (0,))
                layer_outputs.append(layer_out)
            output = tp.cat(layer_outputs, -1) if num_directions == 2 else layer_outputs[0]
        if self.batch_first:
            output = output.transpose(0, 1)
        if self._MODE == 2:
            return output, (h, c)
        return output, h


class LSTM(RNNBase):
    _MODE = 2

    def _make_cell(self, input_size, hidden_size, bias, device, dtype, weight_qparams):
        return LSTMCell(input_size, hidden_size, bias, device, dtype, weight_qparams)


class GRU(RNNBase):
    _MODE = 3

    def _make_cell(self, input_size, hidden_size, bias, device, dtype, weight_qparams):
        return GRUCell(input_size, hidden_size, bias, device, dtype, weight_qparams)