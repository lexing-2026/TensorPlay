"""Quantizable recurrent modules for the eager quantization workflow.

These float modules are conversion targets for quantization-aware training;
they carry a ``qconfig`` so ``prepare`` can attach observers and ``convert``
can swap them for their dynamically quantized counterparts.
"""

import tensorplay as tp
from tensorplay import nn

__all__ = ["LSTM", "LSTMCell"]


class LSTMCell(nn.LSTMCell):
    """A float LSTM cell that participates in the quantization workflow."""

    _FLOAT_MODULE = nn.LSTMCell

    @classmethod
    def from_float(cls, other):
        if type(other) is not cls._FLOAT_MODULE:
            raise AssertionError(f"Expected type {cls._FLOAT_MODULE}, got {type(other)}")
        cell = cls(
            other.input_size,
            other.hidden_size,
            other.bias,
            device=other.weight_ih.device,
            dtype=other.weight_ih.dtype,
        )
        cell.weight_ih = other.weight_ih.detach()
        cell.weight_hh = other.weight_hh.detach()
        if other.bias:
            cell.bias_ih = other.bias_ih.detach()
            cell.bias_hh = other.bias_hh.detach()
        cell.qconfig = getattr(other, "qconfig", None)
        return cell


class LSTM(nn.LSTM):
    """A float LSTM that participates in the quantization workflow."""

    _FLOAT_MODULE = nn.LSTM

    @classmethod
    def from_float(cls, other):
        if type(other) is not cls._FLOAT_MODULE:
            raise AssertionError(f"Expected type {cls._FLOAT_MODULE}, got {type(other)}")
        rnn = cls(
            other.input_size,
            other.hidden_size,
            other.num_layers,
            other.bias,
            other.batch_first,
            other.dropout,
            other.bidirectional,
            device=other.weight_ih_l0.device,
            dtype=other.weight_ih_l0.dtype,
        )
        for name, param in other.named_parameters():
            rnn.get_parameter(name).detach().copy_(param)
        rnn.qconfig = getattr(other, "qconfig", None)
        return rnn
