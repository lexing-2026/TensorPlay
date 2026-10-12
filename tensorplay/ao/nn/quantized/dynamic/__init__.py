"""Dynamic quantization: weights are static, activations quantize at runtime."""

from __future__ import annotations

from .modules import Linear  # noqa: F401

__all__ = ["Linear"]

from .modules.rnn import RNN, LSTM, GRU, RNNCell, LSTMCell, GRUCell
__all__ += ['RNN', 'LSTM', 'GRU', 'RNNCell', 'LSTMCell', 'GRUCell']
