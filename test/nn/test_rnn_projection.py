"""LSTM with a projected hidden state, against a step-by-step NumPy evaluation."""

import numpy as np
import pytest

import tensorplay as tp


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm_reference(module, x, h0, c0):
    """Evaluate the LSTM equations directly: gates i, f, g, o, then h = W_hr h."""

    outputs = []
    layer_input = x
    h_n, c_n = [], []
    for layer in range(module.num_layers):
        w_ih, w_hh, b_ih, b_hh, w_hr = (
            np.asarray(getattr(module, f"{name}_l{layer}").tolist())
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh", "weight_hr")
        )
        h, c = h0[layer], c0[layer]
        size = c.shape[1]
        outputs = []
        for step in layer_input:
            gates = step @ w_ih.T + b_ih + h @ w_hh.T + b_hh
            i, f, g, o = (gates[:, k * size : (k + 1) * size] for k in range(4))
            c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
            h = (_sigmoid(o) * np.tanh(c)) @ w_hr.T
            outputs.append(h)
        layer_input = np.stack(outputs)
        h_n.append(h)
        c_n.append(c)
    return layer_input, np.stack(h_n), np.stack(c_n)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_projected_lstm_keeps_the_cell_state_at_hidden_size(num_layers):
    tp.manual_seed(0)
    module = tp.nn.LSTM(6, 8, num_layers=num_layers, proj_size=3).double()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 2, 6))
    h0 = rng.standard_normal((num_layers, 2, 3))
    c0 = rng.standard_normal((num_layers, 2, 8))

    with tp.no_grad():
        out, (h_n, c_n) = module(
            tp.tensor(x.tolist(), dtype=tp.float64),
            (tp.tensor(h0.tolist(), dtype=tp.float64), tp.tensor(c0.tolist(), dtype=tp.float64)),
        )
    assert tuple(out.shape) == (5, 2, 3)
    assert tuple(h_n.shape) == (num_layers, 2, 3)
    assert tuple(c_n.shape) == (num_layers, 2, 8)

    expected = _lstm_reference(module, x, h0, c0)
    for actual, wanted in zip((out, h_n, c_n), expected):
        np.testing.assert_allclose(np.asarray(actual.tolist()), wanted, rtol=1e-10, atol=1e-12)

    # Without an initial state the cell state still starts at hidden size.
    with tp.no_grad():
        out, (h_n, c_n) = module(tp.tensor(x.tolist(), dtype=tp.float64))
    assert tuple(c_n.shape) == (num_layers, 2, 8)
