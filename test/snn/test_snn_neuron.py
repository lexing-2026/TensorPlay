"""Spiking neuron modules: state, reset, single-step and multi-step."""
import pytest

import tensorplay as tp
from tensorplay import snn


def test_lif_single_step_state_updates():
    neuron = snn.LIFNode(threshold=1.0, decay=0.5, v_reset=0.0)
    x = tp.tensor([[0.8, 0.2]])
    spike = neuron(x)
    assert spike.shape == (1, 2)
    assert neuron.v is not None
    assert neuron.v.shape == (1, 2)
    # Second step accumulates the state.
    spike2 = neuron(x)
    assert spike2.shape == (1, 2)
    neuron.reset()
    assert neuron.v is None


def test_lif_spikes_when_threshold_crossed():
    neuron = snn.LIFNode(threshold=1.0, decay=0.0, v_reset=0.0)
    x = tp.tensor([0.6, 1.2])
    spike = neuron(x)
    tp.testing.assert_close(spike, tp.tensor([0.0, 1.0]))


def test_if_is_lif_with_unit_decay():
    if_node = snn.IFNode(threshold=1.0, v_reset=0.0)
    x = tp.tensor([0.6, 0.6])
    s1 = if_node(x)
    # No leak: the potential keeps accumulating.
    assert float(s1.sum()) == 0.0
    s2 = if_node(x)
    tp.testing.assert_close(s2, tp.tensor([1.0, 1.0]))


def test_lif_multi_step_returns_spike_train():
    neuron = snn.LIFNode(threshold=1.0, decay=0.5, v_reset=0.0, step_mode="m")
    x = tp.tensor(
        [
            [[0.8, 0.2]],
            [[0.8, 0.2]],
            [[0.8, 0.2]],
        ]
    )  # [T=3, B=1, C=2]
    out = neuron(x)
    assert out.shape == (3, 1, 2)
    assert set(tp.unique(out).tolist()) <= {0.0, 1.0}
    assert neuron.v is not None
    neuron.reset()
    assert neuron.v is None


def test_lif_multi_step_matches_manual_iteration():
    neuron = snn.LIFNode(threshold=1.0, decay=0.5, v_reset=0.0, step_mode="m")
    x = tp.rand(4, 2, 3)
    out = neuron(x)

    manual = snn.LIFNode(threshold=1.0, decay=0.5, v_reset=0.0, step_mode="s")
    expected = []
    for t in range(4):
        expected.append(manual(x[t]))
    tp.testing.assert_close(out, tp.stack(expected, dim=0))
    tp.testing.assert_close(neuron.v, manual.v)


def test_parametric_lif_decay_is_learnable():
    neuron = snn.ParametricLIFNode(decay_init=0.3, decay_shape=(1, 4))
    assert isinstance(neuron.decay, tp.nn.Parameter)
    x = tp.rand(2, 4)
    out = neuron(x)
    out.sum().backward()
    assert neuron.decay.grad is not None
    assert neuron.decay.grad.shape == (1, 4)


def test_step_mode_validation():
    neuron = snn.LIFNode(step_mode="bad")
    with pytest.raises(ValueError):
        neuron(tp.rand(2))