"""Extended SNN coverage: ATan/Erf surrogates, broadcast v_reset, learnable
threshold, return_v_seq, ALIF, SpikeLinear, new encoders, monitor, losses."""
import math

import pytest

import tensorplay as tp
from tensorplay import snn


# ---------------------------------------------------------------------------
# ATan / Erf surrogate gradients
# ---------------------------------------------------------------------------

def test_surrogate_grad_atan_matches_analytic():
    x = tp.tensor([-2.0, 0.0, 1.5])
    alpha = 2.0
    got = snn.surrogate_grad(x, surrogate=4, alpha=alpha)
    ax = alpha * x
    expected = alpha / (1.0 + (math.pi * ax) ** 2)
    tp.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)


def test_surrogate_grad_erf_matches_analytic():
    x = tp.tensor([-2.0, 0.0, 1.5])
    alpha = 2.0
    got = snn.surrogate_grad(x, surrogate=5, alpha=alpha)
    expected = alpha / math.sqrt(math.pi) * tp.exp(-(alpha * x) ** 2)
    tp.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)


def test_atan_surrogate_object_backward():
    sur = snn.ATanSurrogate(alpha=2.0)
    x = tp.rand(4, requires_grad=True)
    s = sur(x)
    s.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_erf_surrogate_object_backward():
    sur = snn.ErfSurrogate(alpha=2.0)
    x = tp.rand(4, requires_grad=True)
    s = sur(x)
    s.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# v_reset broadcasting (per-neuron reset voltage)
# ---------------------------------------------------------------------------

def test_lif_step_v_reset_per_neuron_broadcast():
    x = tp.tensor([[0.8, 0.8], [1.2, 1.2]])
    v = tp.zeros_like(x)
    v_reset = tp.tensor([0.0, 0.5])
    spike, v_new = snn.lif_step(
        x, v, decay=0.5, threshold=1.0, v_reset=v_reset
    )
    tp.testing.assert_close(spike, tp.tensor([[0.0, 0.0], [1.0, 1.0]]))
    tp.testing.assert_close(v_new, tp.tensor([[0.4, 0.4], [0.0, 0.5]]))


def test_lif_seq_v_reset_per_neuron():
    x = tp.tensor([[[0.8, 0.8]], [[1.2, 1.2]]])
    spikes, v_final = snn.lif_seq(
        x, decay=0.5, threshold=1.0, v_reset=tp.tensor([0.0, 0.5])
    )
    tp.testing.assert_close(spikes, tp.tensor([[[0.0, 0.0]], [[1.0, 1.0]]]))
    tp.testing.assert_close(v_final, tp.tensor([[0.0, 0.5]]))


def test_lif_step_v_reset_gradient_flow():
    # The spike path depends on v_reset through the hard-reset voltage; make
    # sure autograd flows through the returned v_new when v_reset is a tensor.
    x = tp.tensor([[1.2]])
    v = tp.zeros_like(x)
    v_reset = tp.tensor([0.3], requires_grad=True)
    spike, v_new = snn.lif_step(x, v, decay=0.5, threshold=1.0, v_reset=v_reset)
    v_new.sum().backward()
    assert v_reset.grad is not None


# ---------------------------------------------------------------------------
# learnable threshold
# ---------------------------------------------------------------------------

def test_lif_learnable_threshold_is_parameter():
    neuron = snn.LIFNode(
        threshold=1.0, decay=0.5, v_reset=0.0, learnable_threshold=True
    )
    assert isinstance(neuron.threshold, tp.nn.Parameter)
    x = tp.rand(4, 8) * 2
    out = neuron(x)
    out.sum().backward()
    assert neuron.threshold.grad is not None
    assert neuron.threshold.grad.shape == ()


def test_parametric_lif_learnable_threshold():
    neuron = snn.ParametricLIFNode(
        decay_init=0.5, decay_shape=(1, 4), learnable_threshold=True
    )
    assert isinstance(neuron.threshold, tp.nn.Parameter)
    x = tp.rand(2, 4) * 2
    out = neuron(x)
    out.sum().backward()
    assert neuron.threshold.grad is not None


# ---------------------------------------------------------------------------
# return_v_seq
# ---------------------------------------------------------------------------

def test_lif_return_v_seq_shapes():
    neuron = snn.LIFNode(step_mode="m", return_v_seq=True)
    x = tp.rand(5, 2, 3)
    spikes, v_seq = neuron(x)
    assert spikes.shape == (5, 2, 3)
    assert v_seq.shape == (5, 2, 3)
    assert neuron.v.shape == (2, 3)


def test_lif_return_v_seq_matches_manual():
    neuron = snn.LIFNode(
        threshold=1.0, decay=0.5, v_reset=0.0, step_mode="m", return_v_seq=True
    )
    x = tp.rand(4, 2, 3)
    spikes, v_seq = neuron(x)

    manual = snn.LIFNode(threshold=1.0, decay=0.5, v_reset=0.0, step_mode="s")
    expected = []
    for t in range(4):
        manual(x[t])
        expected.append(manual.v.clone())
    tp.testing.assert_close(v_seq, tp.stack(expected, dim=0))
    tp.testing.assert_close(spikes, snn.lif_seq(x, decay=0.5, threshold=1.0)[0])


def test_lif_return_v_seq_backward():
    neuron = snn.LIFNode(
        threshold=1.0, decay=0.5, v_reset=0.0, step_mode="m", return_v_seq=True
    )
    x = tp.rand(5, 2, 3, requires_grad=True)
    spikes, v_seq = neuron(x)
    (spikes.sum() + v_seq.sum()).backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# ALIFNode
# ---------------------------------------------------------------------------

def test_alif_threshold_rises_after_activity():
    neuron = snn.ALIFNode(base_threshold=1.0, decay=0.8, decay_th=0.8)
    # Strong input fires immediately.
    assert float(neuron(tp.tensor([1.5]))[0]) == 1.0
    # Subthreshold accumulation, then a spike.
    assert float(neuron(tp.tensor([0.6]))[0]) == 0.0
    assert float(neuron(tp.tensor([0.6]))[0]) == 1.0
    # The adaptation term has risen; the same input no longer fires.
    assert float(neuron(tp.tensor([0.6]))[0]) == 0.0
    assert neuron.a is not None


def test_alif_multi_step_matches_single_step_loop():
    neuron = snn.ALIFNode(step_mode="m")
    x = tp.rand(4, 2, 3)
    out = neuron(x)

    manual = snn.ALIFNode(step_mode="s")
    expected = []
    for t in range(4):
        expected.append(manual(x[t]))
    tp.testing.assert_close(out, tp.stack(expected, dim=0))
    tp.testing.assert_close(neuron.v, manual.v)
    tp.testing.assert_close(neuron.a, manual.a)


def test_alif_multi_step_return_v_seq():
    neuron = snn.ALIFNode(step_mode="m", return_v_seq=True)
    x = tp.rand(4, 2, 3)
    spikes, v_seq = neuron(x)
    assert spikes.shape == (4, 2, 3)
    assert v_seq.shape == (4, 2, 3)


# ---------------------------------------------------------------------------
# SpikeLinear
# ---------------------------------------------------------------------------

def test_spike_linear_output():
    layer = snn.SpikeLinear(4, 8)
    x = tp.randint(0, 2, (3, 4)).to(tp.float32)
    out = layer(x)
    assert out.shape == (3, 8)
    out.sum().backward()
    assert layer.linear.weight.grad is not None


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

def test_periodic_encoder_repeats_pattern():
    enc = snn.PeriodicEncoder(tp.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    out = enc(7)
    assert out.shape == (7, 2)
    tp.testing.assert_close(out[0], tp.tensor([1.0, 0.0]))
    tp.testing.assert_close(out[3], tp.tensor([1.0, 0.0]))
    tp.testing.assert_close(out[6], tp.tensor([1.0, 0.0]))
    tp.testing.assert_close(out[1], tp.tensor([0.0, 1.0]))


def test_gaussian_tuning_curve_shape_and_binary():
    mu = tp.tensor([[0.0, 0.5, 1.0]])  # 1 feature, 3 neurons
    enc = snn.GaussianTuningCurveEncoder(mu, sigma=0.3)
    x = tp.tensor([[0.2], [0.8]])  # [2, 1]
    out = enc(x, T=20)
    assert out.shape == (20, 2, 3)
    assert set(tp.unique(out).tolist()) <= {0.0, 1.0}


def test_gaussian_tuning_curve_firing_probability_peaks_at_mu():
    mu = tp.tensor([[0.0, 1.0]])  # 1 feature, 2 neurons
    enc = snn.GaussianTuningCurveEncoder(mu, sigma=0.2)
    x = tp.tensor([[1.0]])  # matches the second center
    out = enc(x, T=2000)
    rates = out.mean(dim=0)  # [1, 2]
    # The neuron centered at 1.0 should fire much more often.
    assert float(rates[0, 1]) > 0.8
    assert float(rates[0, 0]) < 0.2


# ---------------------------------------------------------------------------
# Monitor and spike losses
# ---------------------------------------------------------------------------

def test_spike_monitor_records_and_stats():
    neuron = snn.LIFNode(threshold=1.0, decay=0.5, step_mode="m")
    monitor = snn.SpikeMonitor(neuron)
    x = tp.rand(10, 2, 3)
    out = neuron(x)
    assert len(monitor) == 1
    assert monitor.last_spike().shape == (10, 2, 3)
    rate = monitor.spike_rate()
    assert 0.0 <= rate <= 1.0
    fr = monitor.firing_rate(dim=0)
    assert fr.shape == (2, 3)
    assert monitor.total_spikes() == int(float(out.sum()))
    monitor.reset()
    assert len(monitor) == 0
    monitor.detach()


def test_spike_monitor_handles_return_v_seq_output():
    neuron = snn.LIFNode(step_mode="m", return_v_seq=True)
    monitor = snn.SpikeMonitor(neuron)
    neuron(tp.rand(5, 2, 3))
    assert len(monitor) == 1
    assert monitor.last_spike().shape == (5, 2, 3)


def test_spike_rate_loss_value():
    spikes = tp.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target = tp.tensor([2.0 / 3, 1.0 / 3])
    loss = snn.spike_rate_loss(spikes, target)
    expected = ((tp.tensor([2.0 / 3, 1.0 / 3]) - target) ** 2).mean()
    tp.testing.assert_close(loss, expected)


def test_spike_rate_loss_reductions():
    spikes = tp.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target = tp.tensor([0.5, 0.5])
    none = snn.spike_rate_loss(spikes, target, reduction="none")
    assert none.shape == (2,)
    with pytest.raises(ValueError):
        snn.spike_rate_loss(spikes, target, reduction="bad")