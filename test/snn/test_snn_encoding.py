"""Spike encoders: rate (Bernoulli) and temporal (time-to-first-spike)."""
import tensorplay as tp
from tensorplay import snn


def test_rate_encode_shape_and_binary():
    x = tp.tensor([[0.3, 0.7]])
    out = snn.rate_encode(x, T=50)
    assert out.shape == (50, 1, 2)
    assert set(tp.unique(out).tolist()) <= {0.0, 1.0}


def test_rate_encode_mean_tracks_probability():
    x = tp.tensor([0.0, 0.5, 1.0])
    out = snn.rate_encode(x, T=2000)
    mean = out.mean(dim=0)
    # 2000 Bernoulli draws: mean within ~3 sigma of the probability.
    assert abs(float(mean[0])) < 1e-6
    assert abs(float(mean[1]) - 0.5) < 0.05
    assert abs(float(mean[2]) - 1.0) < 1e-6


def test_temporal_encode_shape_and_one_spike():
    x = tp.tensor([[0.2, 0.8]])
    out = snn.temporal_encode(x, T=10)
    assert out.shape == (10, 1, 2)
    assert set(tp.unique(out).tolist()) <= {0.0, 1.0}
    # Exactly one spike per element.
    assert out.sum(dim=0).tolist() == [[1, 1]]


def test_temporal_encode_larger_input_fires_earlier():
    x = tp.tensor([0.2, 0.9])
    out = snn.temporal_encode(x, T=20)
    spike_time = out.argmax(dim=0)
    assert spike_time[0] > spike_time[1]