"""Coverage for the Vulkan random fills, which run on the device."""

import math

import pytest

import tensorplay as tp

pytestmark = pytest.mark.skipif(
    not tp.is_vulkan_available(), reason="no Vulkan device"
)


def _values(t):
    return tp.reshape(t.cpu(), [-1]).tolist()


def _mean_std(t):
    v = _values(t)
    n = len(v)
    mean = sum(v) / n
    return mean, math.sqrt(sum((x - mean) ** 2 for x in v) / n)


def test_uniform_covers_its_range_evenly():
    tp.manual_seed(5)
    x = tp.zeros(8, 16, 16, 16).to("vulkan")
    x.uniform_(0.0, 1.0)
    values = _values(x)
    mean, std = _mean_std(x)

    assert all(0.0 <= v < 1.0 for v in values)
    assert abs(mean - 0.5) < 0.02
    assert abs(std - (1.0 / math.sqrt(12.0))) < 0.02

    bins = [0] * 10
    for v in values:
        bins[min(9, int(v * 10))] += 1
    for count in bins:
        assert abs(count / len(values) - 0.1) < 0.02


def test_uniform_honours_a_shifted_range():
    tp.manual_seed(6)
    x = tp.zeros(4, 8, 8, 8).to("vulkan")
    x.uniform_(-3.0, 5.0)
    values = _values(x)
    mean, _ = _mean_std(x)
    assert all(-3.0 <= v < 5.0 for v in values)
    assert abs(mean - 1.0) < 0.12


def test_normal_matches_its_mean_and_spread():
    tp.manual_seed(7)
    x = tp.zeros(8, 16, 16, 16).to("vulkan")
    x.normal_(2.0, 3.0)
    mean, std = _mean_std(x)
    assert abs(mean - 2.0) < 0.06
    assert abs(std - 3.0) < 0.08


def test_bernoulli_scalar_probability():
    for p in (0.0, 0.25, 0.75, 1.0):
        tp.manual_seed(11)
        x = tp.zeros(8, 16, 16, 16).to("vulkan")
        x.bernoulli_(p)
        values = _values(x)
        assert set(values) <= {0.0, 1.0}
        assert abs(sum(values) / len(values) - p) < 0.02


def test_bernoulli_tensor_probability_is_per_element():
    x = tp.zeros(2, 4, 8, 8).to("vulkan")
    ones = tp.ones(2, 4, 8, 8).to("vulkan")
    x.bernoulli_(ones)
    assert all(v == 1.0 for v in _values(x))

    zeros = tp.zeros(2, 4, 8, 8).to("vulkan")
    x.bernoulli_(zeros)
    assert all(v == 0.0 for v in _values(x))


def test_neighbouring_channels_get_independent_draws():
    # four elements share one texel; each has to consume its own draw
    tp.manual_seed(13)
    x = tp.zeros(1, 8, 4, 4).to("vulkan")
    x.uniform_(0.0, 1.0)
    lanes = tp.reshape(x.cpu(), [8, 16]).tolist()
    first_of_each_channel = [row[0] for row in lanes]
    assert len(set(first_of_each_channel)) == len(first_of_each_channel)


def test_a_seed_reproduces_the_same_fill():
    tp.manual_seed(99)
    a = tp.zeros(2, 4, 4, 4).to("vulkan")
    a.uniform_(0.0, 1.0)
    tp.manual_seed(99)
    b = tp.zeros(2, 4, 4, 4).to("vulkan")
    b.uniform_(0.0, 1.0)
    assert _values(a) == _values(b)


def test_successive_fills_differ():
    tp.manual_seed(101)
    a = tp.zeros(2, 4, 4, 4).to("vulkan")
    a.uniform_(0.0, 1.0)
    b = tp.zeros(2, 4, 4, 4).to("vulkan")
    b.uniform_(0.0, 1.0)
    assert _values(a) != _values(b)


def test_rand_like_and_randn_like():
    base = tp.zeros(4, 8, 8, 8).to("vulkan")
    mean, std = _mean_std(tp.rand_like(base))
    assert abs(mean - 0.5) < 0.03
    mean, std = _mean_std(tp.randn_like(base))
    assert abs(mean) < 0.06
    assert abs(std - 1.0) < 0.06
