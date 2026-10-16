"""Coverage for the Vulkan random fills, which run on the device."""

import math

import numpy as np
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


def _vk_prob(values):
    return tp.from_numpy(np.asarray(values, dtype=np.float32)).to("vulkan")


def test_multinomial_deterministic_single_category():
    prob = _vk_prob([1.0, 0.0, 0.0])
    samples = tp.multinomial(prob, 1, replacement=True)
    assert samples.dtype == tp.int64
    assert samples.cpu().tolist() == [0]


def test_multinomial_1d_shape_and_range():
    prob = _vk_prob([0.2, 0.3, 0.5])
    samples = tp.multinomial(prob, 64, replacement=True)
    assert tuple(samples.shape) == (64,)
    vals = samples.cpu().numpy()
    assert vals.min() >= 0 and vals.max() <= 2


def test_multinomial_2d_shape_and_rows():
    prob = _vk_prob([[0.5, 0.5], [0.9, 0.1]])
    samples = tp.multinomial(prob, 32, replacement=True)
    assert tuple(samples.shape) == (2, 32)
    row0 = samples.cpu().numpy()[0]
    row1 = samples.cpu().numpy()[1]
    assert (row0 == 0).sum() > 0 and (row0 == 1).sum() > 0
    assert (row1 == 0).sum() > (row1 == 1).sum()


def test_multinomial_distribution_unbiased():
    prob = _vk_prob([0.5, 0.5])
    samples = tp.multinomial(prob, 2000, replacement=True).cpu().numpy()
    assert abs((samples == 0).sum() - 1000) < 120


def _vk_logits(values):
    return tp.from_numpy(np.asarray(values, dtype=np.float32)).to("vulkan")


def test_sample_deterministic_peak():
    logits = _vk_logits([[10.0, 0.0, -100.0, -100.0]])
    samples = tp.sample(logits, 1.0, 0, 1.0)
    assert samples.dtype == tp.int64
    assert samples.cpu().tolist() == [0]


def test_sample_top_k_forces_top1():
    logits = _vk_logits([[0.4, 0.3, 0.2, 0.1]])
    samples = tp.sample(logits, 1.0, 1, 1.0)
    assert samples.cpu().tolist() == [0]


def test_sample_top_p_nucleus_cutoff():
    logits = _vk_logits([[0.4, 0.3, 0.2, 0.1]])
    samples = tp.sample(logits, 1.0, 0, 0.1)
    assert samples.cpu().tolist() == [0]


def test_sample_batch_rows():
    logits = _vk_logits([[10.0, 0.0], [0.0, 10.0]])
    samples = tp.sample(logits, 1.0, 0, 1.0)
    assert samples.cpu().tolist() == [0, 1]


def test_sample_distribution_unbiased():
    logits = _vk_logits([[0.0, 0.0]])  # softmax = [0.5, 0.5]
    counts = {0: 0, 1: 0}
    for _ in range(1000):
        idx = tp.sample(logits, 1.0, 0, 1.0).cpu().item()
        counts[idx] += 1
    assert abs(counts[0] - 500) < 100


def test_sample_top_k_restricts_domain():
    logits = _vk_logits([[0.4, 0.3, 0.2, 0.1]])
    vals = [
        tp.sample(logits, 1.0, 2, 1.0).cpu().item() for _ in range(200)
    ]
    assert set(vals) <= {0, 1}


def test_sample_feed_embedding_chain():
    """Sampled Int64 tokens must feed the embedding table directly."""
    emb = tp.randn(128, 16, device="vulkan")
    logits = tp.randn(1, 128, device="vulkan")
    tok = tp.sample(logits, 0.8, 50, 0.95)
    assert tok.dtype == tp.int64
    assert tok.shape == (1,)
    x = tp._C.embedding(emb, tok)                      # [1, 16]
    assert x.shape == (1, 16)
    ref = tp._C.embedding(emb.cpu(), tok.cpu()).cpu().numpy()
    assert np.allclose(x.cpu().numpy(), ref, atol=1e-6)
