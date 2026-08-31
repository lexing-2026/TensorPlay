"""CUDA bagged-embedding and rank-generic convolution operators.

Cross-checks the CUDA kernels against the CPU ones for the same inputs, which
is the tighter of the two available references: both must also agree with the
reference framework, and the CPU side is covered by
test_embedding_bag_native.py / test_convolution_native.py.
"""

import numpy as np
import pytest
import torch

import tensorplay as tp
import tensorplay.nn.functional as F
import tensorplay._C as _C


pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA not available")

SUM, MEAN, MAX = 0, 1, 2
MODES = [("sum", SUM), ("mean", MEAN), ("max", MAX)]


def _cpu(array):
    return tp.tensor(np.ascontiguousarray(array))


def _cu(array):
    return _cpu(array).to(tp.device("cuda"))


def _np(t):
    if t is None or not t.defined():
        return None
    return np.asarray(t.to(tp.device("cpu")).tolist(), dtype=np.float64)


def _close(actual, expected, rtol=1e-5, atol=1e-6, msg=""):
    got = _np(actual) if isinstance(actual, tp.Tensor) else np.asarray(actual, np.float64)
    want = _np(expected) if isinstance(expected, tp.Tensor) else np.asarray(expected, np.float64)
    assert got.shape == want.shape, f"{msg}: shape {got.shape} != {want.shape}"
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=msg)


def _ints(t):
    return t.to(tp.device("cpu")).tolist()


def _weight(rows=64, dim=16, seed=0, dtype=np.float32):
    return np.random.RandomState(seed).randn(rows, dim).astype(dtype)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

class TestEmbeddingBagForwardCuda:
    @pytest.mark.parametrize("name,mode", MODES)
    def test_matches_cpu_and_reference(self, name, mode):
        w = _weight(seed=1)
        idx = np.array([3, 1, 4, 1, 5, 9, 2, 6, 0, 7], dtype=np.int64)
        offsets = np.array([0, 3, 3, 7], dtype=np.int64)

        host = _C._embedding_bag(_cpu(w), _cpu(idx), _cpu(offsets), False, mode,
                                 False, None, False, -1)
        dev = _C._embedding_bag(_cu(w), _cu(idx), _cu(offsets), False, mode,
                                False, None, False, -1)
        _close(dev[0], host[0], msg=f"cuda vs cpu output {name}")
        assert _ints(dev[1]) == _ints(host[1]), "offset2bag"
        assert _ints(dev[2]) == _ints(host[2]), "bag_size"
        assert _ints(dev[3]) == _ints(host[3]), "max_indices"

        want = torch.nn.functional.embedding_bag(
            torch.tensor(idx), torch.tensor(w), offsets=torch.tensor(offsets), mode=name)
        _close(dev[0], want.detach().numpy(), msg=f"cuda vs reference {name}")

    @pytest.mark.parametrize("name,mode", MODES)
    def test_padding_idx(self, name, mode):
        w = _weight(seed=2)
        idx = np.array([4, 4, 2, 4, 6, 4, 1, 4], dtype=np.int64)
        offsets = np.array([0, 3, 6], dtype=np.int64)

        dev = _C._embedding_bag(_cu(w), _cu(idx), _cu(offsets), False, mode,
                                False, None, False, 4)
        want = torch.nn.functional.embedding_bag(
            torch.tensor(idx), torch.tensor(w), offsets=torch.tensor(offsets),
            mode=name, padding_idx=4)
        _close(dev[0], want.detach().numpy(), msg=f"cuda padding_idx {name}")

    @pytest.mark.parametrize("name,mode", MODES)
    def test_include_last_offset(self, name, mode):
        w = _weight(seed=3)
        idx = np.array([0, 7, 7, 2, 8, 8, 1], dtype=np.int64)
        offsets = np.array([0, 2, 5, 7], dtype=np.int64)

        dev = _C._embedding_bag(_cu(w), _cu(idx), _cu(offsets), False, mode,
                                False, None, True, -1)
        want = torch.nn.functional.embedding_bag(
            torch.tensor(idx), torch.tensor(w), offsets=torch.tensor(offsets),
            mode=name, include_last_offset=True)
        _close(dev[0], want.detach().numpy(), msg=f"cuda include_last_offset {name}")

    def test_per_sample_weights(self):
        w = _weight(seed=4)
        idx = np.array([1, 3, 5, 7, 9, 11], dtype=np.int64)
        offsets = np.array([0, 2, 4], dtype=np.int64)
        psw = np.array([0.5, -1.5, 2.0, 0.25, 3.0, -0.75], dtype=np.float32)

        dev = _C._embedding_bag(_cu(w), _cu(idx), _cu(offsets), False, SUM,
                                False, _cu(psw), False, -1)
        want = torch.nn.functional.embedding_bag(
            torch.tensor(idx), torch.tensor(w), offsets=torch.tensor(offsets),
            mode="sum", per_sample_weights=torch.tensor(psw))
        _close(dev[0], want.detach().numpy(), msg="cuda per_sample_weights")

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtypes(self, dtype):
        w = _weight(seed=5, dtype=dtype)
        idx = np.array([0, 2, 4, 6, 8], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)
        host = _C._embedding_bag(_cpu(w), _cpu(idx), _cpu(offsets), False, MEAN,
                                 False, None, False, -1)
        dev = _C._embedding_bag(_cu(w), _cu(idx), _cu(offsets), False, MEAN,
                                False, None, False, -1)
        _close(dev[0], host[0], rtol=1e-6, atol=1e-7, msg=str(dtype))

    def test_int32_indices(self):
        w = _weight(seed=6)
        idx = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        offsets = np.array([0, 2], dtype=np.int32)
        host = _C._embedding_bag(_cpu(w), _cpu(idx), _cpu(offsets), False, SUM,
                                 False, None, False, -1)
        dev = _C._embedding_bag(_cu(w), _cu(idx), _cu(offsets), False, SUM,
                                False, None, False, -1)
        _close(dev[0], host[0], msg="cuda int32 indices")
        assert _ints(dev[1]) == _ints(host[1])

    def test_empty_and_uncovered_bags(self):
        w = _weight(seed=7)
        idx = np.array([2, 5, 9, 4], dtype=np.int64)
        offsets = np.array([0, 0, 2], dtype=np.int64)
        for _, mode in MODES:
            host = _C._embedding_bag(_cpu(w), _cpu(idx), _cpu(offsets), False, mode,
                                     False, None, False, -1)
            dev = _C._embedding_bag(_cu(w), _cu(idx), _cu(offsets), False, mode,
                                    False, None, False, -1)
            _close(dev[0], host[0], msg=f"cuda empty bags mode={mode}")
            assert _ints(dev[1]) == _ints(host[1])
            assert _ints(dev[2]) == _ints(host[2])

    def test_wide_embedding_crosses_thread_stride(self):
        # Wider than one block's x-extent, so the kernel's feature-stride loop
        # and multi-block-y path both run.
        w = _weight(rows=128, dim=300, seed=8)
        idx = np.random.RandomState(9).randint(0, 128, size=200).astype(np.int64)
        offsets = (np.arange(20) * 10).astype(np.int64)
        for _, mode in MODES:
            host = _C._embedding_bag(_cpu(w), _cpu(idx), _cpu(offsets), False, mode,
                                     False, None, False, -1)
            dev = _C._embedding_bag(_cu(w), _cu(idx), _cu(offsets), False, mode,
                                    False, None, False, -1)
            _close(dev[0], host[0], rtol=1e-5, atol=1e-5, msg=f"cuda wide mode={mode}")
            assert _ints(dev[3]) == _ints(host[3]), f"max_indices mode={mode}"

    def test_forward_only_matches_full(self):
        w = _weight(seed=10)
        idx = np.array([9, 0, 3, 3, 7], dtype=np.int64)
        offsets = np.array([0, 1, 3], dtype=np.int64)
        for _, mode in MODES:
            full = _C._embedding_bag(_cu(w), _cu(idx), _cu(offsets), False, mode,
                                     False, None, False, -1)
            only = _C._embedding_bag_forward_only(_cu(w), _cu(idx), _cu(offsets), False,
                                                  mode, False, None, False, -1)
            for a, b in zip(full, only):
                assert _ints(a) == _ints(b)


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

class TestEmbeddingBagBackwardCuda:
    @pytest.mark.parametrize("name,mode", MODES)
    def test_dense_backward_matches_cpu(self, name, mode):
        w = _weight(seed=11)
        idx = np.array([2, 2, 5, 0, 9, 5, 5, 3], dtype=np.int64)
        offsets = np.array([0, 3, 5], dtype=np.int64)

        host = _C._embedding_bag(_cpu(w), _cpu(idx), _cpu(offsets), False, mode,
                                 False, None, False, -1)
        dev = _C._embedding_bag(_cu(w), _cu(idx), _cu(offsets), False, mode,
                                False, None, False, -1)
        gh = _C._embedding_bag_dense_backward(
            tp.ones_like(host[0]), _cpu(idx), host[1], host[2], host[3],
            w.shape[0], False, mode, None, -1)
        gd = _C._embedding_bag_dense_backward(
            tp.ones_like(dev[0]), _cu(idx), dev[1], dev[2], dev[3],
            w.shape[0], False, mode, None, -1)
        _close(gd, gh, msg=f"cuda dense backward {name}")

    def test_dense_backward_scale_by_freq(self):
        w = _weight(seed=12)
        idx = np.array([1, 1, 1, 4, 4, 7], dtype=np.int64)
        offsets = np.array([0, 3], dtype=np.int64)
        host = _C._embedding_bag(_cpu(w), _cpu(idx), _cpu(offsets), True, SUM,
                                 False, None, False, -1)
        dev = _C._embedding_bag(_cu(w), _cu(idx), _cu(offsets), True, SUM,
                                False, None, False, -1)
        gh = _C._embedding_bag_dense_backward(
            tp.ones_like(host[0]), _cpu(idx), host[1], host[2], host[3],
            w.shape[0], True, SUM, None, -1)
        gd = _C._embedding_bag_dense_backward(
            tp.ones_like(dev[0]), _cu(idx), dev[1], dev[2], dev[3],
            w.shape[0], True, SUM, None, -1)
        _close(gd, gh, msg="cuda scale_grad_by_freq")

    def test_dense_backward_padding_and_mean(self):
        w = _weight(seed=13)
        idx = np.array([3, 6, 3, 1, 3], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)
        host = _C._embedding_bag(_cpu(w), _cpu(idx), _cpu(offsets), False, MEAN,
                                 False, None, False, 3)
        dev = _C._embedding_bag(_cu(w), _cu(idx), _cu(offsets), False, MEAN,
                                False, None, False, 3)
        gh = _C._embedding_bag_dense_backward(
            tp.ones_like(host[0]), _cpu(idx), host[1], host[2], host[3],
            w.shape[0], False, MEAN, None, 3)
        gd = _C._embedding_bag_dense_backward(
            tp.ones_like(dev[0]), _cu(idx), dev[1], dev[2], dev[3],
            w.shape[0], False, MEAN, None, 3)
        _close(gd, gh, msg="cuda mean backward with padding_idx")

    def test_per_sample_weights_backward(self):
        w = _weight(seed=14)
        idx = np.array([0, 4, 8, 2, 6], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)
        psw = np.array([1.0, -0.5, 0.25, 2.0, 0.75], dtype=np.float32)

        host = _C._embedding_bag(_cpu(w), _cpu(idx), _cpu(offsets), False, SUM,
                                 False, _cpu(psw), False, -1)
        dev = _C._embedding_bag(_cu(w), _cu(idx), _cu(offsets), False, SUM,
                                False, _cu(psw), False, -1)
        gh = _C._embedding_bag_per_sample_weights_backward(
            tp.ones_like(host[0]), _cpu(w), _cpu(idx), _cpu(offsets), host[1], SUM, -1)
        gd = _C._embedding_bag_per_sample_weights_backward(
            tp.ones_like(dev[0]), _cu(w), _cu(idx), _cu(offsets), dev[1], SUM, -1)
        _close(gd, gh, msg="cuda per_sample_weights backward")

    def test_per_sample_weights_backward_rebuilds_offset2bag(self):
        w = _weight(seed=15)
        idx = np.array([1, 5, 3, 2], dtype=np.int64)
        offsets = np.array([0, 2], dtype=np.int64)
        dev = _C._embedding_bag(_cu(w), _cu(idx), _cu(offsets), False, SUM,
                                False, None, False, -1)
        empty = tp.zeros([0], dtype=tp.int64).to(tp.device("cuda"))
        from_empty = _C._embedding_bag_per_sample_weights_backward(
            tp.ones_like(dev[0]), _cu(w), _cu(idx), _cu(offsets), empty, SUM, -1)
        from_map = _C._embedding_bag_per_sample_weights_backward(
            tp.ones_like(dev[0]), _cu(w), _cu(idx), _cu(offsets), dev[1], SUM, -1)
        _close(from_empty, from_map, msg="cuda rebuilt offset2bag")

    @pytest.mark.parametrize("name,mode", MODES)
    def test_autograd_matches_reference(self, name, mode):
        w = _weight(seed=16).astype(np.float64)
        idx = np.array([0, 3, 3, 8, 1, 5], dtype=np.int64)
        offsets = np.array([0, 2, 4], dtype=np.int64)

        tw = _cu(w)
        tw.requires_grad_(True)
        out = _C._embedding_bag(tw, _cu(idx), _cu(offsets), False, mode, False,
                                None, False, -1)[0]
        out.sum().backward()

        rw = torch.tensor(w, requires_grad=True)
        torch.nn.functional.embedding_bag(
            torch.tensor(idx), rw, offsets=torch.tensor(offsets), mode=name
        ).sum().backward()
        _close(tw.grad, rw.grad.numpy(), rtol=1e-9, atol=1e-11,
               msg=f"cuda autograd {name}")

    def test_functional_surface(self):
        w = _weight(seed=17)
        idx = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        for name in ("sum", "mean", "max"):
            got = F.embedding_bag(_cu(idx), _cu(w), mode=name)
            want = torch.nn.functional.embedding_bag(
                torch.tensor(idx), torch.tensor(w), mode=name)
            _close(got, want.detach().numpy(), msg=f"cuda F.embedding_bag {name}")


# ---------------------------------------------------------------------------
# Rank-generic convolution
# ---------------------------------------------------------------------------

DIRECT = [
    ("1d", (2, 3, 9), (4, 3, 3), [2], [1], [1], [0], 1),
    ("2d", (2, 4, 7, 6), (6, 2, 3, 3), [1, 2], [1, 0], [1, 1], [0, 0], 2),
    ("3d", (1, 2, 5, 5, 5), (3, 2, 3, 3, 3), [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], 1),
]
TRANSPOSED = [
    ("1d", (2, 3, 5), (3, 4, 3), [2], [1], [1], [1], 1),
    ("2d", (2, 4, 5, 4), (4, 3, 3, 3), [2, 1], [1, 1], [1, 1], [1, 0], 2),
    ("3d", (1, 2, 3, 3, 3), (2, 2, 3, 3, 3), [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], 1),
]


def _rand(shape, seed):
    return np.random.RandomState(seed).randn(*shape).astype(np.float32)


class TestConvolutionCuda:
    @pytest.mark.parametrize("case", DIRECT, ids=[c[0] for c in DIRECT])
    def test_forward_direct(self, case):
        name, xs, ws, stride, padding, dilation, output_padding, groups = case
        x, w = _rand(xs, 20), _rand(ws, 21)
        b = _rand((ws[0],), 22)
        host = _C.convolution(_cpu(x), _cpu(w), _cpu(b), stride, padding, dilation,
                              False, output_padding, groups)
        dev = _C.convolution(_cu(x), _cu(w), _cu(b), stride, padding, dilation,
                             False, output_padding, groups)
        _close(dev, host, rtol=1e-4, atol=1e-4, msg=f"cuda convolution {name}")

    @pytest.mark.parametrize("case", TRANSPOSED, ids=[c[0] for c in TRANSPOSED])
    def test_forward_transposed(self, case):
        name, xs, ws, stride, padding, dilation, output_padding, groups = case
        x, w = _rand(xs, 23), _rand(ws, 24)
        b = _rand((ws[1] * groups,), 25)
        host = _C.convolution(_cpu(x), _cpu(w), _cpu(b), stride, padding, dilation,
                              True, output_padding, groups)
        dev = _C.convolution(_cu(x), _cu(w), _cu(b), stride, padding, dilation,
                             True, output_padding, groups)
        _close(dev, host, rtol=1e-4, atol=1e-4, msg=f"cuda convolution t{name}")

    @pytest.mark.parametrize("case", DIRECT, ids=[c[0] for c in DIRECT])
    def test_backward_direct(self, case):
        name, xs, ws, stride, padding, dilation, output_padding, groups = case
        x, w = _rand(xs, 26), _rand(ws, 27)
        b = _rand((ws[0],), 28)
        out = _C.convolution(_cpu(x), _cpu(w), _cpu(b), stride, padding, dilation,
                             False, output_padding, groups)
        grad = np.ones(tuple(out.shape), dtype=np.float32)
        args = (stride, padding, dilation, False, output_padding, groups,
                [True, True, True])
        host = _C.convolution_backward(_cpu(grad), _cpu(x), _cpu(w), [b.shape[0]], *args)
        dev = _C.convolution_backward(_cu(grad), _cu(x), _cu(w), [b.shape[0]], *args)
        for i, what in enumerate(("grad_input", "grad_weight", "grad_bias")):
            _close(dev[i], host[i], rtol=1e-4, atol=1e-4, msg=f"cuda {what} {name}")

    def test_backward_output_mask(self):
        x, w = _rand((1, 2, 5, 5), 29), _rand((3, 2, 3, 3), 30)
        grad = np.ones((1, 3, 3, 3), dtype=np.float32)
        args = ([1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        only_input = _C.convolution_backward(
            _cu(grad), _cu(x), _cu(w), None, *args, [True, False, False])
        assert only_input[0].defined()
        assert not only_input[1].defined()
        assert not only_input[2].defined()

    def test_overrideable_matches(self):
        x, w = _rand((1, 2, 5, 5), 31), _rand((3, 2, 3, 3), 32)
        b = _rand((3,), 33)
        args = ([1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        base = _C.convolution(_cu(x), _cu(w), _cu(b), *args)
        over = _C.convolution_overrideable(_cu(x), _cu(w), _cu(b), *args)
        _close(over, base, msg="cuda convolution_overrideable")

    def test_autograd(self):
        x, w = _rand((2, 4, 7, 6), 34).astype(np.float64), _rand((6, 2, 3, 3), 35).astype(np.float64)
        b = _rand((6,), 36).astype(np.float64)
        stride, padding, dilation, groups = [1, 2], [1, 0], [1, 1], 2

        tx, tw, tb = _cu(x), _cu(w), _cu(b)
        for t in (tx, tw, tb):
            t.requires_grad_(True)
        _C.convolution(tx, tw, tb, stride, padding, dilation, False, [0, 0], groups).sum().backward()

        rx = torch.tensor(x, requires_grad=True)
        rw = torch.tensor(w, requires_grad=True)
        rb = torch.tensor(b, requires_grad=True)
        torch.nn.functional.conv2d(rx, rw, rb, stride, padding, dilation, groups).sum().backward()
        _close(tx.grad, rx.grad.numpy(), rtol=1e-6, atol=1e-8, msg="cuda conv grad_input")
        _close(tw.grad, rw.grad.numpy(), rtol=1e-6, atol=1e-8, msg="cuda conv grad_weight")
        _close(tb.grad, rb.grad.numpy(), rtol=1e-6, atol=1e-8, msg="cuda conv grad_bias")
