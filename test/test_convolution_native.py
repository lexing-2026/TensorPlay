"""Rank-generic native convolution operators.

`convolution` and `convolution_backward` (plus the `*_overrideable`
extension-point spellings) route 1-D / 2-D / 3-D, direct and transposed
convolutions to the rank-specialized kernels.  These tests pin the routing,
the output-mask contract, and numerical agreement with the reference
framework installed in the test environment.
"""

import numpy as np
import pytest
import torch

import tensorplay as tp
import tensorplay._C as _C


def _mk(array):
    return tp.tensor(np.ascontiguousarray(array))


def _np(t):
    return np.asarray(t.tolist(), dtype=np.float64)


def _close(actual, expected, rtol=1e-9, atol=1e-9, msg=""):
    got = _np(actual)
    want = np.asarray(expected, dtype=np.float64)
    assert got.shape == want.shape, f"{msg}: shape {got.shape} != {want.shape}"
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=msg)


def _rand(shape, seed):
    return np.random.RandomState(seed).randn(*shape).astype(np.float64)


# (name, input shape, weight shape, stride, padding, dilation, output_padding, groups)
DIRECT_CASES = [
    ("1d", (2, 3, 9), (4, 3, 3), [2], [1], [1], [0], 1),
    ("2d", (2, 4, 7, 6), (6, 2, 3, 3), [1, 2], [1, 0], [1, 1], [0, 0], 2),
    ("2d-dilated", (1, 2, 9, 9), (3, 2, 3, 3), [1, 1], [2, 2], [2, 2], [0, 0], 1),
    ("3d", (1, 2, 5, 5, 5), (3, 2, 3, 3, 3), [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], 1),
]

TRANSPOSED_CASES = [
    ("1d", (2, 3, 5), (3, 4, 3), [2], [1], [1], [1], 1),
    ("2d", (2, 4, 5, 4), (4, 3, 3, 3), [2, 1], [1, 1], [1, 1], [1, 0], 2),
    ("3d", (1, 2, 3, 3, 3), (2, 2, 3, 3, 3), [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], 1),
]


def _torch_forward(x, w, b, stride, padding, dilation, transposed,
                   output_padding, groups):
    k = x.ndim - 2
    tx, tw = torch.tensor(x), torch.tensor(w)
    tb = None if b is None else torch.tensor(b)
    if not transposed:
        fn = (torch.nn.functional.conv1d, torch.nn.functional.conv2d,
              torch.nn.functional.conv3d)[k - 1]
        return fn(tx, tw, tb, stride, padding, dilation, groups)
    fn = (torch.nn.functional.conv_transpose1d,
          torch.nn.functional.conv_transpose2d,
          torch.nn.functional.conv_transpose3d)[k - 1]
    return fn(tx, tw, tb, stride, padding, output_padding, groups, dilation)


class TestConvolutionForward:
    @pytest.mark.parametrize("case", DIRECT_CASES, ids=[c[0] for c in DIRECT_CASES])
    def test_direct(self, case):
        name, xs, ws, stride, padding, dilation, output_padding, groups = case
        x, w = _rand(xs, 1), _rand(ws, 2)
        b = _rand((ws[0],), 3)

        got = _C.convolution(_mk(x), _mk(w), _mk(b), stride, padding, dilation,
                             False, output_padding, groups)
        want = _torch_forward(x, w, b, stride, padding, dilation, False,
                              output_padding, groups)
        _close(got, want.detach().numpy(), msg=f"convolution {name}")

    @pytest.mark.parametrize("case", TRANSPOSED_CASES,
                             ids=[c[0] for c in TRANSPOSED_CASES])
    def test_transposed(self, case):
        name, xs, ws, stride, padding, dilation, output_padding, groups = case
        x, w = _rand(xs, 4), _rand(ws, 5)
        b = _rand((ws[1] * groups,), 6)

        got = _C.convolution(_mk(x), _mk(w), _mk(b), stride, padding, dilation,
                             True, output_padding, groups)
        want = _torch_forward(x, w, b, stride, padding, dilation, True,
                              output_padding, groups)
        _close(got, want.detach().numpy(), msg=f"convolution transposed {name}")

    def test_without_bias(self):
        x, w = _rand((2, 3, 6, 6), 7), _rand((5, 3, 3, 3), 8)
        got = _C.convolution(_mk(x), _mk(w), None, [1, 1], [1, 1], [1, 1],
                             False, [0, 0], 1)
        want = _torch_forward(x, w, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        _close(got, want.detach().numpy(), msg="convolution without bias")

    def test_overrideable_matches_convolution(self):
        x, w = _rand((1, 2, 5, 5), 9), _rand((3, 2, 3, 3), 10)
        b = _rand((3,), 11)
        args = ([1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        base = _C.convolution(_mk(x), _mk(w), _mk(b), *args)
        over = _C.convolution_overrideable(_mk(x), _mk(w), _mk(b), *args)
        _close(over, _np(base), msg="convolution_overrideable")

    def test_rejects_unsupported_rank(self):
        x, w = _rand((2, 3), 12), _rand((4, 3), 13)
        with pytest.raises(Exception):
            _C.convolution(_mk(x), _mk(w), None, [1], [0], [1], False, [0], 1)

    def test_rejects_rank_mismatch(self):
        x, w = _rand((1, 2, 5, 5), 14), _rand((3, 2, 3), 15)
        with pytest.raises(Exception):
            _C.convolution(_mk(x), _mk(w), None, [1, 1], [0, 0], [1, 1], False,
                           [0, 0], 1)


def _torch_backward(x, w, b, stride, padding, dilation, transposed,
                    output_padding, groups):
    tx = torch.tensor(x, requires_grad=True)
    tw = torch.tensor(w, requires_grad=True)
    tb = torch.tensor(b, requires_grad=True)
    out = _torch_forward_tensors(tx, tw, tb, stride, padding, dilation,
                                 transposed, output_padding, groups)
    out.sum().backward()
    return tx.grad.numpy(), tw.grad.numpy(), tb.grad.numpy(), out.shape


def _torch_forward_tensors(tx, tw, tb, stride, padding, dilation, transposed,
                           output_padding, groups):
    k = tx.dim() - 2
    if not transposed:
        fn = (torch.nn.functional.conv1d, torch.nn.functional.conv2d,
              torch.nn.functional.conv3d)[k - 1]
        return fn(tx, tw, tb, stride, padding, dilation, groups)
    fn = (torch.nn.functional.conv_transpose1d,
          torch.nn.functional.conv_transpose2d,
          torch.nn.functional.conv_transpose3d)[k - 1]
    return fn(tx, tw, tb, stride, padding, output_padding, groups, dilation)


class TestConvolutionBackward:
    @pytest.mark.parametrize("case", DIRECT_CASES, ids=[c[0] for c in DIRECT_CASES])
    def test_direct(self, case):
        name, xs, ws, stride, padding, dilation, output_padding, groups = case
        x, w = _rand(xs, 16), _rand(ws, 17)
        b = _rand((ws[0],), 18)
        gi, gw, gb, out_shape = _torch_backward(
            x, w, b, stride, padding, dilation, False, output_padding, groups)
        grad = np.ones(tuple(out_shape), dtype=np.float64)

        got = _C.convolution_backward(
            _mk(grad), _mk(x), _mk(w), [b.shape[0]], stride, padding, dilation,
            False, output_padding, groups, [True, True, True])
        _close(got[0], gi, rtol=1e-8, atol=1e-8, msg=f"grad_input {name}")
        _close(got[1], gw, rtol=1e-8, atol=1e-8, msg=f"grad_weight {name}")
        _close(got[2], gb, rtol=1e-8, atol=1e-8, msg=f"grad_bias {name}")

    @pytest.mark.parametrize("case", TRANSPOSED_CASES,
                             ids=[c[0] for c in TRANSPOSED_CASES])
    def test_transposed(self, case):
        name, xs, ws, stride, padding, dilation, output_padding, groups = case
        x, w = _rand(xs, 19), _rand(ws, 20)
        b = _rand((ws[1] * groups,), 21)
        gi, gw, gb, out_shape = _torch_backward(
            x, w, b, stride, padding, dilation, True, output_padding, groups)
        grad = np.ones(tuple(out_shape), dtype=np.float64)

        got = _C.convolution_backward(
            _mk(grad), _mk(x), _mk(w), [b.shape[0]], stride, padding, dilation,
            True, output_padding, groups, [True, True, True])
        _close(got[0], gi, rtol=1e-8, atol=1e-8, msg=f"t-grad_input {name}")
        _close(got[1], gw, rtol=1e-8, atol=1e-8, msg=f"t-grad_weight {name}")
        _close(got[2], gb, rtol=1e-8, atol=1e-8, msg=f"t-grad_bias {name}")

    def test_output_mask_leaves_slots_undefined(self):
        x, w = _rand((1, 2, 5, 5), 22), _rand((3, 2, 3, 3), 23)
        grad = np.ones((1, 3, 3, 3), dtype=np.float64)
        args = ([1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        only_weight = _C.convolution_backward(
            _mk(grad), _mk(x), _mk(w), None, *args, [False, True, False])
        assert not only_weight[0].defined()
        assert only_weight[1].defined()
        assert not only_weight[2].defined()

        full = _C.convolution_backward(
            _mk(grad), _mk(x), _mk(w), None, *args, [True, True, True])
        _close(only_weight[1], _np(full[1]), msg="masked grad_weight")

    def test_rejects_mismatched_bias_sizes(self):
        x, w = _rand((1, 2, 5, 5), 24), _rand((3, 2, 3, 3), 25)
        grad = np.ones((1, 3, 3, 3), dtype=np.float64)
        with pytest.raises(Exception):
            _C.convolution_backward(
                _mk(grad), _mk(x), _mk(w), [7], [1, 1], [0, 0], [1, 1], False,
                [0, 0], 1, [False, False, True])

    def test_overrideable_matches_convolution_backward(self):
        x, w = _rand((1, 2, 5, 5), 26), _rand((3, 2, 3, 3), 27)
        grad = np.ones((1, 3, 3, 3), dtype=np.float64)
        args = ([1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
        base = _C.convolution_backward(_mk(grad), _mk(x), _mk(w), None, *args)
        over = _C.convolution_backward_overrideable(_mk(grad), _mk(x), _mk(w), *args)
        for i in range(3):
            _close(over[i], _np(base[i]), msg=f"overrideable slot {i}")


class TestConvolutionAutograd:
    @pytest.mark.parametrize("transposed", [False, True])
    def test_grads_match_reference(self, transposed):
        if transposed:
            xs, ws, stride, padding, dilation, output_padding, groups = \
                (2, 4, 5, 4), (4, 3, 3, 3), [2, 1], [1, 1], [1, 1], [1, 0], 2
            bias_len = ws[1] * groups
        else:
            xs, ws, stride, padding, dilation, output_padding, groups = \
                (2, 4, 7, 6), (6, 2, 3, 3), [1, 2], [1, 0], [1, 1], [0, 0], 2
            bias_len = ws[0]
        x, w = _rand(xs, 28), _rand(ws, 29)
        b = _rand((bias_len,), 30)

        tx, tw, tb = _mk(x), _mk(w), _mk(b)
        for t in (tx, tw, tb):
            t.requires_grad_(True)
        out = _C.convolution(tx, tw, tb, stride, padding, dilation, transposed,
                             output_padding, groups)
        out.sum().backward()

        gi, gw, gb, _ = _torch_backward(x, w, b, stride, padding, dilation,
                                        transposed, output_padding, groups)
        _close(tx.grad, gi, rtol=1e-8, atol=1e-8, msg="autograd grad_input")
        _close(tw.grad, gw, rtol=1e-8, atol=1e-8, msg="autograd grad_weight")
        _close(tb.grad, gb, rtol=1e-8, atol=1e-8, msg="autograd grad_bias")

    def test_overrideable_is_differentiable(self):
        x, w = _rand((1, 2, 5, 5), 31), _rand((3, 2, 3, 3), 32)
        tx, tw = _mk(x), _mk(w)
        tx.requires_grad_(True)
        tw.requires_grad_(True)
        out = _C.convolution_overrideable(tx, tw, None, [1, 1], [1, 1], [1, 1],
                                          False, [0, 0], 1)
        out.sum().backward()
        assert tx.grad is not None and tw.grad is not None
