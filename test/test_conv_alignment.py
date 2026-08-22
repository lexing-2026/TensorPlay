"""Conv-family alignment tests against PyTorch.

Every case compares forward values and (where differentiable) all gradients
against the installed torch, mirroring the alignment contract:
conv1d/2d/3d, conv_transpose1d/2d/3d, unfold/fold, pad modes, padding_mode
in nn.Conv*, Tensor.unfold and conv_tbc.
"""
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as TF

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.nn.functional as F

TOL = 1e-5
TOL64 = 1e-12


def _grads_torch(fn, *inputs):
    t_inputs = [x.clone().requires_grad_(True) if x.dtype.is_floating_point else x
                for x in inputs]
    out = fn(*t_inputs)
    grad = torch.randn_like(out)
    out.backward(grad)
    return [x.grad if x.dtype.is_floating_point else None for x in t_inputs], out.detach(), grad


def _grads_tp(fn, *inputs):
    t_inputs = [x.clone().requires_grad_(True) for x in inputs]
    out = fn(*t_inputs)
    grad = tp.randn(*out.shape)
    out.backward(grad)
    return [x.grad for x in t_inputs], out.detach(), grad


def _make(shape, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=dtype)


def _to_tp(t):
    return tp.tensor(t.detach().numpy())


def _to_torch(t):
    return torch.tensor(t.detach().numpy())


def _assert_close(a_tp, a_torch, msg, tol=TOL):
    a = a_tp.detach().numpy() if hasattr(a_tp, 'numpy') else np.asarray(a_tp)
    b = a_torch.detach().numpy()
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol, err_msg=msg)


class TestConvTranspose1d(unittest.TestCase):
    def test_forward_and_grads(self):
        x_t = _make((2, 4, 10))
        w_t = _make((4, 6, 3))
        b_t = _make((6,))
        ref = lambda x, w, b: TF.conv_transpose1d(x, w, b, stride=2, padding=1,
                                                  output_padding=1, groups=1)
        got = lambda x, w, b: F.conv_transpose1d(x, w, b, stride=2, padding=1,
                                                 output_padding=1, groups=1)
        grads_t, out_t, g_t = _grads_torch(ref, x_t, w_t, b_t)
        grads_p, out_p, g_p = _grads_tp(got, _to_tp(x_t), _to_tp(w_t), _to_tp(b_t))
        _assert_close(out_p, out_t, "conv_transpose1d forward")
        for i, name in enumerate(["input", "weight", "bias"]):
            _assert_close(grads_p[i], grads_t[i], f"conv_transpose1d grad {name}")

    def test_module(self):
        torch.manual_seed(3)
        layer = nn.ConvTranspose1d(4, 6, 3, stride=2, padding=1)
        layer.weight = nn.Parameter(_to_tp(torch.randn(4, 6, 3)))
        layer.bias = nn.Parameter(_to_tp(torch.randn(6)))
        ref_layer = torch.nn.ConvTranspose1d(4, 6, 3, stride=2, padding=1)
        ref_layer.weight.data = torch.tensor(layer.weight.detach().numpy())
        ref_layer.bias.data = torch.tensor(layer.bias.detach().numpy())
        x = _make((2, 4, 10))
        out = layer(_to_tp(x))
        _assert_close(out, ref_layer(x), "nn.ConvTranspose1d")


class TestUnfoldFold(unittest.TestCase):
    def test_unfold_forward_and_grad(self):
        x_t = _make((2, 3, 8, 8))
        ref = lambda x: TF.unfold(x, kernel_size=3, padding=1, stride=2, dilation=1)
        got = lambda x: F.unfold(x, kernel_size=3, padding=1, stride=2, dilation=1)
        grads_t, out_t, _ = _grads_torch(ref, x_t)
        grads_p, out_p, _ = _grads_tp(got, _to_tp(x_t))
        _assert_close(out_p, out_t, "F.unfold forward")
        _assert_close(grads_p[0], grads_t[0], "F.unfold backward")

    def test_fold_forward_and_grad(self):
        # fold input built from torch's own unfold so shapes are canonical
        x_img = _make((2, 3, 8, 8), seed=5)
        col = TF.unfold(x_img, kernel_size=3, padding=1, stride=2)
        ref = lambda c: TF.fold(c, output_size=(8, 8), kernel_size=3, padding=1, stride=2)
        got = lambda c: F.fold(c, output_size=(8, 8), kernel_size=3, padding=1, stride=2)
        grads_t, out_t, _ = _grads_torch(ref, col)
        grads_p, out_p, _ = _grads_tp(got, _to_tp(col))
        _assert_close(out_p, out_t, "F.fold forward")
        _assert_close(grads_p[0], grads_t[0], "F.fold backward")

    def test_unfold_modules(self):
        u = nn.Unfold(kernel_size=(2, 3))
        f = nn.Fold(output_size=(4, 5), kernel_size=(2, 2))
        x = _make((2, 5, 3, 4))
        out = u(_to_tp(x))
        self.assertEqual(tuple(out.shape), (2, 30, 4))
        rec = f(tp.randn(2, 20, 6))
        self.assertEqual(tuple(rec.shape), (2, 5, 4, 5))


class TestPadModes(unittest.TestCase):
    def _check(self, shape, pad, mode, seed=1):
        x_t = _make(shape, seed=seed)
        ref = lambda x: TF.pad(x, pad, mode=mode)
        got = lambda x: F.pad(x, pad, mode=mode)
        grads_t, out_t, _ = _grads_torch(ref, x_t)
        grads_p, out_p, _ = _grads_tp(got, _to_tp(x_t))
        _assert_close(out_p, out_t, f"F.pad {mode} forward {shape} {pad}")
        _assert_close(grads_p[0], grads_t[0], f"F.pad {mode} backward {shape} {pad}")

    def test_reflect(self):
        self._check((2, 3, 6, 7), [1, 2, 0, 1], 'reflect')
        self._check((2, 3, 5), [2, 1], 'reflect')

    def test_replicate(self):
        self._check((2, 3, 6, 7), [1, 2, 3, 0], 'replicate')
        self._check((2, 3, 5), [4, 2], 'replicate')

    def test_circular(self):
        self._check((2, 3, 6, 7), [1, 2, 0, 1], 'circular')
        self._check((2, 3, 6, 7), [6, 6, 1, 1], 'circular')  # one full wrap allowed

    def test_reflect_pad_too_large_raises(self):
        x = _to_tp(_make((2, 3, 3)))
        with self.assertRaises(Exception):
            F.pad(x, [3, 1], mode='reflect')

    def test_circular_wrap_twice_raises(self):
        x = _to_tp(_make((2, 3, 3)))
        with self.assertRaises(Exception):
            F.pad(x, [4, 0], mode='circular')


class TestConvPaddingMode(unittest.TestCase):
    def _check(self, module_cls, torch_cls, shape, kernel, mode, seed=2):
        torch.manual_seed(seed)
        k = dict(kernel_size=kernel, stride=1, padding=kernel // 2,
                 padding_mode=mode, bias=True)
        layer = module_cls(shape[1], 5, **k)
        ref = torch_cls(shape[1], 5, **k)
        ref.weight.data = torch.tensor(layer.weight.detach().numpy())
        ref.bias.data = torch.tensor(layer.bias.detach().numpy())
        x = _make(shape)
        out = layer(_to_tp(x))
        _assert_close(out, ref(x), f"{module_cls.__name__} padding_mode={mode}")

    def test_conv2d_modes(self):
        for mode in ('reflect', 'replicate', 'circular'):
            self._check(nn.Conv2d, torch.nn.Conv2d, (2, 3, 7, 8), 3, mode)

    def test_conv1d_modes(self):
        for mode in ('reflect', 'replicate', 'circular'):
            self._check(nn.Conv1d, torch.nn.Conv1d, (2, 3, 9), 3, mode)

    def test_conv3d_reflect(self):
        self._check(nn.Conv3d, torch.nn.Conv3d, (1, 2, 4, 5, 6), 3, 'replicate')

    def test_transpose_rejects_non_zeros(self):
        with self.assertRaises(ValueError):
            nn.ConvTranspose2d(3, 5, 3, padding_mode='reflect')


class TestTensorUnfoldView(unittest.TestCase):
    def test_matches_torch(self):
        x = torch.arange(1., 25.).reshape(2, 3, 4)
        ref = x.unfold(1, 2, 1)
        got = _to_tp(x).unfold(1, 2, 1)
        self.assertEqual(tuple(got.shape), tuple(ref.shape))
        _assert_close(got, ref, "Tensor.unfold view")

    def test_step(self):
        x = torch.arange(0., 12.).reshape(3, 4)
        ref = x.unfold(-1, 3, 2)
        got = _to_tp(x).unfold(-1, 3, 2)
        self.assertEqual(tuple(got.shape), tuple(ref.shape))
        _assert_close(got, ref, "Tensor.unfold negative dim + step")


class TestConvTbc(unittest.TestCase):
    def test_matches_torch(self):
        x_t = _make((6, 2, 3))       # (T, B, C)
        w_t = _make((3, 3))          # (k, C)
        b_t = _make((3,))
        ref = torch.conv_tbc(x_t, w_t, b_t, 2)
        got = F.conv_tbc(_to_tp(x_t), _to_tp(w_t), _to_tp(b_t), 2)
        _assert_close(got, ref, "conv_tbc")


class TestLazyConv(unittest.TestCase):
    def test_lazy_conv2d_becomes_conv2d(self):
        layer = nn.LazyConv2d(6, 3, padding=1)
        self.assertTrue(nn.parameter.is_lazy(layer.weight))
        out = layer(_to_tp(_make((2, 4, 8, 8), seed=7)))
        self.assertEqual(tuple(out.shape), (2, 6, 8, 8))
        self.assertFalse(nn.parameter.is_lazy(layer.weight))
        self.assertEqual(layer.weight.shape, (6, 4, 3, 3))

    def test_lazy_conv_transpose2d(self):
        layer = nn.LazyConvTranspose2d(6, 3, stride=2)
        out = layer(_to_tp(_make((2, 4, 4, 4), seed=8)))
        self.assertEqual(tuple(out.shape), (2, 6, 9, 9))
        self.assertEqual(layer.weight.shape, (4, 6, 3, 3))


class TestFloat64Conv(unittest.TestCase):
    def test_conv2d_double(self):
        x_t = _make((2, 3, 6, 6), dtype=torch.float64, seed=9)
        w_t = _make((4, 3, 3, 3), dtype=torch.float64, seed=10)
        b_t = _make((4,), dtype=torch.float64, seed=11)
        ref = lambda x, w, b: TF.conv2d(x, w, b, stride=1, padding=1)
        got = lambda x, w, b: F.conv2d(x, w, b, stride=1, padding=1)
        grads_t, out_t, _ = _grads_torch(ref, x_t, w_t, b_t)
        grads_p, out_p, _ = _grads_tp(got, _to_tp(x_t), _to_tp(w_t), _to_tp(b_t))
        _assert_close(out_p, out_t, "conv2d f64 forward", tol=TOL64)
        for i, name in enumerate(["input", "weight", "bias"]):
            _assert_close(grads_p[i], grads_t[i], f"conv2d f64 grad {name}", tol=TOL64)

    def test_conv3d_double(self):
        x_t = _make((1, 2, 4, 5, 5), dtype=torch.float64, seed=12)
        w_t = _make((3, 2, 3, 3, 3), dtype=torch.float64, seed=13)
        ref = lambda x, w: TF.conv3d(x, w, None, stride=1, padding=1)
        got = lambda x, w: F.conv3d(x, w, None, stride=1, padding=1)
        grads_t, out_t, _ = _grads_torch(ref, x_t, w_t)
        grads_p, out_p, _ = _grads_tp(got, _to_tp(x_t), _to_tp(w_t))
        _assert_close(out_p, out_t, "conv3d f64 forward", tol=TOL64)
        _assert_close(grads_p[0], grads_t[0], "conv3d f64 grad input", tol=TOL64)
        _assert_close(grads_p[1], grads_t[1], "conv3d f64 grad weight", tol=TOL64)

    def test_conv_transpose2d_double(self):
        x_t = _make((2, 3, 4, 4), dtype=torch.float64, seed=14)
        w_t = _make((3, 4, 3, 3), dtype=torch.float64, seed=15)
        ref = TF.conv_transpose2d(x_t, w_t, None, stride=2, padding=1, output_padding=1)
        got = F.conv_transpose2d(_to_tp(x_t), _to_tp(w_t), None, stride=2, padding=1,
                                 output_padding=1)
        _assert_close(got, ref, "conv_transpose2d f64", tol=TOL64)


class TestLowPrecisionCPU(unittest.TestCase):
    def _check(self, fwd_tp, fwd_torch, inputs, tol):
        outs_t = [fwd_torch(*[torch.tensor(x.detach().numpy()) for x in inputs])]
        outs_p = [fwd_tp(*inputs)]
        _assert_close(outs_p[0], outs_t[0], "low-precision conv", tol=tol)

    def test_conv2d_half(self):
        x = torch.randn(2, 3, 6, 6).half()
        w = torch.randn(4, 3, 3, 3).half()
        ref = TF.conv2d(x, w, None, padding=1)
        got = F.conv2d(_to_tp(x), _to_tp(w), None, padding=1)
        self.assertEqual(got.dtype, tp.float16)
        _assert_close(got.float(), ref.float(), "conv2d fp16", tol=2e-2)

    def test_conv2d_bf16(self):
        x = torch.randn(2, 3, 6, 6).bfloat16()
        w = torch.randn(4, 3, 3, 3).bfloat16()
        ref = TF.conv2d(x, w, None, padding=1)
        got = F.conv2d(_to_tp(x), _to_tp(w), None, padding=1)
        self.assertEqual(got.dtype, tp.bfloat16)
        _assert_close(got.float(), ref.float(), "conv2d bf16", tol=2e-1)


class TestConv3dScratchpadRegression(unittest.TestCase):
    """The pre-fix conv3d corrupted the heap via the missing oneDNN scratchpad;
    several consecutive fwd+bwd iterations must survive."""

    def test_repeated_conv3d_training_steps(self):
        x = _to_tp(_make((2, 3, 5, 8, 8), seed=21))
        layer = nn.Conv3d(3, 4, 3, padding=1)
        opt = tp.optim.SGD(layer.parameters(), lr=0.01) if hasattr(tp, 'optim') else None
        for step in range(5):
            out = layer(x)
            loss = (out * out).mean()
            loss.backward()
            if opt is not None:
                opt.step()
                opt.zero_grad()
            else:
                layer.zero_grad()
        self.assertEqual(tuple(out.shape), (2, 4, 5, 8, 8))


if __name__ == '__main__':
    unittest.main()
