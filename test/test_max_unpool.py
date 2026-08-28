"""Native max_unpool2d / max_unpool3d alignment vs torch.

Covers the max_unpool family added natively to close the gap against ATen
(MaxUnpooling.cpp / cpu/MaxUnpoolKernel.cpp / cuda/MaxUnpooling.cu): forward
scatter at the flat in-plane indices recorded by max_pool*_with_indices,
backward gather through autograd, direct native-op calls, round trips through
F.max_unpool{1,2,3}d, and batched/unbatched inputs.
"""
import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as torch_F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay import Tensor


def _np(t):
    return t.detach().cpu().numpy()


def _assert_close(case, tp_t, torch_t, rtol=1e-5, atol=1e-6, msg=""):
    np.testing.assert_allclose(_np(tp_t), _np(torch_t), rtol=rtol, atol=atol,
                               err_msg=msg)


def _devices():
    devs = ["cpu"]
    if tp.cuda.is_available():
        devs.append("cuda")
    return devs


def _tp_tensor(torch_t, device, requires_grad=False):
    t = tp.tensor(torch_t.detach().numpy(), device=device)
    if requires_grad:
        t = t.requires_grad_(True)
    return t


class TestMaxUnpool2d(unittest.TestCase):
    def _run(self, shape, kernel_size, stride, padding, dev, seed):
        torch.manual_seed(seed)
        input_t = torch.randn(*shape)
        g_t = None

        input_ref = input_t.clone().requires_grad_(True)
        ref_pool, ref_idx = torch_F.max_pool2d(
            input_ref, kernel_size, stride=stride, padding=padding,
            return_indices=True)
        out_size = list(input_t.shape[-2:])
        ref_unpool = torch_F.max_unpool2d(
            ref_pool, ref_idx, kernel_size, stride=stride, padding=padding,
            output_size=out_size)
        g_t = torch.randn_like(ref_unpool)
        ref_unpool.backward(g_t)

        x = _tp_tensor(input_t, dev, requires_grad=True)
        pool, idx = F.max_pool2d(
            x, kernel_size, stride=stride, padding=padding, return_indices=True)
        unpool = F.max_unpool2d(
            pool, idx, kernel_size, stride=stride, padding=padding,
            output_size=out_size)
        unpool.backward(_tp_tensor(g_t, dev))

        tag = f"shape={shape} k={kernel_size} s={stride} p={padding} ({dev})"
        _assert_close(self, pool, ref_pool, msg=f"pool values {tag}")
        np.testing.assert_array_equal(_np(idx), _np(ref_idx),
                                      err_msg=f"pool indices {tag}")
        _assert_close(self, unpool, ref_unpool, msg=f"unpool values {tag}")
        _assert_close(self, x.grad, input_ref.grad, msg=f"unpool grad {tag}")

    def test_configs(self):
        for dev in _devices():
            self._run((2, 3, 8, 8), 2, None, 0, dev, 0)
            self._run((2, 3, 9, 7), 2, 2, 0, dev, 1)
            self._run((1, 4, 10, 10), 3, 2, 1, dev, 2)
            self._run((3, 8, 8), 2, None, 0, dev, 3)          # unbatched
            self._run((2, 2, 5, 5), (3, 2), (2, 1), 0, dev, 4)

    def test_native_op_direct(self):
        from tensorplay import _C
        torch.manual_seed(5)
        input_t = torch.randn(2, 3, 8, 8)
        for dev in _devices():
            pool_t, idx_t = torch_F.max_pool2d(input_t, 2, return_indices=True)
            x = _tp_tensor(pool_t, dev)
            ix = _tp_tensor(idx_t, dev)
            out = _C.max_unpool2d(x, ix, [8, 8])
            ref = torch._C._nn.max_unpool2d(pool_t, idx_t, [8, 8])
            _assert_close(self, out, ref, msg=f"native max_unpool2d fwd ({dev})")

            g_t = torch.randn_like(ref)
            gi = _C.max_unpool2d_backward(_tp_tensor(g_t, dev), ix, [8, 8])
            pool_ref = pool_t.clone().requires_grad_(True)
            torch._C._nn.max_unpool2d(pool_ref, idx_t, [8, 8]).backward(g_t)
            _assert_close(self, gi, pool_ref.grad, msg=f"native max_unpool2d bwd ({dev})")

    def test_round_trip(self):
        # unpool(pool(x)) places each window max back at its argmax position;
        # every pooled value must survive the round trip at its index.
        torch.manual_seed(6)
        input_t = torch.randn(2, 2, 7, 9)
        for dev in _devices():
            pool_t, idx_t = torch_F.max_pool2d(input_t, 2, return_indices=True)
            out = F.max_unpool2d(_tp_tensor(pool_t, dev), _tp_tensor(idx_t, dev), 2,
                                 output_size=list(input_t.shape[-2:]))
            ref = torch_F.max_unpool2d(pool_t, idx_t, 2,
                                       output_size=list(input_t.shape[-2:]))
            _assert_close(self, out, ref, msg=f"round trip values ({dev})")
            flat_in = input_t.reshape(-1)
            for b in range(2):
                for c in range(2):
                    p = idx_t[b, c].reshape(-1)
                    v = pool_t[b, c].reshape(-1)
                    for i in range(p.numel()):
                        self.assertEqual(
                            float(out[b, c].reshape(-1)[int(p[i])].item()),
                            float(v[i]),
                            msg=f"round trip placement b={b} c={c} i={i} ({dev})")

    def test_module_smoke(self):
        torch.manual_seed(7)
        input_t = torch.randn(2, 2, 8, 8)
        for dev in _devices():
            pool = tp.nn.MaxPool2d(2, return_indices=True)
            unpool = tp.nn.MaxUnpool2d(2)
            x = _tp_tensor(input_t, dev, requires_grad=True)
            pooled, idx = pool(x)
            out = unpool(pooled, idx)
            self.assertEqual(tuple(out.shape), (2, 2, 8, 8))
            out.backward(_tp_tensor(torch.randn(2, 2, 8, 8), dev))
            self.assertIsNotNone(x.grad)
            self.assertEqual(tuple(x.grad.shape), tuple(input_t.shape))


class TestMaxUnpool1d(unittest.TestCase):
    def test_configs(self):
        for dev in _devices():
            for shape, k, stride, padding, seed in [
                    ((2, 3, 10), 2, None, 0, 20),
                    ((2, 2, 11), 3, 2, 1, 21),
                    ((4, 9), 2, None, 0, 22),           # unbatched
            ]:
                torch.manual_seed(seed)
                input_t = torch.randn(*shape)
                input_ref = input_t.clone().requires_grad_(True)
                ref_pool, ref_idx = torch_F.max_pool1d(
                    input_ref, k, stride=stride, padding=padding,
                    return_indices=True)
                out_size = list(input_t.shape[-1:])
                ref_unpool = torch_F.max_unpool1d(
                    ref_pool, ref_idx, k, stride=stride, padding=padding,
                    output_size=out_size)
                g_t = torch.randn_like(ref_unpool)
                ref_unpool.backward(g_t)

                x = _tp_tensor(input_t, dev, requires_grad=True)
                pool, idx = F.max_pool1d(
                    x, k, stride=stride, padding=padding, return_indices=True)
                unpool = F.max_unpool1d(
                    pool, idx, k, stride=stride, padding=padding,
                    output_size=out_size)
                unpool.backward(_tp_tensor(g_t, dev))

                tag = f"shape={shape} k={k} s={stride} p={padding} ({dev})"
                _assert_close(self, unpool, ref_unpool, msg=f"unpool1d values {tag}")
                _assert_close(self, x.grad, input_ref.grad, msg=f"unpool1d grad {tag}")


class TestMaxUnpool3d(unittest.TestCase):
    def _run(self, shape, kernel_size, stride, padding, dev, seed):
        torch.manual_seed(seed)
        input_t = torch.randn(*shape)

        input_ref = input_t.clone().requires_grad_(True)
        ref_pool, ref_idx = torch_F.max_pool3d(
            input_ref, kernel_size, stride=stride, padding=padding,
            return_indices=True)
        out_size = list(input_t.shape[-3:])
        ref_unpool = torch_F.max_unpool3d(
            ref_pool, ref_idx, kernel_size, stride=stride, padding=padding,
            output_size=out_size)
        g_t = torch.randn_like(ref_unpool)
        ref_unpool.backward(g_t)

        x = _tp_tensor(input_t, dev, requires_grad=True)
        pool, idx = F.max_pool3d(
            x, kernel_size, stride=stride, padding=padding, return_indices=True)
        unpool = F.max_unpool3d(
            pool, idx, kernel_size, stride=stride, padding=padding,
            output_size=out_size)
        unpool.backward(_tp_tensor(g_t, dev))

        tag = f"shape={shape} k={kernel_size} s={stride} p={padding} ({dev})"
        _assert_close(self, pool, ref_pool, msg=f"pool values {tag}")
        np.testing.assert_array_equal(_np(idx), _np(ref_idx),
                                      err_msg=f"pool indices {tag}")
        _assert_close(self, unpool, ref_unpool, msg=f"unpool values {tag}")
        _assert_close(self, x.grad, input_ref.grad, msg=f"unpool grad {tag}")

    def test_configs(self):
        for dev in _devices():
            self._run((2, 2, 6, 6, 6), 2, None, 0, dev, 10)
            self._run((1, 3, 7, 5, 5), 2, 2, 0, dev, 11)
            self._run((2, 6, 6, 6), 2, None, 0, dev, 12)        # unbatched
            self._run((1, 2, 5, 7, 6), (2, 3, 2), (1, 2, 1), 0, dev, 13)

    def test_native_op_direct(self):
        from tensorplay import _C
        torch.manual_seed(14)
        input_t = torch.randn(2, 2, 4, 6, 6)
        for dev in _devices():
            pool_t, idx_t = torch_F.max_pool3d(input_t, 2, return_indices=True)
            x = _tp_tensor(pool_t, dev)
            ix = _tp_tensor(idx_t, dev)
            out = _C.max_unpool3d(x, ix, [4, 6, 6], [2, 2, 2], [0, 0, 0])
            ref = torch._C._nn.max_unpool3d(
                pool_t, idx_t, [4, 6, 6], [2, 2, 2], [0, 0, 0])
            _assert_close(self, out, ref, msg=f"native max_unpool3d fwd ({dev})")

            g_t = torch.randn_like(ref)
            gi = _C.max_unpool3d_backward(_tp_tensor(g_t, dev), ix, [4, 6, 6])
            pool_ref = pool_t.clone().requires_grad_(True)
            torch._C._nn.max_unpool3d(
                pool_ref, idx_t, [4, 6, 6], [2, 2, 2], [0, 0, 0]).backward(g_t)
            _assert_close(self, gi, pool_ref.grad, msg=f"native max_unpool3d bwd ({dev})")

    def test_module_smoke(self):
        torch.manual_seed(15)
        input_t = torch.randn(1, 2, 4, 4, 4)
        for dev in _devices():
            pool = tp.nn.MaxPool3d(2, return_indices=True)
            unpool = tp.nn.MaxUnpool3d(2)
            x = _tp_tensor(input_t, dev, requires_grad=True)
            pooled, idx = pool(x)
            out = unpool(pooled, idx)
            self.assertEqual(tuple(out.shape), (1, 2, 4, 4, 4))
            out.backward(_tp_tensor(torch.randn(1, 2, 4, 4, 4), dev))
            self.assertIsNotNone(x.grad)
            self.assertEqual(tuple(x.grad.shape), tuple(input_t.shape))


if __name__ == "__main__":
    unittest.main(verbosity=2)
