"""

LossMultiLabelMargin.cpp, cuda/MultiMarginLoss.cu /
MultiLabelMarginCriterion.cu): forward with weight + none/mean/sum
reductions, batched and unbatched inputs, the multilabel is_target mask,
native backward ops, autograd through tensorplay.nn.functional, and the
nn module smoke tests. All backwards use explicit grads (no .sum().backward())
so the suite is immune to unrelated reduction regressions.
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


_REDUCTION_ENUM = {"none": 0, "mean": 1, "sum": 2}


class TestMultiMarginLossForward(unittest.TestCase):
    def _run(self, shape, target_shape, p, margin, weighted, reduction, dtype,
             dev, seed):
        torch.manual_seed(seed)
        input_t = torch.randn(*shape, dtype=dtype)
        C = shape[-1]
        target_t = torch.randint(0, C, target_shape)
        weight_t = torch.rand(C, dtype=dtype) if weighted else None

        ref = torch_F.multi_margin_loss(
            input_t, target_t, p=p, margin=margin, weight=weight_t,
            reduction=reduction)

        x = _tp_tensor(input_t, dev)
        t = _tp_tensor(target_t, dev)
        w = _tp_tensor(weight_t, dev) if weighted else None
        got = F.multi_margin_loss(x, t, p=p, margin=margin, weight=w,
                                  reduction=reduction)

        tag = (f"shape={shape} target={target_shape} p={p} margin={margin} "
               f"weighted={weighted} reduction={reduction} dtype={dtype} ({dev})")
        _assert_close(self, got, ref, msg=f"multi_margin_loss {tag}")

    def test_configs(self):
        for dev in _devices():
            for dtype in (torch.float32, torch.float64):
                for reduction in ("mean", "sum", "none"):
                    for p in (1, 2):
                        self._run((4, 6), (4,), p, 1.0, False, reduction,
                                  dtype, dev, 1)
                self._run((4, 6), (4,), 1, 0.7, True, "mean", dtype, dev, 2)
                self._run((4, 6), (4,), 2, 1.5, True, "sum", dtype, dev, 3)
                self._run((3, 5), (3,), 1, 1.0, False, "none", dtype, dev, 4)
            # unbatched: 1-D input with 0-dim and 1-element targets
            for tshape in ((), (1,)):
                for reduction in ("mean", "none"):
                    self._run((5,), tshape, 1, 1.0, False, reduction,
                              torch.float32, dev, 5)
                    self._run((5,), tshape, 2, 0.5, True, reduction,
                              torch.float32, dev, 6)

    def test_native_op_direct(self):
        from tensorplay import _C
        for dev in _devices():
            torch.manual_seed(7)
            input_t = torch.randn(4, 7)
            target_t = torch.randint(0, 7, (4,))
            weight_t = torch.rand(7)
            for p in (1, 2):
                for reduction in ("none", "mean", "sum"):
                    ren = _REDUCTION_ENUM[reduction]
                    ref = torch.ops.aten.multi_margin_loss(
                        input_t, target_t, p, 1.2, weight_t, ren)
                    got = _C.multi_margin_loss(
                        _tp_tensor(input_t, dev), _tp_tensor(target_t, dev),
                        p, 1.2, _tp_tensor(weight_t, dev), ren)
                    tag = f"native p={p} reduction={reduction} ({dev})"
                    _assert_close(self, got, ref, msg=tag)


class TestMultiMarginLossBackward(unittest.TestCase):
    def _run(self, shape, target_shape, p, margin, weighted, reduction, dev,
             seed):
        torch.manual_seed(seed)
        input_t = torch.randn(*shape)
        C = shape[-1]
        target_t = torch.randint(0, C, target_shape)
        weight_t = torch.rand(C) if weighted else None

        ref_in = input_t.clone().requires_grad_(True)
        ref_out = torch_F.multi_margin_loss(
            ref_in, target_t, p=p, margin=margin, weight=weight_t,
            reduction=reduction)
        g_t = torch.randn_like(ref_out)
        (ref_grad,) = torch.autograd.grad(ref_out, ref_in, grad_outputs=g_t)

        x = _tp_tensor(input_t, dev, requires_grad=True)
        t = _tp_tensor(target_t, dev)
        w = _tp_tensor(weight_t, dev) if weighted else None
        out = F.multi_margin_loss(x, t, p=p, margin=margin, weight=w,
                                  reduction=reduction)
        out.backward(_tp_tensor(g_t, dev))

        tag = (f"shape={shape} target={target_shape} p={p} margin={margin} "
               f"weighted={weighted} reduction={reduction} ({dev})")
        _assert_close(self, out, ref_out, msg=f"multi_margin_loss fwd {tag}")
        _assert_close(self, x.grad, ref_grad, msg=f"multi_margin_loss grad {tag}")

    def test_configs(self):
        for dev in _devices():
            for reduction in ("mean", "sum", "none"):
                for p in (1, 2):
                    self._run((4, 6), (4,), p, 1.0, False, reduction, dev, 11)
            self._run((4, 6), (4,), 1, 0.7, True, "mean", dev, 12)
            self._run((4, 6), (4,), 2, 1.5, True, "none", dev, 13)
            for tshape in ((), (1,)):
                self._run((5,), tshape, 1, 1.0, False, "mean", dev, 14)
                self._run((5,), tshape, 2, 0.5, True, "none", dev, 15)

    def test_native_backward_direct(self):
        from tensorplay import _C
        for dev in _devices():
            torch.manual_seed(16)
            input_t = torch.randn(3, 6)
            target_t = torch.randint(0, 6, (3,))
            weight_t = torch.rand(6)
            for p in (1, 2):
                for reduction in ("none", "mean", "sum"):
                    ren = _REDUCTION_ENUM[reduction]
                    ref_out = torch.ops.aten.multi_margin_loss(
                        input_t, target_t, p, 0.8, weight_t, ren)
                    g_t = torch.randn_like(ref_out)
                    ref_grad = torch.ops.aten.multi_margin_loss_backward(
                        g_t, input_t, target_t, p, 0.8, weight_t, ren)
                    got = _C.multi_margin_loss_backward(
                        _tp_tensor(g_t, dev), _tp_tensor(input_t, dev),
                        _tp_tensor(target_t, dev), p, 0.8,
                        _tp_tensor(weight_t, dev), ren)
                    tag = f"native bwd p={p} reduction={reduction} ({dev})"
                    _assert_close(self, got, ref_grad, msg=tag)


class TestMultilabelMarginLossForward(unittest.TestCase):
    def _targets(self, nframe, dim, seed):
        g = torch.Generator().manual_seed(seed)
        t = torch.full((nframe, dim), -1, dtype=torch.int64)
        for i in range(nframe):
            k = int(torch.randint(0, dim + 1, (1,), generator=g).item())
            perm = torch.randperm(dim, generator=g)[:k]
            t[i, :k] = perm
        return t

    def _run(self, shape, reduction, dev, seed):
        torch.manual_seed(seed)
        input_t = torch.randn(*shape)
        C = shape[-1]
        if len(shape) == 2:
            target_t = self._targets(shape[0], C, seed + 100)
        else:
            target_t = self._targets(1, C, seed + 100).reshape(-1)

        ref = torch_F.multilabel_margin_loss(input_t, target_t,
                                             reduction=reduction)

        x = _tp_tensor(input_t, dev)
        t = _tp_tensor(target_t, dev)
        got = F.multilabel_margin_loss(x, t, reduction=reduction)

        tag = f"shape={shape} reduction={reduction} ({dev})"
        _assert_close(self, got, ref, msg=f"multilabel_margin_loss {tag}")

    def test_configs(self):
        for dev in _devices():
            for reduction in ("mean", "sum", "none"):
                self._run((4, 6), reduction, dev, 21)
                self._run((3, 5), reduction, dev, 22)
                self._run((6,), reduction, dev, 23)
            # rows with no active targets contribute zero
            input_t = torch.randn(2, 4)
            target_t = torch.full((2, 4), -1, dtype=torch.int64)
            ref = torch_F.multilabel_margin_loss(input_t, target_t,
                                                 reduction="mean")
            got = F.multilabel_margin_loss(_tp_tensor(input_t, dev),
                                           _tp_tensor(target_t, dev),
                                           reduction="mean")
            _assert_close(self, got, ref, msg=f"all -1 targets ({dev})")

    def test_native_forward_is_target(self):
        from tensorplay import _C
        for dev in _devices():
            torch.manual_seed(24)
            input_t = torch.randn(4, 7)
            target_t = self._targets(4, 7, 124)
            for reduction in ("none", "mean", "sum"):
                ren = _REDUCTION_ENUM[reduction]
                ref_out, ref_is = torch.ops.aten.multilabel_margin_loss_forward(
                    input_t, target_t, ren)
                got_out, got_is = _C.multilabel_margin_loss_forward(
                    _tp_tensor(input_t, dev), _tp_tensor(target_t, dev), ren)
                tag = f"native fwd reduction={reduction} ({dev})"
                _assert_close(self, got_out, ref_out, msg=tag)
                _assert_close(self, got_is, ref_is, msg=f"is_target {tag}")
            # unbatched
            x1 = torch.randn(5)
            t1 = torch.tensor([0, 2, -1, -1, -1])
            ref_out, ref_is = torch.ops.aten.multilabel_margin_loss_forward(
                x1, t1, 1)
            got_out, got_is = _C.multilabel_margin_loss_forward(
                _tp_tensor(x1, dev), _tp_tensor(t1, dev), 1)
            _assert_close(self, got_out, ref_out, msg=f"native fwd 1d ({dev})")
            _assert_close(self, got_is, ref_is, msg=f"native fwd 1d is_target ({dev})")


class TestMultilabelMarginLossBackward(unittest.TestCase):
    def _targets(self, nframe, dim, seed):
        g = torch.Generator().manual_seed(seed)
        t = torch.full((nframe, dim), -1, dtype=torch.int64)
        for i in range(nframe):
            k = int(torch.randint(0, dim + 1, (1,), generator=g).item())
            perm = torch.randperm(dim, generator=g)[:k]
            t[i, :k] = perm
        return t

    def _run(self, shape, reduction, dev, seed):
        torch.manual_seed(seed)
        input_t = torch.randn(*shape)
        C = shape[-1]
        if len(shape) == 2:
            target_t = self._targets(shape[0], C, seed + 200)
        else:
            target_t = self._targets(1, C, seed + 200).reshape(-1)

        ref_in = input_t.clone().requires_grad_(True)
        ref_out = torch_F.multilabel_margin_loss(ref_in, target_t,
                                                 reduction=reduction)
        g_t = torch.randn_like(ref_out)
        (ref_grad,) = torch.autograd.grad(ref_out, ref_in, grad_outputs=g_t)

        x = _tp_tensor(input_t, dev, requires_grad=True)
        t = _tp_tensor(target_t, dev)
        out = F.multilabel_margin_loss(x, t, reduction=reduction)
        out.backward(_tp_tensor(g_t, dev))

        tag = f"shape={shape} reduction={reduction} ({dev})"
        _assert_close(self, out, ref_out, msg=f"multilabel fwd {tag}")
        _assert_close(self, x.grad, ref_grad, msg=f"multilabel grad {tag}")

    def test_configs(self):
        for dev in _devices():
            for reduction in ("mean", "sum", "none"):
                self._run((4, 6), reduction, dev, 31)
                self._run((3, 5), reduction, dev, 32)
                self._run((6,), reduction, dev, 33)

    def test_native_backward_direct(self):
        from tensorplay import _C
        for dev in _devices():
            torch.manual_seed(34)
            input_t = torch.randn(3, 6)
            target_t = self._targets(3, 6, 234)
            for reduction in ("none", "mean", "sum"):
                ren = _REDUCTION_ENUM[reduction]
                ref_out, ref_is = torch.ops.aten.multilabel_margin_loss_forward(
                    input_t, target_t, ren)
                g_t = torch.randn_like(ref_out)
                ref_grad = torch.ops.aten.multilabel_margin_loss_backward(
                    g_t, input_t, target_t, ren, ref_is)
                got = _C.multilabel_margin_loss_backward(
                    _tp_tensor(g_t, dev), _tp_tensor(input_t, dev),
                    _tp_tensor(target_t, dev), ren, _tp_tensor(ref_is, dev))
                tag = f"native bwd reduction={reduction} ({dev})"
                _assert_close(self, got, ref_grad, msg=tag)


class TestModuleSmoke(unittest.TestCase):
    def test_modules(self):
        for dev in _devices():
            torch.manual_seed(41)
            input_t = torch.randn(4, 6)
            target_t = torch.randint(0, 6, (4,))

            ref_in = input_t.clone().requires_grad_(True)
            ref_mod = torch.nn.MultiMarginLoss(margin=0.8)
            ref_out = ref_mod(ref_in, target_t)
            g_t = torch.randn_like(ref_out)
            (ref_grad,) = torch.autograd.grad(ref_out, ref_in,
                                              grad_outputs=g_t)

            mod = tp.nn.MultiMarginLoss(margin=0.8)
            x = _tp_tensor(input_t, dev, requires_grad=True)
            out = mod(x, _tp_tensor(target_t, dev))
            out.backward(_tp_tensor(g_t, dev))
            _assert_close(self, out, ref_out, msg=f"MultiMarginLoss fwd ({dev})")
            _assert_close(self, x.grad, ref_grad,
                          msg=f"MultiMarginLoss grad ({dev})")

            torch.manual_seed(42)
            input_t = torch.randn(3, 5)
            g = torch.Generator().manual_seed(142)
            target_t = torch.full((3, 5), -1, dtype=torch.int64)
            for i in range(3):
                perm = torch.randperm(5, generator=g)[:2]
                target_t[i, :2] = perm

            ref_in = input_t.clone().requires_grad_(True)
            ref_mod = torch.nn.MultiLabelMarginLoss(reduction="sum")
            ref_out = ref_mod(ref_in, target_t)
            g_t = torch.randn_like(ref_out)
            (ref_grad,) = torch.autograd.grad(ref_out, ref_in,
                                              grad_outputs=g_t)

            mod = tp.nn.MultiLabelMarginLoss(reduction="sum")
            x = _tp_tensor(input_t, dev, requires_grad=True)
            out = mod(x, _tp_tensor(target_t, dev))
            out.backward(_tp_tensor(g_t, dev))
            _assert_close(self, out, ref_out,
                          msg=f"MultiLabelMarginLoss fwd ({dev})")
            _assert_close(self, x.grad, ref_grad,
                          msg=f"MultiLabelMarginLoss grad ({dev})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
