"""Forward value and gradient checks for the ROI / detection operator family.

The native CPU kernels live in ``p10/src/backend/cpu/{RoiAlign,RoiPool,
PsRoiAlign,Nms,DeformConv}Kernels.cpp``; gradients are attached by the
autograd nodes generated from ``config/derivatives.yaml``. All comparisons
run the same inputs through an independently installed reference stack and
compare numerically.
"""
import unittest

import numpy as np
import torch
import torchvision.ops as tv_ops

import tensorplay as tp
from tensorplay.vision import ops as tp_vops

TOL_F32 = 5e-5


def _make(shape, seed=0, lo=-1.0, hi=1.0):
    rng = np.random.default_rng(seed)
    return ((hi - lo) * rng.random(shape) + lo).astype(np.float32)


def _to_tp(a):
    return tp.tensor(np.asarray(a))


def _assert_close(a_tp, b_torch, msg, tol=TOL_F32):
    a = a_tp.detach().cpu().numpy() if hasattr(a_tp, "numpy") else np.asarray(a_tp)
    b = b_torch.detach().cpu().numpy()
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol, err_msg=msg)


def _tp_grads(build_fn, inputs, tangent_np):
    tp_inputs = [_to_tp(x).requires_grad_(True) for x in inputs]
    out = build_fn(*tp_inputs)
    out.backward(_to_tp(tangent_np))
    return [x.grad.numpy() for x in tp_inputs]


def _torch_grads(build_fn, inputs, tangent):
    torch_inputs = [torch.tensor(x, requires_grad=True) for x in inputs]
    out = build_fn(*torch_inputs)
    out.backward(torch.tensor(tangent))
    return [x.grad.numpy() for x in torch_inputs]


def _random_rois(rng, n, h, w, spatial_scale, num_rois=4):
    """ROIs inside the image in original (pre-scale) coordinates."""
    scale = 1.0 / spatial_scale
    x1 = rng.random(num_rois) * (w * scale - 2)
    y1 = rng.random(num_rois) * (h * scale - 2)
    x2 = x1 + rng.random(num_rois) * (w * scale - x1 - 1) + 1
    y2 = y1 + rng.random(num_rois) * (h * scale - y1 - 1) + 1
    batch = rng.integers(0, n, size=num_rois).astype(np.float32)
    return np.stack([batch, x1, y1, x2, y2], axis=1).astype(np.float32)


class TestNms(unittest.TestCase):
    def test_forward_values(self):
        rng = np.random.default_rng(42)
        for trial in range(5):
            n = 40
            boxes = _make((n, 4), seed=trial, lo=0, hi=50)
            boxes[:, 2] = boxes[:, 0] + np.abs(boxes[:, 2]) + 1
            boxes[:, 3] = boxes[:, 1] + np.abs(boxes[:, 3]) + 1
            scores = np.sort(rng.random(n))[::-1].astype(np.float32)
            ref = tv_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5)
            got = tp_vops.nms(_to_tp(boxes), _to_tp(scores), 0.5)
            np.testing.assert_array_equal(
                got.numpy(), ref.numpy(), err_msg=f"nms trial {trial}"
            )

    def test_ties_keep_input_order(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12]], dtype=np.float32)
        scores = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        ref = tv_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.3)
        got = tp_vops.nms(_to_tp(boxes), _to_tp(scores), 0.3)
        np.testing.assert_array_equal(got.numpy(), ref.numpy())

    def test_empty(self):
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        got = tp_vops.nms(_to_tp(boxes), _to_tp(scores), 0.5)
        self.assertEqual(got.numel(), 0)
        self.assertEqual(got.dtype, tp.int64)


class TestBoxUtils(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_box_iou(self):
        for _ in range(3):
            a = _make((6, 4), seed=int(self.rng.integers(1000)), lo=0, hi=30)
            a[:, 2] = a[:, 0] + np.abs(a[:, 2]) + 1
            a[:, 3] = a[:, 1] + np.abs(a[:, 3]) + 1
            b = _make((5, 4), seed=int(self.rng.integers(1000)), lo=0, hi=30)
            b[:, 2] = b[:, 0] + np.abs(b[:, 2]) + 1
            b[:, 3] = b[:, 1] + np.abs(b[:, 3]) + 1
            ref = tv_ops.box_iou(torch.tensor(a), torch.tensor(b))
            got = tp_vops.box_iou(_to_tp(a), _to_tp(b))
            _assert_close(got, ref, "box_iou")

    def test_generalized_distance_complete_iou(self):
        a = _make((4, 4), seed=11, lo=0, hi=20)
        a[:, 2] = a[:, 0] + np.abs(a[:, 2]) + 1
        a[:, 3] = a[:, 1] + np.abs(a[:, 3]) + 1
        b = _make((4, 4), seed=12, lo=0, hi=20)
        b[:, 2] = b[:, 0] + np.abs(b[:, 2]) + 1
        b[:, 3] = b[:, 1] + np.abs(b[:, 3]) + 1
        _assert_close(
            tp_vops.generalized_box_iou(_to_tp(a), _to_tp(b)),
            tv_ops.generalized_box_iou(torch.tensor(a), torch.tensor(b)),
            "giou",
        )
        _assert_close(
            tp_vops.distance_box_iou(_to_tp(a), _to_tp(b)),
            tv_ops.distance_box_iou(torch.tensor(a), torch.tensor(b)),
            "diou",
        )
        _assert_close(
            tp_vops.complete_box_iou(_to_tp(a), _to_tp(b)),
            tv_ops.complete_box_iou(torch.tensor(a), torch.tensor(b)),
            "ciou",
        )

    def test_box_convert_roundtrip(self):
        a = _make((6, 4), seed=13, lo=0, hi=20)
        a[:, 2] = a[:, 0] + np.abs(a[:, 2]) + 1
        a[:, 3] = a[:, 1] + np.abs(a[:, 3]) + 1
        t = torch.tensor(a)
        for in_fmt, out_fmt in [("xyxy", "xywh"), ("xywh", "xyxy"),
                                ("xyxy", "cxcywh"), ("cxcywh", "xyxy"),
                                ("xywh", "cxcywh"), ("cxcywh", "xywh")]:
            ref = tv_ops.box_convert(t, in_fmt, out_fmt)
            got = tp_vops.box_convert(_to_tp(a), in_fmt, out_fmt)
            _assert_close(got, ref, f"convert {in_fmt}->{out_fmt}")

    def test_area_small_boxes_clip_batched_nms(self):
        a = _make((8, 4), seed=21, lo=0, hi=40)
        a[:, 2] = a[:, 0] + np.abs(a[:, 2]) + 1
        a[:, 3] = a[:, 1] + np.abs(a[:, 3]) + 1
        t = torch.tensor(a)
        tp_t = _to_tp(a)
        _assert_close(tp_vops.box_area(tp_t), tv_ops.box_area(t), "box_area")

        keep = tp_vops.remove_small_boxes(tp_t, 2.0)
        ref_keep = tv_ops.remove_small_boxes(t, 2.0)
        np.testing.assert_array_equal(keep.numpy(), ref_keep.numpy())

        clipped = tp_vops.clip_boxes_to_image(tp_t, (20, 25))
        ref_clipped = tv_ops.clip_boxes_to_image(t, (20, 25))
        _assert_close(clipped, ref_clipped, "clip")

        scores = np.sort(self.rng.random(8))[::-1].astype(np.float32)
        idxs = self.rng.integers(0, 3, size=8).astype(np.int64)
        ref = tv_ops.batched_nms(t, torch.tensor(scores), torch.tensor(idxs), 0.4)
        got = tp_vops.batched_nms(tp_t, _to_tp(scores), _to_tp(idxs), 0.4)
        np.testing.assert_array_equal(got.numpy(), ref.numpy())

    def test_masks_to_boxes(self):
        masks = (self.rng.random((3, 8, 9)) > 0.7).astype(np.float32)
        masks[2] = 0.0
        ref = tv_ops.masks_to_boxes(torch.tensor(masks))
        got = tp_vops.masks_to_boxes(_to_tp(masks))
        _assert_close(got, ref, "masks_to_boxes")


class TestRoiAlign(unittest.TestCase):
    def _run(self, sampling_ratio, aligned, spatial_scale, pooled, seed):
        rng = np.random.default_rng(seed)
        n, c, h, w = 2, 3, 10, 12
        x = _make((n, c, h, w), seed=seed)
        rois = _random_rois(rng, n, h, w, spatial_scale, num_rois=5)
        ph, pw = pooled
        ref = tv_ops.roi_align(
            torch.tensor(x), torch.tensor(rois), (ph, pw), spatial_scale, sampling_ratio, aligned
        )
        got = tp_vops.roi_align(_to_tp(x), _to_tp(rois), (ph, pw), spatial_scale, sampling_ratio, aligned)
        _assert_close(got, ref, f"roi_align sr={sampling_ratio} aligned={aligned} scale={spatial_scale}")
        return x, rois, ph, pw, spatial_scale, sampling_ratio, aligned

    def test_forward(self):
        for sr in (-1, 2, 4):
            for aligned in (False, True):
                self._run(sr, aligned, 1.0, (3, 4), seed=abs(sr) + int(aligned) * 10)
        self._run(2, False, 0.5, (5, 5), seed=99)
        self._run(0, False, 1.0, (2, 2), seed=100)  # 0 behaves like adaptive

    def test_backward(self):
        rng = np.random.default_rng(5)
        n, c, h, w = 2, 2, 8, 8
        x = _make((n, c, h, w), seed=31)
        rois = _random_rois(rng, n, h, w, 1.0, num_rois=3)
        ph, pw = 3, 3
        out_shape = (rois.shape[0], c, ph, pw)
        tangent = _make(out_shape, seed=32)

        got = _tp_grads(
            lambda x_: tp_vops.roi_align(x_, _to_tp(rois), (ph, pw), 1.0, 2, False),
            [x], tangent,
        )[0]
        ref = _torch_grads(
            lambda x_: tv_ops.roi_align(x_, torch.tensor(rois), (ph, pw), 1.0, 2, False),
            [x], tangent,
        )[0]
        np.testing.assert_allclose(got, ref, rtol=TOL_F32, atol=TOL_F32, err_msg="roi_align grad")

    def test_boxes_list_input(self):
        rng = np.random.default_rng(9)
        n, c, h, w = 2, 2, 8, 9
        x = _make((n, c, h, w), seed=41)
        per_image_np = [_random_rois(rng, 1, h, w, 1.0, num_rois=2) for _ in range(n)]
        ref = tv_ops.roi_align(
            torch.tensor(x), [torch.tensor(r[:, 1:]) for r in per_image_np], (3, 3), 1.0, 2, True
        )
        got = tp_vops.roi_align(
            _to_tp(x), [_to_tp(r[:, 1:]) for r in per_image_np], (3, 3), 1.0, 2, True
        )
        _assert_close(got, ref, "roi_align list-of-per-image boxes")


class TestRoiPool(unittest.TestCase):
    def test_forward(self):
        rng = np.random.default_rng(17)
        n, c, h, w = 2, 3, 9, 10
        x = _make((n, c, h, w), seed=51)
        for scale in (1.0, 0.5):
            rois = _random_rois(rng, n, h, w, scale, num_rois=4)
            ref = tv_ops.roi_pool(torch.tensor(x), torch.tensor(rois), (3, 4), scale)
            got = tp_vops.roi_pool(_to_tp(x), _to_tp(rois), (3, 4), scale)
            _assert_close(got, ref, f"roi_pool scale={scale}")

    def test_backward(self):
        rng = np.random.default_rng(18)
        n, c, h, w = 2, 2, 7, 7
        x = _make((n, c, h, w), seed=61)
        rois = _random_rois(rng, n, h, w, 1.0, num_rois=3)
        tangent = _make((rois.shape[0], c, 2, 2), seed=62)
        got = _tp_grads(
            lambda x_: tp_vops.roi_pool(x_, _to_tp(rois), (2, 2), 1.0), [x], tangent
        )[0]
        ref = _torch_grads(
            lambda x_: tv_ops.roi_pool(x_, torch.tensor(rois), (2, 2), 1.0), [x], tangent
        )[0]
        np.testing.assert_allclose(got, ref, rtol=TOL_F32, atol=TOL_F32, err_msg="roi_pool grad")


class TestPsRoi(unittest.TestCase):
    def test_ps_roi_align(self):
        rng = np.random.default_rng(23)
        n = 2
        pooled = 3
        c = 3 * pooled * pooled
        h, w = 9, 9
        x = _make((n, c, h, w), seed=71)
        rois = _random_rois(rng, n, h, w, 1.0, num_rois=3)
        for sr in (-1, 2):
            ref = tv_ops.ps_roi_align(torch.tensor(x), torch.tensor(rois), pooled, 1.0, sr)
            got = tp_vops.ps_roi_align(_to_tp(x), _to_tp(rois), pooled, 1.0, sr)
            _assert_close(got, ref, f"ps_roi_align sr={sr}")
        tangent = _make((rois.shape[0], 3, pooled, pooled), seed=72)
        got = _tp_grads(
            lambda x_: tp_vops.ps_roi_align(x_, _to_tp(rois), pooled, 1.0, 2), [x], tangent
        )[0]
        ref = _torch_grads(
            lambda x_: tv_ops.ps_roi_align(x_, torch.tensor(rois), pooled, 1.0, 2), [x], tangent
        )[0]
        np.testing.assert_allclose(got, ref, rtol=TOL_F32, atol=TOL_F32, err_msg="ps_roi_align grad")

    def test_ps_roi_pool(self):
        rng = np.random.default_rng(24)
        n = 2
        pooled = 2
        c = 3 * pooled * pooled
        h, w = 8, 8
        x = _make((n, c, h, w), seed=81)
        rois = _random_rois(rng, n, h, w, 1.0, num_rois=3)
        ref = tv_ops.ps_roi_pool(torch.tensor(x), torch.tensor(rois), pooled, 1.0)
        got = tp_vops.ps_roi_pool(_to_tp(x), _to_tp(rois), pooled, 1.0)
        _assert_close(got, ref, "ps_roi_pool")
        tangent = _make((rois.shape[0], 3, pooled, pooled), seed=82)
        got = _tp_grads(
            lambda x_: tp_vops.ps_roi_pool(x_, _to_tp(rois), pooled, 1.0), [x], tangent
        )[0]
        ref = _torch_grads(
            lambda x_: tv_ops.ps_roi_pool(x_, torch.tensor(rois), pooled, 1.0), [x], tangent
        )[0]
        np.testing.assert_allclose(got, ref, rtol=TOL_F32, atol=TOL_F32, err_msg="ps_roi_pool grad")


class TestDeformConv2d(unittest.TestCase):
    def _case(self, seed, use_mask=True, use_bias=True, stride=(1, 1),
              padding=(0, 0), dilation=(1, 1), groups=1):
        rng = np.random.default_rng(seed)
        n, c = 2, 4
        oc = 6 if groups == 1 else 4
        kh, kw = 3, 3
        h, w = 9, 9
        x = _make((n, c, h, w), seed=seed)
        weight = _make((oc, c // groups, kh, kw), seed=seed + 1)
        ker_h = dilation[0] * (kh - 1) + 1
        ker_w = dilation[1] * (kw - 1) + 1
        oh = (h + 2 * padding[0] - ker_h) // stride[0] + 1
        ow = (w + 2 * padding[1] - ker_w) // stride[1] + 1
        offset = _make((n, 2 * kh * kw, oh, ow), seed=seed + 2) * 0.9
        mask = _make((n, kh * kw, oh, ow), seed=seed + 3) if use_mask else None
        bias = _make((oc,), seed=seed + 4) if use_bias else None
        return x, weight, offset, mask, bias, stride, padding, dilation, groups

    def _tv_call(self, x, weight, offset, mask, bias, stride, padding, dilation):
        t_mask = None if mask is None else torch.tensor(mask)
        t_bias = None if bias is None else torch.tensor(bias)
        return tv_ops.deform_conv2d(
            torch.tensor(x), torch.tensor(offset), torch.tensor(weight), t_bias,
            stride, padding, dilation, t_mask,
        )

    def _tp_call(self, x, weight, offset, mask, bias, stride, padding, dilation):
        tp_mask = None if mask is None else _to_tp(mask)
        tp_bias = None if bias is None else _to_tp(bias)
        return tp_vops.deform_conv2d(
            _to_tp(x), _to_tp(offset), _to_tp(weight), tp_bias,
            stride, padding, dilation, tp_mask,
        )

    def test_forward(self):
        cases = [
            dict(seed=91),
            dict(seed=92, use_mask=False),
            dict(seed=93, use_bias=False, use_mask=False),
            dict(seed=94, stride=(2, 2)),
            dict(seed=95, padding=(1, 1)),
            dict(seed=96, dilation=(2, 2), padding=(2, 2)),
            dict(seed=97, groups=2),
        ]
        for kw in cases:
            args = self._case(**kw)
            ref = self._tv_call(*args[:8])
            got = self._tp_call(*args[:8])
            _assert_close(
                got, ref,
                f"deform_conv2d {kw}",
            )

    def test_backward(self):
        x, weight, offset, mask, bias, stride, padding, dilation, groups = self._case(101)
        out = self._tv_call(x, weight, offset, mask, bias, stride, padding, dilation)
        tangent = _make(out.shape, seed=102)

        tp_in = [_to_tp(v) for v in (x, weight, offset, mask, bias)]
        for t in tp_in:
            t.requires_grad_(True)
        got_out = tp_vops.deform_conv2d(
            tp_in[0], tp_in[2], tp_in[1], tp_in[4], stride, padding, dilation, tp_in[3]
        )
        got_out.backward(_to_tp(tangent))
        got_grads = [t.grad.numpy() for t in tp_in]

        torch_in = [torch.tensor(v, requires_grad=True) for v in (x, weight, offset, mask, bias)]
        ref_out = tv_ops.deform_conv2d(
            torch_in[0], torch_in[2], torch_in[1], torch_in[4], stride, padding, dilation, torch_in[3]
        )
        ref_out.backward(torch.tensor(tangent))
        ref_grads = [t.grad.numpy() for t in torch_in]

        names = ["input", "weight", "offset", "mask", "bias"]
        for name, g, r in zip(names, got_grads, ref_grads):
            np.testing.assert_allclose(
                g, r, rtol=TOL_F32, atol=TOL_F32, err_msg=f"deform_conv2d grad {name}"
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class TestCudaParity(unittest.TestCase):
    """Same parity checks on the CUDA kernels (atomic scatter adds make the
    gradients order-dependent, so they get a looser tolerance)."""
    TOL = 2e-4

    def _dev(self, a):
        return tp.tensor(np.asarray(a)).to("cuda")

    def test_nms_and_box_iou(self):
        rng = np.random.default_rng(77)
        boxes = _make((30, 4), seed=77, lo=0, hi=50)
        boxes[:, 2] = boxes[:, 0] + np.abs(boxes[:, 2]) + 1
        boxes[:, 3] = boxes[:, 1] + np.abs(boxes[:, 3]) + 1
        scores = np.sort(rng.random(30))[::-1].astype(np.float32)
        ref = tv_ops.nms(torch.tensor(boxes).cuda(), torch.tensor(scores).cuda(), 0.5)
        got = tp_vops.nms(self._dev(boxes), self._dev(scores), 0.5)
        np.testing.assert_array_equal(got.cpu().numpy(), ref.cpu().numpy())
        ref_iou = tv_ops.box_iou(torch.tensor(boxes).cuda(), torch.tensor(boxes).cuda())
        got_iou = tp_vops.box_iou(self._dev(boxes), self._dev(boxes))
        _assert_close(got_iou, ref_iou, "cuda box_iou", self.TOL)

    def test_roi_ops_forward(self):
        rng = np.random.default_rng(78)
        n, c, h, w = 2, 3, 10, 12
        x = _make((n, c, h, w), seed=81)
        rois = _random_rois(rng, n, h, w, 0.5, num_rois=5)
        xt, rt = torch.tensor(x).cuda(), torch.tensor(rois).cuda()
        ref = tv_ops.roi_align(xt, rt, (3, 4), 0.5, 2, True)
        got = tp_vops.roi_align(self._dev(x), self._dev(rois), (3, 4), 0.5, 2, True)
        _assert_close(got, ref, "cuda roi_align", self.TOL)
        ref = tv_ops.roi_pool(xt, rt, (3, 3), 0.5)
        got = tp_vops.roi_pool(self._dev(x), self._dev(rois), (3, 3), 0.5)
        _assert_close(got, ref, "cuda roi_pool", self.TOL)

    def test_roi_ops_backward(self):
        rng = np.random.default_rng(79)
        n, c, h, w = 2, 2, 8, 8
        x = _make((n, c, h, w), seed=91)
        rois = _random_rois(rng, n, h, w, 1.0, num_rois=3)
        tangent = _make((rois.shape[0], c, 3, 3), seed=92)
        tx = torch.tensor(x, requires_grad=True, device="cuda")
        tv_ops.roi_align(tx, torch.tensor(rois).cuda(), (3, 3), 1.0, 2, False).backward(
            torch.tensor(tangent).cuda())
        px = self._dev(x).requires_grad_(True)
        tp_vops.roi_align(px, self._dev(rois), (3, 3), 1.0, 2, False).backward(
            self._dev(tangent))
        _assert_close(px.grad, tx.grad, "cuda roi_align grad", self.TOL)

        tx = torch.tensor(x, requires_grad=True, device="cuda")
        tv_ops.roi_pool(tx, torch.tensor(rois).cuda(), (3, 3), 1.0).backward(
            torch.tensor(tangent).cuda())
        px = self._dev(x).requires_grad_(True)
        tp_vops.roi_pool(px, self._dev(rois), (3, 3), 1.0).backward(self._dev(tangent))
        _assert_close(px.grad, tx.grad, "cuda roi_pool grad", self.TOL)

    def test_ps_roi_forward_and_grad(self):
        rng = np.random.default_rng(82)
        n, pooled = 2, 2
        c = 3 * pooled * pooled
        h, w = 8, 8
        x = _make((n, c, h, w), seed=95)
        rois = _random_rois(rng, n, h, w, 1.0, num_rois=3)
        xt, rt = torch.tensor(x).cuda(), torch.tensor(rois).cuda()
        ref = tv_ops.ps_roi_align(xt, rt, pooled, 1.0, 2)
        got = tp_vops.ps_roi_align(self._dev(x), self._dev(rois), pooled, 1.0, 2)
        _assert_close(got, ref, "cuda ps_roi_align", self.TOL)
        ref = tv_ops.ps_roi_pool(xt, rt, pooled, 1.0)
        got = tp_vops.ps_roi_pool(self._dev(x), self._dev(rois), pooled, 1.0)
        _assert_close(got, ref, "cuda ps_roi_pool", self.TOL)
        tangent = _make((rois.shape[0], 3, pooled, pooled), seed=96)
        tx = torch.tensor(x, requires_grad=True, device="cuda")
        tv_ops.ps_roi_align(tx, rt, pooled, 1.0, 2).backward(torch.tensor(tangent).cuda())
        px = self._dev(x).requires_grad_(True)
        tp_vops.ps_roi_align(px, self._dev(rois), pooled, 1.0, 2).backward(self._dev(tangent))
        _assert_close(px.grad, tx.grad, "cuda ps_roi_align grad", self.TOL)

    def test_deform_conv2d(self):
        rng = np.random.default_rng(83)
        x = _make((2, 4, 9, 9), seed=97)
        weight = _make((6, 4, 3, 3), seed=98)
        offset = (_make((2, 18, 7, 7), seed=99) - 0.5) * 0.8
        mask = _make((2, 9, 7, 7), seed=100) * 0.5 + 0.5
        bias = _make((6,), seed=101)
        args = (x, weight, offset, mask, bias)
        t_args = [torch.tensor(a, requires_grad=True, device="cuda") for a in args]
        out = tv_ops.deform_conv2d(t_args[0], t_args[2], t_args[1], t_args[4],
                                   (1, 1), (0, 0), (1, 1), t_args[3])
        tangent = _make(out.shape, seed=102)
        out.backward(torch.tensor(tangent).cuda())
        p_args = [self._dev(a).requires_grad_(True) for a in args]
        p_out = tp_vops.deform_conv2d(p_args[0], p_args[2], p_args[1], p_args[4],
                                      (1, 1), (0, 0), (1, 1), p_args[3])
        _assert_close(p_out, out, "cuda deform_conv2d", self.TOL)
        p_out.backward(self._dev(tangent))
        for name, pg, tg in zip(["input", "weight", "offset", "mask", "bias"],
                                [a.grad for a in p_args], [a.grad for a in t_args]):
            _assert_close(pg, tg, f"cuda deform_conv2d grad {name}", self.TOL)


if __name__ == "__main__":
    unittest.main()
