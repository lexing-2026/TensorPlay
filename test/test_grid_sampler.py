"""

  grid_sampler_2d / grid_sampler_3d and their backward kernels, exercised
through tp.nn.functional.grid_sample for every interpolation mode, padding
mode and align_corners setting, plus direct native-op calls with output_mask.
"""
import os
import sys
import unittest

import torch
import torch.nn.functional as torch_F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorplay as tp
import tensorplay.nn.functional as F

from tensorplay.testing._internal.reference import assert_reference_close, from_reference, reference_devices


def _run_grid_sample(case, input_t, grid_t, mode, padding_mode, align_corners,
                     dev, rtol=1e-4, atol=1e-5):
    input_t = input_t.detach().clone().requires_grad_(True)
    grid_t = grid_t.detach().clone().requires_grad_(True)
    ref = torch_F.grid_sample(input_t, grid_t, mode=mode,
                              padding_mode=padding_mode,
                              align_corners=align_corners)
    g = torch.randn_like(ref)
    ref.backward(g)

    x = from_reference(input_t, dev, requires_grad=True)
    gr = from_reference(grid_t, dev, requires_grad=True)
    out = F.grid_sample(x, gr, mode=mode, padding_mode=padding_mode,
                        align_corners=align_corners)
    out.backward(from_reference(g, dev))

    tag = f"{mode}/{padding_mode}/ac={align_corners} ({dev})"
    assert_reference_close(out, ref, rtol=rtol, atol=atol, msg=f"grid_sample fwd {tag}")
    assert_reference_close(x.grad, input_t.grad, rtol=rtol, atol=atol,
                  msg=f"grid_sample grad_input {tag}")
    assert_reference_close(gr.grad, grid_t.grad, rtol=rtol, atol=atol,
                  msg=f"grid_sample grad_grid {tag}")


class TestGridSampler2D(unittest.TestCase):
    MODES = ["bilinear", "nearest", "bicubic"]
    PADS = ["zeros", "border", "reflection"]

    def test_modes_paddings(self):
        torch.manual_seed(0)
        input_t = torch.randn(2, 3, 8, 9)
        grid_t = torch.rand(2, 5, 6, 2) * 2 - 1
        for dev in reference_devices():
            for mode in self.MODES:
                for pad in self.PADS:
                    for ac in (False, True):
                        _run_grid_sample(self, input_t, grid_t, mode, pad, ac, dev)

    def test_out_of_bounds_grid(self):
        # Grid coords outside [-1, 1] exercise the zeros/border/reflection
        # coordinate clamping paths.
        torch.manual_seed(1)
        input_t = torch.randn(1, 2, 7, 7)
        grid_t = torch.rand(1, 6, 6, 2) * 4 - 2
        for dev in reference_devices():
            for mode in ("bilinear", "nearest", "bicubic"):
                for pad in self.PADS:
                    _run_grid_sample(self, input_t, grid_t, mode, pad, False, dev)

    def test_unit_spatial(self):
        # Degenerate 1x1 input: reflection span collapses; nearest/bilinear
        # must still match.
        torch.manual_seed(2)
        input_t = torch.randn(1, 2, 1, 1)
        grid_t = torch.rand(1, 3, 3, 2) * 2 - 1
        for dev in reference_devices():
            for mode in ("bilinear", "nearest"):
                for pad in self.PADS:
                    _run_grid_sample(self, input_t, grid_t, mode, pad, True, dev)

    def test_native_op_direct(self):
        from tensorplay import _C
        torch.manual_seed(3)
        input_t = torch.randn(2, 2, 6, 6)
        grid_t = torch.rand(2, 4, 4, 2) * 2 - 1
        g_t = torch.randn(2, 2, 4, 4)
        for dev in reference_devices():
            x = from_reference(input_t, dev)
            gr = from_reference(grid_t, dev)
            out = _C.grid_sampler_2d(x, gr, 0, 0, False)
            ref_t = torch_F.grid_sample(input_t, grid_t, mode="bilinear",
                                        padding_mode="zeros", align_corners=False)
            assert_reference_close(out, ref_t, msg=f"native grid_sampler_2d fwd ({dev})", rtol=1e-4, atol=1e-5)
            gi, gg = _C.grid_sampler_2d_backward(
                from_reference(g_t, dev), x, gr, 0, 0, False, [True, True])
            input_t2 = input_t.clone().requires_grad_(True)
            grid_t2 = grid_t.clone().requires_grad_(True)
            torch_F.grid_sample(input_t2, grid_t2, mode="bilinear",
                                padding_mode="zeros",
                                align_corners=False).backward(g_t)
            assert_reference_close(gi, input_t2.grad, msg=f"native 2d bwd grad_input ({dev})", rtol=1e-4, atol=1e-5)
            assert_reference_close(gg, grid_t2.grad, msg=f"native 2d bwd grad_grid ({dev})", rtol=1e-4, atol=1e-5)

    def test_f64_parity(self):
        torch.manual_seed(4)
        input_t = torch.randn(1, 2, 5, 5, dtype=torch.float64)
        grid_t = (torch.rand(1, 4, 4, 2, dtype=torch.float64) * 2 - 1)
        for dev in reference_devices():
            _run_grid_sample(self, input_t, grid_t, "bilinear", "zeros", False,
                             dev, rtol=1e-10, atol=1e-12)


class TestGridSampler3D(unittest.TestCase):
    MODES = ["bilinear", "nearest"]
    PADS = ["zeros", "border", "reflection"]

    def test_modes_paddings(self):
        torch.manual_seed(5)
        input_t = torch.randn(2, 2, 5, 6, 7)
        grid_t = torch.rand(2, 3, 4, 3, 3) * 2 - 1
        for dev in reference_devices():
            for mode in self.MODES:
                for pad in self.PADS:
                    for ac in (False, True):
                        _run_grid_sample(self, input_t, grid_t, mode, pad, ac, dev)

    def test_out_of_bounds_grid(self):
        torch.manual_seed(6)
        input_t = torch.randn(1, 2, 4, 4, 4)
        grid_t = torch.rand(1, 3, 3, 3, 3) * 4 - 2
        for dev in reference_devices():
            for mode in self.MODES:
                for pad in self.PADS:
                    _run_grid_sample(self, input_t, grid_t, mode, pad, False, dev)

    def test_native_op_direct(self):
        from tensorplay import _C
        torch.manual_seed(7)
        input_t = torch.randn(1, 2, 4, 5, 6)
        grid_t = torch.rand(1, 3, 3, 3, 3) * 2 - 1
        g_t = torch.randn(1, 2, 3, 3, 3)
        for dev in reference_devices():
            x = from_reference(input_t, dev)
            gr = from_reference(grid_t, dev)
            out = _C.grid_sampler_3d(x, gr, 0, 1, True)
            ref_t = torch_F.grid_sample(input_t, grid_t, mode="bilinear",
                                        padding_mode="border", align_corners=True)
            assert_reference_close(out, ref_t, msg=f"native grid_sampler_3d fwd ({dev})", rtol=1e-4, atol=1e-5)
            gi, gg = _C.grid_sampler_3d_backward(
                from_reference(g_t, dev), x, gr, 0, 1, True, [True, True])
            input_t2 = input_t.clone().requires_grad_(True)
            grid_t2 = grid_t.clone().requires_grad_(True)
            torch_F.grid_sample(input_t2, grid_t2, mode="bilinear",
                                padding_mode="border",
                                align_corners=True).backward(g_t)
            assert_reference_close(gi, input_t2.grad, msg=f"native 3d bwd grad_input ({dev})", rtol=1e-4, atol=1e-5)
            assert_reference_close(gg, grid_t2.grad, msg=f"native 3d bwd grad_grid ({dev})", rtol=1e-4, atol=1e-5)

    def test_f64_parity(self):
        torch.manual_seed(8)
        input_t = torch.randn(1, 1, 4, 4, 4, dtype=torch.float64)
        grid_t = (torch.rand(1, 3, 3, 3, 3, dtype=torch.float64) * 2 - 1)
        for dev in reference_devices():
            _run_grid_sample(self, input_t, grid_t, "bilinear", "reflection", True,
                             dev, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
