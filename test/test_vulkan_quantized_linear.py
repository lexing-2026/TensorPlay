import numpy as np
import pytest

import tensorplay as tp
from tensorplay.ao.quantization import int_repr


pytestmark = pytest.mark.skipif(
    not tp.is_vulkan_available(), reason="Vulkan is not available"
)


@pytest.mark.parametrize(
    "m,k,n",
    [
        (1, 1, 1),
        (3, 5, 2),
        (4, 4, 4),
        (8, 8, 8),
        (16, 32, 16),
        (17, 33, 18),
        (31, 31, 33),
        (33, 65, 35),
        (64, 64, 64),
        (7, 129, 65),
    ],
)
@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_quantized_linear_tiled(m, k, n, asymmetric, with_bias):
    rng = np.random.default_rng(1000 * m + 10 * k + n)
    x_codes = rng.integers(-32, 33, (m, k), dtype=np.int32)
    w_codes = rng.integers(-32, 33, (n, k), dtype=np.int32)
    x_scale = 0.125
    x_zp = -7 if asymmetric else 0
    w_scales = np.exp2(-rng.integers(2, 6, n)).astype(np.float32)
    w_zps = (
        rng.integers(-9, 10, n, dtype=np.int32)
        if asymmetric
        else np.zeros(n, dtype=np.int32)
    )
    bias_values = (
        rng.integers(-16, 17, n).astype(np.float32) * 0.125
        if with_bias
        else np.zeros(n, dtype=np.float32)
    )
    x = tp.tensor((x_codes - x_zp) * x_scale, dtype=tp.float32).to("vulkan")
    w = tp.tensor(w_codes, dtype=tp.float32).to("vulkan")
    xq = tp._C.quantize_per_tensor(x, x_scale, x_zp, tp.qint8)
    wq = tp._C.quantize_per_tensor(w, 1.0, 0, tp.qint8)
    scales = tp.tensor(w_scales, dtype=tp.float32)
    zero_points = tp.tensor(w_zps, dtype=tp.int32)
    bias = tp.tensor(bias_values, dtype=tp.float32) if with_bias else None
    if asymmetric:
        scales = scales.to("vulkan")
        zero_points = zero_points.to("vulkan")
        if bias is not None:
            bias = bias.to("vulkan")
    out_scale, out_zp = 0.25, 3
    out = tp._C.quantized_linear(
        xq, wq, x_scale, x_zp, scales, zero_points, bias, out_scale, out_zp
    )
    assert out.dtype == tp.qint8
    assert out.q_scale() == out_scale
    assert out.q_zero_point() == out_zp
    acc = (x_codes - x_zp) @ (w_codes - w_zps[:, None]).T
    expected = np.clip(
        np.rint((acc * x_scale * w_scales + bias_values) / out_scale) + out_zp,
        -128,
        127,
    ).astype(np.int8)
    np.testing.assert_array_equal(int_repr(out).cpu().numpy(), expected)


@pytest.mark.parametrize("x_zp", [-128, 127])
def test_quantized_linear_extreme_codes(x_zp):
    x_codes = np.array(
        [[-128, 127, 0, -1, 1], [127, -128, -1, 1, 0]], dtype=np.int32
    )
    w_codes = np.array(
        [[127, -128, 1, 0, -1], [-128, 127, 0, -1, 1], [0, 1, -1, 127, -128]],
        dtype=np.int32,
    )
    w_zps = np.array([-128, 127, 0], dtype=np.int32)
    w_scales = np.array([1 / 256, 1 / 128, 1 / 64], dtype=np.float32)
    x_scale = 0.125
    x = tp.tensor((x_codes - x_zp) * x_scale, dtype=tp.float32).to("vulkan")
    w = tp.tensor(w_codes, dtype=tp.float32).to("vulkan")
    xq = tp._C.quantize_per_tensor(x, x_scale, x_zp, tp.qint8)
    wq = tp._C.quantize_per_tensor(w, 1.0, 0, tp.qint8)
    out = tp._C.quantized_linear(
        xq,
        wq,
        x_scale,
        x_zp,
        tp.tensor(w_scales, dtype=tp.float32),
        tp.tensor(w_zps, dtype=tp.int32),
        None,
        2.0,
        0,
    )
    acc = (x_codes - x_zp) @ (w_codes - w_zps[:, None]).T
    expected = np.clip(np.rint(acc * x_scale * w_scales / 2), -128, 127)
    np.testing.assert_array_equal(int_repr(out).cpu().numpy(), expected)
