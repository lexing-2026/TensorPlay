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


def _w4a16_operands(m, k, n, rng, with_bias):
    """Builds the four payloads of one W4A16 product on the accelerator.

    Codes stay in [-8, 7], activations in [-4, 4], the per-channel scales are
    powers of two and the addends are whole numbers, so every product and every
    running sum lands on a value a half payload holds exactly: the comparison
    against the host reference needs no tolerance.
    """
    act_codes = rng.integers(-4, 5, (m, k)).astype(np.float32)
    w_codes = rng.integers(-8, 8, (n, k)).astype(np.float32)
    w_scales = np.exp2(-rng.integers(1, 4, n)).astype(np.float32)
    bias_values = (
        rng.integers(-3, 4, n).astype(np.float32) if with_bias else None
    )
    act = tp.tensor(act_codes, dtype=tp.float32).to("vulkan").to(tp.float16)
    weight = tp._C.quantize_per_tensor(
        tp.tensor(w_codes, dtype=tp.float32).to("vulkan"), 1.0, 0, tp.qint8
    )
    scales = tp.tensor(w_scales, dtype=tp.float32).to("vulkan")
    bias = (
        tp.tensor(bias_values, dtype=tp.float32).to("vulkan")
        if with_bias
        else None
    )
    expected = (
        (act_codes @ w_codes.T) * w_scales
        + (bias_values if with_bias else 0.0)
    ).astype(np.float16)
    return act, weight, scales, bias, expected


# One row count above the device's image-axis limit leaves the activations and
# the product with no texel form at all, so the same shapes must agree whether
# they drew an image or a run of words.
@pytest.mark.parametrize(
    "m,k,n",
    [
        (2, 8, 3),
        (17, 13, 5),
        (16383, 8, 8),
        (16385, 8, 8),
        (16385, 1, 5),
        (20000, 13, 16),
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_w4a16_linear_over_both_storages(m, k, n, with_bias):
    rng = np.random.default_rng(1000 * m + 10 * k + n)
    act, weight, scales, bias, expected = _w4a16_operands(
        m, k, n, rng, with_bias)
    out = tp._C.w4a16_linear(act, weight, scales, bias)
    assert out.dtype == tp.float16
    assert tuple(out.shape) == (m, n)
    np.testing.assert_array_equal(out.cpu().numpy(), expected)
