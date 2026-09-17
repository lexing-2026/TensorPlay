"""Tests for the ao package: quantized ops, modules, and the eager workflow.

Native quantized kernels are validated against a dequantize-compute-requantize
reference computed with plain float ops.
"""

import pytest

import tensorplay as tp
import tensorplay.ao.nn.quantized as nnq
from tensorplay.ao.quantization import (
    QConfig,
    QConfigMapping,
    convert,
    default_qconfig,
    fuse_modules,
    prepare,
    quantize_dynamic,
)
from tensorplay.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
from tensorplay import nn


def _requant_reference(x, scale, zero_point, fn, out_scale, out_zero_point):
    """Float-domain reference: dequantize, apply fn, requantize."""
    dx = tp.dequantize(x)
    dy = fn(dx)
    return tp.quantize_per_tensor(dy, out_scale, out_zero_point, tp.qint8)


def _make_qtensor(values, scale=0.1, zero_point=3):
    x = tp.tensor(values, dtype=tp.float32)
    return tp.quantize_per_tensor(x, scale, zero_point, tp.qint8)


# ---------------------------------------------------------------------------
# quantized activations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values,scale,zp,out_scale,out_zp",
    [([1.0, -2.0, 3.5, -4.2], 0.13, 2, 0.21, 4),
     ([-10.0, 0.0, 10.0], 0.3, -1, 0.17, 0)],
)
def test_quantized_leaky_relu(values, scale, zp, out_scale, out_zp):
    x = _make_qtensor(values, scale, zp)
    got = tp._C.quantized_leaky_relu(x, 0.1, out_scale, out_zp)
    want = _requant_reference(x, scale, zp, lambda v: v * (v > 0) + v * 0.1 * (v <= 0),
                              out_scale, out_zp)
    assert tp.equal(got, want)


def test_quantized_relu_keeps_qparams():
    x = _make_qtensor([-1.0, 0.5, 3.0], 0.2, 5)
    got = tp._C.quantized_relu(x)
    assert got.q_scale() == x.q_scale()
    assert got.q_zero_point() == x.q_zero_point()
    # integer-domain: negatives map exactly to the zero point; 0.5 sits on
    # the scale-0.2 grid at code 7 (0.4), not at its float value
    assert tp.dequantize(got).tolist() == pytest.approx([0.0, 0.4, 3.0])


def test_quantized_relu6_grid_position():
    x = _make_qtensor([-1.0, 2.0, 9.0, 6.0], 0.2, 3)
    got = tp._C.quantized_relu6(x)
    dq = tp.dequantize(got).tolist()
    assert dq[0] == 0.0
    assert abs(dq[1] - 2.0) < 1e-6
    assert abs(dq[2] - 6.0) < 0.21  # clamped to the grid position of 6
    assert abs(dq[3] - 6.0) < 1e-6


def test_quantized_elu_matches_float_reference():
    x = _make_qtensor([-2.0, -0.3, 0.0, 1.5], 0.25, 4)
    out_scale, out_zp = 0.19, 2
    got = tp._C.quantized_elu(x, out_scale, out_zp)

    def elu(v):
        return v if v >= 0 else (v.exp() - 1.0)

    dx = tp.dequantize(x)
    want = tp.stack([
        tp.tensor(elu(v), dtype=tp.float32) if v.dim() == 0 else elu(v)
        for v in dx.reshape(-1)
    ]).reshape(dx.shape)
    want = tp.quantize_per_tensor(want, out_scale, out_zp, tp.qint8)
    assert tp.equal(got, want)


def test_quantized_hardswish_hardsigmoid():
    x = _make_qtensor([-3.0, 0.5, 4.0, 7.0], 0.3, 1)
    out_scale, out_zp = 0.11, 2
    hw = tp._C.quantized_hardswish(x, out_scale, out_zp)
    hs = tp._C.quantized_hardsigmoid(x, out_scale, out_zp)
    dx = tp.dequantize(x)
    hw_ref = dx * dx.add(3.0).clamp(0.0, 6.0) / 6.0
    hs_ref = (dx / 6.0 + 0.5).clamp(0.0, 1.0)
    hw_want = tp.quantize_per_tensor(hw_ref, out_scale, out_zp, tp.qint8)
    hs_want = tp.quantize_per_tensor(hs_ref, out_scale, out_zp, tp.qint8)
    assert tp.equal(hw, hw_want)
    assert tp.equal(hs, hs_want)


def test_quantized_sigmoid_tanh():
    x = _make_qtensor([-2.0, 0.0, 2.0], 0.4, 3)
    out_scale, out_zp = 0.07, 1
    sg = tp._C.quantized_sigmoid(x, out_scale, out_zp)
    th = tp._C.quantized_tanh(x, out_scale, out_zp)
    dx = tp.dequantize(x)
    sg_want = tp.quantize_per_tensor(dx.sigmoid(), out_scale, out_zp, tp.qint8)
    th_want = tp.quantize_per_tensor(dx.tanh(), out_scale, out_zp, tp.qint8)
    assert tp.equal(sg, sg_want)
    assert tp.equal(th, th_want)


# ---------------------------------------------------------------------------
# cat and pooling
# ---------------------------------------------------------------------------


def test_quantized_cat_same_qparams_is_byte_exact():
    a = _make_qtensor([1.0, 2.0], 0.2, 3)
    b = _make_qtensor([3.0, 4.0], 0.2, 3)
    got = tp._C.quantized_cat([a, b], 0)
    assert got.q_scale() == 0.2 and got.q_zero_point() == 3
    assert tp.dequantize(got).tolist() == [1.0, 2.0, 3.0, 4.0]


def test_quantized_cat_requantizes_mixed_qparams():
    a = _make_qtensor([1.0, 2.0], 0.2, 3)
    b = _make_qtensor([3.0, 4.0], 0.4, 5)
    out_scale, out_zp = 0.3, 4
    got = tp._C.quantized_cat([a, b], 0, out_scale, out_zp)
    ref = tp.cat([tp.dequantize(a), tp.dequantize(b)])
    want = tp.quantize_per_tensor(ref, out_scale, out_zp, tp.qint8)
    assert tp.equal(got, want)


def test_quantized_cat_relu():
    a = _make_qtensor([-1.0, 2.0], 0.2, 3)
    got = tp._C.quantized_cat_relu([a, a], 0)
    assert tp.dequantize(got).tolist() == [0.0, 2.0, 0.0, 2.0]


def test_quantized_max_pool2d_matches_float():
    x = _make_qtensor(tp.rand(1, 2, 6, 6).mul(4).sub(2).tolist(), 0.15, 2)
    got = tp._C.quantized_max_pool2d(x, [2, 2], [2, 2])
    ref = tp.max_pool2d(tp.dequantize(x), [2, 2], [2, 2])
    # max is order-preserving on the grid: byte-exact equality against the
    # float pool followed by requantization on the same qparams
    want = tp.quantize_per_tensor(ref, x.q_scale(), x.q_zero_point(), tp.qint8)
    assert tp.equal(got, want)


def test_quantized_max_pool1d_via_2d():
    x = _make_qtensor([1.0, 3.0, 2.0, 5.0, 0.0, 4.0], 0.2, 2)
    x = x.reshape(1, 1, 6)
    got = tp._C.quantized_max_pool1d(x, [2], [2])
    want = tp._C.quantized_max_pool2d(x.reshape(1, 1, 1, 6), [1, 2], [1, 2])
    assert tp.equal(got.reshape(want.shape), want)


def test_quantized_max_pool3d_matches_float():
    x = _make_qtensor(tp.rand(1, 1, 4, 4, 4).mul(4).sub(2).tolist(), 0.1, 1)
    got = tp._C.quantized_max_pool3d(x, [2, 2, 2], [2, 2, 2])
    ref = tp.max_pool3d(tp.dequantize(x), [2, 2, 2], [2, 2, 2])
    want = tp.quantize_per_tensor(ref, x.q_scale(), x.q_zero_point(), tp.qint8)
    assert tp.equal(got, want)


# ---------------------------------------------------------------------------
# convolutions
# ---------------------------------------------------------------------------


def test_quantized_conv1d_matches_conv2d_promotion():
    x = _make_qtensor(tp.rand(1, 3, 8).mul(4).sub(2).tolist(), 0.2, 1)
    w = _make_qtensor(tp.rand(4, 3, 3).mul(2).sub(1).tolist(), 0.05, 0)
    out_scale, out_zp = 0.4, 2
    got = tp._C.quantized_conv1d(
        x, w, None, x.q_scale(), x.q_zero_point(),
        w.q_scale(), w.q_zero_point(), out_scale, out_zp,
        [1], [0], [1], 1)
    # reshape drops the quantizer, so re-wrap the promoted operands on the
    # same affine grid (codes are unchanged)
    x2 = tp.quantize_per_tensor(tp.dequantize(x).reshape(1, 3, 1, 8),
                                x.q_scale(), x.q_zero_point(), tp.qint8)
    w2 = tp.quantize_per_tensor(tp.dequantize(w).reshape(4, 3, 1, 3),
                                w.q_scale(), w.q_zero_point(), tp.qint8)
    want = tp._C.quantized_conv2d(
        x2, w2, None, x.q_scale(), x.q_zero_point(),
        w.q_scale(), w.q_zero_point(), out_scale, out_zp,
        [1, 1], [0, 0], [1, 1], 1)
    assert got.shape == (1, 4, 6)
    assert tp.equal(got.reshape(want.shape), want)


def test_quantized_conv3d_matches_float_reference():
    x = _make_qtensor(tp.rand(1, 2, 5, 5, 5).mul(4).sub(2).tolist(), 0.2, 1)
    w = _make_qtensor(tp.rand(3, 2, 3, 3, 3).mul(2).sub(1).tolist(), 0.04, 0)
    bias = tp.rand(3)
    out_scale, out_zp = 0.5, 3
    got = tp._C.quantized_conv3d(
        x, w, bias, x.q_scale(), x.q_zero_point(),
        w.q_scale(), w.q_zero_point(), out_scale, out_zp,
        [1, 1, 1], [0, 0, 0], [1, 1, 1], 1)
    acc = tp.conv3d(tp.dequantize(x), tp.dequantize(w), bias,
                    [1, 1, 1], [0, 0, 0], [1, 1, 1], 1)
    want = tp.quantize_per_tensor(acc, out_scale, out_zp, tp.qint8)
    assert tp.equal(got, want)


def test_quantized_conv2d_module_end_to_end():
    conv = nnq.Conv2d(3, 4, 3, input_scale=0.2, input_zero_point=1,
                      out_scale=0.3, out_zero_point=2)
    # fill the weight with a known quantized pattern
    ref_float = tp.rand(4, 3, 3, 3)
    conv.weight = tp.quantize_per_tensor(
        ref_float, conv.weight_scale, conv.weight_zero_point, tp.qint8)
    x = tp.quantize_per_tensor(tp.rand(1, 3, 8, 8), 0.2, 1, tp.qint8)
    got = conv(x)
    acc = tp.conv2d(tp.dequantize(x), tp.dequantize(conv.weight),
                    conv.bias, [1, 1], [0, 0], [1, 1], 1)
    want = tp.quantize_per_tensor(acc, 0.3, 2, tp.qint8)
    assert tp.equal(got, want)


# ---------------------------------------------------------------------------
# modules and linear
# ---------------------------------------------------------------------------


def test_quantized_linear_matches_float_reference():
    float_linear = nn.Linear(8, 4)
    x = tp.rand(2, 8)
    scale, zp = 0.25, 2
    out_scale, out_zp = 0.1, 5
    qlinear = nnq.Linear.from_float(float_linear, scale, zp,
                                    out_scale=out_scale,
                                    out_zero_point=out_zp)
    xq = tp.quantize_per_tensor(x, scale, zp, tp.qint8)
    got = qlinear(xq)
    # quantized output under the module's output affine parameters
    assert got.is_quantized() and got.dtype == tp.qint8
    assert got.q_scale() == out_scale and got.q_zero_point() == out_zp
    want = float_linear(x)
    assert got.shape == (2, 4)
    assert tp.allclose(got.dequantize(), want, atol=0.2)


def test_quantized_linear_chain_closed_loop():
    # two quantized linears compose: the first emits a QInt8 output that
    # feeds the second directly, matching the packed-output contract
    tp.manual_seed(3)
    fc1 = nn.Linear(8, 6)
    fc2 = nn.Linear(6, 4)
    in_scale, in_zp = 0.25, 2
    mid_scale, mid_zp = 0.2, 0
    out_scale, out_zp = 0.5, 4
    q1 = nnq.Linear.from_float(fc1, in_scale, in_zp,
                               out_scale=mid_scale, out_zero_point=mid_zp)
    q2 = nnq.Linear.from_float(fc2, mid_scale, mid_zp,
                               out_scale=out_scale, out_zero_point=out_zp)
    x = tp.rand(2, 8)
    xq = tp.quantize_per_tensor(x, in_scale, in_zp, tp.qint8)
    got = q2(q1(xq))
    assert got.is_quantized()
    assert got.q_scale() == out_scale and got.q_zero_point() == out_zp
    reference = fc2(fc1(x))
    assert tp.allclose(got.dequantize(), reference, atol=0.4)


def test_quantize_dequantize_modules_roundtrip():
    q = nnq.Quantize(0.2, 3, tp.qint8)
    dq = nnq.DeQuantize()
    x = tp.tensor([-1.0, 0.2, 3.4])
    xq = q(x)
    assert xq.is_quantized()
    recovered = dq(xq)
    assert tp.allclose(recovered, x, atol=0.11)


def test_quantized_activation_modules():
    x = tp.quantize_per_tensor(tp.tensor([-2.0, 0.0, 2.0]), 0.2, 3, tp.qint8)
    relu = nnq.ReLU()
    assert tp.equal(relu(x), tp._C.quantized_relu(x))
    relu6 = nnq.ReLU6()
    assert tp.equal(relu6(x), tp._C.quantized_relu6(x))
    leaky = nnq.LeakyReLU(0.19, 2, negative_slope=0.1)
    assert tp.equal(leaky(x), tp._C.quantized_leaky_relu(x, 0.1, 0.19, 2))


def test_dynamic_linear_matches_float():
    float_linear = nn.Linear(6, 3)
    dynamic = quantize_dynamic(float_linear)
    x = tp.rand(2, 6)
    got = dynamic(x)
    want = float_linear(x)
    assert got.shape == (2, 3)
    # per-channel int8 weights: tight but nonzero tolerance
    assert tp.allclose(got, want, atol=0.1)


def test_qfunctional_add():
    # grid-aligned inputs so the integer-domain sum is exact at scale 0.2
    x = tp.quantize_per_tensor(tp.tensor([1.0, 2.0]), 0.2, 3, tp.qint8)
    y = tp.quantize_per_tensor(tp.tensor([0.6, 0.2]), 0.2, 3, tp.qint8)
    qfun = nnq.QFunctional()
    qfun.scale, qfun.zero_point = 0.2, 3
    got = qfun.add(x, y)
    assert tp.dequantize(got).tolist() == pytest.approx([1.6, 2.2], abs=1e-6)


# ---------------------------------------------------------------------------
# eager quantization workflow
# ---------------------------------------------------------------------------



    qconfig = QConfig(activation=MinMaxObserver, weight=PerChannelMinMaxObserver)
    mapping = QConfigMapping().set_global(default_qconfig).set_module_name("fc1", qconfig)
    resolved = mapping.to_dict()
    assert resolved["module_name"]["fc1"] is qconfig
    assert resolved[""] is default_qconfig


def test_prepare_convert_linear_workflow():
    # two swapped linears chain through a quantized intermediate: fc1 emits
    # a QInt8 output carrying its calibrated affine parameters and fc2
    # consumes it directly
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    model[0].qconfig = default_qconfig
    model[1].qconfig = default_qconfig
    model.eval()
    prepared = prepare(model, inplace=False)
    for _ in range(3):
        prepared(tp.rand(4, 4))
    converted = convert(prepared, inplace=False)
    assert isinstance(converted[0], nnq.QuantizedLinear)
    assert isinstance(converted[1], nnq.QuantizedLinear)
    x = tp.rand(4, 4)
    reference = model(x)
    scale = converted[0].input_scale
    zero_point = converted[0].input_zero_point
    xq = tp.quantize_per_tensor(x, scale, zero_point, tp.qint8)
    got = converted(xq)
    # the chained quantized linears emit a quantized result
    assert got.is_quantized()
    assert got.shape == reference.shape
    # int8 weights keep the output close to the float model
    assert tp.allclose(got.dequantize(), reference, atol=0.2)


def test_fuse_conv_bn_matches_manual_folding():
    conv = nn.Conv2d(3, 4, 3, bias=False)
    bn = nn.BatchNorm2d(4)
    conv.eval()
    bn.eval()
    x = tp.rand(1, 3, 8, 8)
    with tp.no_grad():
        reference = bn(conv(x))
    fused = fuse_modules(
        nn.Sequential(conv, bn), [["0", "1"]], inplace=False)
    with tp.no_grad():
        got = fused(x)
    assert tp.allclose(got, reference, atol=1e-4)


def test_fuse_linear_relu_module():
    linear = nn.Linear(4, 3)
    relu = nn.ReLU()
    fused = fuse_modules(nn.Sequential(linear, relu), [["0", "1"]],
                         inplace=False)
    x = tp.rand(2, 4)
    with tp.no_grad():
        reference = relu(linear(x))
        got = fused(x)
    assert tp.allclose(got, reference, atol=1e-6)
