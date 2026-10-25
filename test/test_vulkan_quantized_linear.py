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


# The output projection of a full vocabulary is a weight with far more rows than
# an image axis holds, so it can only arrive as integer codes in linear storage
# and be named as a quantized payload where it landed.  A row count within the
# limit draws an image instead, which keeps both payload forms of the factory in
# one parametrized sweep.
@pytest.mark.parametrize("n,k", [(16384, 8), (16385, 13), (20000, 8)])
def test_w4a16_linear_from_an_uploaded_weight(n, k):
    rng = np.random.default_rng(n * 7 + k)
    codes = rng.integers(-8, 8, (n, k)).astype(np.int8)
    w_scales = np.exp2(-rng.integers(1, 4, n)).astype(np.float32)
    bias_values = rng.integers(-3, 4, n).astype(np.float32)
    act_codes = rng.integers(-4, 5, (3, k)).astype(np.float32)

    weight = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(codes.copy()).to("vulkan"), 1.0, 0)
    assert weight.dtype == tp.qint8
    assert weight.q_scale() == 1.0
    assert weight.q_zero_point() == 0

    act = tp.tensor(act_codes, dtype=tp.float32).to("vulkan").to(tp.float16)
    out = tp._C.w4a16_linear(
        act,
        weight,
        tp.tensor(w_scales, dtype=tp.float32),
        tp.tensor(bias_values, dtype=tp.float32),
    )
    expected = (
        act_codes @ codes.T.astype(np.float32) * w_scales + bias_values
    ).astype(np.float16)
    assert tuple(out.shape) == (3, n)
    np.testing.assert_array_equal(out.cpu().numpy(), expected)


# A reduction axis longer than an image axis leaves the activation no texel form
# while the product it reduces into still has one, so the fold runs over words
# and stores its result as a texel.  Codes and activations stay at one unit and
# the scales at one, which keeps every running sum a whole number a half payload
# names exactly at these magnitudes.
@pytest.mark.parametrize("m,n,k", [(1, 8, 16384), (17, 5, 16385), (2, 8, 20000)])
def test_w4a16_linear_of_a_long_weight_row(m, n, k):
    rng = np.random.default_rng(m * 13 + n * 17 + k)
    codes = rng.integers(-1, 2, (n, k)).astype(np.int8)
    act_codes = rng.integers(-1, 2, (m, k)).astype(np.float32)
    bias_values = rng.integers(-2, 3, n).astype(np.float32)

    weight = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(codes.copy()).to("vulkan"), 1.0, 0)
    act = tp.tensor(act_codes, dtype=tp.float32).to("vulkan").to(tp.float16)
    out = tp._C.w4a16_linear(
        act,
        weight,
        tp.tensor(np.ones(n, dtype=np.float32), dtype=tp.float32),
        tp.tensor(bias_values, dtype=tp.float32),
    )
    expected = (act_codes @ codes.T.astype(np.float32) + bias_values).astype(
        np.float16
    )
    assert tuple(out.shape) == (m, n)
    np.testing.assert_array_equal(out.cpu().numpy(), expected)


# A code payload whose row count passes the image axis limit has no texel form,
# so the affine undo reads the four byte lanes of every code word instead of
# sampling an image, and the result keeps that linear storage.  An element count
# that does not divide by four leaves a partly used word on both sides, which the
# run must stop before.  The scale stays a power of two while the codes and the
# zero point stay whole, so every result is one single precision names exactly.
@pytest.mark.parametrize("n,k", [(16384, 8), (16385, 13), (20000, 8), (16385, 3)])
@pytest.mark.parametrize("zero_point", [0, -3])
def test_dequantize_of_an_uploaded_weight(n, k, zero_point):
    rng = np.random.default_rng(n * 5 + k + abs(zero_point))
    codes = rng.integers(-8, 8, (n, k)).astype(np.int8)
    weight = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(codes.copy()).to("vulkan"), 0.25, zero_point)
    out = tp.dequantize(weight)
    expected = 0.25 * (codes.astype(np.float32) - zero_point)
    assert out.dtype == tp.float32
    assert tuple(out.shape) == (n, k)
    np.testing.assert_array_equal(out.cpu().numpy(), expected)


# A block-wise affine payload carries one parameter pair per block of the grid
# that block_size names, so reading it means finding the block that spans each
# element.  The grid is numbered in the row-major order of the blocks themselves
# and an entry that does not tile its axis leaves a short block at the end, so
# the expected value below walks the block grid element by element.
def _affine_operands(shape, block_size, seed, unsigned=False):
    rng = np.random.default_rng(seed)
    codes = (
        rng.integers(0, 256, shape).astype(np.uint8)
        if unsigned
        else rng.integers(-128, 128, shape).astype(np.int8)
    )
    counts = [int(np.ceil(s / b)) for s, b in zip(shape, block_size)]
    n_blocks = int(np.prod(counts))
    scales = (rng.random(n_blocks) * 0.1).astype(np.float32)
    zero_points = rng.integers(-3, 4, n_blocks).astype(np.int32)
    return codes, scales, zero_points


def _affine_expect(codes, block_size, scales, zero_points):
    counts = [int(np.ceil(s / b)) for s, b in zip(codes.shape, block_size)]
    strides = [1] * codes.ndim
    acc = 1
    for d in range(codes.ndim - 1, -1, -1):
        strides[d] = acc
        acc *= counts[d]
    expect = np.empty(codes.shape, dtype=np.float32)
    for idx in np.ndindex(codes.shape):
        block = sum((c // b) * s for c, b, s in zip(idx, block_size, strides))
        z = 0 if zero_points is None else int(zero_points[block])
        expect[idx] = (int(codes[idx]) - z) * float(scales[block])
    return expect


@pytest.mark.parametrize(
    "shape,block_size,seed",
    [
        ((8, 16), (1, 4), 11),
        ((8, 16), (8, 16), 12),
        ((5, 17), (1, 5), 13),
        ((5, 17), (3, 5), 14),
        ((2, 6, 8), (1, 1, 4), 15),
        ((2, 3, 5, 7), (1, 1, 5, 7), 16),
        ((5,), (2,), 17),
        ((4, 4), (1, 1), 18),
    ],
)
@pytest.mark.parametrize("with_zero_points", [False, True])
def test_dequantize_affine_reads_the_block_that_spans_each_element(
    shape, block_size, seed, with_zero_points
):
    codes, scales, zero_points = _affine_operands(shape, block_size, seed)
    weight = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(codes.copy()).to("vulkan"), 1.0, 0
    )
    zeros = None
    if with_zero_points:
        zeros = tp.from_numpy(zero_points.copy()).to("vulkan")
    out = tp.dequantize_affine(
        weight,
        list(block_size),
        tp.from_numpy(scales.copy()).to("vulkan"),
        zeros,
    )
    expect = _affine_expect(
        codes, block_size, scales, zero_points if with_zero_points else None
    )
    assert out.dtype == tp.float32
    assert tuple(out.shape) == tuple(shape)
    np.testing.assert_allclose(out.cpu().numpy(), expect, rtol=1e-6, atol=1e-7)


# A code payload whose row count passes the image axis limit has no texel form,
# so the block is found from the linear position of each element and the result
# keeps the linear storage.  The groups tile each row here, so one reshape names
# them, and the whole payload still has to be covered by the run.
@pytest.mark.parametrize(
    "n,k,g", [(20000, 128, 64), (17000, 64, 64), (16385, 512, 128)]
)
def test_dequantize_affine_of_a_weight_with_no_image_form(n, k, g):
    codes, scales, zero_points = _affine_operands((n, k), (1, g), n + k + g)
    weight = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(codes.copy()).to("vulkan"), 1.0, 0
    )
    out = tp.dequantize_affine(
        weight,
        [1, g],
        tp.from_numpy(scales.copy()).to("vulkan"),
        tp.from_numpy(zero_points.copy()).to("vulkan"),
    )
    expect = (
        (
            codes.reshape(n, k // g, g).astype(np.float32)
            - zero_points.reshape(n, k // g, 1)
        )
        * scales.reshape(n, k // g, 1)
    ).reshape(n, k)
    np.testing.assert_allclose(out.cpu().numpy(), expect, rtol=1e-6, atol=1e-7)


# An unsigned payload keeps its codes in the same eight bits as a signed one, so
# the undo has to read them without folding the high bit into a sign.
def test_dequantize_affine_of_an_unsigned_payload():
    codes, scales, zero_points = _affine_operands(
        (8, 16), (1, 4), 21, unsigned=True
    )
    weight = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(codes.copy()).to("vulkan"), 1.0, 0
    )
    out = tp.dequantize_affine(
        weight,
        [1, 4],
        tp.from_numpy(scales.copy()).to("vulkan"),
        tp.from_numpy(zero_points.copy()).to("vulkan"),
    )
    expect = _affine_expect(codes, (1, 4), scales, zero_points)
    np.testing.assert_allclose(out.cpu().numpy(), expect, rtol=1e-6, atol=1e-7)


# One parameter of each kind covers the whole payload, so the block an element
# belongs to stops selecting anything: the grid still numbers the codes, while
# both reads collapse to the first slot.
def test_dequantize_affine_with_one_parameter_of_each_kind():
    rng = np.random.default_rng(22)
    codes = rng.integers(-128, 128, (8, 16)).astype(np.int8)
    weight = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(codes.copy()).to("vulkan"), 1.0, 0
    )
    out = tp.dequantize_affine(
        weight,
        [1, 4],
        tp.from_numpy(np.array([0.0625], dtype=np.float32)).to("vulkan"),
        tp.from_numpy(np.array([-2], dtype=np.int32)).to("vulkan"),
    )
    expect = (codes.astype(np.float32) + 2.0) * 0.0625
    np.testing.assert_array_equal(out.cpu().numpy(), expect)


def test_dequantize_affine_refuses_operands_it_cannot_read():
    rng = np.random.default_rng(23)
    codes = rng.integers(-128, 128, (8, 16)).astype(np.int8)
    scales = (rng.random(32) * 0.1).astype(np.float32)
    weight = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(codes.copy()).to("vulkan"), 1.0, 0
    )
    on_device = tp.from_numpy(scales.copy()).to("vulkan")

    with pytest.raises(RuntimeError):
        # Parameters that stayed on the host would have to be fetched per block.
        tp.dequantize_affine(weight, [1, 4], tp.from_numpy(scales.copy()), None)
    with pytest.raises(RuntimeError):
        tp.dequantize_affine(weight, [1, 4, 1], on_device, None)
    with pytest.raises(RuntimeError):
        tp.dequantize_affine(weight, [1, 0], on_device, None)
    with pytest.raises(RuntimeError):
        short = tp.from_numpy(scales[:8].copy()).to("vulkan")
        tp.dequantize_affine(weight, [1, 4], short, None)


# A matrix of codes spends four bytes of a texel on each code that needs one, so
# the entry that attaches an affine parameter set to a matrix already on the
# device reserves the codes in words instead.  The first two shapes below do have
# an image form, so it is the width of a code and not the shape that keeps the
# payload linear; the third is one no image covers at all, where the reservation
# has to agree with the shape rather than draw a texel back.
@pytest.mark.parametrize("n,k", [(97, 64), (12, 1), (16384, 8)])
def test_a_matrix_of_codes_named_on_the_device_lands_in_words(n, k):
    rng = np.random.default_rng(n * 11 + k)
    codes = rng.integers(-128, 128, (n, k)).astype(np.int8)
    weight = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(codes.copy()).to("vulkan"), 0.25, -4)
    assert weight.data_ptr() != 0, "the codes were not kept in words"
    assert tuple(weight.shape) == (n, k)
    np.testing.assert_array_equal(int_repr(weight).cpu().numpy(), codes)
    expected = 0.25 * (codes.astype(np.float32) + 4.0)
    np.testing.assert_array_equal(
        tp.dequantize(weight).cpu().numpy(), expected)


# That reservation runs through the strided gather, so the order the incoming
# matrix names is the order that lands dense: codes whose storage sits the other
# way round still arrive with the row a product later multiplies, and the
# reference below reads that same row.
def test_a_matrix_of_codes_named_from_a_transposed_arrival_is_dense():
    rng = np.random.default_rng(29)
    codes = rng.integers(-128, 128, (9, 7)).astype(np.int8)
    stored = tp.from_numpy(codes.copy()).to("vulkan")
    weight = tp._C._make_per_tensor_quantized_tensor(
        stored.transpose(0, 1), 0.5, 3)
    assert weight.data_ptr() != 0, "a transposed arrival was not kept in words"
    assert tuple(weight.shape) == (7, 9)
    named = codes.T
    np.testing.assert_array_equal(int_repr(weight).cpu().numpy(), named)
    expected = 0.5 * (named.astype(np.float32) - 3.0)
    np.testing.assert_array_equal(
        tp.dequantize(weight).cpu().numpy(), expected)


# The product packs both operands before its walk reads them, so an operand kept
# in words is packed from the byte lanes of the words its codes share instead of
# first being laid out as an image.  A row whose length does not divide by four
# splits the four codes of one texel across word boundaries, and a width below
# four leaves a texel whose tail lanes are padding, so both ends of that range
# are swept here against the host reference.
@pytest.mark.parametrize("m,k,n", [(4, 16, 8), (17, 33, 5), (5, 3, 7), (2, 1, 4)])
def test_quantized_linear_of_a_weight_named_on_the_device(m, k, n):
    rng = np.random.default_rng(m * 31 + k * 7 + n)
    x_codes = rng.integers(-32, 33, (m, k), dtype=np.int32)
    w_codes = rng.integers(-32, 33, (n, k), dtype=np.int32)
    x_scale, x_zp = 0.125, -3
    w_scales = np.exp2(-rng.integers(2, 6, n)).astype(np.float32)
    w_zps = rng.integers(-4, 5, n, dtype=np.int32)
    bias_values = rng.integers(-8, 9, n).astype(np.float32)

    xq = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(x_codes.astype(np.int8).copy()).to("vulkan"),
        x_scale, x_zp)
    wq = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(w_codes.astype(np.int8).copy()).to("vulkan"), 1.0, 0)
    assert xq.data_ptr() != 0 and wq.data_ptr() != 0

    out = tp._C.quantized_linear(
        xq,
        wq,
        x_scale,
        x_zp,
        tp.tensor(w_scales, dtype=tp.float32),
        tp.tensor(w_zps, dtype=tp.int32),
        tp.tensor(bias_values, dtype=tp.float32),
        0.25,
        3,
    )
    acc = (x_codes - x_zp) @ (w_codes - w_zps[:, None]).T
    expected = np.clip(
        np.rint((acc * x_scale * w_scales + bias_values) / 0.25) + 3,
        -128,
        127,
    ).astype(np.int8)
    assert out.dtype == tp.qint8
    np.testing.assert_array_equal(int_repr(out).cpu().numpy(), expected)


# A matrix the device quantizer writes keeps a byte per code, since four codes
# share the word that one float filled, while a payload whose shape draws an
# image keeps the texel walk.  Both routes round the way the host does: to the
# even neighbour, then clamped into the code range, so the corners below sit far
# outside it.  A width that does not divide by four leaves a partly used word the
# run has to stop before, which is what the short shapes probe.
@pytest.mark.parametrize("shape", [(6,), (4, 16), (5, 7), (2, 3, 4)])
def test_a_device_quantizer_writes_codes_where_its_shape_keeps_them(shape):
    rng = np.random.default_rng(7 + 11 * sum(shape) + 5 * len(shape))
    values = (rng.normal(0.0, 6.0, shape) * 4.0).astype(np.float32)
    values.flat[0] = 1.0e3
    values.flat[1] = -1.0e3
    scale, zero_point = 0.25, 4

    q = tp._C.quantize_per_tensor(
        tp.tensor(values, dtype=tp.float32).to("vulkan"),
        scale,
        zero_point,
        tp.qint8,
    )
    expected = np.clip(
        np.rint(values / scale) + zero_point, -128, 127
    ).astype(np.int8)
    np.testing.assert_array_equal(int_repr(q).cpu().numpy(), expected)
    assert (q.data_ptr() != 0) == (len(shape) == 2), (
        "the codes did not land where the shape keeps them")

    # The undo reads the codes back from the class they landed in, so a round
    # trip names the quantized grid rather than the incoming value.
    np.testing.assert_allclose(
        tp.dequantize(q).cpu().numpy(),
        (expected.astype(np.int64) - zero_point) * scale,
        rtol=0.0,
        atol=1e-6,
    )


# A decode step pairs an activation the device quantizer just wrote with a
# weight that was uploaded once and named.  Both now keep a byte per code, so the
# product reads the two through the same word-bound pack, and the answer is the
# one the host reference gives.
@pytest.mark.parametrize("m,k,n", [(1, 13, 6), (4, 16, 8)])
def test_quantized_linear_of_the_two_routes_a_matrix_of_codes_can_arrive_by(
    m, k, n
):
    rng = np.random.default_rng(m * 17 + k * 11 + n)
    x_codes = rng.integers(-32, 33, (m, k), dtype=np.int32)
    w_codes = rng.integers(-32, 33, (n, k), dtype=np.int32)
    x_scale, x_zp = 0.125, 2
    w_scales = np.exp2(-rng.integers(2, 6, n)).astype(np.float32)
    w_zps = rng.integers(-4, 5, n, dtype=np.int32)

    xq = tp._C.quantize_per_tensor(
        tp.tensor((x_codes - x_zp) * x_scale, dtype=tp.float32).to("vulkan"),
        x_scale, x_zp, tp.qint8)
    wq = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(w_codes.astype(np.int8).copy()).to("vulkan"), 1.0, 0)
    assert xq.data_ptr() != 0, "a device-quantized activation was not kept in words"
    assert wq.data_ptr() != 0, "the weight codes were not kept in words"

    out = tp._C.quantized_linear(
        xq,
        wq,
        x_scale,
        x_zp,
        tp.tensor(w_scales, dtype=tp.float32),
        tp.tensor(w_zps, dtype=tp.int32),
        None,
        0.5,
        -1,
    )
    acc = (x_codes - x_zp) @ (w_codes - w_zps[:, None]).T
    expected = np.clip(
        np.rint(acc * x_scale * w_scales / 0.5) - 1, -128, 127
    ).astype(np.int8)
    np.testing.assert_array_equal(int_repr(out).cpu().numpy(), expected)

    # A payload that reaches the device through the ordinary transfer keeps the
    # texel form its shape alone asks for, so the very same codes arrive in the
    # other class.  Each operand is packed from the class it is in, and both
    # routes have to answer the codes the host reference gives.
    x_texel = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(x_codes.astype(np.int8).copy()), x_scale, x_zp).to("vulkan")
    w_texel = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(w_codes.astype(np.int8).copy()), 1.0, 0).to("vulkan")
    assert x_texel.data_ptr() == 0 and w_texel.data_ptr() == 0, (
        "the transfer did not leave the codes in a texel")
    np.testing.assert_array_equal(
        int_repr(x_texel).cpu().numpy(), int_repr(xq).cpu().numpy())

    out_texel = tp._C.quantized_linear(
        x_texel,
        w_texel,
        x_scale,
        x_zp,
        tp.tensor(w_scales, dtype=tp.float32),
        tp.tensor(w_zps, dtype=tp.int32),
        None,
        0.5,
        -1,
    )
    np.testing.assert_array_equal(
        int_repr(out_texel).cpu().numpy(), expected)


# A fold of two payloads that keep their codes one byte per element walks the
# words those codes share: four consecutive output positions land in one word on
# the result side, while on each operand's side they sit wherever that operand's
# own row-major order puts them.  A singleton axis repeats through the same
# modulo the texel walk uses, which is what the broadcast shapes below try, and
# every pair is folded down both walks so the two answers can be compared byte
# for byte, then against the host.
@pytest.mark.parametrize(
    "a_shape,b_shape",
    [((23, 5), (23, 5)), ((6, 7), (6, 7)), ((23, 5), (1, 5)),
     ((23, 5), (23, 1)), ((4, 9), (1, 1)), ((5, 12), (1, 12))],
)
def test_a_fold_of_words_backed_matrices_stays_in_words(a_shape, b_shape):
    rng = np.random.default_rng(
        7 + 13 * a_shape[0] + 5 * a_shape[1] + 11 * b_shape[0] + b_shape[1])
    a_codes = rng.integers(-128, 129, a_shape).astype(np.int8)
    b_codes = rng.integers(-128, 129, b_shape).astype(np.int8)
    # A divisor that names its own zero point would read as nothing, so the sweep
    # keeps away from it and every quotient stays a finite number.
    b_codes = np.where(b_codes == -2, -1, b_codes).astype(np.int8)

    a_scale, a_zp = 0.25, 4
    b_scale, b_zp = 0.5, -2

    a = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(a_codes.copy()).to("vulkan"), a_scale, a_zp)
    b = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(b_codes.copy()).to("vulkan"), b_scale, b_zp)
    assert a.data_ptr() != 0 and b.data_ptr() != 0

    # The same codes reaching the device through the ordinary transfer keep the
    # texel form their shape alone asks for, so each fold runs down both walks
    # over one pair of payloads and the two answers have to be the same bytes.
    a_texel = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(a_codes.copy()), a_scale, a_zp).to("vulkan")
    b_texel = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(b_codes.copy()), b_scale, b_zp).to("vulkan")
    assert a_texel.data_ptr() == 0 and b_texel.data_ptr() == 0, (
        "the transfer did not leave the codes in a texel")

    a_cpu = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(a_codes.copy()), a_scale, a_zp)
    b_cpu = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(b_codes.copy()), b_scale, b_zp)

    for name, out_scale, out_zp in (
        ("quantized_add", 0.125, 0),
        ("quantized_sub", 0.125, 3),
        ("quantized_mul", 0.5, -1),
        ("quantized_div", 0.25, 2),
    ):
        run = getattr(tp._C, name)
        got = int_repr(
            run(a, b, a_scale, a_zp, b_scale, b_zp, out_scale, out_zp)
        ).cpu().numpy()
        texel = int_repr(
            run(a_texel, b_texel, a_scale, a_zp, b_scale, b_zp, out_scale, out_zp)
        ).cpu().numpy()
        np.testing.assert_array_equal(got, texel)
        assert got.shape == np.broadcast_shapes(a_shape, b_shape)

    # Both scales are a power of two, so a sum, a difference or a product of two
    # dequantized codes names a whole number of the grid exactly and the host's
    # rounding is the walk's own.  A quotient names such a number far less often,
    # and a value that lands between two codes can fall on either side of the tie
    # on the way there, so the quotient is only pinned to the host within one.
    for name, out_scale, out_zp in (
        ("quantized_add", 0.125, 0),
        ("quantized_sub", 0.125, 3),
        ("quantized_mul", 0.5, -1),
    ):
        run = getattr(tp._C, name)
        got = int_repr(
            run(a, b, a_scale, a_zp, b_scale, b_zp, out_scale, out_zp)
        ).cpu().numpy()
        want = int_repr(
            run(a_cpu, b_cpu, a_scale, a_zp, b_scale, b_zp, out_scale, out_zp)
        ).cpu().numpy()
        np.testing.assert_array_equal(got, want)

    quotient = int_repr(
        tp._C.quantized_div(a, b, a_scale, a_zp, b_scale, b_zp, 0.25, 2)
    ).cpu().numpy().astype(np.int64)
    host_quotient = int_repr(
        tp._C.quantized_div(a_cpu, b_cpu, a_scale, a_zp, b_scale, b_zp, 0.25, 2)
    ).cpu().numpy().astype(np.int64)
    assert np.max(np.abs(quotient - host_quotient)) <= 1

    # A pair that mixes the classes still folds: the payload kept in words is the
    # one that has to cross, and the answer cannot depend on which side crossed.
    mixed = int_repr(
        tp._C.quantized_add(a, b_texel, a_scale, a_zp, b_scale, b_zp, 0.125, 0)
    ).cpu().numpy()
    np.testing.assert_array_equal(
        mixed,
        int_repr(
            tp._C.quantized_add(
                a_texel, b_texel, a_scale, a_zp, b_scale, b_zp, 0.125, 0)
        ).cpu().numpy())

    # The words route writes its result into words too, so a fold of a pair that
    # arrived one byte per code does not hand the next consumer a texel to cross.
    stayed = tp._C.quantized_add(a, b, a_scale, a_zp, b_scale, b_zp, 0.125, 0)
    assert stayed.data_ptr() != 0, "the fold left its codes outside the words"


# The bounds fold reads the four lanes of one word and writes the next, so the
# codes it requantizes stay in the storage they arrived in.  The bounds sit in the
# dequantized domain and one pair of parameters maps the grid onto another, so
# both the window and the requantization are exercised.  The two walks of the
# same payload have to agree byte for byte, while the host, which reads the same
# formula in its own width, is only pinned to within one code: a value that lands
# between two codes can fall on either side of the tie on the way there.
@pytest.mark.parametrize(
    "out_scale,out_zp,low,high",
    [
        (0.1, 5, -2.0, 2.0),
        (0.2, -4, -1.0, 1.0),
        (0.05, 0, 0.3, None),
        (0.4, 1, None, 0.75),
    ],
)
def test_a_bounds_fold_of_a_words_payload_keeps_its_codes_in_words(
    out_scale, out_zp, low, high
):
    rng = np.random.default_rng(101 + int(1000 * out_scale) + out_zp)
    codes = rng.integers(-128, 129, (19, 6)).astype(np.int8)

    on_cpu = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(codes.copy()), 0.1, 5)
    on_device = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(codes.copy()).to("vulkan"), 0.1, 5)
    on_texel = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(codes.copy()), 0.1, 5).to("vulkan")
    assert on_device.data_ptr() != 0 and on_texel.data_ptr() == 0

    got = tp._C.quantized_clamp(on_device, 0.1, 5, out_scale, out_zp, low, high)
    texel = tp._C.quantized_clamp(on_texel, 0.1, 5, out_scale, out_zp, low, high)
    want = tp._C.quantized_clamp(on_cpu, 0.1, 5, out_scale, out_zp, low, high)
    np.testing.assert_array_equal(
        int_repr(got).cpu().numpy(), int_repr(texel).cpu().numpy())
    assert got.data_ptr() != 0, "the fold left its codes outside the words"
    difference = (
        int_repr(got).cpu().numpy().astype(np.int64)
        - int_repr(want).cpu().numpy().astype(np.int64))
    assert np.max(np.abs(difference)) <= 1


# A weight whose storage sits the other way round names its rows through strides
# that walk the words by columns, while both packs read bytes by their position
# in the storage.  Nothing here re-lays such a view out, so the product refuses it
# loudly instead of answering with the transposed codes: naming the refusal is
# what keeps a strided weight from quietly becoming a wrong layer.
def test_quantized_linear_refuses_a_weight_whose_storage_sits_the_other_way_round():
    m, k, n = 3, 13, 8
    rng = np.random.default_rng(m * 23 + k * 5 + n)
    x_codes = rng.integers(-32, 33, (m, k), dtype=np.int32)
    stored = rng.integers(-32, 33, (k, n), dtype=np.int32)
    x_scale, x_zp = 0.125, -2
    w_scales = np.exp2(-rng.integers(2, 6, n)).astype(np.float32)
    w_zps = rng.integers(-4, 5, n, dtype=np.int32)

    xq = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(x_codes.astype(np.int8).copy()).to("vulkan"),
        x_scale, x_zp)
    wq = tp._C._make_per_tensor_quantized_tensor(
        tp.from_numpy(stored.astype(np.int8).copy()).to("vulkan"), 1.0, 0)
    weight = wq.transpose(0, 1)
    assert tuple(weight.shape) == (n, k)

    with pytest.raises(RuntimeError, match="dense"):
        tp._C.quantized_linear(
            xq,
            weight,
            x_scale,
            x_zp,
            tp.tensor(w_scales, dtype=tp.float32),
            tp.tensor(w_zps, dtype=tp.int32),
            None,
            0.5,
            4,
        )
