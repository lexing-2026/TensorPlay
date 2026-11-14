"""Shape-only tensors on the meta device.

The meta device models sizes, dtypes and strides without owning memory:
factories and operators return unbacked tensors whose element data cannot be
read. These tests pin the metadata contract: correct shapes, promotion and
device plumbing for factories, pointwise ops, reductions, matmuls and copies,
plus the error paths for anything that would need real data.
"""

import pytest

import tensorplay as tp


def _meta(*size, dtype=None):
    return tp.empty(size, dtype=dtype, device="meta")


# ---------------------------------------------------------------------------
# Device plumbing
# ---------------------------------------------------------------------------


def test_device_parsing():
    dev = tp.device("meta")
    assert dev.type == "meta" or str(dev.type) in ("meta", "Meta")
    assert dev.is_meta()
    assert str(dev) == "meta"
    assert tp.device(tp.device("meta")) == dev


def test_device_no_index():
    with pytest.raises(Exception):
        tp.device("meta:0")


def test_tensor_is_meta_flag():
    assert _meta(2, 3).is_meta
    assert not tp.zeros(2).is_meta


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,args,shape",
    [
        ("zeros", ((2, 3),), (2, 3)),
        ("ones", ((4,),), (4,)),
        ("empty", ((2, 3, 5),), (2, 3, 5)),
        ("rand", ((2, 3),), (2, 3)),
        ("randn", ((2, 3),), (2, 3)),
        ("randint", (0, 10, (2, 3)), (2, 3)),
        ("randperm", (7,), (7,)),
        ("arange", (5,), (5,)),
        ("linspace", (0.0, 1.0, 11), (11,)),
        ("logspace", (0.0, 1.0, 5), (5,)),
    ],
)
def test_factory_shapes(factory, args, shape):
    t = getattr(tp, factory)(*args, device="meta")
    assert t.is_meta
    assert tuple(t.shape) == shape
    assert t.device.type == "meta"


def test_factory_dtypes():
    assert tp.zeros(3, dtype=tp.float64, device="meta").dtype == tp.float64
    assert tp.ones(3, dtype=tp.int32, device="meta").dtype == tp.int32
    # Full carries only its fill value in data, so only the dtype remains.
    assert tp.full((2,), 3.5, device="meta").dtype == tp.float32
    eye = tp.eye(3, device="meta")
    assert tuple(eye.shape) == (3, 3)


def test_arange_default_dtype():
    start = tp.arange(5, device="meta")
    assert start.dtype == tp.int64
    stop = tp.arange(5.0, device="meta")
    assert stop.dtype == tp.float32
    stepped = tp.arange(1, 10, 2, device="meta")
    assert tuple(stepped.shape) == (5,)


def test_like_family():
    t = _meta(2, 4, dtype=tp.float16)
    for like in ("empty_like", "zeros_like", "ones_like", "rand_like", "randn_like"):
        out = getattr(tp, like)(t)
        assert out.is_meta
        assert tuple(out.shape) == (2, 4)
        assert out.dtype == tp.float16
    moved = tp.empty_like(t, device="cpu")
    assert not moved.is_meta
    assert tuple(moved.shape) == (2, 4)
    fl = tp.full_like(t, 2.0)
    assert fl.is_meta and fl.dtype == tp.float16


def test_storage_tracks_size_without_data():
    t = _meta(1000, dtype=tp.float32)
    assert t.numel() == 1000
    assert t.element_size() == 4
    # The unbacked storage hands out no pointer.
    assert t.data_ptr() == 0


# ---------------------------------------------------------------------------
# Pointwise
# ---------------------------------------------------------------------------


def test_add_broadcast_and_promotion():
    a = _meta(3, 1, dtype=tp.float32)
    b = _meta(1, 4, dtype=tp.int64)
    out = a + b
    assert out.is_meta
    assert tuple(out.shape) == (3, 4)
    assert out.dtype == tp.float32
    scalar_out = a + 2
    assert scalar_out.dtype == tp.float32
    int_scalar = _meta(2, dtype=tp.int64) + 1.5
    assert int_scalar.dtype == tp.float32


def test_div_true_division_widens_integers():
    a = _meta(3, dtype=tp.int32)
    b = _meta(3, dtype=tp.int32)
    assert (a / b).dtype == tp.float32
    assert (a / 2).dtype == tp.float32


def test_inplace_keeps_shape():
    a = _meta(2, 3)
    a += _meta(2, 3)
    assert tuple(a.shape) == (2, 3)


def test_unary_ops():
    t = _meta(2, 3)
    for fn in (tp.neg, tp.exp, tp.log, tp.sqrt, tp.clamp):
        out = fn(t)
        assert out.is_meta
        assert tuple(out.shape) == (2, 3)
        assert out.dtype == t.dtype
    clamped = tp.clamp(t, 0.0, 1.0)
    assert tuple(clamped.shape) == (2, 3)


def test_abs_complex_returns_real_dtype():
    z = _meta(4, dtype=tp.complex64)
    assert tp.abs(z).dtype == tp.float32


@pytest.mark.parametrize("op", ["eq", "ne", "lt", "le", "gt", "ge"])
def test_comparisons_produce_bool(op):
    a = _meta(3, 4)
    b = _meta(3, 4)
    out = getattr(a, op)(b)
    assert out.dtype == tp.bool
    assert tuple(out.shape) == (3, 4)
    scalar_out = getattr(a, op)(1)
    assert scalar_out.dtype == tp.bool


def test_where_and_minmax():
    cond = _meta(2, 3, dtype=tp.bool)
    a = _meta(2, 3, dtype=tp.float32)
    b = _meta(2, 3, dtype=tp.float64)
    out = tp.where(cond, a, b)
    assert tuple(out.shape) == (2, 3)
    assert out.dtype == tp.float64
    assert tuple(tp.maximum(a, b).shape) == (2, 3)
    assert tuple(tp.minimum(a, b).shape) == (2, 3)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def test_sum_widens_integers():
    t = _meta(2, 3, dtype=tp.int16)
    whole = t.sum()
    assert tuple(whole.shape) == ()
    assert whole.dtype == tp.int64
    along = t.sum(dim=1)
    assert tuple(along.shape) == (2,)
    kept = t.sum(dim=1, keepdim=True)
    assert tuple(kept.shape) == (2, 1)


def test_mean_floats_for_integers():
    assert _meta(4, dtype=tp.int64).mean().dtype == tp.float32
    assert _meta(4, dtype=tp.float16).mean().dtype == tp.float16


def test_argmax_and_minmax():
    t = _meta(3, 4)
    assert t.argmax().dtype == tp.int64
    assert tuple(t.argmax(dim=1).shape) == (3,)
    assert tuple(t.amax(dim=0).shape) == (4,)
    assert t.amax(dim=0).dtype == t.dtype
    lo, hi = t.aminmax(dim=1)
    assert tuple(lo.shape) == (3,)
    assert tuple(hi.shape) == (3,)


def test_prod_shapes():
    t = _meta(2, 2, 2)
    assert tuple(t.prod(dim=0).shape) == (2, 2)
    assert tuple(t.prod().shape) == ()


# ---------------------------------------------------------------------------
# Matmuls and concatenation
# ---------------------------------------------------------------------------


def test_mm_shapes():
    a = _meta(2, 3)
    b = _meta(3, 4)
    out = a @ b
    assert tuple(out.shape) == (2, 4)
    with pytest.raises(Exception):
        _meta(2, 3) @ _meta(2, 3)


def test_bmm_and_addmm_shapes():
    a = _meta(5, 2, 3)
    b = _meta(5, 3, 4)
    assert tuple((a @ b).shape) == (5, 2, 4)
    m1 = _meta(2, 3)
    m2 = _meta(3, 4)
    bias = _meta(2, 4)
    out = tp.addmm(bias, m1, m2)
    assert tuple(out.shape) == (2, 4)
    # The operands must agree on the dtype.
    with pytest.raises(Exception):
        tp.addmm(bias, m1, _meta(3, 4, dtype=tp.float64))
    # A bias outside the result's shape is rejected unless its weight is zero.
    with pytest.raises(Exception):
        tp.addmm(_meta(5, 4), m1, m2)
    assert tuple(tp.addmm(_meta(5, 4), m1, m2, beta=0).shape) == (2, 4)
    assert tuple(tp.addmm_(_meta(2, 4), m1, m2).shape) == (2, 4)
    with pytest.raises(Exception):
        tp.addmm_(_meta(3, 4), m1, m2)
    # Whole-number operands accumulate in single precision.
    assert tp.addmm(_meta(2, 4, dtype=tp.int32), _meta(2, 3, dtype=tp.int32),
                    _meta(3, 4, dtype=tp.int32)).dtype == tp.float32
    with pytest.raises(Exception):
        tp.addmm(_meta(2, 4, dtype=tp.bool), _meta(2, 3, dtype=tp.bool),
                 _meta(3, 4, dtype=tp.bool))


def test_cat_shapes():
    a = _meta(2, 3)
    b = _meta(4, 3)
    out = tp.cat([a, b], dim=0)
    assert tuple(out.shape) == (6, 3)
    c = _meta(2, 5)
    assert tuple(tp.cat([a, c], dim=1).shape) == (2, 8)
    with pytest.raises(Exception):
        tp.cat([a, _meta(9, 9)], dim=0)


# ---------------------------------------------------------------------------
# Views: metadata-only composites work without data
# ---------------------------------------------------------------------------


def test_view_ops():
    t = _meta(2, 6)
    r = t.reshape((3, 4))
    assert tuple(r.shape) == (3, 4)
    assert r.is_meta
    tr = t.transpose(0, 1)
    assert tuple(tr.shape) == (6, 2)
    assert tuple(tr.stride()) == (1, 6)
    sl = t[0:1, ::2]
    assert tuple(sl.shape) == (1, 3)


# ---------------------------------------------------------------------------
# Copies and data access errors
# ---------------------------------------------------------------------------


def test_to_meta_from_cpu():
    real = tp.arange(6, dtype=tp.float32).reshape((2, 3))
    m = real.to("meta")
    assert m.is_meta
    assert tuple(m.shape) == (2, 3)
    assert m.dtype == tp.float32


def test_copy_out_of_meta_raises():
    m = _meta(4)
    with pytest.raises(Exception):
        m.to("cpu")


def test_item_raises_on_meta():
    with pytest.raises(Exception):
        _meta(3).item()


def test_repr_does_not_crash():
    text = repr(_meta(2, 3))
    assert "meta" in text


def test_mixed_devices_rejected():
    with pytest.raises(Exception):
        _meta(3) + tp.zeros(3)


# ---------------------------------------------------------------------------
# Autograd: gradients are shape-only too
# ---------------------------------------------------------------------------


def test_autograd_on_meta():
    x = tp.empty(4, device="meta", requires_grad=True)
    y = x * 2 + 1
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.is_meta
    assert tuple(x.grad.shape) == (4,)


def test_no_grad_still_runs():
    with tp.no_grad():
        out = _meta(3) * 3
    assert tuple(out.shape) == (3,)


# ---------------------------------------------------------------------------
# Activations and elementary math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        tp.relu,
        tp.gelu,
        tp.tanh,
        tp.sigmoid,
        tp.silu,
        tp.mish,
        tp.elu,
        tp.selu,
        tp.celu,
        tp.leaky_relu,
        tp.hardsigmoid,
        tp.hardswish,
        tp.hardtanh,
        tp.tanhshrink,
        tp.softshrink,
        tp.hardshrink,
        tp.relu6,
        tp.softplus,
        tp.log_sigmoid,
        tp.floor,
        tp.ceil,
        tp.round,
        tp.trunc,
        tp.frac,
        tp.sign,
        tp.sgn,
    ],
)
def test_activation_shapes(fn):
    t = _meta(2, 3)
    out = fn(t)
    assert out.is_meta
    assert tuple(out.shape) == (2, 3)
    assert out.dtype == t.dtype


def test_relu_inplace():
    t = _meta(2, 3)
    out = tp.relu_(t)
    assert out is t
    assert tuple(out.shape) == (2, 3)


def test_pow_shapes_and_dtypes():
    assert tuple((_meta(2, 3) ** 2).shape) == (2, 3)
    assert (_meta(2, 3) ** 2).dtype == tp.float32
    assert (_meta(2, 3, dtype=tp.int64) ** 2).dtype == tp.int64
    assert (_meta(2, 3) ** 2.0).dtype == tp.float32
    base = _meta(2, 3)
    exp = _meta(3,)
    out = tp.pow(base, exp)
    assert tuple(out.shape) == (2, 3)
    with pytest.raises(Exception):
        _meta(2, dtype=tp.bool) ** 2


def test_clamp_min_max():
    t = _meta(2, 3)
    assert tuple(tp.clamp_min(t, 0.0).shape) == (2, 3)
    assert tuple(tp.clamp_max(t, 1.0).shape) == (2, 3)
    lo = _meta(2, 1)
    assert tuple(tp.clamp_min(t, lo).shape) == (2, 3)


def test_glu_and_prelu_shapes():
    assert tuple(tp.glu(_meta(2, 4)).shape) == (2, 2)
    assert tuple(tp.glu(_meta(2, 4), dim=0).shape) == (1, 4)
    with pytest.raises(Exception):
        tp.glu(_meta(2, 3))
    assert tuple(tp.prelu(_meta(2, 3), _meta(3)).shape) == (2, 3)


# ---------------------------------------------------------------------------
# Softmax family
# ---------------------------------------------------------------------------


def test_softmax_shapes():
    t = _meta(2, 3)
    assert tuple(tp.softmax(t, dim=1).shape) == (2, 3)
    assert tuple(tp.log_softmax(t, dim=1).shape) == (2, 3)
    assert tp.softmax(t, dim=1).dtype == t.dtype
    assert tp.softmax(t, dim=1, dtype=tp.float64).dtype == tp.float64
    assert tuple(tp.softmax(_meta(3,), dim=0).shape) == (3,)
    # Integer inputs answer at single precision.
    assert tp.softmax(_meta(2, 3, dtype=tp.int32), dim=1).dtype == tp.float32
    with pytest.raises(Exception):
        tp.softmax(t, dim=2)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_layer_norm_shapes():
    out = tp.nn.functional.layer_norm(_meta(2, 3), (3,))
    assert tuple(out.shape) == (2, 3)
    mean, rstd = tp._C.native_layer_norm(_meta(2, 3), (3,), None, None, 1e-5)[1:]
    assert tuple(mean.shape) == (2,)
    assert tuple(rstd.shape) == (2,)
    with pytest.raises(Exception):
        tp.nn.functional.layer_norm(_meta(2, 3), (4,))


def test_batch_norm_shapes():
    x = _meta(2, 3)
    rm = _meta(3)
    rv = _meta(3)
    out = tp.nn.functional.batch_norm(x, rm, rv, None, None, training=False)
    assert tuple(out.shape) == (2, 3)
    train_out = tp.nn.functional.batch_norm(x, None, None, None, None, training=True)
    assert tuple(train_out.shape) == (2, 3)
    with pytest.raises(Exception):
        tp.nn.functional.batch_norm(x, None, None, None, None, training=False)


# ---------------------------------------------------------------------------
# Embedding and pooling
# ---------------------------------------------------------------------------


def test_embedding_shape():
    weight = _meta(10, 4)
    indices = tp.empty((2, 3), dtype=tp.int64, device="meta")
    out = tp.nn.functional.embedding(indices, weight)
    assert tuple(out.shape) == (2, 3, 4)
    assert out.dtype == weight.dtype
    with pytest.raises(Exception):
        tp.nn.functional.embedding(tp.empty((2,), dtype=tp.float32, device="meta"),
                                   weight)


def test_max_pool2d_shapes():
    x = _meta(1, 3, 8, 8)
    out = tp.nn.functional.max_pool2d(x, 2)
    assert tuple(out.shape) == (1, 3, 4, 4)
    values, indices = tp.nn.functional.max_pool2d(x, 2, return_indices=True)
    assert tuple(indices.shape) == (1, 3, 4, 4)
    assert indices.dtype == tp.int64
    # Unbatched input keeps three dimensions.
    assert tuple(tp.nn.functional.max_pool2d(_meta(3, 8, 8), 2).shape) == (3, 4, 4)


def test_avg_pool2d_shapes():
    x = _meta(1, 3, 8, 8)
    assert tuple(tp.nn.functional.avg_pool2d(x, 2).shape) == (1, 3, 4, 4)
    assert tuple(tp.nn.functional.avg_pool2d(x, 2, ceil_mode=True).shape) == (1, 3, 4, 4)
    assert tuple(tp.nn.functional.adaptive_avg_pool2d(x, (1, 1)).shape) == (1, 3, 1, 1)
    assert tuple(tp.nn.functional.adaptive_max_pool2d(x, (1, 1)).shape) == (1, 3, 1, 1)
    assert tuple(tp.adaptive_max_pool2d(x, [2, 3]).shape) == (1, 3, 2, 3)
    with pytest.raises(Exception):
        tp.adaptive_max_pool2d(x, [0, 1])
    with pytest.raises(Exception):
        tp.nn.functional.avg_pool2d(x, 2, stride=0)


def test_pool1d_shapes_and_checks():
    x = _meta(2, 3, 16)
    assert tuple(tp.max_pool1d(x, 2).shape) == (2, 3, 8)
    assert tuple(tp.max_pool1d(_meta(3, 16), 2).shape) == (3, 8)
    values, indices = tp.max_pool1d_with_indices(x, 2)
    assert tuple(values.shape) == (2, 3, 8)
    assert values.dtype == tp.float32
    assert indices.dtype == tp.int64
    assert tuple(tp.avg_pool1d(x, 3, 2).shape) == (2, 3, 7)
    assert tuple(tp.adaptive_avg_pool1d(x, 5).shape) == (2, 3, 5)
    values, indices = tp.adaptive_max_pool1d(x, 5)
    assert tuple(values.shape) == (2, 3, 5)
    assert indices.dtype == tp.int64

    with pytest.raises(Exception):
        tp.max_pool1d(_meta(16), 2)
    with pytest.raises(Exception):
        tp.max_pool1d_with_indices(_meta(16), 2)
    with pytest.raises(Exception):
        tp.max_pool1d(x, [2, 2])
    with pytest.raises(Exception):
        tp.avg_pool1d(x, [2, 2])
    with pytest.raises(Exception):
        tp.avg_pool1d(x, 3, 0)
    with pytest.raises(Exception):
        tp.adaptive_max_pool1d(_meta(2, 0, 4), 2)


def test_pool3d_shapes_and_checks():
    x = _meta(1, 2, 8, 8, 8)
    assert tuple(tp.max_pool3d(x, 2).shape) == (1, 2, 4, 4, 4)
    values, indices = tp.max_pool3d_with_indices(x, 2)
    assert tuple(values.shape) == (1, 2, 4, 4, 4)
    assert values.dtype == tp.float32
    assert indices.dtype == tp.int64
    assert tuple(tp.max_pool3d(_meta(2, 8, 8, 8), 2).shape) == (2, 4, 4, 4)

    assert tuple(tp.avg_pool3d(x, 2).shape) == (1, 2, 4, 4, 4)
    assert tuple(tp.avg_pool3d(_meta(2, 8, 8, 8), 2).shape) == (2, 4, 4, 4)
    assert tuple(tp.avg_pool3d(x, 3, 2, 0, True).shape) == (1, 2, 4, 4, 4)
    assert tuple(tp.avg_pool3d(x, 3, 2, 0, False).shape) == (1, 2, 3, 3, 3)

    assert tuple(tp.adaptive_avg_pool3d(x, [2, 3, 4]).shape) == (1, 2, 2, 3, 4)
    assert tuple(tp.adaptive_avg_pool3d(_meta(2, 8, 8, 8), [2, 2, 2]).shape) == (
        2, 2, 2, 2)
    values, indices = tp.adaptive_max_pool3d(x, [2, 3, 4])
    assert tuple(values.shape) == (1, 2, 2, 3, 4)
    assert indices.dtype == tp.int64
    assert tuple(tp.adaptive_max_pool3d(_meta(2, 8, 8, 8), [2, 2, 2])[0].shape) == (
        2, 2, 2, 2)

    with pytest.raises(Exception):
        tp.max_pool3d(_meta(1, 2, 3, 8, 8, 8, 8), 2)
    with pytest.raises(Exception):
        tp.avg_pool3d(x, 2, 0)
    with pytest.raises(Exception):
        tp.avg_pool3d(x, [2, 2])
    with pytest.raises(Exception):
        tp.adaptive_avg_pool3d(x, [2, 2])
    with pytest.raises(Exception):
        tp.adaptive_avg_pool3d(x, [2, -1, 2])
    with pytest.raises(Exception):
        tp.adaptive_max_pool3d(x, [2, 2])
    with pytest.raises(Exception):
        tp.adaptive_max_pool3d(x, [0, 2, 2])


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------


def test_permute_and_movedim():
    t = _meta(2, 3, 4)
    assert tuple(t.permute(2, 0, 1).shape) == (4, 2, 3)
    assert tuple(t.movedim(0, 2).shape) == (3, 4, 2)
    with pytest.raises(Exception):
        t.permute(0, 0, 1)


def test_split_and_stack():
    t = _meta(2, 6)
    parts = t.split(2, dim=1)
    assert [tuple(p.shape) for p in parts] == [(2, 2)] * 3
    by_sizes = t.split([1, 2, 3], dim=1)
    assert [tuple(p.shape) for p in by_sizes] == [(2, 1), (2, 2), (2, 3)]
    stacked = tp.stack([_meta(2, 3), _meta(2, 3)], dim=0)
    assert tuple(stacked.shape) == (2, 2, 3)
    stacked_last = tp.stack([_meta(2, 3), _meta(2, 3)], dim=-1)
    assert tuple(stacked_last.shape) == (2, 3, 2)


def test_flip_roll_repeat_expand_narrow_diagonal():
    t = _meta(2, 3)
    assert tuple(t.flip(0).shape) == (2, 3)
    assert tuple(t.roll(1, dims=(1,)).shape) == (2, 3)
    assert tuple(t.repeat(2, 1).shape) == (4, 3)
    assert tuple(_meta(2, 1).expand(2, 4).shape) == (2, 4)
    # A singleton axis gets stride 0; an axis whose extent already matches
    # keeps its input stride.
    assert tuple(_meta(2, 1).expand(2, 4).stride()) == (1, 0)
    assert tuple(_meta(2, 3).expand(2, 3).stride()) == (3, 1)
    assert tuple(_meta(2, 6).narrow(1, 1, 3).shape) == (2, 3)
    assert tuple(_meta(3, 3).diagonal(0).shape) == (3,)


# ---------------------------------------------------------------------------
# Losses and norms
# ---------------------------------------------------------------------------


def test_loss_shapes():
    x = _meta(2, 3)
    y = _meta(2, 3)
    assert tuple(tp.nn.functional.mse_loss(x, y).shape) == ()
    assert tuple(tp.nn.functional.mse_loss(x, y, reduction="none").shape) == (2, 3)
    assert tuple(tp.nn.functional.l1_loss(x, y).shape) == ()
    assert tuple(tp.nn.functional.smooth_l1_loss(x, y).shape) == ()
    assert tuple(tp.nn.functional.binary_cross_entropy(x, y).shape) == ()
    target = tp.empty((2,), dtype=tp.int64, device="meta")
    loss = tp.nn.functional.nll_loss(x, target)
    assert tuple(loss.shape) == ()
    raw, total = tp._C.nll_loss(x, target, None, 1, -100)
    assert tuple(raw.shape) == ()
    assert tuple(total.shape) == ()


def test_rsub_shapes_and_dtypes():
    i = _meta(3, dtype=tp.int32)
    f16 = _meta(3, dtype=tp.float16)
    z = _meta(3, dtype=tp.complex64)
    # A whole-number scalar leaves the tensor dtype alone; a floating scalar
    # answers an integral tensor (truth values included) at single precision.
    assert tp.functional.rsub(i, 1).dtype == tp.int32
    assert tp.functional.rsub(i, 1.0).dtype == tp.float32
    assert tp.functional.rsub(f16, 1).dtype == tp.float16
    assert tp.functional.rsub(f16, 1.0).dtype == tp.float16
    assert tp.functional.rsub(_meta(2, dtype=tp.bool), 1.0).dtype == tp.float32
    # The scalar path has no complex loop: a complex tensor is refused against
    # a whole-number scalar but narrowed to single precision against a float.
    with pytest.raises(TypeError):
        tp.functional.rsub(z, 1)
    assert tp.functional.rsub(z, 1.0).dtype == tp.float32
    # The tensor form broadcasts and promotes both operands, complex included.
    assert tuple(tp.functional.rsub(_meta(2, 1), _meta(2, 4)).shape) == (2, 4)
    assert tp.functional.rsub(f16, i).dtype == tp.float16
    assert tp.functional.rsub(z, i).dtype == tp.complex64


def test_extended_loss_shapes_and_dtypes():
    x = _meta(2, 3)
    y = _meta(2, 3)
    assert tuple(tp.nn.functional.huber_loss(x, y).shape) == ()
    assert tuple(tp.nn.functional.huber_loss(x, y, reduction="none").shape) == (2, 3)
    assert tuple(tp._C.huber_loss(_meta(2, 1), y, 0, 1.0).shape) == (2, 3)
    assert tp.nn.functional.huber_loss(x, y).dtype == tp.float32
    # Whole-number operands answer at single precision because the loss
    # arithmetic blends in floating literals; a half input stays half.
    i = _meta(2, 3, dtype=tp.int32)
    assert tp.nn.functional.huber_loss(i, i, reduction="none").dtype == tp.float32
    half = _meta(2, 3, dtype=tp.float16)
    assert tp.nn.functional.huber_loss(half, half, reduction="none").dtype == tp.float16
    assert tp.nn.functional.huber_loss(half, x, reduction="none").dtype == tp.float32
    with pytest.raises(ValueError):
        tp.nn.functional.huber_loss(x, y, delta=0.0)

    assert tuple(tp.nn.functional.soft_margin_loss(x, y).shape) == ()
    assert tuple(tp.nn.functional.soft_margin_loss(x, y, reduction="none").shape) == (2, 3)
    assert tp.nn.functional.soft_margin_loss(i, i).dtype == tp.float32

    # The hinge loss follows the input dtype; the target only broadcasts.
    assert tp.nn.functional.hinge_embedding_loss(
        x, _meta(2, 3, dtype=tp.int64)).dtype == tp.float32
    assert tp.nn.functional.hinge_embedding_loss(
        _meta(2, 3, dtype=tp.float64), x).dtype == tp.float64

    assert tuple(tp.nn.functional.margin_ranking_loss(
        _meta(2, 1), _meta(2, 3), _meta(3), reduction="none").shape) == (2, 3)
    assert tp.nn.functional.margin_ranking_loss(i, i, i).dtype == tp.float32


def test_cosine_embedding_shapes_and_checks():
    x = _meta(3, 4)
    assert tuple(tp.nn.functional.cosine_embedding_loss(x, x, _meta(3)).shape) == ()
    assert tuple(tp.nn.functional.cosine_embedding_loss(
        x, x, _meta(3, 1), reduction="none").shape) == (3, 3)
    assert tuple(tp.nn.functional.cosine_embedding_loss(
        _meta(), _meta(), _meta(4), reduction="none").shape) == (4,)
    with pytest.raises(IndexError):
        tp.nn.functional.cosine_embedding_loss(_meta(4), _meta(4), _meta(4))
    assert tp.nn.functional.cosine_embedding_loss(
        _meta(3, 4, dtype=tp.int32), _meta(3, 4, dtype=tp.int32),
        _meta(3)).dtype == tp.float32


def test_margin_loss_shapes_and_checks():
    x = _meta(4, 5)
    target = _meta(4, dtype=tp.int64)
    assert tuple(tp.nn.functional.multi_margin_loss(x, target).shape) == ()
    assert tuple(tp.nn.functional.multi_margin_loss(
        x, target, reduction="none").shape) == (4,)
    # A vector input counts as one frame, so the unreduced answer keeps the
    # frame axis of length one.
    assert tuple(tp.nn.functional.multi_margin_loss(
        _meta(5), _meta(1, dtype=tp.int64), reduction="none").shape) == (1,)
    assert tp.nn.functional.multi_margin_loss(
        _meta(4, 5, dtype=tp.float64), target).dtype == tp.float64
    with pytest.raises(NotImplementedError):
        tp.nn.functional.multi_margin_loss(_meta(4, 5, dtype=tp.float16), target)
    with pytest.raises(RuntimeError):
        tp.nn.functional.multi_margin_loss(x, _meta(3, dtype=tp.int64))
    with pytest.raises(RuntimeError):
        tp.nn.functional.multi_margin_loss(_meta(4, 0), target)
    with pytest.raises(RuntimeError):
        tp.nn.functional.multi_margin_loss(x, target, weight=_meta(2))
    with pytest.raises(RuntimeError):
        tp._C.multi_margin_loss(x, target, 3, 1.0, None, 1)

    labels = _meta(4, 5, dtype=tp.int64)
    assert tuple(tp.nn.functional.multilabel_margin_loss(x, labels).shape) == ()
    assert tuple(tp.nn.functional.multilabel_margin_loss(
        x, labels, reduction="none").shape) == (4,)
    assert tuple(tp.nn.functional.multilabel_margin_loss(
        _meta(5), _meta(5, dtype=tp.int64), reduction="none").shape) == ()
    output, is_target = tp._C.multilabel_margin_loss_forward(x, labels, 1)
    assert tuple(output.shape) == ()
    assert tuple(is_target.shape) == (4, 5)
    assert is_target.dtype == tp.float32
    with pytest.raises(NotImplementedError):
        tp.nn.functional.multilabel_margin_loss(
            _meta(4, 5, dtype=tp.float16), labels)
    with pytest.raises(RuntimeError):
        tp.nn.functional.multilabel_margin_loss(x, _meta(4, 2, dtype=tp.int64))


def test_logit_losses_run_on_meta():
    # Both losses subtract the target from one, which is the reversed scalar
    # subtraction the pointwise frontend offers.
    x = _meta(2, 3)
    y = _meta(2, 3)
    assert tuple(tp.nn.functional.binary_cross_entropy_with_logits(x, y).shape) == ()
    assert tuple(tp.nn.functional.binary_cross_entropy_with_logits(
        x, y, reduction="none").shape) == (2, 3)
    assert tuple(tp.nn.functional.multilabel_soft_margin_loss(
        _meta(4, 5), _meta(4, 5)).shape) == ()


def test_norm_and_normalize():
    t = _meta(2, 3)
    assert tuple(t.norm().shape) == ()
    assert tuple(t.norm([1], 2.0, True).shape) == (2, 1)
    assert tuple(tp.nn.functional.normalize(t, dim=1).shape) == (2, 3)
    assert tuple(tp.nn.functional.cosine_similarity(t, t, dim=1).shape) == (2,)
    z = _meta(2, 3, dtype=tp.complex64)
    assert tp.abs(z).dtype == tp.float32
    assert z.norm().dtype == tp.float32


def test_threshold_and_isclose():
    t = _meta(2, 3)
    assert tuple(tp.threshold(t, 0.0, 1.0).shape) == (2, 3)
    assert tp.threshold(t, 0.0, 1.0).dtype == tp.float32
    assert tp.threshold(_meta(3, dtype=tp.float16), 0.0, 1.0).dtype == tp.float16
    assert tp.threshold(_meta(3, dtype=tp.bool), 0.0, 1.0).dtype == tp.bool
    with pytest.raises(TypeError):
        tp.threshold(_meta(3, dtype=tp.complex64), 0.0, 1.0)

    assert tp.isclose(t, t).dtype == tp.bool
    assert tuple(tp.isclose(_meta(2, 1), _meta(2, 4)).shape) == (2, 4)
    with pytest.raises(RuntimeError):
        tp.isclose(t, _meta(2, 3, dtype=tp.float16))
    with pytest.raises(RuntimeError):
        tp.isclose(t, t, rtol=-1.0)
    with pytest.raises(RuntimeError):
        tp.isclose(t, t, atol=-1.0)


def test_diff_shapes_and_checks():
    t = _meta(2, 3)
    assert tuple(tp.diff(t).shape) == (2, 2)
    assert tuple(tp.diff(t, 0).shape) == (2, 3)
    assert tuple(tp.diff(t, 5, 1).shape) == (2, 0)
    assert tuple(tp.diff(t, -1).shape) == (2, 3)
    assert tuple(tp.diff(t, 1, 1, prepend=_meta(2, 1)).shape) == (2, 3)
    assert tuple(tp.diff(t, -1, 1, prepend=_meta(2, 1)).shape) == (2, 4)
    assert tp.diff(t, 1, 1, prepend=_meta(2, 1, dtype=tp.float64)).dtype == tp.float64
    # A zero order answers the operand untouched, so a boundary row of the
    # wrong shape is never inspected.
    assert tuple(tp.diff(t, 0, prepend=_meta(3, 3)).shape) == (2, 3)
    with pytest.raises(IndexError):
        tp.diff(t, 1, 2)
    with pytest.raises(RuntimeError):
        tp.diff(_meta(3, dtype=tp.bool), 1)
    assert tuple(tp.diff(_meta(3, dtype=tp.bool), -1).shape) == (3,)


def test_cross_shapes_and_checks():
    a = _meta(2, 3)
    assert tuple(tp.cross(a, a, dim=1).shape) == (2, 3)
    assert tuple(tp.cross(_meta(2, 1, 3), _meta(2, 4, 3), dim=2).shape) == (2, 4, 3)
    with pytest.raises(RuntimeError):
        tp.cross(_meta(4), _meta(4))
    with pytest.raises(RuntimeError):
        tp.cross(a, _meta(2, 4), dim=1)
    with pytest.raises(RuntimeError):
        tp.cross(_meta(1, 3), _meta(3))
    with pytest.raises(RuntimeError):
        tp.cross(a, _meta(2, 3, dtype=tp.float64), dim=1)
    with pytest.raises(NotImplementedError):
        tp.cross(_meta(3, dtype=tp.bool), _meta(3, dtype=tp.bool))
    assert tuple(tp.linalg.cross(a, a).shape) == (2, 3)
    with pytest.raises(IndexError):
        tp.linalg.cross(_meta(3), _meta(3), dim=2)


def test_instance_norm_shapes_and_checks():
    x = _meta(2, 3, 4)
    assert tuple(tp.nn.functional.instance_norm(x).shape) == (2, 3, 4)
    half = tp.nn.functional.instance_norm(_meta(2, 3, 4, dtype=tp.float16))
    assert half.dtype == tp.float16
    assert tuple(tp.nn.functional.instance_norm(
        x, _meta(3), _meta(3), use_input_stats=False).shape) == (2, 3, 4)
    with pytest.raises(RuntimeError):
        tp.nn.functional.instance_norm(_meta(3, 4))
    with pytest.raises(NotImplementedError):
        tp.nn.functional.instance_norm(_meta(2, 3, 4, dtype=tp.int64))
    with pytest.raises(RuntimeError):
        tp.nn.functional.instance_norm(x, use_input_stats=False)


# ---------------------------------------------------------------------------
# Moments, extrema and ordering
# ---------------------------------------------------------------------------


def test_var_std_shapes_and_dtypes():
    t = _meta(3, 4)
    assert tuple(tp.var(t).shape) == ()
    assert tuple(tp.std(t).shape) == ()
    assert tuple(tp.var(t, [1], correction=0).shape) == (3,)
    assert tuple(tp.std(t, [1], keepdim=True).shape) == (3, 1)
    var, mean = tp.std_mean(t, [0], keepdim=True)
    assert tuple(var.shape) == (1, 4)
    assert tuple(mean.shape) == (1, 4)
    assert tp.var(t).dtype == tp.float32
    assert tp.var(_meta(3, dtype=tp.int32)).dtype == tp.float32
    assert tp.std(_meta(3, dtype=tp.float64)).dtype == tp.float64
    assert tp.var(_meta(3, dtype=tp.complex64)).dtype == tp.float32


def test_extrema_shapes():
    t = _meta(3, 4)
    assert tuple(tp.max(t).shape) == ()
    assert tuple(tp.min(t).shape) == ()
    values, indices = tp.max(t, 1)
    assert tuple(values.shape) == (3,)
    assert values.dtype == t.dtype
    assert indices.dtype == tp.int64
    values, indices = tp.min(t, 0, keepdim=True)
    assert tuple(values.shape) == (1, 4)
    assert indices.dtype == tp.int64


def test_sort_and_topk_shapes():
    t = _meta(3, 5)
    values, indices = tp.sort(t, 1, descending=True)
    assert tuple(values.shape) == (3, 5)
    assert tuple(indices.shape) == (3, 5)
    assert values.dtype == t.dtype
    assert indices.dtype == tp.int64
    assert tuple(tp.argsort(t).shape) == (3, 5)
    assert tp.argsort(t).dtype == tp.int64
    top, top_indices = tp.topk(t, 2, dim=1)
    assert tuple(top.shape) == (3, 2)
    assert tuple(top_indices.shape) == (3, 2)
    assert top_indices.dtype == tp.int64
    with pytest.raises(Exception):
        tp.sort(_meta())
    with pytest.raises(Exception):
        tp.topk(t, 6, dim=1)
    with pytest.raises(Exception):
        tp.topk(_meta(4, dtype=tp.bool), 2)


def test_scan_and_outer_shapes():
    t = _meta(2, 3)
    assert tuple(tp.cumsum(t, 1).shape) == (2, 3)
    assert tp.cumsum(t, 1).dtype == tp.float32
    assert tp.cumsum(_meta(4, dtype=tp.int32), 0).dtype == tp.int64
    assert tp.cumsum(_meta(4, dtype=tp.bool), 0).dtype == tp.int64
    assert tp.cumsum(t, 1, dtype=tp.float64).dtype == tp.float64
    assert tuple(tp.cumprod(_meta(), 0).shape) == ()
    assert tuple(tp.cumprod(t, 0).shape) == (2, 3)
    assert tuple(tp.cumsum_(t, 1).shape) == (2, 3)
    assert tuple(tp.cumprod_(_meta(2, 3, dtype=tp.int32), 0).shape) == (2, 3)
    assert tuple(tp.outer(_meta(3), _meta(4)).shape) == (3, 4)
    assert tp.outer(_meta(3), _meta(4, dtype=tp.int32)).dtype == tp.float32
    assert tp.outer(_meta(3, dtype=tp.int32), _meta(4, dtype=tp.int32)).dtype == tp.int32
    with pytest.raises(Exception):
        tp.outer(_meta(2, 3), _meta(4))
    with pytest.raises(Exception):
        tp.cumsum(_meta(), 1)


def test_logical_and_bitwise():
    a = _meta(3, 1, dtype=tp.int32)
    b = _meta(1, 4, dtype=tp.int64)
    assert tuple(tp.logical_and(a, b).shape) == (3, 4)
    assert tp.logical_and(a, b).dtype == tp.bool
    assert tp.logical_not(a).dtype == tp.bool
    assert tuple(tp.logical_or(a, b).shape) == (3, 4)
    assert tp.bitwise_and(a, b).dtype == tp.int64
    assert tuple(tp.bitwise_xor(a, b).shape) == (3, 4)
    assert tp.bitwise_and(a, 3).dtype == tp.int32
    assert tuple(tp.bitwise_or(3, a).shape) == (3, 1)
    assert tp.bitwise_left_shift(a, 1).dtype == tp.int32
    assert tp.bitwise_right_shift(a, b).dtype == tp.int64
    assert tp.bitwise_not(_meta(4, dtype=tp.bool)).dtype == tp.bool
    with pytest.raises(Exception):
        tp.bitwise_and(_meta(4), _meta(4))
    assert tp.bitwise_left_shift(_meta(4, dtype=tp.bool), 1).dtype == tp.int64

    target = _meta(3, 3, dtype=tp.bool)
    assert target.logical_and_(_meta(3, 1, dtype=tp.bool)).shape == (3, 3)
    assert target.logical_or_(_meta(3, 3, dtype=tp.bool)).shape == (3, 3)
    assert target.logical_xor_(_meta(1, 3, dtype=tp.bool)).shape == (3, 3)
    # An operand that widens the result cannot be applied in place.
    with pytest.raises(RuntimeError):
        _meta(3, 1, dtype=tp.bool).logical_and_(_meta(3, 3, dtype=tp.bool))
    with pytest.raises(RuntimeError):
        _meta(3, 3, dtype=tp.bool).logical_or_(_meta(2, 3, dtype=tp.bool))
    with pytest.raises(Exception):
        tp.bitwise_left_shift(_meta(4, dtype=tp.bool), _meta(4, dtype=tp.bool))


# ---------------------------------------------------------------------------
# Indexed access
# ---------------------------------------------------------------------------


def test_gather_and_index_select_shapes():
    t = _meta(2, 3)
    idx = _meta(2, 4, dtype=tp.int64)
    assert tuple(tp.gather(t, 1, idx).shape) == (2, 4)
    assert tp.gather(t, 1, idx).dtype == t.dtype
    assert tuple(tp.index_select(t, 1, _meta(5, dtype=tp.int64)).shape) == (2, 5)
    assert tuple(tp.index_select(t, 0, _meta(dtype=tp.int64)).shape) == (1, 3)
    with pytest.raises(Exception):
        tp.gather(t, 1, _meta(2, 4, 1, dtype=tp.int64))
    with pytest.raises(Exception):
        tp.gather(t, 0, _meta(2, 5, dtype=tp.int64))
    with pytest.raises(Exception):
        tp.index_select(t, 0, _meta(2, 2, dtype=tp.int64))


def test_scatter_and_index_write_shapes():
    t = _meta(2, 3)
    idx = _meta(2, 3, dtype=tp.int64)
    src = _meta(2, 3)
    assert tuple(tp.scatter(t, 1, idx, src).shape) == (2, 3)
    assert tuple(tp.scatter(t, 1, idx, 0.5).shape) == (2, 3)
    assert tuple(tp.scatter_add(t, 1, idx, src).shape) == (2, 3)
    assert tuple(tp.scatter_reduce(t, 1, idx, src, "mean").shape) == (2, 3)
    assert tuple(tp.scatter_(t, 0, idx, src).shape) == (2, 3)
    wide = _meta(1, 3, dtype=tp.int64)
    assert tuple(tp.scatter(t, 0, wide, _meta(1, 3)).shape) == (2, 3)
    assert tuple(tp.index_add(t, 1, _meta(3, dtype=tp.int64), src).shape) == (2, 3)
    assert tuple(tp.index_copy(t, 1, _meta(3, dtype=tp.int64), src).shape) == (2, 3)
    assert tuple(tp.index_fill(t, 1, _meta(2, dtype=tp.int64), 1.0).shape) == (2, 3)
    filled = tp.index_fill_(t, 1, _meta(2, dtype=tp.int64),
                            tp.empty((), dtype=tp.float32, device="meta"))
    assert tuple(filled.shape) == (2, 3)
    assert tuple(tp.index_reduce(t, 1, _meta(3, dtype=tp.int64), src, "prod").shape) == (2, 3)
    with pytest.raises(Exception):
        tp.scatter(t, 1, idx, _meta(2, 2))
    with pytest.raises(Exception):
        tp.scatter_reduce(t, 1, idx, src, "median")
    with pytest.raises(Exception):
        tp.index_add(t, 1, _meta(3, dtype=tp.int64), _meta(2, 2))
    with pytest.raises(Exception):
        tp.index_reduce(t, 1, _meta(4, dtype=tp.int64), src, "sum")


def test_masked_fill_shapes():
    t = _meta(2, 3)
    mask = _meta(3, dtype=tp.bool)
    assert tuple(tp.masked_fill(t, mask, 0.0).shape) == (2, 3)
    assert tuple(tp.masked_fill_(t, mask, 0.0).shape) == (2, 3)
    scalar = tp.empty((), dtype=tp.float32, device="meta")
    assert tuple(tp.masked_fill(t, mask, scalar).shape) == (2, 3)
    with pytest.raises(Exception):
        tp.masked_fill(t, _meta(3), 0.0)
    with pytest.raises(Exception):
        tp.masked_fill(t, mask, _meta(1))


# ---------------------------------------------------------------------------
# Transpose and unfold
# ---------------------------------------------------------------------------


def test_t_and_unfold():
    t = _meta(2, 3)
    assert tuple(tp.t(t).shape) == (3, 2)
    assert tuple(tp.t(t).stride()) == (1, 3)
    assert tuple(tp.t(_meta(4)).shape) == (4,)
    with pytest.raises(Exception):
        tp.t(_meta(2, 3, 4))
    windowed = tp.unfold(_meta(2, 6), 1, 3, 2)
    assert tuple(windowed.shape) == (2, 2, 3)
    assert tuple(windowed.stride()) == (6, 2, 1)
    assert tuple(tp.unfold(_meta(2, 6), 0, 2, 1).shape) == (1, 6, 2)
    scalar = tp.unfold(_meta(), 0, 1, 1)
    assert tuple(scalar.shape) == (1,)
    assert tuple(scalar.stride()) == (1,)
    with pytest.raises(Exception):
        tp.unfold(_meta(2, 6), 1, 7, 1)
    with pytest.raises(Exception):
        tp.unfold(_meta(2, 6), 1, 2, 0)


# ---------------------------------------------------------------------------
# Pointwise unary family, in-place forms and interpolation
# ---------------------------------------------------------------------------


def test_pointwise_unary_family():
    t = _meta(2, 3)
    for name in ("exp2", "expm1", "log2", "log10", "log1p", "tan", "asin",
                 "acos", "atan", "sinh", "cosh", "asinh", "acosh", "atanh",
                 "reciprocal", "square", "sinc", "erf", "erfc", "erfinv",
                 "lgamma", "digamma", "i0", "deg2rad", "rad2deg",
                 "absolute"):
        out = getattr(tp, name)(t)
        assert tuple(out.shape) == (2, 3)
        assert out.dtype == t.dtype
    assert tp.abs(_meta(2, dtype=tp.complex64)).dtype == tp.float32
    assert tp.absolute(_meta(2, dtype=tp.complex64)).dtype == tp.float32

    x = _meta(2, 3)
    for name in ("abs_", "sqrt_", "rsqrt_", "exp_", "exp2_", "expm1_",
                 "log_", "log2_", "log10_", "log1p_", "sin_", "cos_", "tan_",
                 "asin_", "acos_", "atan_", "sinh_", "cosh_", "asinh_",
                 "acosh_", "atanh_", "ceil_", "floor_", "round_",
                 "reciprocal_", "square_", "sinc_", "erf_", "erfc_",
                 "erfinv_", "lgamma_", "digamma_", "i0_", "deg2rad_",
                 "rad2deg_"):
        assert getattr(x, name)() is x


def test_lerp_shapes_and_dtypes():
    t = _meta(2, 3)
    assert tuple(tp.lerp(t, _meta(2, 3), 0.5).shape) == (2, 3)
    assert tp.lerp(t, _meta(3), 0.5).dtype == tp.float32
    assert tp.lerp(_meta(2, 3, dtype=tp.int32),
                   _meta(3, dtype=tp.int32), 2).dtype == tp.float32
    assert tp.lerp(_meta(2, 3, dtype=tp.int32),
                   _meta(3, dtype=tp.int32), 0.5).dtype == tp.float32
    assert tuple(tp.lerp(t, _meta(3), _meta(3)).shape) == (2, 3)
    assert tp.lerp(t, t, _meta(3)).dtype == tp.float32
    assert tp.lerp(t, t, _meta(3, dtype=tp.float64)).dtype == tp.float64
    assert tp.lerp_(t, _meta(3), 0.5) is t
    assert tp.lerp_(t, _meta(3), _meta(3)) is t
    with pytest.raises(Exception):
        tp.lerp(t, t, _meta(2, 4))
    with pytest.raises(Exception):
        tp.lerp_(_meta(3), _meta(2, 3), 0.5)
    with pytest.raises(Exception):
        tp.lerp_(_meta(3), _meta(2, 3), _meta(2, 3))


# ---------------------------------------------------------------------------
# Triangular masks, diagonal embedding and plane rotations
# ---------------------------------------------------------------------------


def test_tril_triu_and_diagonals():
    t = _meta(2, 4)
    assert tuple(tp.tril(t).shape) == (2, 4)
    assert tuple(tp.triu(t, 1).shape) == (2, 4)
    assert tp.tril_(t) is t
    assert tp.triu_(t, -1) is t
    with pytest.raises(Exception):
        tp.tril(_meta(4))

    assert tuple(tp.diag_embed(_meta(3)).shape) == (3, 3)
    assert tuple(tp.diag_embed(_meta(3), 1).shape) == (4, 4)
    assert tuple(tp.diag_embed(_meta(2, 3)).shape) == (2, 3, 3)
    assert tuple(tp.diag_embed(_meta(3), 0, 0, 1).shape) == (3, 3)
    with pytest.raises(Exception):
        tp.diag_embed(_meta(3), 0, 1, 1)
    assert tuple(tp.diagflat(_meta(2, 2)).shape) == (4, 4)
    assert tuple(tp.diagflat(_meta(2, 2), -1).shape) == (5, 5)
    assert tuple(tp.diag(_meta(3)).shape) == (3, 3)
    assert tuple(tp.diag(_meta(2, 4)).shape) == (2,)
    with pytest.raises(Exception):
        tp.diag(_meta(2, 3, 4))


def test_rot90_and_cartesian_prod():
    t = _meta(2, 3, 4)
    turned = tp.rot90(t)
    assert tuple(turned.shape) == (3, 2, 4)
    assert tuple(turned.stride()) == (4, 12, 1)
    assert tuple(tp.rot90(t, 2).shape) == (2, 3, 4)
    assert tuple(tp.rot90(t, 3).shape) == (3, 2, 4)
    assert tuple(tp.rot90(t, 4).shape) == (2, 3, 4)
    assert tuple(tp.rot90(t, 1, [0, 2]).shape) == (4, 3, 2)
    with pytest.raises(Exception):
        tp.rot90(t, 0, [0, 0])
    with pytest.raises(Exception):
        tp.rot90(t, 0, [0])
    with pytest.raises(Exception):
        tp.rot90(_meta(3), 0)

    assert tuple(tp.cartesian_prod(_meta(3), _meta(4)).shape) == (12, 2)
    assert tp.cartesian_prod(_meta(3, dtype=tp.int32),
                             _meta(4)).dtype == tp.float32
    assert tuple(tp.cartesian_prod(_meta(3)).shape) == (3,)
    with pytest.raises(Exception):
        tp.cartesian_prod(_meta(2, 3), _meta(4))
    with pytest.raises(Exception):
        tp.cartesian_prod()


# ---------------------------------------------------------------------------
# Predicates and positional summaries
# ---------------------------------------------------------------------------


def test_all_any_and_logsumexp():
    t = _meta(2, 3)
    assert tuple(tp.all(t).shape) == ()
    assert tp.all(t).dtype == tp.bool
    assert tuple(tp.all(t, 1).shape) == (2,)
    assert tuple(tp.all(t, 1, True).shape) == (2, 1)
    assert tuple(tp.any(t).shape) == ()
    assert tuple(tp.any(t, 0).shape) == (3,)
    assert tp.any(t, 0).dtype == tp.bool

    assert tuple(tp.logsumexp(t, 1).shape) == (2,)
    assert tp.logsumexp(t, 1).dtype == tp.float32
    assert tp.logsumexp(_meta(2, 3, dtype=tp.int32), 1).dtype == tp.float32
    assert tp.logsumexp(_meta(2, 3, dtype=tp.float64), 1).dtype == tp.float64
    assert tuple(tp.logsumexp(t, (0, 1)).shape) == ()
    assert tuple(tp.logsumexp(_meta(), (0,)).shape) == ()
    with pytest.raises(Exception):
        tp.logsumexp(t, [0, 0])
    with pytest.raises(Exception):
        tp.logsumexp(_meta(2, dtype=tp.complex64), 0)


def test_mode_and_kthvalue():
    t = _meta(2, 3)
    values, indices = tp.mode(t, 1)
    assert tuple(values.shape) == (2,)
    assert values.dtype == t.dtype
    assert tuple(indices.shape) == (2,)
    assert indices.dtype == tp.int64
    values, indices = tp.mode(t, -1, True)
    assert tuple(values.shape) == (2, 1)
    assert tuple(tp.mode(_meta(), 0)[0].shape) == ()
    with pytest.raises(Exception):
        tp.mode(_meta(2, 0), 1)

    values, indices = tp.kthvalue(t, 2, 1)
    assert tuple(values.shape) == (2,)
    assert values.dtype == t.dtype
    assert indices.dtype == tp.int64
    assert tuple(tp.kthvalue(t, 1, 0, True)[0].shape) == (1, 3)
    assert tuple(tp.kthvalue(_meta(), 1, 0)[0].shape) == ()
    with pytest.raises(Exception):
        tp.kthvalue(t, 4, 1)
    with pytest.raises(Exception):
        tp.kthvalue(t, 0, 1)


# ---------------------------------------------------------------------------
# Flat gather, lookup tables and whole-tensor writes
# ---------------------------------------------------------------------------


def test_take_searchsorted_and_bucketize():
    t = _meta(2, 3)
    assert tuple(tp.take(t, _meta(4, dtype=tp.int64)).shape) == (4,)
    assert tuple(tp.take(t, _meta(2, 2, dtype=tp.int64)).shape) == (2, 2)
    assert tuple(tp.take(t, _meta(2, 2, 2, dtype=tp.int64)).shape) == (2, 2, 2)
    assert tuple(tp.take(t, _meta(dtype=tp.int64)).shape) == ()

    seq = _meta(5)
    query = _meta(2, 3)
    assert tuple(tp.searchsorted(seq, query).shape) == (2, 3)
    assert tp.searchsorted(seq, query).dtype == tp.int64
    assert tp.searchsorted(seq, query, out_int32=True).dtype == tp.int32
    assert tuple(tp.searchsorted(seq, 2.0).shape) == ()
    assert tuple(tp.searchsorted(_meta(2, 5), _meta(2, 4)).shape) == (2, 4)
    with pytest.raises(Exception):
        tp.searchsorted(_meta(1, 2), _meta(2, 2))
    with pytest.raises(Exception):
        tp.searchsorted(seq, query, side="middle")

    assert tuple(tp.bucketize(query, seq).shape) == (2, 3)
    assert tp.bucketize(query, seq, out_int32=True, right=True).dtype == tp.int32
    assert tuple(tp.bucketize(1.0, seq).shape) == ()
    with pytest.raises(Exception):
        tp.bucketize(query, _meta(2, 3))


def test_index_put_shapes():
    t = _meta(4, 3)
    rows = (_meta(2, dtype=tp.int64),)
    written = tp.index_put(t, rows, _meta(2, 3))
    assert tuple(written.shape) == (4, 3)
    assert written.dtype == t.dtype
    assert tuple(tp.index_put(t, rows, _meta(1, 3)).shape) == (4, 3)
    assert tuple(tp.index_put(t, (_meta(4, 3, dtype=tp.bool),),
                              tp.empty((), dtype=tp.float32, device="meta")).shape) == (4, 3)
    assert tuple(tp.index_put_(t, rows, _meta(2, 3)).shape) == (4, 3)
    with pytest.raises(Exception):
        tp.index_put(t, rows, _meta(3, 3))
    with pytest.raises(Exception):
        tp.index_put(t, rows, _meta(2, 3, dtype=tp.int32))
    with pytest.raises(Exception):
        tp.index_put(t, (_meta(2, dtype=tp.int64),
                         _meta(2, dtype=tp.int64),
                         _meta(2, dtype=tp.int64),
                         _meta(2, dtype=tp.int64)), _meta(2))


# ---------------------------------------------------------------------------
# Pointwise maps beside the arithmetic family
# ---------------------------------------------------------------------------


def test_modulo_and_division_family():
    a = _meta(2, 3, dtype=tp.int32)
    b = _meta(2, 3, dtype=tp.int64)
    assert tp.fmod(a, b).dtype == tp.int64
    assert tuple(tp.fmod(a, _meta(3, dtype=tp.float32)).shape) == (2, 3)
    assert tp.fmod(a, _meta(3, dtype=tp.float32)).dtype == tp.float32
    assert tp.remainder(a, 2.5).dtype == tp.float32
    assert tuple(tp.remainder(2.5, a).shape) == (2, 3)
    assert tp.floor_divide(a, b).dtype == tp.int64
    assert tp.floor_divide(a, 2.5).dtype == tp.float32
    assert tp.true_divide(a, b).dtype == tp.float32
    assert tp.true_divide(a, 2.0).dtype == tp.float32
    with pytest.raises(Exception):
        tp.fmod(_meta(3, dtype=tp.complex64), _meta(3, dtype=tp.complex64))
    with pytest.raises(Exception):
        tp.floor_divide(a, tp.tensor(1 + 2j, device="meta"))


def test_predicate_family_shapes():
    t = _meta(2, 3)
    for predicate in (tp.isnan, tp.isinf, tp.isfinite, tp.isneginf,
                      tp.isposinf, tp.signbit):
        answer = predicate(t)
        assert tuple(answer.shape) == (2, 3)
        assert answer.dtype == tp.bool
    assert tp.isreal(_meta(2, dtype=tp.complex64)).dtype == tp.bool
    assert tp.isreal(t).dtype == tp.bool
    assert tp.is_complex(t) is False
    assert tp.is_complex(_meta(2, dtype=tp.complex64)) is True
    with pytest.raises(Exception):
        tp.isnan(_meta(2, dtype=tp.complex64))


def test_positive_entr_logit_nan_to_num():
    t = _meta(2, 3, dtype=tp.int32)
    assert tp.positive(t).dtype == tp.int32
    assert tuple(tp.positive(t).shape) == (2, 3)
    assert tp.entr(t).dtype == tp.float32
    assert tp.entr(_meta(2, dtype=tp.float64)).dtype == tp.float64
    assert tp.entr(_meta(2, dtype=tp.complex64)).dtype == tp.float32
    assert tp.logit(t).dtype == tp.float32
    assert tp.logit(_meta(2, dtype=tp.float64), eps=1e-6).dtype == tp.float64
    assert tp.nan_to_num(t).dtype == tp.int32
    assert tuple(tp.nan_to_num(_meta(2), nan=0.0, posinf=1.0).shape) == (2,)
    with pytest.raises(Exception):
        tp.nan_to_num(_meta(2, dtype=tp.complex64))


def test_sign_and_transcendental_pairs():
    a = _meta(2, 3, dtype=tp.int32)
    b = _meta(2, 3, dtype=tp.int64)
    assert tp.copysign(a, b).dtype == tp.float32
    assert tp.nextafter(a, b).dtype == tp.float32
    assert tp.hypot(a, b).dtype == tp.float32
    assert tp.atan2(a, b).dtype == tp.float32
    assert tp.heaviside(a, b).dtype == tp.int64
    assert tp.xlogy(_meta(3, dtype=tp.float64),
                    _meta(3, dtype=tp.float64)).dtype == tp.float64
    assert tp.copysign(_meta(3, dtype=tp.float32), 1.0).dtype == tp.float32
    assert tuple(tp.hypot(_meta(2, 1), _meta(1, 4)).shape) == (2, 4)
    assert tuple(tp.atan2_(_meta(2, 3), _meta(2, 3)).shape) == (2, 3)
    assert tuple(tp.heaviside_(_meta(2, 3), _meta(2, 3)).shape) == (2, 3)


# ---------------------------------------------------------------------------
# Vector products and the batch accumulators
# ---------------------------------------------------------------------------


def test_dot_mv_inner_ger():
    assert tuple(tp.dot(_meta(3), _meta(3)).shape) == ()
    assert tp.dot(_meta(3, dtype=tp.int32),
                  _meta(3, dtype=tp.int32)).dtype == tp.int32
    with pytest.raises(Exception):
        tp.dot(_meta(3), _meta(3, dtype=tp.int32))
    with pytest.raises(Exception):
        tp.dot(_meta(2, 3), _meta(2, 3))

    assert tuple(tp.mv(_meta(2, 3), _meta(3)).shape) == (2,)
    with pytest.raises(Exception):
        tp.mv(_meta(2, 3), _meta(2))
    with pytest.raises(Exception):
        tp.mv(_meta(2, 3), _meta(3, dtype=tp.float64))

    assert tuple(tp.inner(_meta(2, 3), _meta(4, 3)).shape) == (2, 4)
    assert tuple(tp.inner(_meta(2, 3), _meta(3)).shape) == (2,)
    assert tp.inner(_meta(2, 3), _meta(3)).dtype == tp.float32
    with pytest.raises(Exception):
        tp.inner(_meta(2, 3, dtype=tp.float32),
                 _meta(4, 3, dtype=tp.float64))
    with pytest.raises(Exception):
        tp.inner(_meta(2, 3), _meta(4, 5))

    assert tuple(tp.ger(_meta(2), _meta(4)).shape) == (2, 4)
    assert tp.ger(_meta(2, dtype=tp.bool),
                  _meta(3, dtype=tp.bool)).dtype == tp.bool


def test_baddbmm_and_addbmm_shapes():
    batch1 = _meta(4, 3, 5, dtype=tp.float32)
    batch2 = _meta(4, 5, 6, dtype=tp.float32)
    seed = _meta(4, 3, 6, dtype=tp.float32)
    assert tuple(tp.baddbmm(seed, batch1, batch2).shape) == (4, 3, 6)
    # The addend may broadcast into the batch.
    assert tuple(tp.baddbmm(_meta(3, 6), batch1, batch2).shape) == (4, 3, 6)
    whole = tp.baddbmm(_meta(4, 3, 6, dtype=tp.int32),
                       _meta(4, 3, 5, dtype=tp.int32),
                       _meta(4, 5, 6, dtype=tp.int32))
    assert whole.dtype == tp.float32
    with pytest.raises(Exception):
        tp.baddbmm(seed, batch1, _meta(4, 5, 6, dtype=tp.float64))
    with pytest.raises(Exception):
        tp.baddbmm(_meta(4, 2, 6), batch1, batch2)

    assert tuple(tp.addbmm(_meta(3, 6), batch1, batch2).shape) == (3, 6)
    assert tp.addbmm(_meta(3, 6, dtype=tp.int32),
                     _meta(4, 3, 5, dtype=tp.int32),
                     _meta(4, 5, 6, dtype=tp.int32)).dtype == tp.int32
    with pytest.raises(Exception):
        tp.addbmm(_meta(1, 3, 6), batch1, batch2)
    with pytest.raises(Exception):
        tp.addbmm(_meta(3, 6, dtype=tp.int32),
                  _meta(4, 3, 5, dtype=tp.int32),
                  _meta(4, 5, 6, dtype=tp.int32), beta=0.5)


def test_addr_and_addmv_shapes():
    assert tuple(tp.addr(_meta(2, 3), _meta(2), _meta(3)).shape) == (2, 3)
    assert tuple(tp.addr(_meta(), _meta(2), _meta(3)).shape) == (2, 3)
    assert tp.addr(_meta(2, 3, dtype=tp.bool), _meta(2, dtype=tp.bool),
                   _meta(3, dtype=tp.bool)).dtype == tp.bool
    with pytest.raises(Exception):
        tp.addr(_meta(4, 4), _meta(2), _meta(3))
    with pytest.raises(Exception):
        tp.addr(_meta(2, 1, dtype=tp.int32), _meta(2, 1, dtype=tp.int32),
                _meta(3, dtype=tp.int32))
    # An integral addend with a zero weight skips the broadcast check.
    assert tuple(tp.addr(_meta(4, 4, dtype=tp.int32),
                         _meta(2, dtype=tp.int32),
                         _meta(3, dtype=tp.int32), beta=0).shape) == (2, 3)
    assert tuple(tp.addr_(_meta(2, 3), _meta(2), _meta(3)).shape) == (2, 3)
    with pytest.raises(Exception):
        tp.addr_(_meta(4, 4), _meta(2), _meta(3))

    assert tuple(tp.addmv(_meta(2), _meta(2, 3), _meta(3)).shape) == (2,)
    assert tuple(tp.addmv(_meta(), _meta(2, 3), _meta(3)).shape) == (2,)
    assert tp.addmv(_meta(2, dtype=tp.int32), _meta(2, 3, dtype=tp.int32),
                    _meta(3, dtype=tp.int32)).dtype == tp.int32
    assert tuple(tp.addmv(_meta(4, 4, dtype=tp.int32),
                          _meta(2, 3, dtype=tp.int32),
                          _meta(3, dtype=tp.int32), beta=0).shape) == (2,)
    assert tuple(tp.addmv_(_meta(2), _meta(2, 3), _meta(3)).shape) == (2,)
    with pytest.raises(Exception):
        tp.addmv_(_meta(3), _meta(2, 3), _meta(3))
    with pytest.raises(Exception):
        tp.addmv(_meta(2), _meta(3, 3), _meta(3))
    with pytest.raises(Exception):
        tp.addmv(_meta(2), _meta(2, 3), _meta(2))
    with pytest.raises(Exception):
        tp.addmv(_meta(2, dtype=tp.bool), _meta(2, 3, dtype=tp.bool),
                 _meta(3, dtype=tp.bool))


def test_addcmul_addcdiv_shapes():
    assert tuple(tp.addcmul(_meta(2, 1), _meta(1, 3), _meta(2, 3)).shape) == (2, 3)
    assert tp.addcmul(_meta(2, 3, dtype=tp.int32), _meta(2, 3, dtype=tp.int32),
                      _meta(2, 3, dtype=tp.int32)).dtype == tp.int32
    assert tp.addcmul(_meta(2, dtype=tp.int32), _meta(2, dtype=tp.float32),
                      _meta(2, dtype=tp.int32)).dtype == tp.float32
    assert tp.addcmul_(_meta(2, 3, dtype=tp.int32),
                       _meta(2, 3, dtype=tp.float32),
                       _meta(2, 3, dtype=tp.int32)).dtype == tp.int32
    with pytest.raises(Exception):
        tp.addcmul_(_meta(2, 1), _meta(2, 3), _meta(2, 3))

    with pytest.raises(Exception):
        tp.addcdiv(_meta(2, 3, dtype=tp.int32), _meta(2, 3, dtype=tp.int32),
                   _meta(2, 3, dtype=tp.int32))
    assert tp.addcdiv(_meta(2, 3, dtype=tp.int32), _meta(2, 3, dtype=tp.int32),
                      _meta(2, 3, dtype=tp.float32)).dtype == tp.float32
    assert tuple(tp.addcdiv(_meta(2, 1), _meta(1, 3), _meta(2, 3)).shape) == (2, 3)
    assert tuple(tp.addcdiv_(_meta(2, 3), _meta(2, 3), _meta(2, 3)).shape) == (2, 3)
    with pytest.raises(Exception):
        tp.addcdiv_(_meta(2, 1), _meta(2, 3), _meta(2, 3))


def test_angle_arange_start_and_private_aminmax():
    assert tp.angle(_meta(2, 3)).dtype == tp.float32
    assert tp.angle(_meta(2, 3, dtype=tp.float64)).dtype == tp.float64
    assert tp.angle(_meta(2, 3, dtype=tp.int32)).dtype == tp.float32
    assert tp.angle(_meta(2, 3, dtype=tp.complex64)).dtype == tp.float32
    assert tuple(tp.angle(_meta(2, 3)).shape) == (2, 3)
    with pytest.raises(Exception):
        tp.angle(_meta(2, dtype=tp.bool))

    assert tuple(tp.arange(2, 9).shape) == (7,)
    assert tp.arange(2, 9).dtype == tp.int64
    assert tuple(tp.arange(2, 9, 2).shape) == (4,)
    assert tuple(tp.arange(5, 2).shape) == (0,)
    steps = tp.arange(0.5, 2.0, 0.5)
    assert tuple(steps.shape) == (3,)
    assert steps.dtype == tp.float32
    with pytest.raises(Exception):
        tp.arange(0, 5, 0)

    low, high = tp._C._aminmax(_meta(3, 4))
    assert tuple(low.shape) == ()
    assert high.dtype == tp.float32
    low, high = tp._C._aminmax(_meta(3, 4), 1, True)
    assert tuple(low.shape) == (3, 1)
    assert tuple(high.shape) == (3, 1)
    low, high = tp._C._aminmax(_meta(3, 4, dtype=tp.int32))
    assert low.dtype == tp.int32
    assert high.dtype == tp.int32
    with pytest.raises(Exception):
        tp._C._aminmax(_meta(3, 4), 2, False)


# ---------------------------------------------------------------------------
# Random fills and module construction
# ---------------------------------------------------------------------------


def test_random_fills_keep_metadata():
    t = _meta(4, 5)
    assert t.uniform_(0.0, 1.0) is t
    assert t.normal_(0.0, 1.0) is t
    assert t.normal_() is t
    assert t.exponential_(1.5) is t
    assert t.log_normal_(0.0, 0.5) is t
    assert t.cauchy_(0.0, 1.0) is t
    assert t.bernoulli_(0.5) is t
    assert t.bernoulli_(_meta(4, 5)) is t
    assert tuple(t.shape) == (4, 5)
    assert t.dtype == tp.float32
    assert t.is_meta

    whole = _meta(3, 4, dtype=tp.int32)
    assert whole.random_() is whole
    assert whole.random_(10) is whole
    assert whole.random_(1, 10) is whole
    assert tuple(whole.shape) == (3, 4)
    assert whole.dtype == tp.int32
    assert tuple(_meta(3, dtype=tp.int64).geometric_(0.5).shape) == (3,)
    assert tuple(_meta(2, 2, dtype=tp.float64).uniform_(1.0, 2.0).shape) == (2, 2)


def test_random_fill_checks():
    t = _meta(2, 3)
    with pytest.raises(Exception):
        _meta(2, 3, dtype=tp.int32).uniform_(0.0, 1.0)
    with pytest.raises(Exception):
        t.uniform_(1.0, 0.0)
    with pytest.raises(Exception):
        t.uniform_(0.0, 1e300)
    with pytest.raises(Exception):
        t.normal_(0.0, -1.0)
    with pytest.raises(Exception):
        t.exponential_(0.0)
    with pytest.raises(Exception):
        t.log_normal_(0.0, 0.0)
    with pytest.raises(Exception):
        t.cauchy_(0.0, 0.0)
    with pytest.raises(Exception):
        t.geometric_(1.0)
    with pytest.raises(Exception):
        t.bernoulli_(1.5)
    with pytest.raises(Exception):
        t.bernoulli_(_meta(2, 3, dtype=tp.bool))
    with pytest.raises(Exception):
        _meta(2, 3, dtype=tp.int32).normal_(0.0, 1.0)
    with pytest.raises(Exception):
        _meta(2, 3, dtype=tp.int32).random_(10, 10)
    with pytest.raises(Exception):
        _meta(2, 3, dtype=tp.int8).random_(0, 1000)
    expanded = _meta(1, 3).expand(2, 3)
    with pytest.raises(Exception):
        expanded.normal_()


def test_module_construction_on_meta():
    with tp.device("meta"):
        linear = tp.nn.Linear(4, 3)
        conv = tp.nn.Conv2d(3, 8, 3, padding=1)
        embedding = tp.nn.Embedding(10, 5)
        norm = tp.nn.LayerNorm(6)
        batchnorm = tp.nn.BatchNorm1d(6)
    assert linear.weight.is_meta
    assert tuple(linear.weight.shape) == (3, 4)
    assert tuple(linear.bias.shape) == (3,)
    assert tuple(conv.weight.shape) == (8, 3, 3, 3)
    assert tuple(embedding.weight.shape) == (10, 5)
    assert tuple(norm.weight.shape) == (6,)
    assert tuple(batchnorm.running_mean.shape) == (6,)
    with tp.device("meta"):
        y = linear(tp.randn(2, 4))
    assert y.is_meta
    assert tuple(y.shape) == (2, 3)


# ---------------------------------------------------------------------------
# Convolutions, resampling and renormalization
# ---------------------------------------------------------------------------


def test_conv1d_conv3d_conv_transpose2d():
    assert tuple(tp.conv1d(_meta(1, 2, 8), _meta(3, 2, 3)).shape) == (1, 3, 6)
    assert tuple(tp.conv1d(_meta(1, 2, 8), _meta(3, 2, 3), _meta(3),
                           stride=2, padding=1).shape) == (1, 3, 4)
    assert tuple(tp.conv3d(_meta(1, 2, 4, 4, 4),
                           _meta(3, 2, 2, 2, 2)).shape) == (1, 3, 3, 3, 3)
    assert tuple(tp.conv_transpose2d(_meta(1, 2, 4, 4),
                                     _meta(2, 3, 3, 3),
                                     stride=2).shape) == (1, 3, 9, 9)
    assert tuple(tp.conv_transpose2d(_meta(1, 2, 4, 4),
                                     _meta(2, 3, 3, 3), stride=1,
                                     padding=1).shape) == (1, 3, 4, 4)
    with pytest.raises(Exception):
        tp.conv1d(_meta(1, 2, 8), _meta(3, 3, 3))
    with pytest.raises(Exception):
        tp.conv3d(_meta(1, 2, 4, 4), _meta(3, 2, 2, 2, 2))


def test_conv_transpose1d_and_3d():
    # (in - 1) * stride - 2 * padding + dilation * (kernel - 1) + op + 1
    assert tuple(tp.conv_transpose1d(_meta(2, 3, 8),
                                     _meta(3, 2, 3)).shape) == (2, 2, 10)
    assert tuple(tp.conv_transpose1d(_meta(2, 3, 8), _meta(3, 2, 3), _meta(2),
                                     stride=2, padding=1, dilation=2).shape) == \
        (2, 2, 17)
    assert tuple(tp.conv_transpose3d(_meta(1, 2, 4, 4, 4),
                                     _meta(2, 3, 2, 2, 2),
                                     stride=2).shape) == (1, 3, 8, 8, 8)
    assert tuple(tp.conv_transpose3d(_meta(1, 2, 4, 4, 4),
                                     _meta(2, 3, 3, 3, 3),
                                     stride=[2, 1, 1], padding=1,
                                     output_padding=[1, 0, 0]).shape) == \
        (1, 3, 8, 4, 4)
    # groups split the weight's trailing channel extent
    assert tuple(tp.conv_transpose1d(_meta(1, 4, 8), _meta(4, 2, 3),
                                     groups=2).shape) == (1, 4, 10)
    # a zero output extent is an empty tensor, a negative one cannot exist
    assert tuple(tp.conv_transpose1d(_meta(2, 3, 0),
                                     _meta(3, 2, 1)).shape) == (2, 2, 0)
    with pytest.raises(Exception):
        tp.conv_transpose1d(_meta(2, 8), _meta(3, 2, 3))
    with pytest.raises(Exception):
        tp.conv_transpose1d(_meta(2, 3, 8), _meta(2, 3, 3))
    with pytest.raises(Exception):
        tp.conv_transpose3d(_meta(1, 2, 4, 4), _meta(2, 3, 2, 2, 2))
    with pytest.raises(Exception):
        tp.conv_transpose3d(_meta(1, 2, 4, 4, 4), _meta(2, 3, 2, 2, 2),
                            stride=[1, 2])


def test_conv_transpose2d_list_shapes():
    # the per-side form of a parameter list keeps its axis entries
    assert tuple(tp.conv_transpose2d(_meta(1, 2, 4, 4), _meta(2, 3, 3, 3),
                                     stride=[2, 1, 2, 1]).shape) == (1, 3, 9, 6)
    # no extent is policed beyond the formula: what a materialized backend
    # computes, meta names
    assert tuple(tp.conv_transpose2d(_meta(1, 2, 4, 4), _meta(2, 3, 3, 3),
                                     stride=2, output_padding=3).shape) == \
        (1, 3, 12, 12)
    with pytest.raises(RuntimeError):
        tp.conv_transpose2d(_meta(2, 3, 8, 8), _meta(3, 2, 3, 3), padding=10)
    with pytest.raises(RuntimeError):
        tp.conv_transpose2d(_meta(2, 3, 8, 8), _meta(3, 2, 3, 3),
                            stride=[1, 2, 3])
    with pytest.raises(RuntimeError):
        tp.conv_transpose2d(_meta(2, 3, 8, 8), _meta(3, 2, 3, 3), stride=[])


def test_upsample_bilinear2d_and_pixel_shuffle():
    assert tuple(tp.upsample_bilinear2d(_meta(2, 3, 4, 5), [8, 10],
                                        align_corners=False).shape) == (2, 3, 8, 10)
    with pytest.raises(Exception):
        tp.upsample_bilinear2d(_meta(2, 3, 4, 5), [-2, 10],
                               align_corners=False)
    assert tuple(tp.pixel_shuffle(_meta(2, 8, 3, 4), 2).shape) == (2, 2, 6, 8)
    with pytest.raises(Exception):
        tp.pixel_shuffle(_meta(2, 2), 2)
    with pytest.raises(Exception):
        tp.pixel_shuffle(_meta(2, 5, 3, 4), 2)


def test_upsample_forward_family_shapes_and_gates():
    plain = [
        ("upsample_nearest1d", (2, 3, 4), [8], ()),
        ("upsample_linear1d", (2, 3, 4), [8], (False,)),
        ("upsample_nearest2d", (2, 3, 4, 5), [6, 7], ()),
        ("upsample_bilinear2d", (2, 3, 4, 5), [6, 7], (False,)),
        ("upsample_bicubic2d", (2, 3, 4, 5), [6, 7], (False,)),
        ("upsample_nearest3d", (1, 2, 2, 3, 4), [5, 6, 7], ()),
        ("upsample_trilinear3d", (1, 2, 2, 3, 4), [5, 6, 7], (False,)),
    ]
    for name, shape, size, extra in plain:
        out = getattr(tp, name)(_meta(*shape), size, *extra)
        assert tuple(out.shape) == tuple(shape[:2]) + tuple(size)
        assert out.dtype == tp.float32 and out.is_meta
        assert getattr(tp, name)(_meta(*shape, dtype=tp.float64), size,
                                 *extra).dtype == tp.float64
        with pytest.raises(NotImplementedError):
            getattr(tp, name)(_meta(*shape, dtype=tp.int64), size, *extra)
    # negative extents are rejected while the answer is being sized
    with pytest.raises(RuntimeError):
        tp.upsample_nearest1d(_meta(2, 3, 4), [-1])
    # extra recorded entries ride along in the answer
    assert tuple(tp.upsample_nearest2d(_meta(2, 3, 4, 5), [6, 7, 8]).shape) == (2, 3, 6, 7, 8)
    # empty operands and zero extents return before any values are needed
    assert tuple(tp.upsample_nearest2d(_meta(0, 3, 4, 5), [6, 7]).shape) == (0, 3, 6, 7)
    assert tuple(tp.upsample_bilinear2d(_meta(2, 3, 4, 5), [0, 7],
                                        align_corners=False).shape) == (2, 3, 0, 7)
    assert tuple(tp.upsample_nearest2d(_meta(2, 3, 4, 5, dtype=tp.int64),
                                       [0, 0]).shape) == (2, 3, 0, 0)
    # every output size entry lands in the answer, so a trailing zero shrinks
    # it under the sweep the kernel would perform
    with pytest.raises(RuntimeError):
        tp.upsample_nearest1d(_meta(2, 3, 4), [8, 0])
    with pytest.raises(RuntimeError):
        tp.upsample_nearest2d(_meta(2, 3, 4, 5), [6, 7, 0])
    # sizes below the spatial rank cannot be read at all
    with pytest.raises(ValueError):
        tp.upsample_nearest1d(_meta(2, 3, 4), [])
    with pytest.raises(ValueError):
        tp.upsample_nearest2d(_meta(2, 3, 4, 5), [6])
    with pytest.raises(RuntimeError):
        tp.upsample_nearest1d(_meta(3, 4), [8])
    with pytest.raises(RuntimeError):
        tp.upsample_nearest2d(_meta(2, 3, 4), [6, 7])


def test_upsample_private_forward_names():
    f = tp.functional
    assert tuple(f._upsample_nearest_exact1d(_meta(2, 3, 4), [8], None).shape) == (2, 3, 8)
    assert tuple(f._upsample_nearest_exact2d(_meta(2, 3, 4, 5), [6, 7], None).shape) == (2, 3, 6, 7)
    assert tuple(f._upsample_nearest_exact3d(_meta(1, 2, 2, 3, 4), [5, 6, 7],
                                             None).shape) == (1, 2, 5, 6, 7)
    assert tuple(f._upsample_bilinear2d_aa(_meta(2, 3, 4, 5), [6, 7], False,
                                           None).shape) == (2, 3, 6, 7)
    assert tuple(f._upsample_bicubic2d_aa(_meta(2, 3, 4, 5), [6, 7], False,
                                          None).shape) == (2, 3, 6, 7)
    with pytest.raises(NotImplementedError):
        f._upsample_bilinear2d_aa(_meta(2, 3, 4, 5, dtype=tp.int64), [6, 7],
                                  False, None)


def test_upsample_aa_vec_forms():
    f = tp.functional
    x = _meta(2, 3, 4, 5)
    # an explicit output size wins over scale factors
    assert tuple(f._upsample_bilinear2d_aa(x, [6, 7], False,
                                           [2.0, 2.0]).shape) == (2, 3, 6, 7)
    # scale factors floor to the output extents
    assert tuple(f._upsample_bilinear2d_aa(x, None, False,
                                           [2.0, 2.0]).shape) == (2, 3, 8, 10)
    assert tuple(f._upsample_bicubic2d_aa(x, None, False,
                                          [0.5, 0.5]).shape) == (2, 3, 2, 2)
    # with neither argument the call is rejected outright
    with pytest.raises(RuntimeError):
        f._upsample_bilinear2d_aa(x, None, False, None)
    # both lists must carry exactly two entries; a valid output size answers
    # before scale factors are ever inspected
    with pytest.raises(RuntimeError):
        f._upsample_bilinear2d_aa(x, [6], False, [2.0, 2.0])
    with pytest.raises(RuntimeError):
        f._upsample_bicubic2d_aa(x, None, False, [2.0])
    assert tuple(f._upsample_bicubic2d_aa(x, [6, 7], False,
                                          [2.0]).shape) == (2, 3, 6, 7)
    # scale factors below one floor the extents to zero through the plain path
    with pytest.raises(RuntimeError):
        f._upsample_bilinear2d_aa(x, None, False, [-1.0, -1.0])


def test_upsample_backward_family_shapes_and_checks():
    cases = [
        ("upsample_nearest1d_backward", (2, 3, 8), [8], (2, 3, 4), ()),
        ("upsample_linear1d_backward", (2, 3, 8), [8], (2, 3, 4), (False,)),
        ("upsample_nearest2d_backward", (2, 3, 8, 9), [8, 9], (2, 3, 4, 5), ()),
        ("upsample_bilinear2d_backward", (2, 3, 8, 9), [8, 9], (2, 3, 4, 5), (False,)),
        ("upsample_bicubic2d_backward", (2, 3, 8, 9), [8, 9], (2, 3, 4, 5), (False,)),
        ("upsample_nearest3d_backward", (1, 2, 5, 6, 7), [5, 6, 7], (1, 2, 2, 3, 4), ()),
        ("upsample_trilinear3d_backward", (1, 2, 5, 6, 7), [5, 6, 7],
         (1, 2, 2, 3, 4), (False,)),
    ]
    for name, goshape, size, insize, extra in cases:
        out = getattr(tp, name)(_meta(*goshape), size, list(insize), *extra)
        assert tuple(out.shape) == tuple(insize)
        assert out.dtype == tp.float32 and out.is_meta
        with pytest.raises(NotImplementedError):
            getattr(tp, name)(_meta(*goshape, dtype=tp.int64), size,
                              list(insize), *extra)
        with pytest.raises(RuntimeError):
            getattr(tp, name)(_meta(*goshape[:-1]), size, list(insize), *extra)
    with pytest.raises(RuntimeError):
        tp.upsample_nearest2d_backward(_meta(2, 3, 8, 9), [8, 9], [2, 3, -1, 5])
    # empty gradients and zero extents skip straight to the zero answer
    assert tuple(tp.upsample_nearest2d_backward(_meta(0, 3, 8, 9), [8, 9],
                                                [2, 3, 4, 5]).shape) == (2, 3, 4, 5)
    assert tuple(tp.upsample_bilinear2d_backward(_meta(2, 3, 8, 9), [0, 9],
                                                 [2, 3, 4, 5],
                                                 align_corners=False).shape) == (2, 3, 4, 5)
    f = tp.functional
    assert tuple(f._upsample_bilinear2d_aa_backward(_meta(2, 3, 8, 9), [8, 9],
                                                    [2, 3, 4, 5],
                                                    False).shape) == (2, 3, 4, 5)
    assert tuple(f._upsample_bicubic2d_aa_backward(_meta(2, 3, 8, 9), [8, 9],
                                                   [2, 3, 4, 5],
                                                   False).shape) == (2, 3, 4, 5)
    assert tuple(f._upsample_nearest_exact1d_backward(_meta(2, 3, 8), [8],
                                                      [2, 3, 4],
                                                      None).shape) == (2, 3, 4)
    with pytest.raises(NotImplementedError):
        f._upsample_bilinear2d_aa_backward(_meta(2, 3, 8, 9, dtype=tp.int64),
                                           [8, 9], [2, 3, 4, 5], False)


def test_upsample_backward_polices_recorded_geometry():
    # a gradient sweep that would read past the recorded gradient, or write
    # past the recorded answer, is refused on meta
    with pytest.raises(RuntimeError):
        tp.upsample_nearest2d_backward(_meta(2, 3, 8, 9), [100, 100],
                                       [2, 3, 4, 5])
    with pytest.raises(RuntimeError):
        tp.upsample_bilinear2d_backward(_meta(2, 6, 8, 9), [8, 9],
                                        [2, 3, 4, 5], align_corners=False)
    # output sizes below the spatial rank cannot be read
    with pytest.raises(ValueError):
        tp.upsample_nearest2d_backward(_meta(2, 3, 8, 9), [8], [2, 3, 4, 5])
    with pytest.raises(ValueError):
        tp.upsample_bilinear2d_backward(_meta(2, 3, 8, 9), [8], [2, 3, 4, 5],
                                        align_corners=False)
    # gradients below rank 2 read out of bounds on a materialized device
    with pytest.raises(RuntimeError):
        tp.upsample_nearest2d_backward(_meta(8), [], [])
    with pytest.raises(RuntimeError):
        tp.upsample_nearest1d_backward(_meta(8), [8], [4])


def test_activation_backward_family_shapes():
    grad = _meta(2, 3, 4)
    # gradient-shaped answers at the gradient's dtype, for every scalar family
    plain = [
        lambda dt: tp.gelu_backward(_meta(2, 3, 4, dtype=dt), _meta(2, 3, 4, dtype=dt)),
        lambda dt: tp.hardtanh_backward(_meta(2, 3, 4, dtype=dt),
                                        _meta(2, 3, 4, dtype=dt), -1.0, 1.0),
        lambda dt: tp.hardswish_backward(_meta(2, 3, 4, dtype=dt),
                                         _meta(2, 3, 4, dtype=dt)),
        lambda dt: tp.silu_backward(_meta(2, 3, 4, dtype=dt), _meta(2, 3, 4, dtype=dt)),
        lambda dt: tp.hardsigmoid_backward(_meta(2, 3, 4, dtype=dt),
                                           _meta(2, 3, 4, dtype=dt)),
        lambda dt: tp.leaky_relu_backward(_meta(2, 3, 4, dtype=dt),
                                          _meta(2, 3, 4, dtype=dt), 0.01, False),
        lambda dt: tp.elu_backward(_meta(2, 3, 4, dtype=dt), 1.0, 1.25, 1.0, False,
                                   _meta(2, 3, 4, dtype=dt)),
        lambda dt: tp.softplus_backward(_meta(2, 3, 4, dtype=dt),
                                        _meta(2, 3, 4, dtype=dt), 1.0, 20.0),
        lambda dt: tp.mish_backward(_meta(2, 3, 4, dtype=dt), _meta(2, 3, 4, dtype=dt)),
        lambda dt: tp.threshold_backward(_meta(2, 3, 4, dtype=dt),
                                         _meta(2, 3, 4, dtype=dt), 0.5),
    ]
    for call in plain:
        out = call(tp.float32)
        assert tuple(out.shape) == (2, 3, 4) and out.dtype == tp.float32
        assert call(tp.float64).dtype == tp.float64
        assert call(tp.int64).dtype == tp.int64
    # an empty gradient returns before the saved plane is ever touched
    empty = _meta(0)
    assert tuple(tp.gelu_backward(empty, _meta(2, 3, 4, dtype=tp.float64)).shape) == (0,)
    assert tuple(tp.silu_backward(empty, empty).shape) == (0,)
    assert tuple(tp.threshold_backward(_meta(2, 0, 3), _meta(2, 0, 3), 0.5).shape) == (2, 0, 3)
    # gelu's approximation name is validated before anything else
    assert tp.gelu_backward(grad, _meta(2, 3, 4), "tanh").dtype == tp.float32
    with pytest.raises(ValueError):
        tp.gelu_backward(grad, _meta(2, 3, 4), "relu")
    # the saved plane feeds a linear read at the gradient's width: an
    # empty gradient returns before any of that happens
    assert tuple(tp.gelu_backward(_meta(0), _meta(2, 3, 4, dtype=tp.float64)).shape) == (0,)
    with pytest.raises(RuntimeError):
        tp.silu_backward(grad, _meta(2, 3, 4, dtype=tp.float64))
    with pytest.raises(RuntimeError):
        tp.hardtanh_backward(grad, _meta(2, 2), -1.0, 1.0)


def test_broadcasting_and_gated_activation_backwards():
    grad = _meta(2, 3)
    # sigmoid/tanh/logit backwards broadcast and promote like any binary op
    out = tp.sigmoid_backward(_meta(2, 1, 4), _meta(3, 4))
    assert tuple(out.shape) == (2, 3, 4) and out.dtype == tp.float32
    out = tp.tanh_backward(_meta(2, 3, dtype=tp.float64), _meta(3))
    assert tuple(out.shape) == (2, 3) and out.dtype == tp.float64
    assert tp.logit_backward(grad, _meta(2, 3, dtype=tp.float64), 0.1).dtype == tp.float64
    with pytest.raises(RuntimeError):
        tp.logit_backward(grad, _meta(2, 2))
    # the shrink pair gates the saved plane's dtype first
    assert tuple(tp.hardshrink_backward(grad, _meta(2, 3), 0.5).shape) == (2, 3)
    assert tp.softshrink_backward(_meta(2, 3, dtype=tp.float16),
                                  _meta(2, 3, dtype=tp.float16), 0.5).dtype == tp.float16
    with pytest.raises(NotImplementedError):
        tp.hardshrink_backward(grad, _meta(2, 3, dtype=tp.int64), 0.5)
    with pytest.raises(NotImplementedError):
        tp.softshrink_backward(grad, _meta(2, 3, dtype=tp.bool), 0.5)
    # log_sigmoid's sweep runs in floating widths only, after empty inputs pass
    assert tuple(tp.log_sigmoid_backward(_meta(0, dtype=tp.int64),
                                         _meta(0, dtype=tp.int64)).shape) == (0,)
    with pytest.raises(TypeError):
        tp.log_sigmoid_backward(_meta(2, 3, dtype=tp.int64),
                                _meta(2, 3, dtype=tp.int64))
    with pytest.raises(RuntimeError):
        tp.log_sigmoid_backward(grad, _meta(2, 3, dtype=tp.int64))


def test_glu_backward_composition_checks():
    out = tp.glu_backward(_meta(2, 3, 2), _meta(2, 3, 4))
    assert tuple(out.shape) == (2, 3, 4) and out.dtype == tp.float32
    assert tuple(tp.glu_backward(_meta(4, 3), _meta(8, 3), 0).shape) == (8, 3)
    assert tuple(tp.glu_backward(_meta(2, 2, 3), _meta(2, 4, 3), -2).shape) == (2, 4, 3)
    with pytest.raises(RuntimeError):
        tp.glu_backward(_meta(2, 3, 2), _meta(2, 3, 3))
    with pytest.raises(RuntimeError):
        tp.glu_backward(_meta(), _meta())
    with pytest.raises(IndexError):
        tp.glu_backward(_meta(2, 2), _meta(2, 4), 2)
    with pytest.raises(IndexError):
        tp.glu_backward(_meta(2, 2), _meta(2, 4), -3)


def test_rrelu_with_noise_backward_layout():
    noise = _meta(2, 3, 4)
    out = tp.rrelu_with_noise_backward(_meta(2, 3, 4), _meta(2, 3, 4), noise,
                                       0.125, 0.334, True, False)
    assert tuple(out.shape) == (2, 3, 4) and out.dtype == tp.float32
    # eval mode leaves the noise plane unread: only the gradient and the input
    # must agree
    assert tuple(tp.rrelu_with_noise_backward(_meta(2, 3), _meta(2, 3), _meta(5, 5),
                                              0.0, 1.0, False, False).shape) == (2, 3)
    with pytest.raises(RuntimeError):
        tp.rrelu_with_noise_backward(_meta(2, 3, 4), _meta(2, 3, 4), _meta(2, 3, 5),
                                     0.125, 0.334, True, False)
    with pytest.raises(RuntimeError):
        tp.rrelu_with_noise_backward(_meta(2, 3, 4), _meta(2, 3, 5), _meta(2, 3, 4),
                                     0.125, 0.334, False, False)
    with pytest.raises(RuntimeError):
        tp.rrelu_with_noise_backward(_meta(2, 3), _meta(2, 3, dtype=tp.float64),
                                     _meta(2, 3), 0.0, 1.0, False, False)


def test_cat_zero_sized_geometry():
    # a zero-valued plane still describes geometry: its length along the
    # joined dimension counts, and only size-[0] 1-d inputs are exempt from
    # the wrap and shape checks
    a = _meta(2, 0, 3)
    assert tuple(tp.cat([a, a], -1).shape) == (2, 0, 6)
    assert tuple(tp.cat([a, a], 0).shape) == (4, 0, 3)
    # the wrap dimension is taken from the first non-exempt input
    assert tuple(tp.cat([_meta(0), _meta(2, 3)], -1).shape) == (2, 3)
    assert tuple(tp.cat([_meta(0), _meta(0)], 0).shape) == (0,)
    # promotion covers every operand, exempt ones included
    assert tp.cat([_meta(3, dtype=tp.float64), _meta(2, dtype=tp.float32)],
                  0).dtype == tp.float64
    with pytest.raises(RuntimeError):
        tp.cat([_meta(2, 3), _meta(3, 4)], 0)
    with pytest.raises(RuntimeError):
        tp.cat([_meta(2, 3), _meta(2, 2)], 0)
    with pytest.raises(RuntimeError):
        tp.cat([_meta(2, 3), _meta(2, 0, 3)], 0)
    with pytest.raises(IndexError):
        tp.cat([_meta(2, 3)], -3)


def test_dist_renorm_and_rms_norm():
    distance = tp.dist(_meta(3, dtype=tp.int32), _meta(3, dtype=tp.int32))
    assert tuple(distance.shape) == ()
    assert distance.dtype == tp.float32
    assert tp.dist(_meta(3), _meta(3), p=1.0).dtype == tp.float32
    assert tp.dist(_meta(3, dtype=tp.complex64),
                   _meta(3, dtype=tp.complex64)).dtype == tp.float32
    with pytest.raises(Exception):
        tp.dist(_meta(3, dtype=tp.bool), _meta(3, dtype=tp.bool))

    t = _meta(2, 3)
    assert tuple(tp.renorm(t, 2.0, 0, 5.0).shape) == (2, 3)
    assert tuple(t.renorm_(2.0, 0, 5.0).shape) == (2, 3)
    with pytest.raises(Exception):
        tp.renorm(_meta(2, 3, dtype=tp.int32), 2.0, 0, 5.0)
    with pytest.raises(Exception):
        tp.renorm(t, 0.0, 0, 5.0)
    with pytest.raises(Exception):
        tp.renorm(_meta(3), 2.0, 0, 5.0)

    assert tuple(tp.rms_norm(_meta(2, 3), [3]).shape) == (2, 3)
    assert tp.rms_norm(_meta(2, 3), [3]).dtype == tp.float32
    with pytest.raises(Exception):
        tp.rms_norm(_meta(2, 3), [4])
    with pytest.raises(Exception):
        tp.rms_norm(_meta(2, 3), [3, 3])
    with pytest.raises(Exception):
        tp.rms_norm(_meta(2, 3, dtype=tp.int32), [3])


# ---------------------------------------------------------------------------
# Splitting, label encodings and diagonal products
# ---------------------------------------------------------------------------


def test_unbind_and_split_with_sizes():
    t = _meta(2, 3, 4)
    parts = tp.unbind(t, 1)
    assert len(parts) == 3
    assert tuple(parts[0].shape) == (2, 4)
    assert len(tp.unbind(t, -1)) == 4
    assert tp.unbind(t, 0)[0].dtype == t.dtype
    with pytest.raises(Exception):
        tp.unbind(_meta(()), 0)

    splits = tp.split_with_sizes(t, [1, 2], 1)
    assert [tuple(p.shape) for p in splits] == [(2, 1, 4), (2, 2, 4)]
    with pytest.raises(Exception):
        tp.split_with_sizes(t, [4], 1)
    with pytest.raises(Exception):
        tp.split_with_sizes(t, [1, 1], 1)
    with pytest.raises(Exception):
        tp.split_with_sizes(t, [-1, 4], 1)


def test_one_hot_kron_and_trace():
    index = _meta(2, 3, dtype=tp.int64)
    encoded = tp.one_hot(index, 5)
    assert tuple(encoded.shape) == (2, 3, 5)
    assert encoded.dtype == tp.int64
    assert tuple(tp.one_hot(_meta(0, dtype=tp.int64), 4).shape) == (0, 4)
    with pytest.raises(Exception):
        tp.one_hot(_meta(0, dtype=tp.int64), -1)
    with pytest.raises(Exception):
        tp.one_hot(index, -1)
    with pytest.raises(Exception):
        tp.one_hot(_meta(2, dtype=tp.float32), 3)

    assert tuple(tp.kron(_meta(2, 3), _meta(4)).shape) == (2, 12)
    assert tuple(tp.kron(_meta(2, 3), _meta(4, 5)).shape) == (8, 15)
    assert tp.kron(_meta(2, dtype=tp.int32),
                   _meta(3, dtype=tp.float32)).dtype == tp.float32

    traced = tp.trace(_meta(3, 4, dtype=tp.int32))
    assert tuple(traced.shape) == ()
    assert traced.dtype == tp.int64
    assert tp.trace(_meta(3, 4)).dtype == tp.float32
    with pytest.raises(Exception):
        tp.trace(_meta(3))


def test_masked_scatter_shapes():
    t = _meta(3, 4)
    mask = _meta(3, 4, dtype=tp.bool)
    source = _meta(5)
    scattered = tp.masked_scatter(t, mask, source)
    assert tuple(scattered.shape) == (3, 4)
    assert scattered.dtype == t.dtype
    assert tuple(tp.masked_scatter(t, _meta(1, 4, dtype=tp.bool),
                                   source).shape) == (3, 4)
    assert tuple(tp.masked_scatter_(t, mask, source).shape) == (3, 4)
    with pytest.raises(Exception):
        tp.masked_scatter(t, mask, _meta(5, dtype=tp.int32))
    with pytest.raises(Exception):
        tp.masked_scatter(t, _meta(5, 4, dtype=tp.bool), source)


# ---------------------------------------------------------------------------
# Order statistics and running extrema
# ---------------------------------------------------------------------------


def test_median_nanmedian_and_running_extrema():
    t = _meta(3, 4, dtype=tp.float32)
    assert tuple(tp.median(t).shape) == ()
    assert tp.median(_meta(3, 4, dtype=tp.int32)).dtype == tp.int32
    assert tp.nanmedian(_meta(3, 4, dtype=tp.int32)).dtype == tp.int64

    values, indices = tp.nanmedian(t, dim=1, keepdim=True)
    assert tuple(values.shape) == (3, 1)
    assert values.dtype == tp.float32
    assert tuple(indices.shape) == (3, 1)
    assert indices.dtype == tp.int64
    with pytest.raises(Exception):
        tp.nanmedian(_meta(3, 4, dtype=tp.int32), dim=1)

    running, positions = tp.cummax(t, dim=0)
    assert tuple(running.shape) == (3, 4)
    assert running.dtype == tp.float32
    assert tuple(positions.shape) == (3, 4)
    assert positions.dtype == tp.int64
    running, positions = tp.cummin(_meta(2, 3, dtype=tp.int32), dim=-1)
    assert running.dtype == tp.int32
    assert positions.dtype == tp.int64
    with pytest.raises(Exception):
        tp.cummax(_meta(2, dtype=tp.complex64), dim=0)


# ---------------------------------------------------------------------------
# Linear algebra
#
# Every factorization and solve reports the geometry of its result plus the
# integer status planes that describe the value-dependent outcome.  Failures
# that depend on the stored numbers cannot fire here; argument errors do.
# ---------------------------------------------------------------------------


def test_linalg_factorization_shapes_and_info():
    a = _meta(2, 3, 3)
    assert tuple(tp.linalg.cholesky(a).shape) == (2, 3, 3)
    lower, info = tp.linalg.cholesky_ex(a)
    assert tuple(lower.shape) == (2, 3, 3)
    assert info.dtype == tp.int32 and tuple(info.shape) == (2,)
    assert tuple(tp.linalg.inv(a).shape) == (2, 3, 3)
    inverse, inv_info = tp.linalg.inv_ex(a)
    assert tuple(inverse.shape) == (2, 3, 3)
    assert inv_info.dtype == tp.int32 and tuple(inv_info.shape) == (2,)

    det = tp.linalg.det(a)
    assert tuple(det.shape) == (2,) and det.dtype == tp.float32
    assert tuple(tp.linalg.det(_meta(3, 3)).shape) == ()
    sign, logabsdet = tp.linalg.slogdet(a)
    assert tuple(sign.shape) == (2,)
    assert tuple(logabsdet.shape) == (2,)
    sign, logabsdet = tp.linalg.slogdet(_meta(4, 4, dtype=tp.complex64))
    assert sign.dtype == tp.complex64
    assert logabsdet.dtype == tp.float32

    with pytest.raises(Exception):
        tp.linalg.det(_meta(3, 4))
    with pytest.raises(Exception):
        tp.linalg.det(_meta(3))
    with pytest.raises(Exception):
        tp.linalg.cholesky(_meta(3, 3, dtype=tp.int32))
    with pytest.raises(Exception):
        tp.linalg.inv(_meta(2, 4))


def test_linalg_lu_family():
    a = _meta(2, 4, 5)
    lu, pivots, info = tp.linalg.lu_factor_ex(a)
    assert tuple(lu.shape) == (2, 4, 5)
    assert pivots.dtype == tp.int32 and tuple(pivots.shape) == (2, 4)
    assert info.dtype == tp.int32 and tuple(info.shape) == (2,)

    P, L, U = tp.linalg.lu(a)
    assert (tuple(P.shape), tuple(L.shape), tuple(U.shape)) == (
        (2, 4, 4), (2, 4, 4), (2, 4, 5))
    with pytest.raises(Exception):
        tp.linalg.lu(a, pivot=False)

    # Only a square factorization can be solved against.
    with pytest.raises(Exception):
        tp.linalg.lu_solve(lu, pivots, _meta(2, 5, 4))
    square = _meta(2, 3, 3)
    slu, spivots, _ = tp.linalg.lu_factor_ex(square)
    assert tuple(tp.linalg.lu_solve(slu, spivots, _meta(2, 3, 4)).shape) == (2, 3, 4)
    unbatched, unbatched_pivots, _ = tp.linalg.lu_factor_ex(_meta(3, 3))
    assert tuple(tp.linalg.lu_solve(unbatched, unbatched_pivots,
                                    _meta(3)).shape) == (3,)
    with pytest.raises(Exception):
        tp.linalg.lu_solve(slu, _meta(2, 3, dtype=tp.int64), _meta(3))
    p, l, u = tp.lu_unpack(lu, pivots)
    assert (tuple(p.shape), tuple(l.shape), tuple(u.shape)) == (
        (2, 4, 4), (2, 4, 4), (2, 4, 5))
    p, l, u = tp.lu_unpack(lu, pivots, unpack_pivots=False)
    assert tuple(p.shape) == (0,) and tuple(l.shape) == (2, 4, 4)
    p, l, u = tp.lu_unpack(lu, pivots, unpack_data=False)
    assert tuple(l.shape) == (0,) and tuple(u.shape) == (0,)
    with pytest.raises(Exception):
        tp.lu_unpack(lu, _meta(2, 4, dtype=tp.int64))


def test_linalg_solves():
    a = _meta(2, 3, 3)
    x, info = tp.linalg.solve_ex(a, _meta(2, 3, 4))
    assert tuple(x.shape) == (2, 3, 4)
    assert info.dtype == tp.int32 and tuple(info.shape) == (2,)
    x, _ = tp.linalg.solve_ex(a, _meta(2, 3))
    assert tuple(x.shape) == (2, 3)
    assert tuple(tp.linalg.solve(a, _meta(2, 3, 4)).shape) == (2, 3, 4)
    assert tuple(tp.linalg.solve(a, _meta(2, 3)).shape) == (2, 3)

    tri = tp.linalg.solve_triangular(a, _meta(2, 3, 4), upper=True)
    assert tuple(tri.shape) == (2, 3, 4)
    tri = tp.linalg.solve_triangular(a, _meta(2, 4, 3), upper=False, left=False)
    assert tuple(tri.shape) == (2, 4, 3)

    with pytest.raises(Exception):
        tp.linalg.solve(_meta(3, 4), _meta(3))
    with pytest.raises(Exception):
        tp.linalg.solve(a, _meta(2, 4, 2))
    with pytest.raises(Exception):
        tp.linalg.solve_ex(a, _meta(2, 3), left=False)
    with pytest.raises(Exception):
        tp.linalg.solve_triangular(a, _meta(2, 3), upper=True)


def test_linalg_least_squares_and_ldl():
    # The default driver is the rank-revealing one: it reports a batch of
    # ranks and no residuals.
    solution, residuals, rank, singular = tp.linalg.lstsq(
        _meta(2, 5, 3), _meta(2, 5, 2))
    assert tuple(solution.shape) == (2, 3, 2)
    assert tuple(residuals.shape) == (0,)
    assert rank.dtype == tp.int64 and tuple(rank.shape) == (2,)
    assert tuple(singular.shape) == (0,)

    solution, residuals, rank, singular = tp.linalg.lstsq(
        _meta(2, 5, 3), _meta(2, 5, 2), driver="gels")
    assert tuple(solution.shape) == (2, 3, 2)
    assert tuple(residuals.shape) == (2, 2)
    assert tuple(rank.shape) == (0,)
    assert tuple(singular.shape) == (0,)

    _, _, rank, singular = tp.linalg.lstsq(
        _meta(2, 5, 3), _meta(2, 5, 2), driver="gelsd")
    assert tuple(rank.shape) == (2,)
    assert tuple(singular.shape) == (2, 3)

    solution, residuals, rank, _ = tp.linalg.lstsq(_meta(5, 3), _meta(5))
    assert tuple(solution.shape) == (3,)
    assert tuple(residuals.shape) == (0,)
    assert tuple(rank.shape) == ()
    # An underdetermined system reports the wide solution plane.
    solution, *_ = tp.linalg.lstsq(_meta(3, 5), _meta(3, 2))
    assert tuple(solution.shape) == (5, 2)

    with pytest.raises(Exception):
        tp.linalg.lstsq(_meta(3, 3), _meta(4, 2))
    with pytest.raises(Exception):
        tp.linalg.lstsq(_meta(3, 3), _meta(3, 2), driver="bogus")

    a = _meta(2, 3, 3)
    ld, pivots = tp.linalg.ldl_factor(a)
    assert tuple(ld.shape) == (2, 3, 3)
    assert pivots.dtype == tp.int32 and tuple(pivots.shape) == (2, 3)
    ld, pivots, info = tp.linalg.ldl_factor_ex(a)
    assert info.dtype == tp.int32 and tuple(info.shape) == (2,)
    assert tuple(tp.linalg.ldl_solve(ld, pivots, _meta(2, 3, 4)).shape) == (2, 3, 4)


def test_linalg_eigen_and_svd():
    values, vectors = tp.linalg.eigh(_meta(2, 3, 3))
    assert tuple(values.shape) == (2, 3) and values.dtype == tp.float32
    assert tuple(vectors.shape) == (2, 3, 3) and vectors.dtype == tp.float32
    values, vectors = tp.linalg.eigh(_meta(2, 3, 3, dtype=tp.complex64))
    assert values.dtype == tp.float32
    assert vectors.dtype == tp.complex64
    assert tuple(tp.linalg.eigvalsh(_meta(2, 3, 3, dtype=tp.float64)).shape) == (2, 3)

    values, vectors = tp.linalg.eig(_meta(2, 3, 3))
    assert tuple(values.shape) == (2, 3) and values.dtype == tp.complex64
    assert tuple(vectors.shape) == (2, 3, 3) and vectors.dtype == tp.complex64
    assert tp.linalg.eigvals(_meta(3, 3, dtype=tp.float64)).dtype == tp.complex128

    u, s, vh = tp.linalg.svd(_meta(2, 4, 3), full_matrices=True)
    assert (tuple(u.shape), tuple(s.shape), tuple(vh.shape)) == (
        (2, 4, 4), (2, 3), (2, 3, 3))
    u, s, vh = tp.linalg.svd(_meta(2, 4, 3), full_matrices=False)
    assert (tuple(u.shape), tuple(s.shape), tuple(vh.shape)) == (
        (2, 4, 3), (2, 3), (2, 3, 3))
    u, s, vh = tp.linalg.svd(_meta(4, 3, dtype=tp.complex64), full_matrices=False)
    assert u.dtype == tp.complex64 and s.dtype == tp.float32
    assert vh.dtype == tp.complex64
    assert tuple(tp.linalg.svdvals(_meta(2, 4, 3)).shape) == (2, 3)

    # An empty matrix has nothing to factor, so the integer gate is skipped
    # and the empty factors keep the input dtype.
    u, s, vh = tp.linalg.svd(_meta(0, 3, dtype=tp.int64))
    assert (tuple(u.shape), tuple(s.shape), tuple(vh.shape)) == (
        (0, 0), (0,), (3, 3))
    assert u.dtype == tp.int64 and vh.dtype == tp.int64
    assert tuple(tp.linalg.svdvals(_meta(0, 0)).shape) == (0,)

    with pytest.raises(Exception):
        tp.linalg.eigh(_meta(3, 4))
    with pytest.raises(Exception):
        tp.linalg.eigh(_meta(3, 3), UPLO="X")
    with pytest.raises(Exception):
        tp.linalg.eig(_meta(3, 3, dtype=tp.int32))
    with pytest.raises(Exception):
        tp.linalg.svd(_meta(3, 3), driver="gesvdj")
    with pytest.raises(Exception):
        tp.linalg.svd(_meta(3, 3, dtype=tp.int32))


def test_linalg_qr_and_householder():
    a = _meta(2, 4, 3)
    q, r = tp.linalg.qr(a)
    assert (tuple(q.shape), tuple(r.shape)) == ((2, 4, 3), (2, 3, 3))
    q, r = tp.linalg.qr(a, mode="complete")
    assert (tuple(q.shape), tuple(r.shape)) == ((2, 4, 4), (2, 4, 3))
    q, r = tp.linalg.qr(a, mode="r")
    assert (tuple(q.shape), tuple(r.shape)) == ((2, 4, 0), (2, 3, 3))
    with pytest.raises(Exception):
        tp.linalg.qr(a, mode="bogus")
    with pytest.raises(Exception):
        tp.linalg.qr(_meta(3))

    # The product replaces the reflectors of an m x n input with n <= m.
    assert tuple(tp.linalg.householder_product(a, _meta(2, 3)).shape) == (2, 4, 3)
    with pytest.raises(Exception):
        tp.linalg.householder_product(_meta(3, 4), _meta(3))
    with pytest.raises(Exception):
        tp.linalg.householder_product(a, _meta(2, 2))


def test_linalg_pinv_diagonal_and_empty():
    assert tuple(tp.linalg.pinv(_meta(2, 4, 3)).shape) == (2, 3, 4)
    assert tuple(tp.linalg.pinv(_meta(3, 4)).shape) == (4, 3)
    assert tuple(tp.linalg.pinv(_meta(2, 3, 3), hermitian=True).shape) == (2, 3, 3)

    # A tolerance that carries a batch broadcasts into the result batch; a
    # tolerance without one leaves the input batch alone.
    a = _meta(2, 4, 3)
    assert tuple(tp.linalg.pinv(a, rtol=_meta(2, 1)).shape) == (2, 2, 3, 4)
    assert tuple(tp.linalg.pinv(a, atol=_meta(2, 1)).shape) == (2, 2, 3, 4)
    assert tuple(tp.linalg.pinv(a, rtol=_meta(2)).shape) == (2, 3, 4)
    # An empty input short-circuits before any tolerance batch is merged.
    assert tuple(tp.linalg.pinv(_meta(3, 0), rtol=_meta(2, 1)).shape) == (0, 3)
    with pytest.raises(Exception):
        tp.linalg.pinv(_meta(2, 3, dtype=tp.int32))
    with pytest.raises(Exception):
        tp.linalg.pinv(_meta(2, 4, 3), hermitian=True)

    assert tuple(tp.linalg.diagonal(_meta(2, 3, 4)).shape) == (2, 3)
    assert tuple(tp.linalg.diagonal(_meta(2, 3, 4), offset=1).shape) == (2, 3)
    assert tuple(tp.linalg.diagonal(_meta(2, 3, 4), offset=-2).shape) == (2, 1)
    with pytest.raises(Exception):
        tp.linalg.diagonal(_meta(2, 3, 4), dim1=1, dim2=1)
    with pytest.raises(Exception):
        tp.linalg.diagonal(_meta(2, 3, 4), dim1=0, dim2=5)


# ---------------------------------------------------------------------------
# Complex views and factories
# ---------------------------------------------------------------------------


def test_complex_views_and_factories():
    c = _meta(2, 3, dtype=tp.complex64)
    assert tuple(tp.real(c).shape) == (2, 3) and tp.real(c).dtype == tp.float32
    assert tuple(tp.imag(c).shape) == (2, 3) and tp.imag(c).dtype == tp.float32
    assert tp.real(_meta(2, dtype=tp.complex128)).dtype == tp.float64
    assert tp.imag(_meta(2, dtype=tp.complex128)).dtype == tp.float64
    # A real tensor is its own real part; its imaginary part is zeros.
    x = _meta(2, 3, dtype=tp.float32)
    assert tuple(tp.real(x).shape) == (2, 3) and tp.real(x).dtype == tp.float32
    assert tuple(tp.imag(x).shape) == (2, 3) and tp.imag(x).dtype == tp.float32
    # conj of a real operand is a view with the input's layout; a complex
    # operand answers the materialized geometry.
    assert tuple(tp.conj(x).stride()) == tuple(x.stride())
    assert tuple(tp.conj(c).shape) == (2, 3) and tp.conj(c).dtype == tp.complex64
    assert tuple(tp.conj(_meta(4)).shape) == (4,)
    # adjoint is transpose(-2, -1) composed with conj; ndim <= 1 is plain conj.
    assert tuple(tp.adjoint(_meta(2, 3, 4)).shape) == (2, 4, 3)
    assert tuple(tp.adjoint(_meta(4)).shape) == (4,)
    adj = tp.adjoint(c)
    assert tuple(adj.shape) == (3, 2) and adj.dtype == tp.complex64
    # factories answer the broadcast in the complex twin dtype.
    out = tp.complex(_meta(2, 1, dtype=tp.float32), _meta(3, dtype=tp.float32))
    assert tuple(out.shape) == (2, 3) and out.dtype == tp.complex64
    pol = tp.polar(_meta(2, 3, dtype=tp.float64), _meta(2, 3, dtype=tp.float64))
    assert tuple(pol.shape) == (2, 3) and pol.dtype == tp.complex128
    with pytest.raises(Exception):
        tp.complex(_meta(2, dtype=tp.int32), _meta(2, dtype=tp.int32))
    with pytest.raises(Exception):
        tp.complex(_meta(2, dtype=tp.float32), _meta(2, dtype=tp.float64))
    with pytest.raises(Exception):
        tp.complex(_meta(2, dtype=tp.float32), _meta(2, dtype=tp.float32, device="cpu"))
    with pytest.raises(Exception):
        tp.polar(_meta(2, dtype=tp.float32), _meta(3, dtype=tp.int32))


def test_view_as_real_and_complex():
    c = _meta(2, 3, dtype=tp.complex64)
    v = tp.view_as_real(c)
    assert tuple(v.shape) == (2, 3, 2)
    assert tuple(v.stride()) == (6, 2, 1)
    assert v.dtype == tp.float32
    # every source stride doubles, on non-contiguous layouts too
    tv = tp.view_as_real(tp.t(c))
    assert tuple(tv.shape) == (3, 2, 2)
    assert tuple(tv.stride()) == (2, 6, 1)
    assert tp.view_as_real(_meta(2, dtype=tp.complex128)).dtype == tp.float64
    back = tp.view_as_complex(v)
    assert tuple(back.shape) == (2, 3) and back.dtype == tp.complex64
    pair = _meta(2, dtype=tp.float32)
    assert tuple(tp.view_as_complex(pair).shape) == ()
    with pytest.raises(Exception):
        tp.view_as_real(_meta(2, dtype=tp.float32))
    # the last dimension must hold exactly the pair, with unit stride
    with pytest.raises(Exception):
        tp.view_as_complex(_meta(2, 3, dtype=tp.float32))
    with pytest.raises(Exception):
        tp.view_as_complex(_meta(2, dtype=tp.int32))
    # an odd outer stride would split a pair
    transposed = tp.t(_meta(2, 3, dtype=tp.float32))  # strides (1, 2)
    with pytest.raises(Exception):
        tp.view_as_complex(transposed)
    # an odd entry position would start inside a pair
    odd_offset = tp.narrow(_meta(6, dtype=tp.float32), 0, 1, 2)
    with pytest.raises(Exception):
        tp.view_as_complex(odd_offset)


# ---------------------------------------------------------------------------
# Non-constant padding
# ---------------------------------------------------------------------------


def test_nonconstant_padding_shapes():
    x = _meta(2, 3, 4)
    assert tuple(tp.reflection_pad1d(x, [1, 2]).shape) == (2, 3, 7)
    assert tuple(tp.reflection_pad2d(x, [1, 2, 1, 2]).shape) == (2, 6, 7)
    assert tuple(tp.reflection_pad3d(x, [1, 1, 1, 1, 1, 1]).shape) == (4, 5, 6)
    assert tuple(tp.reflection_pad_nd(x, [1, 1, 1, 1]).shape) == (2, 5, 6)
    assert tuple(tp.replication_pad2d(x, [1, 0, 0, 1]).shape) == (2, 4, 5)
    assert tuple(tp.circular_pad_nd(x, [1, 1, 1, 1]).shape) == (2, 5, 6)
    g = _meta(2, 5, 7, dtype=tp.float32)
    assert tuple(tp.reflection_pad2d_backward(g, x, [1, 2, 1, 2]).shape) == (2, 3, 4)
    assert tuple(tp.replication_pad1d_backward(g, _meta(2, 5), [1, 1]).shape) == (2, 5)
    assert tuple(tp.circular_pad_nd_backward(g, x, [1, 2, 1, 2]).shape) == (2, 3, 4)
    assert tp.replication_pad1d(_meta(2, 3, dtype=tp.int32), [1, 1]).dtype == tp.int32
    # pad-list and mode validations
    with pytest.raises(Exception):
        tp.reflection_pad1d(x, [1])
    with pytest.raises(Exception):
        tp.reflection_pad2d(_meta(2), [1, 1, 1, 1])
    with pytest.raises(Exception):
        tp.reflection_pad1d(_meta(2), [2, 1])
    with pytest.raises(Exception):
        tp.circular_pad_nd(_meta(2), [3, 1])
    # the forward accepts a wider dtype set than the backward
    with pytest.raises(Exception):
        tp.reflection_pad1d(_meta(2, 3, dtype=tp.int64), [1, 1])
    with pytest.raises(Exception):
        tp.reflection_pad1d_backward(_meta(2, 5, dtype=tp.int32),
                                     _meta(2, 3, dtype=tp.int32), [1, 1])


# ---------------------------------------------------------------------------
# NaN-aware reductions and the accumulate scan
# ---------------------------------------------------------------------------


def test_nansum_nanmean_logcumsumexp():
    x = _meta(2, 3)
    assert tuple(tp.nansum(x, dim=[1]).shape) == (2,)
    assert tuple(tp.nansum(x, dim=[1], keepdim=True).shape) == (2, 1)
    assert tuple(tp.nansum(x).shape) == ()
    assert tp.nansum(x).dtype == tp.float32
    assert tp.nansum(_meta(2, 3, dtype=tp.int32)).dtype == tp.int64
    assert tp.nansum(_meta(2, dtype=tp.float64)).dtype == tp.float64
    with pytest.raises(Exception):
        tp.nansum(_meta(2, dtype=tp.complex64))

    assert tuple(tp.nanmean(x, dim=0).shape) == (3,)
    assert tuple(tp.nanmean(x, dim=1, keepdim=True).shape) == (2, 1)
    assert tuple(tp.nanmean(x).shape) == ()
    # narrow floating types accumulate at single precision
    assert tp.nanmean(_meta(2, dtype=tp.float16)).dtype == tp.float32
    assert tp.nanmean(_meta(2, dtype=tp.float64)).dtype == tp.float64
    assert tp.nanmean(x, dim=0, dtype=tp.float64).dtype == tp.float64
    with pytest.raises(Exception):
        tp.nanmean(_meta(2, dtype=tp.int32))
    with pytest.raises(Exception):
        tp.nanmean(x, dim=0, dtype=tp.int32)
    with pytest.raises(Exception):
        tp.nanmean(_meta(2, dtype=tp.complex64))
    # a complex input works once a real accumulation type is pinned
    assert tp.nanmean(_meta(2, dtype=tp.complex64),
                      dtype=tp.float32).dtype == tp.float32

    y = _meta(2, 3)
    assert tuple(tp.logcumsumexp(y, dim=0).shape) == (2, 3)
    assert tp.logcumsumexp(y, dim=1).dtype == tp.float32
    assert tp.logcumsumexp(_meta(2, dtype=tp.float16), dim=0).dtype == tp.float16
    assert tp.logcumsumexp(y, dim=0, dtype=tp.float64).dtype == tp.float64
    assert tp.logcumsumexp(_meta(2, dtype=tp.complex64), dim=0).dtype == tp.complex64
    with pytest.raises(Exception):
        tp.logcumsumexp(tp.empty([], device="meta"), dim=2)
    with pytest.raises(Exception):
        tp.logcumsumexp(y, dim=5)
    with pytest.raises(Exception):
        tp.logcumsumexp(_meta(2, dtype=tp.int32), dim=0)


# ---------------------------------------------------------------------------
# Tensor-list builders
# ---------------------------------------------------------------------------


def test_block_diag_and_broadcast_tensors():
    a = _meta(2, 3)
    b = _meta(3, dtype=tp.float64)
    out = tp.block_diag(a, b)
    assert tuple(out.shape) == (3, 6)
    assert out.dtype == tp.float64
    # a rank-1 block spans one row; the empty call answers the placeholder
    assert tuple(tp.block_diag(tp.empty([], device="meta"), a).shape) == (3, 4)
    assert tuple(tp.block_diag().shape) == (1, 0)
    assert tp.block_diag().dtype == tp.float32
    with pytest.raises(Exception):
        tp.block_diag(a, _meta(2, 2, 2))
    with pytest.raises(Exception):
        tp.block_diag(a, _meta(2, 2, device="cpu"))

    t1 = _meta(2, 1)
    t2 = _meta(3)
    outs = tp.broadcast_tensors(t1, t2)
    assert len(outs) == 2
    assert all(tuple(o.shape) == (2, 3) for o in outs)
    # the broadcast dimension repeats elements, so its stride is zero
    assert tuple(outs[0].stride()) == (1, 0)
    assert tuple(outs[1].stride()) == (0, 1)
    same = _meta(2, 3)
    outs2 = tp.broadcast_tensors(same, _meta(2, 3))
    assert all(tuple(o.shape) == (2, 3) for o in outs2)


# ---------------------------------------------------------------------------
# Assorted value-free kernels
# ---------------------------------------------------------------------------


def test_alpha_dropout_histc_frexp():
    x = _meta(2, 3)
    out, mask = tp.native_alpha_dropout(x, 0.5)
    assert tuple(out.shape) == (2, 3) and out.dtype == tp.float32
    assert tuple(mask.shape) == (2, 3) and mask.dtype == tp.float32
    assert tp.native_alpha_dropout(_meta(2, dtype=tp.float64), 0.2)[0].dtype == tp.float64
    assert tp.native_alpha_dropout(_meta(2, dtype=tp.int32), 0.2)[0].dtype == tp.float32
    with pytest.raises(Exception):
        tp.native_alpha_dropout(x, 1.0)
    with pytest.raises(Exception):
        tp.native_alpha_dropout(x, -0.1)

    h = tp.histc(x, 10, 0.0, 1.0)
    assert tuple(h.shape) == (10,) and h.dtype == tp.float32
    # an equal range derives the span from the data, which meta cannot read;
    # the shape is still answered
    assert tuple(tp.histc(x, 5, 2.0, 2.0).shape) == (5,)
    assert tp.histc(_meta(4, dtype=tp.float64), 3, 0.0, 1.0).dtype == tp.float64
    with pytest.raises(Exception):
        tp.histc(x, 0, 0.0, 1.0)
    with pytest.raises(Exception):
        tp.histc(_meta(2, dtype=tp.int32), 4, 0.0, 1.0)
    with pytest.raises(Exception):
        tp.histc(x, 4, 2.0, 1.0)
    with pytest.raises(Exception):
        tp.histc(x, 4, float("nan"), 1.0)

    m, e = tp.frexp(x)
    assert tuple(m.shape) == (2, 3) and m.dtype == tp.float32
    assert tuple(e.shape) == (2, 3) and e.dtype == tp.int32
    assert tp.frexp(_meta(2, dtype=tp.float64))[1].dtype == tp.int32
    with pytest.raises(Exception):
        tp.frexp(_meta(2, dtype=tp.int32))


def test_pointwise_alias_and_pair_names():
    x = _meta(3, 4)
    for name in ("arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh"):
        f = getattr(tp.functional, name)
        assert tuple(f(x).shape) == (3, 4)
        assert f(_meta(3, 4, dtype=tp.float64)).dtype == tp.float64
        assert f(_meta(3, 4, dtype=tp.int64)).dtype == tp.float32
    assert tuple(tp.functional.negative(x).shape) == (3, 4)
    assert tp.functional.negative(_meta(3, dtype=tp.int64)).dtype == tp.int64
    y = _meta(3, 4, dtype=tp.int64)
    for name in ("multiply", "subtract", "divide"):
        f = getattr(tp.functional, name)
        assert tuple(f(x, x).shape) == (3, 4)
        assert f(x, y).dtype == tp.float32
    for name in ("logaddexp", "logaddexp2"):
        f = getattr(tp.functional, name)
        assert tuple(f(x, x).shape) == (3, 4)
        assert f(x, y).dtype == tp.float32
    for name in ("greater", "greater_equal", "less", "less_equal", "not_equal"):
        f = getattr(tp.functional, name)
        out = f(_meta(3, 4), _meta(1, 4))
        assert tuple(out.shape) == (3, 4) and out.dtype == tp.bool


def test_pixel_unshuffle_and_channel_shuffle():
    out = tp.functional.pixel_unshuffle(_meta(1, 8, 4, 4), 2)
    assert tuple(out.shape) == (1, 32, 2, 2)
    with pytest.raises(Exception):
        tp.functional.pixel_unshuffle(_meta(1, 8, 5, 4), 2)
    with pytest.raises(Exception):
        tp.functional.pixel_unshuffle(_meta(1, 8, 4, 4), 0)
    with pytest.raises(Exception):
        tp.functional.pixel_unshuffle(_meta(1, 8, 4, 4), 2**40)
    with pytest.raises(Exception):
        tp.functional.pixel_unshuffle(_meta(8), 2)

    assert tuple(tp.functional.channel_shuffle(_meta(2, 8, 4, 4), 2).shape) == (2, 8, 4, 4)
    with pytest.raises(Exception):
        tp.functional.channel_shuffle(_meta(2, 8, 4, 4), 3)
    with pytest.raises(Exception):
        tp.functional.channel_shuffle(_meta(2, 8, 4, 4), 0)
    with pytest.raises(Exception):
        tp.functional.channel_shuffle(_meta(2, 8), 2)


def test_scatter_forms_check_their_slices():
    self_ = _meta(2, 3, 4)
    assert tuple(tp.functional.select_scatter(self_, _meta(2, 4), 1, 1).shape) == (2, 3, 4)
    with pytest.raises(Exception):
        tp.functional.select_scatter(self_, _meta(2, 4), 1, 5)
    with pytest.raises(Exception):
        tp.functional.select_scatter(self_, _meta(2, 3), 1, 1)

    assert tuple(tp.functional.slice_scatter(self_, _meta(2, 2, 4), 1, 0, 2).shape) == (2, 3, 4)
    with pytest.raises(Exception):
        tp.functional.slice_scatter(self_, _meta(2, 2, 4), 1, 0, 2, 0)
    with pytest.raises(Exception):
        tp.functional.slice_scatter(self_, _meta(2, 1, 4), 1, 0, 2)

    cube = _meta(3, 4, 4)
    assert tuple(tp.functional.diagonal_scatter(cube, _meta(4, 3), 0).shape) == (3, 4, 4)
    assert tuple(tp.functional.diagonal_scatter(_meta(3, 5, 5), _meta(5, 3), 2).shape) == (3, 5, 5)
    with pytest.raises(Exception):
        tp.functional.diagonal_scatter(cube, _meta(3, 4), 0)
    with pytest.raises(Exception):
        tp.functional.diagonal_scatter(_meta(3, 5, 5), _meta(3, 3), 2)
    with pytest.raises(Exception):
        tp.functional.diagonal_scatter(cube, _meta(4, 3), 0, 1, 1)


def test_vander_extents_and_dtype():
    x = _meta(4)
    assert tuple(tp.functional.vander(x).shape) == (4, 4)
    assert tuple(tp.functional.vander(x, 3).shape) == (4, 3)
    assert tp.functional.vander(_meta(4, dtype=tp.int64)).dtype == tp.int64
    assert tp.functional.vander(x, 3).dtype == tp.float32
    with pytest.raises(Exception):
        tp.functional.vander(x, -1)
    with pytest.raises(Exception):
        tp.functional.vander(_meta(2, 4))


def test_trapezoid_family_shapes():
    y = _meta(5)
    assert tuple(tp.functional.trapezoid(y).shape) == ()
    assert tuple(tp.functional.trapezoid(y, _meta(5)).shape) == ()
    assert tuple(tp.functional.trapezoid(_meta(2, 5), dim=1).shape) == (2,)
    assert tp.functional.trapezoid(_meta(2, 5), dim=1).dtype == tp.float32
    with pytest.raises(Exception):
        tp.functional.trapezoid(y, _meta(3))
    with pytest.raises(Exception):
        tp.functional.trapezoid(_meta(2, 5), _meta(3), dim=1)

    assert tuple(tp.functional.cumulative_trapezoid(y).shape) == (4,)
    assert tuple(tp.functional.cumulative_trapezoid(_meta(2, 5), _meta(2, 5), dim=1).shape) == (2, 4)
    with pytest.raises(Exception):
        tp.functional.cumulative_trapezoid(y, _meta(3))


def test_cov_corrcoef_count_nonzero_shapes():
    m = _meta(3, 5)
    assert tuple(tp.functional.cov(m).shape) == (3, 3)
    assert tuple(tp.functional.cov(_meta(5)).shape) == ()
    with pytest.raises(Exception):
        tp.functional.cov(_meta(2, 3, 5))
    with pytest.raises(NotImplementedError):
        tp.functional.cov(_meta(3, 5, dtype=tp.bool))
    with pytest.raises(Exception):
        tp.functional.cov(m, fweights=_meta(3, 1))
    with pytest.raises(Exception):
        tp.functional.cov(m, fweights=_meta(4))
    assert tuple(tp.functional.corrcoef(m).shape) == (3, 3)
    with pytest.raises(Exception):
        tp.functional.corrcoef(_meta(2, 3, 5))

    assert tuple(tp.functional.count_nonzero(_meta(2, 3)).shape) == ()
    assert tuple(tp.functional.count_nonzero(_meta(2, 3), dim=[1]).shape) == (2,)
    assert tp.functional.count_nonzero(_meta(2, 3)).dtype == tp.int64


def test_histogram_bin_contract():
    hist, edges = tp.functional.histogram(_meta(10), 4)
    assert tuple(hist.shape) == (4,) and hist.dtype == tp.float32
    assert tuple(edges.shape) == (5,) and edges.dtype == tp.float32
    with pytest.raises(Exception):
        tp.functional.histogram(_meta(10), 0)
    with pytest.raises(Exception):
        tp.functional.histogram(_meta(10, dtype=tp.int64), 4)
    with pytest.raises(Exception):
        tp.functional.histogram(_meta(10), 4, weight=_meta(5))
    with pytest.raises(Exception):
        tp._C.histogram(_meta(10), 4, range=[0.0, 1.0, 2.0])
    with pytest.raises(Exception):
        tp._C.histogram(_meta(10), 4, range=[1.0, 0.0])
    with pytest.raises(Exception):
        tp._C.histogram(_meta(10), 4, range=[float("nan"), 1.0])


def test_pdist_and_matrix_power():
    out = tp.functional.pdist(_meta(5, 3))
    assert tuple(out.shape) == (10,) and out.dtype == tp.float32
    assert tuple(tp.functional.pdist(_meta(1, 3)).shape) == (0,)
    with pytest.raises(Exception):
        tp.functional.pdist(_meta(5))
    with pytest.raises(Exception):
        tp.functional.pdist(_meta(5, 3, dtype=tp.int64))
    with pytest.raises(Exception):
        tp.functional.pdist(_meta(5, 3), p=-1.0)

    assert tuple(tp.functional.matrix_power(_meta(3, 3), 2).shape) == (3, 3)
    assert tuple(tp.functional.matrix_power(_meta(3, 3), -1).shape) == (3, 3)
    assert tuple(tp.functional.matrix_power(_meta(2, 3, 3), 2).shape) == (2, 3, 3)
    assert tp.functional.matrix_power(_meta(3, 3, dtype=tp.int64), 2).dtype == tp.int64
    with pytest.raises(Exception):
        tp.functional.matrix_power(_meta(3, 4), 2)


def test_fixed_mean_losses_answer_scalars():
    x, z = _meta(3, 4), _meta(3, 4)
    zi = _meta(3, 4, dtype=tp.int64)
    f = tp.functional
    cases = [
        ("l1_loss", lambda: f.l1_loss(x, z)),
        ("kl_div", lambda: f.kl_div(x, z)),
        ("poisson_nll_loss", lambda: f.poisson_nll_loss(x, z)),
        ("soft_margin_loss", lambda: f.soft_margin_loss(x, z)),
        ("multilabel_soft_margin_loss", lambda: f.multilabel_soft_margin_loss(x, z)),
        ("cosine_embedding_loss", lambda: f.cosine_embedding_loss(_meta(3, 4), _meta(3, 4), _meta(3))),
        ("margin_ranking_loss", lambda: f.margin_ranking_loss(x, z, _meta(3, 4))),
        ("hinge_embedding_loss", lambda: f.hinge_embedding_loss(x, z)),
        ("triplet_margin_loss", lambda: f.triplet_margin_loss(_meta(3, 4), _meta(3, 4), _meta(3, 4))),
        ("binary_cross_entropy_with_logits", lambda: f.binary_cross_entropy_with_logits(x, z)),
    ]
    for _, call in cases:
        out = call()
        assert tuple(out.shape) == () and out.dtype == tp.float32
    assert f.l1_loss(_meta(3, 4, dtype=tp.float64), z).dtype == tp.float64
    assert f.l1_loss(zi, zi).dtype == tp.float32
    assert f.binary_cross_entropy_with_logits(zi, z).dtype == tp.float32
    assert f.binary_cross_entropy_with_logits(x, z, weight=_meta(3, 4)).dtype == tp.float32
    assert f.binary_cross_entropy_with_logits(x, z, weight=_meta(0)).dtype == tp.float32

    with pytest.raises(Exception):
        f.l1_loss(x, _meta(5))
    with pytest.raises(Exception):
        f.kl_div(x, _meta(5))
    with pytest.raises(Exception):
        f.soft_margin_loss(x, _meta(5))
    with pytest.raises(Exception):
        f.multilabel_soft_margin_loss(x, _meta(5))
    with pytest.raises(Exception):
        f.margin_ranking_loss(x, _meta(5), _meta(3, 4))
    with pytest.raises(Exception):
        f.margin_ranking_loss(x, z, _meta(5))
    with pytest.raises(Exception):
        f.hinge_embedding_loss(x, _meta(5))
    with pytest.raises(Exception):
        f.binary_cross_entropy_with_logits(x, _meta(5))
    with pytest.raises(Exception):
        f.binary_cross_entropy_with_logits(x, z, weight=_meta(5))


# ---------------------------------------------------------------------------
# Special functions, polygamma, out= forms and resize_
# ---------------------------------------------------------------------------


SPECIAL_UNARY = (
    "airy_ai", "bessel_j0", "bessel_j1", "bessel_y0", "bessel_y1",
    "spherical_bessel_j0", "modified_bessel_i0", "modified_bessel_i1",
    "modified_bessel_k0", "modified_bessel_k1", "scaled_modified_bessel_k0",
    "scaled_modified_bessel_k1", "erfcx", "ndtr", "ndtri", "log_ndtr",
    "i0e", "i1", "i1e",
)

SPECIAL_BINARY = (
    "zeta", "gammainc", "gammaincc", "igamma", "igammac", "xlog1py",
    "chebyshev_polynomial_t", "chebyshev_polynomial_u",
    "chebyshev_polynomial_v", "chebyshev_polynomial_w",
    "shifted_chebyshev_polynomial_t", "shifted_chebyshev_polynomial_u",
    "shifted_chebyshev_polynomial_v", "shifted_chebyshev_polynomial_w",
    "hermite_polynomial_h", "hermite_polynomial_he",
    "laguerre_polynomial_l", "legendre_polynomial_p",
)


def test_special_unary_family():
    t = _meta(2, 3)
    for name in SPECIAL_UNARY:
        f = getattr(tp._C, name)
        out = f(t)
        assert tuple(out.shape) == (2, 3)
        assert out.dtype == tp.float32
        assert f(_meta(2, 3, dtype=tp.float64)).dtype == tp.float64
        # integers, booleans and complex planes are answered in single
        # precision
        assert f(_meta(2, 3, dtype=tp.int64)).dtype == tp.float32
        assert f(_meta(2, 3, dtype=tp.bool)).dtype == tp.float32
        assert f(_meta(2, 3, dtype=tp.complex128)).dtype == tp.float32
        assert f(_meta(0)).shape == (0,)
        assert f(_meta(2, 3, dtype=tp.float16)).dtype == tp.float16


def test_special_binary_family():
    for name in SPECIAL_BINARY:
        f = getattr(tp._C, name)
        out = f(_meta(2, 3), _meta(2, 3))
        assert tuple(out.shape) == (2, 3)
        assert out.dtype == tp.float32
        assert f(_meta(2, 3, dtype=tp.float64),
                 _meta(2, 3, dtype=tp.float64)).dtype == tp.float64
        # the promoted pair is lifted to single precision when no floating
        # width survives
        assert f(_meta(2, 3, dtype=tp.int64),
                 _meta(2, 3, dtype=tp.int64)).dtype == tp.float32
        assert f(_meta(2, 3, dtype=tp.complex64),
                 _meta(2, 3, dtype=tp.complex64)).dtype == tp.float32
        assert f(_meta(2, 1, 4), _meta(3, 4)).shape == (2, 3, 4)
        assert f(_meta(0), _meta(0)).shape == (0,)
        with pytest.raises(RuntimeError):
            f(_meta(2, 3), tp.zeros(2, 3))


def test_polygamma_orders():
    for n in (0, 1, 2, 7):
        out = tp._C.polygamma(n, _meta(2, 3))
        assert tuple(out.shape) == (2, 3)
        assert out.dtype == tp.float32
    assert tp._C.polygamma(3, _meta(2, 3, dtype=tp.float64)).dtype == tp.float64
    assert tp._C.polygamma(2, _meta(2, 3, dtype=tp.int64)).dtype == tp.float32
    assert tp._C.polygamma(2, _meta(2, 3, dtype=tp.complex128)).dtype == tp.float32
    with pytest.raises(RuntimeError):
        tp._C.polygamma(-1, _meta(2, 3))
    with pytest.raises(RuntimeError):
        tp._C.polygamma(2**40, _meta(2, 3))
    with pytest.raises(RuntimeError):
        tp._C.special_polygamma(-2, _meta(2, 3))


def test_incomplete_gamma_answers_on_meta():
    # the incomplete-gamma pair is answered directly on the meta device
    for name in ("igamma", "igammac", "gammainc", "gammaincc"):
        f = getattr(tp._C, name)
        assert tuple(f(_meta(2, 3), _meta(2, 3)).shape) == (2, 3)
        assert f(_meta(2, 3), _meta(1, 3)).shape == (2, 3)
        assert f(_meta(2, 3, dtype=tp.float64),
                 _meta(2, 3)).dtype == tp.float64


def test_special_aliases_on_meta():
    x = _meta(2, 3)
    assert tuple(tp._C.special_xlogy(x, x).shape) == (2, 3)
    assert tp._C.special_xlogy(x, x).dtype == tp.float32
    assert tp._C.special_xlogy(2.0, _meta(2, 3, dtype=tp.int64)).shape == (2, 3)
    assert tp._C.special_xlogy(_meta(2, 3), 3.0).dtype == tp.float32
    assert tp._C.special_xlog1py(x, x).dtype == tp.float32
    assert tp._C.special_zeta(2.0, _meta(2, 3, dtype=tp.float64)).dtype == tp.float64
    assert tp._C.special_chebyshev_polynomial_t(2.0, x).shape == (2, 3)
    assert tp._C.special_chebyshev_polynomial_t(x, 2.0).shape == (2, 3)
    assert tp._C.special_hermite_polynomial_he(x, 2.0).shape == (2, 3)
    assert tuple(tp._C.special_multigammaln(x, 3).shape) == (2, 3)
    assert tp._C.special_multigammaln(_meta(2, 3, dtype=tp.float64), 2).dtype == tp.float64
    with pytest.raises(RuntimeError):
        tp._C.special_multigammaln(x, 0)
    with pytest.raises(RuntimeError):
        tp._C.mvlgamma(x, -1)
    assert tuple(tp._C.special_softmax(x, 1).shape) == (2, 3)
    assert tuple(tp._C.special_log_softmax(x, -1).shape) == (2, 3)
    assert tuple(tp._C.special_logsumexp(x, [1], False).shape) == (2,)
    assert tp._C.special_digamma(_meta(2, 3, dtype=tp.int64)).dtype == tp.float32
    assert tp._C.special_psi(x).shape == (2, 3)
    assert tp._C.special_expit(x).dtype == tp.float32
    assert tp._C.special_gammaln(_meta(2, 3, dtype=tp.int64)).dtype == tp.float32
    assert tp._C.special_sinc(_meta(2, 3, dtype=tp.complex64)).dtype == tp.float32


def test_special_out_overloads():
    assert tuple(tp._C.digamma(_meta(2, 3), out=_meta(2, 3)).shape) == (2, 3)
    # a destination that names another geometry is resized by the write
    small = _meta(4)
    tp._C.digamma(_meta(2, 3), out=small)
    assert tuple(small.shape) == (2, 3)
    # a widened destination is filled from the cast answer
    wide = _meta(2, 3, dtype=tp.float64)
    tp._C.digamma(_meta(2, 3), out=wide)
    assert wide.dtype == tp.float64
    assert tuple(tp._C.i0(_meta(2, 3), out=_meta(2, 3)).shape) == (2, 3)
    assert tuple(tp._C.polygamma(2, _meta(2, 3), out=_meta(3, 3)).shape) == (2, 3)
    assert tuple(tp._C.igamma(_meta(2, 3), _meta(2, 3), out=_meta(5)).shape) == (2, 3)
    assert tuple(tp._C.igammac(_meta(2, 3), _meta(2, 3), out=_meta(5)).shape) == (2, 3)
    # an order that cannot be run is refused before any write
    with pytest.raises(RuntimeError):
        tp._C.polygamma(-1, _meta(2, 3), out=_meta(2, 3))
    # an integral destination cannot be cast to safely
    with pytest.raises(TypeError):
        tp._C.i0(_meta(2, 3), out=_meta(2, 3, dtype=tp.int64))
    with pytest.raises(TypeError):
        tp._C.igamma(_meta(2, 3), _meta(2, 3), out=_meta(2, 3, dtype=tp.bool))


def test_resize_on_meta():
    # growing and shrinking rewrites the sizes contiguously
    t = _meta(4)
    assert t.resize_((2, 3)) is t
    assert tuple(t.shape) == (2, 3)
    assert tuple(t.stride()) == (3, 1)
    t.resize_((4,))
    assert tuple(t.shape) == (4,)

    # a call that keeps the current sizes is a no-op for the strides
    t = _meta(3, 4)
    assert tuple(t.stride()) == (4, 1)
    t.resize_((3, 4))
    assert tuple(t.stride()) == (4, 1)

    # a same-size call on an expanded tensor keeps its broadcast strides
    e = _meta(1, 4).expand(3, 4)
    assert tuple(e.stride()) == (0, 1)
    e.resize_((3, 4))
    assert tuple(e.stride()) == (0, 1)
    assert tuple(e.shape) == (3, 4)

    assert tuple(_meta(2, 3).resize_((0,)).shape) == (0,)
    assert tuple(_meta(2, 3).resize_(tuple()).shape) == ()

    with pytest.raises(ValueError):
        _meta(2, 3).resize_((2, -3))
    with pytest.raises(RuntimeError):
        _meta(2).resize_((2**61, 2**61))


def test_special_float_math_dtype_contracts():
    # the float-math sweeps refuse complex and boolean operands for the
    # members that have no single-precision pass over those planes
    for name in ("erf", "erfc", "lgamma"):
        f = getattr(tp._C, name)
        with pytest.raises(TypeError):
            f(_meta(2, 3, dtype=tp.complex128))
        with pytest.raises(TypeError):
            f(_meta(2, 3, dtype=tp.bool))
        assert f(_meta(2, 3, dtype=tp.int64)).dtype == tp.float32
    for name in ("round", "floor", "ceil", "trunc", "frac"):
        with pytest.raises(TypeError):
            getattr(tp._C, name)(_meta(2, 3, dtype=tp.complex128))
    assert tp._C.round(_meta(2, 3, dtype=tp.bool)).dtype == tp.bool
    assert tp._C.round(_meta(2, 3, dtype=tp.int64)).dtype == tp.int64
    # the remaining sweeps answer every non-floating plane in single precision
    for name in ("exp2", "sinc", "erfinv", "digamma", "i0", "entr", "logit"):
        assert getattr(tp._C, name)(
            _meta(2, 3, dtype=tp.complex128)).dtype == tp.float32
        assert getattr(tp._C, name)(_meta(2, 3, dtype=tp.bool)).dtype == tp.float32
    # the pair forms of the product family follow the same lift
    assert tp._C.xlogy(_meta(2, 3, dtype=tp.complex128),
                       _meta(2, 3, dtype=tp.complex128)).dtype == tp.float32
    assert tp._C.xlog1py(_meta(2, 3, dtype=tp.complex128),
                         _meta(2, 3, dtype=tp.complex128)).dtype == tp.float32


def test_softmax_and_complex_passthrough_contracts():
    # the normalization runs on real numbers only, after the axis is named
    for f in (tp._C.softmax, tp._C.log_softmax):
        with pytest.raises(TypeError):
            f(_meta(2, 3, dtype=tp.complex128), 1)
        with pytest.raises(TypeError):
            f(_meta(2, 3, dtype=tp.bool), 1)
    # a bad axis is named before the dtype is policed
    with pytest.raises(IndexError):
        tp._C.softmax(_meta(2, 3, dtype=tp.complex128), 99)
    # the complex-capable activations keep their own planes
    for name in ("sigmoid", "tanh", "expm1", "log1p"):
        assert getattr(tp._C, name)(
            _meta(2, 3, dtype=tp.complex128)).dtype == tp.complex128
    # the reduction refuses complex planes outright
    with pytest.raises(RuntimeError):
        tp._C.logsumexp(_meta(2, 3, dtype=tp.complex128), [1])


REAL_MATH_UNARY = (
    "exp", "expm1", "log", "log2", "log10", "log1p", "sqrt", "rsqrt",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "asinh", "acosh", "atanh",
    "tanh", "sigmoid", "tanhshrink",
)


def test_real_math_sweeps_keep_their_planes():
    # the logarithm / exponential / trigonometric sweeps compute over real
    # numbers or a complex lane of their own: a whole-number operand is
    # answered in single precision, a boolean or 8-bit float operand has no
    # form to run in, and a floating or complex operand keeps its width
    f8 = getattr(tp, "float8_e4m3fn", None)
    for name in REAL_MATH_UNARY:
        f = getattr(tp._C, name)
        with pytest.raises(TypeError):
            f(_meta(2, 3, dtype=tp.bool))
        if f8 is not None:
            with pytest.raises(TypeError):
                f(_meta(2, 3, dtype=f8))
        assert f(_meta(2, 3, dtype=tp.int64)).dtype == tp.float32
        assert f(_meta(2, 3, dtype=tp.complex128)).dtype == tp.complex128
        assert f(_meta(2, 3, dtype=tp.float16)).dtype == tp.float16


REAL_ONLY_ACTIVATIONS = (
    ("gelu", ()), ("mish", ()), ("softplus", (1.0, 20.0)), ("elu", (1.0,)),
    ("selu", ()), ("celu", (1.0,)), ("leaky_relu", (0.01,)),
    ("hardsigmoid", ()), ("hardswish", ()), ("relu6", ()),
    ("log_sigmoid", ()), ("hardtanh", (-1.0, 1.0)),
)


def test_real_only_activations_refuse_planes():
    # the activation family runs a single real-number form: boolean, complex
    # and 8-bit float operands are refused, whole numbers are answered in
    # single precision
    f8 = getattr(tp, "float8_e4m3fn", None)
    for name, args in REAL_ONLY_ACTIVATIONS:
        f = getattr(tp._C, name)
        with pytest.raises(TypeError):
            f(_meta(2, 3, dtype=tp.bool), *args)
        with pytest.raises(TypeError):
            f(_meta(2, 3, dtype=tp.complex128), *args)
        if f8 is not None:
            with pytest.raises(TypeError):
                f(_meta(2, 3, dtype=f8), *args)
        assert f(_meta(2, 3, dtype=tp.int64), *args).dtype == tp.float32


def test_pointwise_inplace_dtype_gates():
    # sweeps with a complex lane refuse boolean and whole-number targets but
    # accept a complex target of their own
    for name in ("exp_", "log_", "sin_", "tanh_", "sigmoid_", "deg2rad_"):
        t = _meta(2, 3, dtype=tp.int64)
        with pytest.raises(TypeError):
            getattr(tp._C, name)(t)
        t = _meta(2, 3, dtype=tp.bool)
        with pytest.raises(TypeError):
            getattr(tp._C, name)(t)
        t = _meta(2, 3, dtype=tp.complex128)
        assert getattr(tp._C, name)(t) is t
    # the single-form sweeps and activations refuse complex targets as well
    for name in ("erf_", "erfc_", "lgamma_", "silu_", "mish_",
                 "hardsigmoid_", "hardswish_"):
        t = _meta(2, 3, dtype=tp.complex128)
        with pytest.raises(TypeError):
            getattr(tp._C, name)(t)
        t = _meta(2, 3, dtype=tp.int64)
        with pytest.raises(TypeError):
            getattr(tp._C, name)(t)
    # the copy-back activations accept a whole-number target and keep it
    for name, args in (("gelu_", ()), ("elu_", (1.0,)), ("selu_", ()),
                       ("celu_", (1.0,)), ("leaky_relu_", (0.01,)),
                       ("relu6_", ()), ("hardtanh_", (-1.0, 1.0))):
        t = _meta(2, 3, dtype=tp.int64)
        assert getattr(tp._C, name)(t, *args) is t
        t = _meta(2, 3, dtype=tp.complex128)
        with pytest.raises(TypeError):
            getattr(tp._C, name)(t, *args)
        t = _meta(2, 3, dtype=tp.bool)
        with pytest.raises(TypeError):
            getattr(tp._C, name)(t, *args)
    # the rectangle clamp only refuses complex and 8-bit float targets
    for name in ("relu_",):
        for dt in (tp.int64, tp.bool):
            t = _meta(2, 3, dtype=dt)
            assert getattr(tp._C, name)(t) is t
        t = _meta(2, 3, dtype=tp.complex128)
        with pytest.raises(TypeError):
            getattr(tp._C, name)(t)
    # the rounding family keeps whole planes and refuses complex ones
    for name in ("ceil_", "floor_", "round_", "trunc_"):
        t = _meta(2, 3, dtype=tp.int64)
        assert getattr(tp._C, name)(t) is t
        t = _meta(2, 3, dtype=tp.complex128)
        with pytest.raises(TypeError):
            getattr(tp._C, name)(t)
    # the fractional part has no whole-number or complex form
    with pytest.raises(NotImplementedError):
        tp._C.frac_(_meta(2, 3, dtype=tp.int64))
    with pytest.raises(TypeError):
        tp._C.frac_(_meta(2, 3, dtype=tp.complex128))


def test_pair_and_clamp_dtype_edges():
    # the power family keeps plain boolean promotion and refuses a boolean
    # base with a scalar exponent
    assert tp._C.pow(_meta(2, 3, dtype=tp.bool),
                     _meta(2, 3, dtype=tp.bool)).dtype == tp.bool
    assert tp._C.pow(2.0, _meta(2, 3, dtype=tp.bool)).dtype == tp.float32
    with pytest.raises(TypeError):
        tp._C.pow(_meta(2, 3, dtype=tp.bool), 2.0)
    # the clamp family polices complex and 8-bit float operands and keeps
    # boolean and whole-number planes
    f8 = getattr(tp, "float8_e4m3fn", None)
    for name, args in (("clamp", (-1.0, 1.0)), ("clamp_min", (-1.0,)),
                       ("clamp_max", (1.0,))):
        f = getattr(tp._C, name)
        with pytest.raises(TypeError):
            f(_meta(2, 3, dtype=tp.complex128), *args)
        if f8 is not None:
            with pytest.raises(TypeError):
                f(_meta(2, 3, dtype=f8), *args)
        assert f(_meta(2, 3, dtype=tp.int64), *args).dtype == tp.int64
        assert f(_meta(2, 3, dtype=tp.bool), *args).dtype == tp.bool
    # sign answers a complex operand in its own plane
    assert tp._C.sign(_meta(2, 3, dtype=tp.complex128)).dtype == tp.complex128
    assert tp._C.sgn(_meta(2, 3, dtype=tp.complex128)).dtype == tp.complex128
    # the step family rejects a complex pair outright
    with pytest.raises(TypeError):
        tp._C.heaviside(_meta(2, 3, dtype=tp.complex128),
                        _meta(2, 3, dtype=tp.complex128))
    with pytest.raises(TypeError):
        tp._C.heaviside_(_meta(2, 3, dtype=tp.complex128),
                         _meta(2, 3, dtype=tp.complex128))
    t = _meta(2, 3, dtype=tp.int64)
    assert tp._C.heaviside_(t, _meta(2, 3, dtype=tp.int64)) is t
    # the rectangle activation refuses a complex plane in both forms
    with pytest.raises(TypeError):
        tp._C.relu(_meta(2, 3, dtype=tp.complex128))
    with pytest.raises(TypeError):
        tp._C.relu_(_meta(2, 3, dtype=tp.complex128))
    # the fractional part of a whole-number plane has no kernel form
    with pytest.raises(NotImplementedError):
        tp._C.frac(_meta(2, 3, dtype=tp.int64))
    with pytest.raises(TypeError):
        tp._C.frac(_meta(2, 3, dtype=tp.complex128))
    # the angle conversions have no complex form and answer in single
    # precision
    assert tp._C.deg2rad(_meta(2, 3, dtype=tp.complex128)).dtype == tp.float32
    assert tp._C.rad2deg(_meta(2, 3, dtype=tp.complex128)).dtype == tp.float32


def test_where_condition_must_be_boolean():
    # every selection form reads the condition element by element, so a
    # non-boolean condition is refused before the geometry is answered
    for cond_dt in (tp.float32, tp.int64):
        cond = _meta(2, 3, dtype=cond_dt)
        with pytest.raises(TypeError):
            tp._C.where(cond, _meta(2, 3), _meta(2, 3))
        with pytest.raises(TypeError):
            tp._C.where(cond, _meta(2, 3), 1.0)
        with pytest.raises(TypeError):
            tp._C.where(cond, 1.0, _meta(2, 3))
        with pytest.raises(TypeError):
            tp._C.where(cond, 1.0, 2.0)
    # a boolean condition selects over the promoted operand types
    cond = _meta(2, 3, dtype=tp.bool)
    assert tp._C.where(cond, _meta(2, 3, dtype=tp.int64), 1.0).dtype == \
        tp.float32
    assert tp._C.where(cond, 1.0, 2.0).dtype == tp.float64


def test_extrema_and_median_refuse_complex():
    # the pairwise extrema have no complex form, and the shape checks run
    # before the dtype is policed
    for name, args in (("max", ()), ("min", ()), ("max", (1,)), ("min", (1,)),
                       ("argmax", ()), ("argmax", (1,)), ("argmin", (1,)),
                       ("median", ()), ("median", (1,))):
        # median carries a flat form that refuses complex as unimplemented
        # and a dim form with no kernel at all, which surfaces as a type
        # error; the rest refuse complex as unimplemented everywhere
        expected = TypeError if name == "median" and args == (1,) \
            else NotImplementedError
        with pytest.raises(expected):
            getattr(tp._C, name)(_meta(2, 3, dtype=tp.complex64), *args)
    for name in ("amax", "amin"):
        with pytest.raises(TypeError):
            getattr(tp._C, name)(_meta(2, 3, dtype=tp.complex64), [1])
    with pytest.raises(TypeError):
        tp._C.aminmax(_meta(2, 3, dtype=tp.complex64), 1)
    # an out-of-range dimension is named before the dtype is policed
    with pytest.raises(IndexError):
        tp._C.amax(_meta(2, 3, dtype=tp.complex64), [5])
    with pytest.raises(IndexError):
        tp._C.aminmax(_meta(2, 3, dtype=tp.complex64), 5)
    # whole-number and boolean planes keep their extrema forms
    for dt in (tp.int64, tp.bool):
        t = _meta(2, 3, dtype=dt)
        assert tp._C.max(t).dtype == dt
        assert tp._C.min(t, 1)[0].dtype == dt
        assert tp._C.amax(t, [1]).dtype == dt
        assert tp._C.aminmax(t, 1)[0].dtype == dt
        assert tp._C.argmax(t).dtype == tp.int64


def test_nanmedian_dtype_contract():
    # the flat form answers every operand in a comparable type: a floating
    # plane keeps its own width, anything else answers as a whole-number
    # scalar
    for dt in (tp.complex64, tp.int64, tp.bool):
        r = tp._C.nanmedian(_meta(2, 3, dtype=dt))
        assert r.dtype == tp.int64
        assert tuple(r.shape) == ()
    for dt in (tp.float16, tp.bfloat16, tp.float32, tp.float64):
        assert tp._C.nanmedian(_meta(2, 3, dtype=dt)).dtype == dt
    # the dim form computes over floating planes only: whole-number planes
    # are refused as unrunnable and the half-width planes have no kernel
    with pytest.raises(RuntimeError):
        tp._C.nanmedian(_meta(2, 3, dtype=tp.int64), 1)
    with pytest.raises(TypeError):
        tp._C.nanmedian(_meta(2, 3, dtype=tp.float16), 1)
    with pytest.raises(TypeError):
        tp._C.nanmedian(_meta(2, 3, dtype=tp.bfloat16), 1)
    assert tp._C.nanmedian(_meta(2, 3, dtype=tp.float32), 1)[0].dtype == \
        tp.float32


def test_norm_dtype_contract():
    # a whole-number operand runs the norm as a chain of powers: order 1
    # keeps the operand type, any other finite order answers in single
    # precision, an infinite order short-circuits to the largest element and
    # keeps the operand type, and a negative order has no whole-number power
    i64 = _meta(2, 3, dtype=tp.int64)
    assert tp._C.norm(i64, 1.0).dtype == tp.int64
    assert tp._C.norm(i64, 2.0).dtype == tp.float32
    assert tp._C.norm(i64, float("inf")).dtype == tp.int64
    assert tp._C.norm(i64, float("-inf")).dtype == tp.int64
    assert tp._C.norm(i64, [1], float("inf"), True).dtype == tp.int64
    assert tp._C.norm(i64, [1], 2.0, True).dtype == tp.float32
    with pytest.raises(RuntimeError):
        tp._C.norm(i64, -2.0)
    # a boolean operand has no power form: only the infinite orders run
    with pytest.raises(TypeError):
        tp._C.norm(_meta(2, 3, dtype=tp.bool), 2.0)
    assert tp._C.norm(_meta(2, 3, dtype=tp.bool),
                      float("inf")).dtype == tp.bool
    assert tp._C.norm(_meta(2, 3, dtype=tp.bool), [1], float("inf"),
                      True).dtype == tp.bool
    # floating planes keep their width and complex planes answer in their
    # real component type
    assert tp._C.norm(_meta(2, 3, dtype=tp.float16)).dtype == tp.float16
    assert tp._C.norm(_meta(2, 3, dtype=tp.float64), 2.0).dtype == tp.float64
    assert tp._C.norm(_meta(2, 3, dtype=tp.complex64)).dtype == tp.float32
    assert tp._C.norm(_meta(2, 3, dtype=tp.complex128)).dtype == tp.float64


def test_pdist_dtype_contract():
    # the pairwise distances accumulate at single precision, so a half-width
    # operand answers in single precision and wider planes keep their width
    assert tp._C.pdist(_meta(4, 3, dtype=tp.float16)).dtype == tp.float32
    assert tp._C.pdist(_meta(4, 3, dtype=tp.bfloat16)).dtype == tp.float32
    assert tp._C.pdist(_meta(4, 3, dtype=tp.float32)).dtype == tp.float32
    assert tp._C.pdist(_meta(4, 3, dtype=tp.float64)).dtype == tp.float64
    # a distance has no whole-number or complex form
    for dt in (tp.int64, tp.bool, tp.complex64):
        with pytest.raises(TypeError):
            tp._C.pdist(_meta(4, 3, dtype=dt))


def test_var_std_answer_real_and_mean_keeps_plane():
    # the spread measures square real deviations, so a complex operand
    # answers its spread in the real component type while the mean keeps
    # the complex plane
    for dt, real in ((tp.complex64, tp.float32),
                     (tp.complex128, tp.float64)):
        v, m = tp._C.var_mean(_meta(2, 3, dtype=dt), [1])
        assert v.dtype == real
        assert m.dtype == dt
        s, m2 = tp._C.std_mean(_meta(2, 3, dtype=dt), [1])
        assert s.dtype == real
        assert m2.dtype == dt
    # a floating operand answers both members in its own type
    v, m = tp._C.var_mean(_meta(2, 3, dtype=tp.float32), [1])
    assert v.dtype == tp.float32
    assert m.dtype == tp.float32


def test_ordered_comparisons_refuse_complex():
    # an ordering is undefined over complex planes: the tensor forms refuse
    # a complex operand outright while equality keeps one
    for name in ("lt", "le", "gt", "ge"):
        f = getattr(tp._C, name)
        with pytest.raises(NotImplementedError):
            f(_meta(2, 3, dtype=tp.complex64), _meta(2, 3, dtype=tp.complex64))
        with pytest.raises(NotImplementedError):
            f(_meta(2, 3, dtype=tp.complex64), _meta(2, 3, dtype=tp.float32))
        # the scalar forms compare element-wise against a real value
        assert f(_meta(2, 3, dtype=tp.complex64), 0.5).dtype == tp.bool
    assert tp._C.eq(_meta(2, 3, dtype=tp.complex64),
                    _meta(2, 3, dtype=tp.complex64)).dtype == tp.bool
    assert tp._C.ne(_meta(2, 3, dtype=tp.complex64), 0.5).dtype == tp.bool


# ---------------------------------------------------------------------------
# Family refusal paths: dimension wrapping, dtype tables and geometry checks
# ---------------------------------------------------------------------------


def test_index_family_wraps_negative_and_rejects_unknown_dims():
    t = _meta(2, 3)
    vec = _meta(2, dtype=tp.int64)
    # The gathered reads and writes share one dimension wrap: a negative axis
    # counts from the end, and an axis past the end is refused after the wrap.
    for call in (
        lambda d: tp.gather(t, d, _meta(2, 2, dtype=tp.int64)),
        lambda d: tp.index_select(t, d, _meta(2, dtype=tp.int64)),
        lambda d: tp._C.scatter(t, d, _meta(2, 2, dtype=tp.int64),
                                _meta(2, 2)),
        lambda d: tp._C.scatter_add(t, d, _meta(2, 2, dtype=tp.int64),
                                    _meta(2, 2)),
        lambda d: tp._C.index_add(t, d, _meta(2, dtype=tp.int64), _meta(2, 2)),
        lambda d: tp._C.index_fill(t, d, _meta(2, dtype=tp.int64), 1.0),
    ):
        with pytest.raises(RuntimeError):
            call(5)
        assert call(-1) is not None
    # The flat read walks real and complex planes alike.
    c = _meta(2, 3, dtype=tp.complex64)
    assert tp.take(c, _meta(2, dtype=tp.int64)).dtype == tp.complex64


def test_topk_zero_and_empty_selections():
    # A selection that writes nothing still produces the answer planes for
    # every dtype, including those the ordering walk itself refuses.
    for dt in (tp.bool, tp.complex64, tp.int64, tp.float16):
        values, indices = tp._C.topk(_meta(2, 3, dtype=dt), 0, 1, True, True)
        assert tuple(values.shape) == (2, 0)
        assert values.dtype == dt
        assert indices.dtype == tp.int64
    empty_bool = _meta(0, 3, dtype=tp.bool)
    values, indices = tp._C.topk(empty_bool, 3, 1, True, True)
    assert tuple(values.shape) == (0, 3)
    assert values.dtype == tp.bool
    assert indices.dtype == tp.int64
    # A walk that would actually order elements refuses booleans and
    # complex planes.
    with pytest.raises(NotImplementedError):
        tp._C.topk(_meta(2, 3, dtype=tp.bool), 2, 1, True, True)
    with pytest.raises(NotImplementedError):
        tp._C.topk(_meta(2, 3, dtype=tp.complex64), 2, 1, True, True)
    with pytest.raises(RuntimeError):
        tp._C.topk(_meta(2, 3), 5, 1, True, True)


def test_softmax_native_and_backward_tables():
    t = _meta(2, 3)
    i = _meta(2, 3, dtype=tp.int64)
    # The lower-level forward refuses a set half-to-float flag before it
    # reads anything else.
    for x in (t, i, _meta(2, 3, dtype=tp.complex64)):
        with pytest.raises(RuntimeError):
            tp._C._softmax(x, 1, True)
        with pytest.raises(RuntimeError):
            tp._C._log_softmax(x, 1, True)
    # Whole-number planes answer at single precision; boolean and complex
    # planes have no normalization.
    assert tp._C._softmax(i, 1, False).dtype == tp.float32
    assert tp._C._log_softmax(i, 1, False).dtype == tp.float32
    assert tp._C._softmax(t, 1, False).dtype == tp.float32
    with pytest.raises(TypeError):
        tp._C._softmax(_meta(2, 3, dtype=tp.bool), 1, False)
    with pytest.raises(TypeError):
        tp._C._softmax(_meta(2, 3, dtype=tp.complex64), 1, False)
    with pytest.raises(IndexError):
        tp._C._softmax(t, 5, False)
    # The backward answers at the incoming gradient's own type, except a
    # half forward whose gradient arrived in single precision narrows back.
    g16 = _meta(2, 3, dtype=tp.float16)
    g32 = _meta(2, 3, dtype=tp.float32)
    out16 = _meta(2, 3, dtype=tp.float16)
    assert tp._C._softmax_backward_data(g32, out16, 1, tp.float16).dtype == tp.float16
    assert tp._C._softmax_backward_data(g16, out16, 1, tp.float16).dtype == tp.float16
    assert tp._C._softmax_backward_data(
        _meta(2, 3, dtype=tp.float64), t, 1, tp.float32).dtype == tp.float64
    assert tp._C._softmax_backward_data(
        _meta(2, 3, dtype=tp.bfloat16), _meta(2, 3, dtype=tp.bfloat16), 1,
        tp.bfloat16).dtype == tp.bfloat16
    assert tp._C._log_softmax_backward_data(g32, out16, 1, tp.float16).dtype == tp.float16


def _attn(dtype=tp.float32, *shape):
    return _meta(*(shape or (1, 2, 3, 4)), dtype=dtype)


def test_attention_geometry_checks_and_dtypes():
    q = _attn()
    out = tp._C.scaled_dot_product_attention(q, _attn(), _attn())
    assert tuple(out.shape) == (1, 2, 3, 4)
    assert out.dtype == tp.float32
    for dt in (tp.float16, tp.bfloat16, tp.float64):
        assert tp._C.scaled_dot_product_attention(
            _attn(dt), _attn(dt), _attn(dt)).dtype == dt
    for dt in (tp.complex64, tp.int64, tp.bool):
        with pytest.raises(NotImplementedError):
            tp._C.scaled_dot_product_attention(_attn(dt), _attn(dt), _attn(dt))
    # A mask of any broadcastable rank joins the scores in the working
    # precision: a wide mask is narrowed, so the answer keeps its own type.
    for mask_shape, mask_dt in (((3,), tp.float32), ((3, 3), tp.float32),
                                ((1, 1, 3, 3), tp.bool), ((3, 3), tp.float64)):
        assert tp._C.scaled_dot_product_attention(
            q, _attn(), _attn(), _meta(*mask_shape, dtype=mask_dt)
        ).dtype == tp.float32
    with pytest.raises(RuntimeError):
        tp._C.scaled_dot_product_attention(q, _attn(), _attn(), _meta(2, 2))
    with pytest.raises(RuntimeError):
        tp._C.scaled_dot_product_attention(q, _attn(), _attn(),
                                           _meta(1, 3, 3, 3))
    # One head dimension is read from the query and must be non-zero, match
    # the key, and sit inside matching key/value weights.
    with pytest.raises(ValueError):
        tp._C.scaled_dot_product_attention(_attn(tp.float32, 1, 2, 3, 0),
                                           _attn(tp.float32, 1, 2, 3, 0),
                                           _attn(tp.float32, 1, 2, 3, 0))
    with pytest.raises(ValueError):
        tp._C.scaled_dot_product_attention(_attn(), _attn(tp.float32, 1, 2, 4, 4),
                                           _attn(tp.float32, 1, 2, 3, 4))
    with pytest.raises(ValueError):
        tp._C.scaled_dot_product_attention(_attn(tp.float32, 1, 4, 3, 4),
                                           _attn(tp.float32, 1, 3, 3, 4),
                                           _attn(tp.float32, 1, 3, 3, 4))
    # Mixed operand types reach the score product's own refusal.
    with pytest.raises(RuntimeError):
        tp._C.scaled_dot_product_attention(_attn(tp.float64), _attn(),
                                           _attn())
    with pytest.raises(RuntimeError):
        tp._C.scaled_dot_product_attention(_attn(), _attn(tp.float64),
                                           _attn(tp.float64))
    # Ranks other than three and four follow the score product's rules: the
    # key and value must broadcast against the query's batch geometry.
    assert tuple(tp._C.scaled_dot_product_attention(
        _attn(tp.float32, 2, 3, 4), _attn(tp.float32, 2, 3, 4),
        _attn(tp.float32, 2, 3, 4)).shape) == (2, 3, 4)
    assert tuple(tp._C.scaled_dot_product_attention(
        _attn(tp.float32, 3, 4), _attn(tp.float32, 3, 4),
        _attn(tp.float32, 3, 4)).shape) == (3, 4)
    assert tuple(tp._C.scaled_dot_product_attention(
        _attn(tp.float32, 4), _attn(tp.float32, 3, 4),
        _attn(tp.float32, 3, 4)).shape) == (4,)
    # A lower-rank key/value pair broadcasts against the query's batch, and
    # a one-dimensional key whose value is also flat has no matrix axis to
    # transpose.
    assert tuple(tp._C.scaled_dot_product_attention(
        _attn(tp.float32, 2, 3, 4), _attn(tp.float32, 3, 4),
        _attn(tp.float32, 3, 4)).shape) == (2, 3, 4)
    with pytest.raises(IndexError):
        tp._C.scaled_dot_product_attention(_attn(tp.float32, 2, 3, 4),
                                           _attn(tp.float32, 4),
                                           _attn(tp.float32, 4))
    with pytest.raises(RuntimeError):
        tp._C.scaled_dot_product_attention(
            _attn(tp.float32, 2, 3, 3, 4), _attn(tp.float32, 5, 3, 4),
            _attn(tp.float32, 5, 3, 4))
    # A query batch of one expands to the key/value batch instead.
    assert tuple(tp._C.scaled_dot_product_attention(
        _attn(tp.float32, 2, 1, 3, 4), _attn(tp.float32, 1, 3, 4),
        _attn(tp.float32, 1, 3, 4)).shape) == (2, 1, 3, 4)
    assert tuple(tp._C.scaled_dot_product_attention(
        _attn(tp.float32, 2, 3, 4, 5, 6), _attn(tp.float32, 2, 3, 4, 5, 6),
        _attn(tp.float32, 2, 3, 4, 5, 6)).shape) == (2, 3, 4, 5, 6)


def test_attention_lse_planes():
    q, k, v = _attn(), _attn(), _attn()
    out, lse = tp._C._scaled_dot_product_attention_with_lse(q, k, v, False, 0)
    assert tuple(out.shape) == (1, 2, 3, 4)
    assert out.dtype == tp.float32
    assert tuple(lse.shape) == (1, 2, 3)
    assert lse.dtype == tp.float32
    with pytest.raises(NotImplementedError):
        tp._C._scaled_dot_product_attention_with_lse(
            _attn(tp.int64), _attn(tp.int64), _attn(tp.int64), False, 0)


def test_linalg_whole_number_and_complex_refusals():
    eye = tp.eye(3, device="meta")
    assert tp._C.matrix_power(eye, -1).dtype == tp.float32
    # A negative power inverts the operand, which runs over the linalg
    # element types only.
    for dt in (tp.int64, tp.float16, tp.bool):
        with pytest.raises(NotImplementedError):
            tp._C.matrix_power(tp.eye(3, dtype=dt, device="meta"), -1)
    assert tp._C.matrix_power(
        tp.eye(3, dtype=tp.int64, device="meta"), 2).dtype == tp.int64
    # The factorization unpacks a permutation written in the operand's own
    # type, so complex planes have no form for it.
    for shape in ((3, 3), (2, 3)):
        with pytest.raises(TypeError):
            tp._C.linalg_lu(_meta(*shape, dtype=tp.complex64))


def test_loss_dtype_tables():
    # The squared difference runs in the pair's promoted type: a whole-number
    # pair answers its mean at single precision and keeps its pair type for
    # the elementwise and summed forms.
    i = _meta(2, 3, dtype=tp.int64)
    assert tp._C.mse_loss(i, i, 1).dtype == tp.float32
    assert tp._C.mse_loss(i, i, 0).dtype == tp.int64
    assert tp._C.mse_loss(i, i, 2).dtype == tp.int64
    assert tp._C.mse_loss(_meta(2, 3, dtype=tp.complex64),
                          _meta(2, 3, dtype=tp.complex64), 1).dtype == tp.complex64
    assert tp._C.mse_loss(_meta(2, 3, dtype=tp.float32), i, 1).dtype == tp.float32
    assert tp._C.mse_loss(_meta(2, 3, dtype=tp.float32),
                          _meta(2, 3, dtype=tp.float64), 1).dtype == tp.float64
    with pytest.raises(RuntimeError):
        tp._C.mse_loss(_meta(2, 3, dtype=tp.bool), _meta(2, 3, dtype=tp.bool), 1)
    assert tp._C.mse_loss(_meta(2, 3, dtype=tp.bool), t32 := _meta(2, 3), 1).dtype == tp.float32
    with pytest.raises(ValueError):
        tp._C.mse_loss(i, i, 5)
    # The switched absolute blend multiplies by non-integral scalars in both
    # branches, so a whole-number pair answers at single precision at every
    # reduction.
    for reduction in (0, 1, 2):
        assert tp._C.smooth_l1_loss(i, i, 0.0, reduction).dtype == tp.float32
    assert tp._C.smooth_l1_loss(_meta(2, 3, dtype=tp.complex64),
                                _meta(2, 3, dtype=tp.complex64), 0.0,
                                1).dtype == tp.complex64
    with pytest.raises(RuntimeError):
        tp._C.smooth_l1_loss(_meta(2, 3, dtype=tp.bool),
                             _meta(2, 3, dtype=tp.bool), 0.0, 1)
    with pytest.raises(ValueError):
        tp._C.smooth_l1_loss(t32, t32, 0.0, -1.0)
    # The scored loss runs over the four wide element types only.
    tgt = _meta(2, dtype=tp.int64)
    raw, total = tp._C.nll_loss(_meta(2, 4, dtype=tp.float16), tgt, None, 1, -100)
    assert raw.dtype == tp.float16
    assert total.dtype == tp.float16
    raw, total = tp._C.nll_loss(_meta(2, 4, dtype=tp.bfloat16), tgt, None, 1, -100)
    assert raw.dtype == tp.bfloat16
    assert total.dtype == tp.bfloat16
    for dt in (tp.int64, tp.bool):
        with pytest.raises(NotImplementedError):
            tp._C.nll_loss(_meta(2, 4, dtype=dt), tgt, None, 1, -100)
    # The logarithmic loss blends a logarithm and a fractional shift, so
    # whole-number planes answer at single precision; the weight joins the
    # promotion.
    assert tp._C.binary_cross_entropy(i, i, None, 1).dtype == tp.float32
    assert tp._C.binary_cross_entropy(
        _meta(2, 3, dtype=tp.float16), _meta(2, 3, dtype=tp.float16), None,
        1).dtype == tp.float16
    assert tp._C.binary_cross_entropy(i, i, i, 1).dtype == tp.float32


# ---------------------------------------------------------------------------
# Cross-checks against the reference framework when it is installed
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def _torch_meta(*size, dtype=None):
    return torch.empty(size, dtype=dtype, device="meta")


@pytest.mark.parametrize(
    "shape_a,shape_b",
    [((3, 1), (1, 4)), ((5,), (5,)), ((2, 3, 4), (4,))],
)
def test_pointwise_metadata_matches_reference(shape_a, shape_b):
    ours = _meta(*shape_a, dtype=tp.float32) + _meta(*shape_b, dtype=tp.float32)
    theirs = _torch_meta(*shape_a) + _torch_meta(*shape_b)
    assert tuple(ours.shape) == tuple(theirs.shape)
    assert ours.dtype == tp.float32


def test_reduction_metadata_matches_reference():
    ours = _meta(3, 4, dtype=tp.int32).sum(dim=1)
    theirs = _torch_meta(3, 4, dtype=torch.int32).sum(dim=1)
    assert tuple(ours.shape) == tuple(theirs.shape)
    assert ours.dtype == tp.int64


def test_mm_metadata_matches_reference():
    ours = _meta(2, 3) @ _meta(3, 5)
    theirs = _torch_meta(2, 3) @ _torch_meta(3, 5)
    assert tuple(ours.shape) == tuple(theirs.shape)


# ---------------------------------------------------------------------------
# Convolution family
# ---------------------------------------------------------------------------


def test_convolution_family_metadata():
    C = tp._C

    def meta(*shape, dt=tp.float32):
        return tp.zeros(list(shape), dtype=dt, device="meta")

    conv = C.convolution(meta(1, 3, 8), meta(2, 3, 2), None,
                         [1], [0], [1], False, [0], 1)
    assert tuple(conv.shape) == (1, 2, 7)
    conv2d = C.convolution(meta(1, 3, 6, 6), meta(2, 3, 2, 2), None,
                           [1], [0], [1], False, [0], 1)
    assert tuple(conv2d.shape) == (1, 2, 5, 5)
    assert tuple(C.convolution(meta(1, 3, 6, 6, 6), meta(2, 3, 2, 2, 2), None,
                               [1], [0], [1], False, [0], 1).shape) == (1, 2, 5, 5, 5)
    assert tuple(C.convolution(meta(1, 3, 6, 6), meta(2, 3, 2, 2), meta(2),
                               [1], [0], [1], False, [0], 1).shape) == (1, 2, 5, 5)
    assert tuple(C.convolution(meta(1, 3, 6, 6), meta(2, 3, 2, 2), None,
                               [2], [0], [1], False, [0], 1).shape) == (1, 2, 3, 3)
    assert tuple(C.convolution(meta(1, 4, 6, 6), meta(4, 1, 2, 2), None,
                               [1], [0], [1], False, [0], 4).shape) == (1, 4, 5, 5)
    assert tuple(C.convolution(meta(1, 2, 6), meta(2, 3, 2), None,
                               [1], [0], [1], True, [0], 1).shape) == (1, 3, 7)
    assert tuple(C.convolution(meta(1, 2, 6, 6), meta(2, 3, 2, 2), None,
                               [1], [0], [1], True, [0], 1).shape) == (1, 3, 7, 7)
    assert tuple(C.convolution(meta(1, 2, 6), meta(2, 3, 2), None,
                               [2], [0], [1], True, [0], 1).shape) == (1, 3, 12)
    f16 = C.convolution(meta(1, 3, 6, 6, dt=tp.float16),
                        meta(2, 3, 2, 2, dt=tp.float16), None,
                        [1], [0], [1], False, [0], 1)
    assert f16.dtype == tp.float16

    # overrideable spellings share the same geometry
    assert tuple(C.convolution_overrideable(meta(1, 3, 6, 6), meta(2, 3, 2, 2), None,
                                            [1], [0], [1], False, [0], 1).shape) == (1, 2, 5, 5)

    with pytest.raises(Exception):
        C.convolution(meta(1, 3, 6), meta(2, 3, 2, 2), None,
                      [1], [0], [1], False, [0], 1)
    with pytest.raises(Exception):
        C.convolution(meta(1, 3, 6, 6), meta(2, 4, 2, 2), None,
                      [1], [0], [1], False, [0], 1)


def test_convolution_backward_overrideable_metadata():
    C = tp._C

    def meta(*shape):
        return tp.zeros(list(shape), dtype=tp.float32, device="meta")

    gi, gw, gb = C.convolution_backward_overrideable(
        meta(1, 2, 5, 5), meta(1, 3, 6, 6), meta(2, 3, 2, 2),
        [1], [0], [1], False, [0], 1, [True, True, True])
    assert tuple(gi.shape) == (1, 3, 6, 6)
    assert tuple(gw.shape) == (2, 3, 2, 2)
    assert tuple(gb.shape) == (2,)
    gi, gw, gb = C.convolution_backward_overrideable(
        meta(1, 2, 7), meta(1, 3, 8), meta(2, 3, 2),
        [1], [0], [1], False, [0], 1, [True, True, True])
    assert tuple(gi.shape) == (1, 3, 8)
    assert tuple(gw.shape) == (2, 3, 2)
    assert tuple(gb.shape) == (2,)
    gi, gw, gb = C.convolution_backward_overrideable(
        meta(1, 2, 5, 5), meta(1, 3, 6, 6), meta(2, 3, 2, 2),
        [1], [0], [1], False, [0], 1, [True, False, False])
    assert tuple(gi.shape) == (1, 3, 6, 6)
    assert not gw.defined() and not gb.defined()
    gi, gw, gb = C.convolution_backward_overrideable(
        meta(1, 2, 5, 5), meta(1, 3, 6, 6), meta(2, 3, 2, 2),
        [1], [0], [1], False, [0], 1, [False, False, False])
    assert not gi.defined() and not gw.defined() and not gb.defined()
    # transposed backward answers the weight geometry without the forward
    # geometry cross-check; the bias gradient spans the transposed output's
    # channels, which are the backward grad_output's
    gi, gw, gb = C.convolution_backward_overrideable(
        meta(1, 3, 7), meta(1, 2, 6), meta(2, 3, 2),
        [1], [0], [1], True, [0], 1, [True, True, True])
    assert tuple(gw.shape) == (2, 3, 2)
    assert tuple(gb.shape) == (3,)


# ---------------------------------------------------------------------------
# Pooling backwards
# ---------------------------------------------------------------------------


def test_avg_pool3d_backward_metadata():
    C = tp._C

    def meta(*shape):
        return tp.zeros(list(shape), dtype=tp.float32, device="meta")

    g = C.avg_pool3d_backward(meta(1, 2, 3, 3, 3), meta(1, 2, 6, 6, 6),
                              [2], [2], [0], False, True, None)
    assert tuple(g.shape) == (1, 2, 6, 6, 6)
    assert tuple(C.avg_pool3d_backward(meta(1, 2, 3, 3, 3), meta(1, 2, 6, 6, 6),
                                       [2], [1], [0], False, True, None).shape) == (1, 2, 6, 6, 6)
    assert tuple(C.avg_pool3d_backward(meta(1, 2, 3, 3, 3), meta(1, 2, 7, 7, 7),
                                       [2], [2], [0], True, True, None).shape) == (1, 2, 7, 7, 7)
    assert tuple(C.avg_pool3d_backward(meta(1, 2, 3, 3, 3), meta(1, 2, 6, 6, 6),
                                       [2], [2], [1], False, False, None).shape) == (1, 2, 6, 6, 6)
    assert tuple(C.avg_pool3d_backward(meta(1, 2, 3, 3, 3), meta(1, 2, 6, 6, 6),
                                       [2], [2], [0], False, True, 4).shape) == (1, 2, 6, 6, 6)
    # 4D inputs enter the batched path and answer the same geometry
    assert tuple(C.avg_pool3d_backward(meta(2, 3, 3, 3), meta(2, 4, 4, 4),
                                       [2], [2], [0], False, True, None).shape) == (2, 4, 4, 4)
    with pytest.raises(Exception):
        C.avg_pool3d_backward(meta(1, 2, 3, 3, 3), meta(1, 2, 6, 6, 6),
                              [2], [2], [0], False, True, 0)


def test_adaptive_pooling_backwards_metadata():
    C = tp._C

    def meta(*shape):
        return tp.zeros(list(shape), dtype=tp.float32, device="meta")

    g = C._adaptive_avg_pool3d_backward(meta(1, 2, 3, 3, 3), meta(1, 2, 6, 6, 6))
    assert tuple(g.shape) == (1, 2, 6, 6, 6)
    assert tuple(C._adaptive_avg_pool3d_backward(meta(2, 3, 3, 3),
                                                 meta(2, 4, 4, 4)).shape) == (2, 4, 4, 4)

    assert tuple(C.adaptive_max_pool2d_backward(meta(1, 2, 3, 3),
                                                meta(1, 2, 6, 6)).shape) == (1, 2, 6, 6)
    assert tuple(C.adaptive_max_pool2d_backward(meta(2, 3, 3),
                                                meta(2, 6, 6)).shape) == (2, 6, 6)
    assert tuple(C.adaptive_max_pool2d_backward(meta(2, 3, 3, 3),
                                                meta(2, 6, 6, 6)).shape) == (2, 6, 6, 6)

    g = C.adaptive_max_pool3d_backward(meta(1, 2, 3, 3, 3), meta(1, 2, 6, 6, 6),
                                       tp.zeros([1, 2, 3, 3, 3], dtype=tp.int64,
                                                device="meta"))
    assert tuple(g.shape) == (1, 2, 6, 6, 6)


def test_max_pool_with_indices_backwards_metadata():
    C = tp._C

    def meta(*shape, dt=tp.float32):
        return tp.zeros(list(shape), dtype=dt, device="meta")

    idx2d = meta(1, 2, 3, 3, dt=tp.int64)
    g = C.max_pool2d_with_indices_backward(meta(1, 2, 3, 3), meta(1, 2, 6, 6),
                                           [2], [2], [0], [1], False, idx2d)
    assert tuple(g.shape) == (1, 2, 6, 6)
    # a 3D input enters the batched path and answers the same geometry
    g = C.max_pool2d_with_indices_backward(meta(2, 3, 3), meta(2, 6, 6),
                                           [2], [2], [0], [1], False,
                                           meta(2, 3, 3, dt=tp.int64))
    assert tuple(g.shape) == (2, 6, 6)
    with pytest.raises(Exception):
        C.max_pool2d_with_indices_backward(meta(1, 2, 3, 3), meta(1, 2, 6, 6),
                                           [2], [2], [0], [1], False, None)
    with pytest.raises(Exception):
        C.max_pool2d_with_indices_backward(meta(1, 2, 3, 3), meta(1, 2, 6, 6),
                                           [2], [2], [0], [1], False,
                                           meta(1, 2, 4, 4, dt=tp.int64))

    g = C.max_pool3d_with_indices_backward(meta(1, 2, 3, 3, 3),
                                           meta(1, 2, 6, 6, 6),
                                           [2], [2], [0], [1], False,
                                           meta(1, 2, 3, 3, 3, dt=tp.int64))
    assert tuple(g.shape) == (1, 2, 6, 6, 6)


# ---------------------------------------------------------------------------
# Embedding backwards
# ---------------------------------------------------------------------------


def test_embedding_dense_backward_metadata():
    C = tp._C

    def meta(*shape, dt=tp.float32):
        return tp.zeros(list(shape), dtype=dt, device="meta")

    def meta_i(*shape):
        return tp.zeros(list(shape), dtype=tp.int64, device="meta")

    g = C.embedding_dense_backward(meta(6, 5), meta_i(8), 6, -1, False)
    assert tuple(g.shape) == (6, 5)
    g = C.embedding_dense_backward(meta(6, 5, 4), meta_i(2, 3), 6, -1, False)
    assert tuple(g.shape) == (6, 4)
    g = C.embedding_dense_backward(meta(6, 5, dt=tp.float16), meta_i(8), 6, -1, False)
    assert g.dtype == tp.float16
    g = C.embedding_dense_backward(meta(6, 5), tp.zeros([0], dtype=tp.int64,
                                                        device="meta"), 6, -1, False)
    assert tuple(g.shape) == (6, 5)
    with pytest.raises(Exception):
        C.embedding_dense_backward(meta(6, 5), meta(8, dtype=tp.float32), 6, -1, False)
    with pytest.raises(Exception):
        C.embedding_dense_backward(meta(6, 5, dtype=tp.bool), meta_i(8), 6, -1, False)


def test_embedding_bag_backwards_metadata():
    C = tp._C

    def meta(*shape, dt=tp.float32):
        return tp.zeros(list(shape), dtype=dt, device="meta")

    def meta_i(*shape):
        return tp.zeros(list(shape), dtype=tp.int64, device="meta")

    # modes: 0 sum, 1 mean, 2 max
    for mode in (0, 1):
        g = C._embedding_bag_dense_backward(meta(2, 5), meta_i(8), meta(8),
                                            meta_i(2), meta_i(2), 6, False,
                                            mode, None, -1)
        assert tuple(g.shape) == (6, 5)
    g = C._embedding_bag_dense_backward(meta(2, 5), meta_i(8), meta(8),
                                        meta_i(2), meta_i(2, 5), 6, False,
                                        2, None, -1)
    assert tuple(g.shape) == (6, 5)
    with pytest.raises(Exception):
        C._embedding_bag_dense_backward(meta(2, 5, 4), meta_i(8), meta(8),
                                        meta_i(2), meta_i(2), 6, False, 0, None, -1)
    with pytest.raises(Exception):
        C._embedding_bag_dense_backward(meta(2, 5), meta_i(8), meta(8),
                                        meta_i(2), meta_i(2), -1, False, 0, None, -1)
    with pytest.raises(Exception):
        C._embedding_bag_dense_backward(meta(2, 5), meta_i(8), meta(4),
                                        meta_i(2), meta_i(2), 6, False, 0, None, -1)
    # per-sample weights ride along with the sum reduction
    g = C._embedding_bag_dense_backward(meta(2, 5), meta_i(8), meta(8),
                                        meta_i(2), meta_i(2), 6, False, 0,
                                        meta(8), -1)
    assert tuple(g.shape) == (6, 5)

    g = C._embedding_bag_per_sample_weights_backward(
        meta(2, 5), meta(6, 5), meta_i(8), meta_i(2), meta(8), 0, -1)
    assert tuple(g.shape) == (8,)
    with pytest.raises(Exception):
        C._embedding_bag_per_sample_weights_backward(
            meta(2, 5), meta(6, 5), meta_i(8), meta_i(2), meta(8), 2, -1)


# ---------------------------------------------------------------------------
# Segment reduce and grid sampler backwards
# ---------------------------------------------------------------------------


def test_segment_reduce_backward_metadata():
    C = tp._C

    def meta(*shape):
        return tp.zeros(list(shape), dtype=tp.float32, device="meta")

    lengths = tp.zeros([2], dtype=tp.int64, device="meta")
    for reduce in ("sum", "mean"):
        g = C._segment_reduce_backward(meta(2, 5), meta(2, 5), meta(4, 5),
                                       reduce, lengths=lengths)
        assert tuple(g.shape) == (4, 5)
    with pytest.raises(Exception):
        C._segment_reduce_backward(meta(2, 5), meta(2, 5), meta(4, 5),
                                   "bogus", lengths=lengths)
    with pytest.raises(Exception):
        C._segment_reduce_backward(meta(2, 5), meta(2, 5), meta(4, 5), "sum")
    with pytest.raises(Exception):
        C._segment_reduce_backward(meta(2, 5), meta(2, 5), meta(4, 5),
                                   "sum", lengths=lengths, axis=2)


def test_grid_sampler_3d_backward_metadata():
    C = tp._C

    def meta(*shape, dt=tp.float32):
        return tp.zeros(list(shape), dtype=dt, device="meta")

    gi, gg = C.grid_sampler_3d_backward(meta(1, 2, 3, 3, 3), meta(1, 2, 6, 6, 6),
                                        meta(1, 3, 3, 3, 3), 0, 0, True,
                                        [True, True])
    assert tuple(gi.shape) == (1, 2, 6, 6, 6)
    assert tuple(gg.shape) == (1, 3, 3, 3, 3)
    assert gg.dtype == tp.float32
    # the output mask only steers the buffer fill on computing backends
    gi, _ = C.grid_sampler_3d_backward(meta(1, 2, 3, 3, 3), meta(1, 2, 6, 6, 6),
                                       meta(1, 3, 3, 3, 3), 0, 0, True,
                                       [True, False])
    assert tuple(gi.shape) == (1, 2, 6, 6, 6)
    with pytest.raises(Exception):
        C.grid_sampler_3d_backward(meta(1, 2, 3, 3, 3), meta(1, 2, 6, 6, 6),
                                   meta(1, 3, 3, 3, 3), 2, 0, True, [True, True])
    with pytest.raises(Exception):
        C.grid_sampler_3d_backward(meta(1, 2, 6, 6, 6, dt=tp.int32),
                                   meta(1, 2, 6, 6, 6, dt=tp.int32),
                                   meta(1, 3, 3, 3, 3, dt=tp.int32), 0, 0, True,
                                   [True, True])


# ---------------------------------------------------------------------------
# Sparse COO factories
# ---------------------------------------------------------------------------


def test_sparse_coo_tensor_on_meta():
    C = tp._C

    def meta_i(*shape):
        return tp.zeros(list(shape), dtype=tp.int64, device="meta")

    def meta_f(*shape, dt=tp.float32):
        return tp.zeros(list(shape), dtype=dt, device="meta")

    sp = C.sparse_coo_tensor(meta_i(1, 4), meta_f(4, 3), [2, 3])
    assert sp.is_sparse
    assert sp.is_meta
    assert tuple(sp.shape) == (2, 3)
    assert tuple(sp._indices().shape) == (1, 4)
    assert tuple(sp._values().shape) == (4, 3)
    assert sp.sparse_dim() == 1
    sp = C.sparse_coo_tensor(meta_i(1, 4), meta_f(4, 3, dt=tp.float64), [2, 3])
    assert sp.dtype == tp.float64
    sp = C.sparse_coo_tensor(meta_i(1, 4), meta_i(4, 3), [2, 3])
    assert sp.dtype == tp.int64
    with pytest.raises(Exception):
        C.sparse_coo_tensor(meta_i(1, 4), meta_f(4, 2), [2, 3])
    with pytest.raises(Exception):
        C.sparse_coo_tensor(meta_i(1, 4), meta_f(3, 3), [2, 3])
    # a hybrid COO (2 sparse dims plus one dense trailing extent) is valid
    sp = C.sparse_coo_tensor(meta_i(2, 4), meta_f(4, 3), [2, 3, 3])
    assert tuple(sp.shape) == (2, 3, 3)
    assert sp.sparse_dim() == 2


def test_sparse_sum_on_meta():
    C = tp._C

    def sp(n):
        return C.sparse_coo_tensor(tp.zeros([1, n], dtype=tp.int64, device="meta"),
                                   tp.zeros([n, 3], dtype=tp.float32, device="meta"),
                                   [2, 3])

    out = C._sparse_sum(sp(4))
    assert tuple(out.shape) == ()
    assert out.dtype == tp.float32
    out = C._sparse_sum(C.sparse_coo_tensor(
        tp.zeros([1, 4], dtype=tp.int64, device="meta"),
        tp.zeros([4, 3], dtype=tp.float64, device="meta"), [2, 3]))
    assert out.dtype == tp.float64
    # a dense operand reaches no kernel at all, matching the dense-key
    # outcome on computing backends
    with pytest.raises(NotImplementedError):
        C._sparse_sum(tp.zeros([2, 3], dtype=tp.float32, device="meta"))


def test_sparse_addmm_on_meta():
    C = tp._C

    def meta(*shape, dt=tp.float32):
        return tp.zeros(list(shape), dtype=dt, device="meta")

    # a fully sparse 3x4 matrix: two sparse dims, 1-D values plane
    def sp():
        return C.sparse_coo_tensor(tp.zeros([2, 4], dtype=tp.int64, device="meta"),
                                   meta(4), [3, 4])

    out = C._sparse_addmm(meta(3, 5), sp(), meta(4, 5))
    assert tuple(out.shape) == (3, 5)
    assert out.dtype == tp.float32
    out = C._sparse_addmm(meta(3, 5), sp(), meta(4, 5), beta=0, alpha=1)
    assert tuple(out.shape) == (3, 5)
    # a sparse self keeps the dense-mat1 gate; a dense self answers the product
    with pytest.raises(RuntimeError):
        C._sparse_addmm(sp(), meta(2, 3), meta(3, 3))
    with pytest.raises(Exception):
        C._sparse_addmm(meta(3, 5), sp(), meta(5, 6))


# ---------------------------------------------------------------------------
# Foreach copy and mm
# ---------------------------------------------------------------------------


def test_foreach_copy_and_mm_on_meta():
    C = tp._C

    def meta(*shape, dt=tp.float32):
        return tp.zeros(list(shape), dtype=dt, device="meta")

    outs = C._foreach_copy([meta(2, 3)], [meta(2, 3)], False)
    assert tuple(outs[0].shape) == (2, 3)
    outs = C._foreach_copy([meta(2, 3), meta(4)], [meta(2, 3), meta(4)], False)
    assert [tuple(o.shape) for o in outs] == [(2, 3), (4,)]
    # the copy answers the source's metadata: dtype comes from the source
    outs = C._foreach_copy([meta(2, 3)], [meta(2, 3, dt=tp.float64)], False)
    assert outs[0].dtype == tp.float64
    outs = C._foreach_copy([meta(2, 3, dt=tp.float16)], [meta(2, 3)], False)
    assert outs[0].dtype == tp.float32
    with pytest.raises(ValueError):
        C._foreach_copy([meta(2, 3), meta(4)], [meta(2, 3)], False)

    outs = C._foreach_mm([meta(2, 3)], [meta(3, 4)])
    assert tuple(outs[0].shape) == (2, 4)
    outs = C._foreach_mm([meta(2, 3), meta(4, 5)], [meta(3, 4), meta(5, 6)])
    assert [tuple(o.shape) for o in outs] == [(2, 4), (4, 6)]
    with pytest.raises(Exception):
        C._foreach_mm([meta(2, 3)], [meta(4, 5)])
