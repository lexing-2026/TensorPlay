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
