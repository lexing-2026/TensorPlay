"""Regions are lowered from the operators they run, and come out as eager does.

Every region -- one with nothing to differentiate as much as a training one --
is traced to operator overloads before it is lowered, so a function's name
never stands in for the overload it reached.  Each case is compiled strictly
(a region that cannot be built fails here instead of running op by op) and
compared with the eager result, a second call on fresh values included where a
value could have been baked in at trace time.
"""

import pytest

import tensorplay as tp
import tensorplay.nn.functional as F

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def _check(fn, *inputs, tol=1e-5, second=None):
    want = fn(*[x.clone() if isinstance(x, tp.Tensor) else x for x in inputs])
    compiled = tp.compile(fn, strict_native=True)
    got = compiled(*[x.clone() if isinstance(x, tp.Tensor) else x for x in inputs])
    _same(got, want, tol)
    if second is not None:
        _same(compiled(*second), fn(*second), tol)
    return got


def _same(got, want, tol):
    if isinstance(want, (tuple, list)):
        assert len(got) == len(want)
        for g, w in zip(got, want):
            _same(g, w, tol)
        return
    assert tuple(got.shape) == tuple(want.shape)
    assert got.dtype == want.dtype
    if want.numel():
        err = (got.double() - want.double()).abs().max().item()
        assert err <= tol * max(1.0, want.double().abs().max().item())


@pytest.mark.parametrize("device", DEVICES)
def test_overloads_not_names(device):
    tp.manual_seed(0)
    x = tp.randn(37, 129, device=device)
    # A count of truths is a whole number, accumulated as one.
    _check(lambda x: (x > 0).sum(1), x)
    # ``max`` over an axis is not ``max`` of everything.
    _check(lambda x: tp.max(x, 1)[0] + tp.max(x, 1)[1], x)
    _check(lambda x: (x.sum(1), x.max(1)[0], x.softmax(1)), x)


@pytest.mark.parametrize("device", DEVICES)
def test_reshape_of_a_padded_buffer(device):
    # Rows longer than the padding threshold are padded apart when the buffer
    # is laid down, and a reshape reads them through its index arithmetic.
    tp.manual_seed(0)
    y = tp.randn(4, 16, 9, 9, device=device)
    _check(lambda y: (y.reshape(4, 4, 4, 81) * 2).reshape(4, 16, 9, 9), y)
    w, b = tp.randn(16, device=device), tp.randn(16, device=device)
    _check(lambda y, w, b: F.group_norm(y, 4, w, b), y, w, b, tol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_writes_are_kept_and_ordered(device):
    tp.manual_seed(0)
    x, y = tp.randn(37, 129, device=device), tp.randn(37, 129, device=device)
    # The returned value is the sum, not the sum read back after the input
    # was overwritten with it; the input is overwritten too.
    x_eager, x_comp = x.clone(), x.clone()
    want = x_eager.add_(y)
    got = tp.compile(lambda a, b: a.add_(b), strict_native=True)(x_comp, y)
    _same(got, want, 0)
    _same(x_comp, x_eager, 0)

    def assign(x, v):
        out = x.clone()
        out[:, 2:7] = v
        out[1] += v[1, 0]
        out[:, 0].mul_(3)
        return out

    _check(assign, x, tp.randn(37, 5, device=device))
    # Mutating operators with no functional form are written as copies and
    # the operator writing into them.
    _check(lambda x: x.clone().zero_() + 1, x)
    _check(lambda x: x[:9, :9].clone().fill_diagonal_(0.0), x)


@pytest.mark.parametrize("device", DEVICES)
def test_outputs_do_not_share_memory_the_program_did_not(device):
    x = tp.randn(4, 9, device=device)
    for fn in (lambda x: x.clone(), lambda x: x + 0, lambda x: (x * 2, (x * 2).clone())):
        got = tp.compile(fn)(x)
        outs = got if isinstance(got, tuple) else (got,)
        ptrs = [o.data_ptr() for o in outs]
        assert x.data_ptr() not in ptrs
        assert len(set(ptrs)) == len(ptrs)
    # A view the program returns is still the input's memory.
    assert tp.compile(lambda x: x.view(36))(x).data_ptr() == x.data_ptr()


@pytest.mark.parametrize("device", DEVICES)
def test_dtype_views_and_scalar_promotion(device):
    tp.manual_seed(0)
    x = tp.randn(17, 33, device=device)
    fresh = (tp.randn(17, 33, device=device),)
    # Reinterpreting the bits is traced, not baked in as a constant.
    _check(lambda x: x.view(tp.int32) & 0xFF, x, second=fresh)
    # A number contributes its kind, never its width.
    _check(lambda x: (x * 100).to(tp.int32) & 255, x)
    _check(lambda x: tp.fmod((x * 100).to(tp.int32), 2.5), x)
    _check(lambda x: tp.fmod((x * 100).to(tp.int32), 3), x)
    # Bounds are values of the clamped type, not numbers spliced in as written.
    _check(lambda x: x.clamp(min=-0.5, max=0.5) + (x * 10).long().clamp(-2, 2) * 1.5, x)


@pytest.mark.parametrize("device", DEVICES)
def test_factories(device):
    x = tp.randn(17, 33, device=device)
    _check(lambda x: x + tp.arange(0, 3.3, 0.1, device=x.device)[:33], x, tol=1e-6)
    _check(lambda x: x[:, :10] + tp.arange(1.5, 11.5, device=x.device), x)
    _check(lambda x: x @ tp.eye(33, device=x.device), x, tol=1e-6)
    _check(lambda x: x[:, :5] @ tp.eye(5, 7, device=x.device), x, tol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_calls_handed_over(device):
    tp.manual_seed(0)
    x = tp.randn(37, 129, device=device)
    # Several results of one framework call, each taken out of what it returns.
    _check(lambda x: tp.topk(x, 5, 1), x)
    _check(lambda x: tp.kthvalue(x, 5, 1)[0] + tp.median(x, 1)[0], x)
    w, i = tp.randn(50, 16, device=device), tp.randint(0, 50, (20,), device=device)
    offsets = tp.tensor([0, 5, 12], device=device)
    _check(lambda w, i, o: F.embedding_bag(i, w, o, mode="mean"), w, i, offsets)
    # A mask selects as many as it holds truths: the framework counts them.
    _check(lambda x: x[x > 0], x, second=(tp.randn(37, 129, device=device),))
    # Complex values are the framework's; the real parts around them are not.
    re, im = tp.randn(17, 33, device=device), tp.randn(17, 33, device=device)
    _check(lambda a, b: tp.view_as_real(tp.complex(a, b) * 2) + 1, re, im)
    _check(lambda a: tp.fft.rfft(a).abs() * 2, re, tol=1e-4)
    # A view by explicit strides, on an input that starts inside its storage.
    flat = tp.randn(100, device=device)
    _check(lambda x: x.as_strided((5, 5), (3, 1), 2) * 2, flat[10:])


@pytest.mark.parametrize("device", DEVICES)
def test_batched_products_read_their_operands_views(device):
    tp.manual_seed(0)
    x, w = tp.randn(2, 8, 9, 3, device=device), tp.randn(2, 2, 3, device=device)
    _check(lambda x, w: tp.bmm(x.reshape(2, 72, 3), w.transpose(1, 2)), x, w, tol=1e-5)
    _check(lambda x, w: tp.bmm((x * 2).reshape(2, 72, 3), w.transpose(1, 2)), x, w, tol=1e-5)
    b = tp.randn(4, 8, 16, device=device)
    p, q = tp.randn(4, 32, 8, device=device), tp.randn(4, 32, 16, device=device)
    _check(lambda b, p, q: tp.baddbmm(b, p.transpose(1, 2), q, beta=0.5, alpha=2), b, p, q, tol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_scans_sorts_and_cpu_helpers(device):
    tp.manual_seed(0)
    x = tp.randn(37, 129, device=device)
    _check(lambda x: tp.cumsum(x, 1), x, tol=1e-5)
    _check(lambda x: tp.sort(x, 1)[0], x)
    _check(lambda x: tp.sort(x, 1, descending=True)[1], x)
    # Reductions and whole-number arithmetic whose helpers the generated CPU
    # code calls by name.
    _check(lambda x: tp.softmax(x, -1), x, tol=1e-5)
    _check(lambda x: (tp.argmax(x, 1), tp.argmin(x, 0), x.amax(1)), x)
    i = tp.randint(0, 100, (17, 33), device=device)
    _check(lambda i: tp.remainder(i - 50, (i % 7) + 1), i)
    _check(lambda i: tp.div(i - 50, (i % 7) + 1, rounding_mode="floor"), i)


@pytest.mark.parametrize("device", DEVICES)
def test_python_functions_seen_whole_by_capture(device):
    tp.manual_seed(0)
    x, y = tp.randn(17, 33, device=device), tp.randn(17, 33, device=device)
    # Functions that compare shapes during capture.
    _check(lambda x, y: F.smooth_l1_loss(x, y, beta=0.5), x, y)
    _check(lambda x, y: F.binary_cross_entropy_with_logits(x, y.sigmoid()), x, y)
    lp = tp.randn(20, 2, 6, device=device)
    tgt = tp.randint(1, 6, (2, 5), device=device)
    _check(
        lambda lp, t: F.ctc_loss(
            lp.log_softmax(-1), t, tp.tensor([20, 20], device=lp.device),
            tp.tensor([5, 4], device=lp.device)),
        lp, tgt, tol=1e-4,
    )
