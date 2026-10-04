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
    if isinstance(want, dict):
        assert set(got) == set(want)
        for key in want:
            _same(got[key], want[key], tol)
        return
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "shape,out",
    [
        ((20, 1), (7, 1)),  # rows uneven, one column: the one-axis pool
        ((20, 14), (7, 7)),  # rows uneven, columns even, as many of each
        ((14, 20), (7, 7)),  # columns uneven, rows even
        ((20, 20), (7, 7)),  # both uneven
        ((10, 9), (3, 4)),
    ],
)
def test_adaptive_average_windows_line_up_with_their_axis(device, shape, out):
    # A window grid whose windows differ in length along one axis divides
    # each window by its own length along that axis and no other.
    tp.manual_seed(0)
    x = tp.randn(2, 3, *shape, device=device)
    _check(lambda x: F.adaptive_avg_pool2d(x, out), x)
    if shape[1] == 1:
        y = x.squeeze(3)
        _check(lambda y: F.adaptive_avg_pool1d(y, out[0]), y)


@pytest.mark.parametrize("device", DEVICES)
def test_control_flow_runs_its_pieces(device):
    # Branches and loops are captured as one operator holding their pieces,
    # recorded whole by the region's trace, and lowered as pieces of their
    # own: the branch taken, the number of passes and the per-step results are
    # the program's, on fresh values too.
    from tensorplay import cond, while_loop
    from tensorplay._higher_order_ops import map as map_steps, scan

    tp.manual_seed(0)
    x = tp.randn(4, 3, device=device).abs()
    _check(
        lambda x: cond(x.sum() > 0, lambda v: v.sin(), lambda v: v.cos(), (x,)) * 2,
        x, second=(-x,),
    )
    _check(lambda x: map_steps(lambda r: r.exp() + 1, x), x, second=(x * 2,))
    _check(
        lambda x: while_loop(
            lambda i, y: i < 3, lambda i, y: (i + 1, y * 1.5),
            (tp.tensor(0, device=x.device), x),
        )[1],
        x, second=(x * 2,),
    )
    _check(
        lambda x: scan(lambda c, v: (c + v, c * v), tp.zeros(3, device=x.device), x)[1],
        x, second=(x * 2,),
    )
    # A carry of several tensors in a structure comes back in that structure.
    _check(
        lambda x: scan(
            lambda c, v: ({"a": c["a"] + v, "b": c["b"] * 0.5}, c["a"] * v),
            {"a": tp.zeros(3, device=x.device), "b": tp.ones(3, device=x.device)},
            x,
        ),
        x, second=(x * 2,),
    )
    # A loop body that reads a tensor from its enclosing scope and calls an
    # operation the region does not write itself.
    w = tp.randn(3, 3, device=device) * 0.3
    _check(
        lambda x: while_loop(
            lambda i, y: i < 4, lambda i, y: (i + 1, (y @ w).tanh() + y),
            (tp.tensor(0, device=x.device), x),
        ),
        x, second=(x * 2,),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_control_flow_differentiates_what_its_pieces_read(device):
    # A gradient through a branch, a map or a scan reaches the tensors the
    # pieces close over as well as the ones they are handed, and a branch
    # chosen differently on a later call is differentiated as that branch.
    from tensorplay import cond
    from tensorplay._higher_order_ops import map as map_steps, scan

    tp.manual_seed(0)
    w = tp.randn(3, 3, device=device, requires_grad=True)
    cases = {
        "cond": lambda x: cond(
            x.sum() > 0, lambda v: (v @ w).sin(), lambda v: v.cos(), (x,)
        ).sum(),
        "map": lambda x: map_steps(lambda r: (r @ w).exp(), x).sum(),
        "scan": lambda x: scan(
            lambda c, v: ((c @ w).tanh() + v, c * v), tp.zeros(3, device=x.device), x
        )[1].sum(),
    }
    for name, fn in cases.items():
        compiled = tp.compile(fn, strict_native=True)
        for sign in (1.0, -1.0):
            x = (sign * tp.randn(4, 3, device=device).abs()).requires_grad_()
            want = tp.autograd.grad(fn(x), (x, w), allow_unused=True)
            got = tp.autograd.grad(compiled(x), (x, w), allow_unused=True)
            for g, e in zip(got, want):
                e = tp.zeros_like(w) if e is None else e
                assert g is not None, name
                _same(g, e, 1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("op", ["quantile", "nanquantile"])
def test_quantile_differentiated_through_its_parts(device, op):
    # An operation without a derivative of its own is differentiated through
    # the operations it is made of, compiled as much as eager.
    tp.manual_seed(0)
    reduce = getattr(tp, op)
    q1 = tp.tensor([0.25, 0.5, 0.9], device=device)
    q0 = tp.tensor(0.3, device=device)
    cases = [
        (q1, None, False, "linear"),
        (q1, 1, True, "lower"),
        (q0, 0, False, "higher"),
        (q1, -1, False, "midpoint"),
        (q0, None, True, "nearest"),
    ]
    for q, dim, keepdim, interpolation in cases:
        fn = lambda a: (
            reduce(a, q, dim=dim, keepdim=keepdim, interpolation=interpolation) ** 2
        ).sum()
        compiled = tp.compile(fn, strict_native=True)
        for _ in range(2):
            x = tp.randn(4, 7, device=device).requires_grad_()
            (want,) = tp.autograd.grad(fn(x), [x])
            got_value = compiled(x)
            _same(got_value, fn(x), 1e-5)
            (got,) = tp.autograd.grad(got_value, [x])
            _same(got, want, 1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_detached_value_follows_its_input(device):
    # A value detached from one that requires grad is still computed from it:
    # a later call reads that call's input, not the trace's.
    fn = lambda x: (x.detach() * 3 + x).sum()
    compiled = tp.compile(fn, strict_native=True)
    for _ in range(3):
        x = tp.randn(5, device=device, requires_grad=True)
        got = compiled(x)
        _same(got, fn(x), 1e-5)
        (grad,) = tp.autograd.grad(got, [x])
        _same(grad, tp.ones_like(x), 1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_comparisons_answer_for_each_call(device):
    # A comparison answering with one truth value is computed from the call's
    # operands, not from the ones it was traced with.
    fn = lambda a, b: (tp.allclose(a, b), tp.equal(a, b), tp.equal(a, b[:2]))
    compiled = tp.compile(fn, strict_native=True)
    a = tp.randn(4, 5, device=device)
    for b in (a.clone(), a + 1e-3, a.clone()):
        assert [bool(v) for v in compiled(a, b)] == list(fn(a, b))
