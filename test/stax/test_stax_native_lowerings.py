"""Operators the compiler writes as its own loops, checked against eager.

Elementwise activations and their gradients, row and column statistics
(layer norm, softmax, batch norm) and windows named by indexing are lowered
into generated code rather than called out to.  Each case runs the same
function compiled and eagerly and compares values and gradients; a counter
on the call-out path checks that the region was really generated.
"""
import collections

import pytest

import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay.compiler.backends.stax.graph_lowering import GraphLowering

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])
needs_cuda = pytest.mark.skipif(not tp.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def call_outs(monkeypatch):
    """The operators the compiled regions called out to, by name."""

    seen = collections.Counter()
    original = GraphLowering.make_extern

    def counting(self, node, args, kwargs):
        seen[str(node.target)] += 1
        return original(self, node, args, kwargs)

    monkeypatch.setattr(GraphLowering, "make_extern", counting)
    return seen


def _check(fn, inputs, *, train, atol=1e-5, rtol=1e-4):
    """Runs ``fn`` compiled and eagerly on copies of ``inputs``; compares the
    results and, when ``train``, the gradients of their squared sum."""

    mine = [t.detach().clone().requires_grad_(train and t.is_floating_point()) for t in inputs]
    ref = [t.detach().clone().requires_grad_(train and t.is_floating_point()) for t in inputs]
    out = tp.compile(fn)(*mine)
    expected = fn(*ref)
    assert out.dtype == expected.dtype
    assert tp.allclose(out.float(), expected.float(), atol=atol, rtol=rtol, equal_nan=True), (
        (out.float() - expected.float()).abs().max().item())
    if train:
        out.float().pow(2).sum().backward()
        expected.float().pow(2).sum().backward()
        for a, b in zip(mine, ref):
            if not b.requires_grad:
                continue
            assert (a.grad is None) == (b.grad is None)
            if b.grad is not None:
                assert tp.allclose(a.grad, b.grad, atol=atol * 10, rtol=rtol * 10, equal_nan=True), (
                    (a.grad - b.grad).abs().max().item())


ELEMENTWISE = {
    "erf": lambda a, b: tp.erf(a),
    "expm1": lambda a, b: tp.expm1(a * 0.1),
    "log1p": lambda a, b: tp.log1p(a.abs()),
    "sinh": lambda a, b: tp.sinh(a),
    "atanh": lambda a, b: tp.atanh(a.tanh() * 0.9),
    "frac": lambda a, b: tp.frac(a),
    "square": lambda a, b: tp.square(a),
    "atan2": lambda a, b: tp.atan2(a, b),
    "hypot": lambda a, b: tp.hypot(a, b),
    "clamp_min": lambda a, b: tp.clamp_min(a, 0.3),
    "clamp_max": lambda a, b: tp.clamp_max(a, b),
    "lerp": lambda a, b: tp.lerp(a, b, b.sigmoid()),
    "gelu": lambda a, b: F.gelu(a),
    "gelu_tanh": lambda a, b: F.gelu(a, approximate="tanh"),
    "softplus": lambda a, b: F.softplus(a, beta=2.0, threshold=3.0),
    "elu": lambda a, b: F.elu(a, 0.7),
    "leaky_relu": lambda a, b: F.leaky_relu(a, 0.2),
    "hardtanh": lambda a, b: F.hardtanh(a, -0.5, 2.0),
    "hardsigmoid": lambda a, b: F.hardsigmoid(a),
    "hardswish": lambda a, b: F.hardswish(a),
    "threshold": lambda a, b: F.threshold(a, 0.5, -2.0),
    "tanh": lambda a, b: tp.tanh(a) * b,
    "sigmoid": lambda a, b: tp.sigmoid(a) * b,
    "nan_to_num": lambda a, b: tp.nan_to_num(a.log()),
    "isfinite": lambda a, b: tp.where(tp.isfinite(a.log()), a, b),
    "signbit": lambda a, b: tp.where(tp.signbit(a), a, b),
    "fmod": lambda a, b: tp.fmod(a, 1.7),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(ELEMENTWISE))
def test_elementwise_values_and_gradients(device, name, call_outs):
    tp.manual_seed(0)
    a = tp.randn(16, 33, device=device) * 3
    b = tp.randn(16, 33, device=device)
    _check(ELEMENTWISE[name], [a, b], train=True)
    assert not call_outs, dict(call_outs)


@pytest.mark.parametrize("device", DEVICES)
def test_comparisons_broadcast_their_operands(device, call_outs):
    # The comparison is the region's result: each operand is read at its own
    # positions of the broadcast shape, the classes along the last axis.
    labels = tp.tensor([[4, 0, 8], [1, 1, 7]], device=device)
    out = tp.compile(lambda: labels.unsqueeze(-1).eq(tp.arange(9, device=device)))()
    expected = [[[c == k for k in range(9)] for c in row] for row in labels.tolist()]
    assert out.tolist() == expected
    assert tp.equal(tp.compile(lambda: F.one_hot(labels, 9))(), F.one_hot(labels, 9))
    assert tp.equal(tp.compile(lambda: 5 & labels)(), labels & 5)
    assert not call_outs, dict(call_outs)


@pytest.mark.parametrize("device", DEVICES)
def test_constants_shaped_like_a_value(device, call_outs):
    x = tp.tensor([[1.0, -2.0, 3.0], [0.5, 0.0, -1.0]], device=device)
    fn = lambda x: (
        tp.full_like(x, 3.0) * x + tp.zeros_like(x) + tp.ones_like(x, dtype=tp.float64).float()
        + tp.eye(2, 3, device=device) + tp.where(x > 0, 1.0, 0.0)
    )
    out = tp.compile(fn)(x)
    expected = [[3 * v + 1 + (r == c) + (v > 0) for c, v in enumerate(row)] for r, row in enumerate(x.tolist())]
    assert out.tolist() == expected
    assert not call_outs, dict(call_outs)


@pytest.mark.parametrize("device", DEVICES)
def test_integer_inputs_keep_their_answers(device):
    i = tp.arange(-20, 20, device=device).reshape(5, 8)
    for fn in (tp.floor, tp.bitwise_not, tp.isnan, lambda v: tp.erf(v)):
        out = tp.compile(fn)(i)
        expected = fn(i)
        assert out.dtype == expected.dtype
        assert tp.allclose(out.double(), expected.double(), atol=1e-6)


NORMS = {
    "layer_norm": (lambda x, w, b: F.layer_norm(x, (32,), w, b), (4, 7, 32)),
    "layer_norm_two_axes": (
        lambda x, w, b: F.layer_norm(x, (7, 32), w[None].expand(7, 32), b[None].expand(7, 32)), (3, 7, 32)),
    "softmax": (lambda x, w, b: tp.softmax(x * w, -1) + b, (6, 9, 32)),
    "softmax_inner_axis": (lambda x, w, b: tp.softmax(x, 1) * w + b, (6, 9, 32)),
    "log_softmax": (lambda x, w, b: F.log_softmax(x * w, -1) + b, (6, 9, 32)),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(NORMS))
def test_row_statistics_are_generated(device, name, call_outs):
    tp.manual_seed(0)
    fn, shape = NORMS[name]
    inputs = [tp.randn(*shape, device=device), tp.randn(32, device=device), tp.randn(32, device=device)]
    _check(fn, inputs, train=True, atol=1e-4)
    assert not call_outs, dict(call_outs)


@pytest.mark.parametrize("device", DEVICES)
def test_layer_norm_with_a_frozen_weight(device):
    # The backward is asked for the input's gradient alone: the outputs it
    # does not compute stand for nothing and are not read.
    tp.manual_seed(0)
    x = tp.randn(5, 16, device=device, requires_grad=True)
    w = tp.randn(16, device=device)
    out = tp.compile(lambda x: F.layer_norm(x, (16,), w))(x)
    out.pow(2).sum().backward()
    x2 = x.detach().clone().requires_grad_(True)
    F.layer_norm(x2, (16,), w).pow(2).sum().backward()
    assert tp.allclose(x.grad, x2.grad, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_batch_norm_reading_running_statistics(device, call_outs):
    tp.manual_seed(0)
    bn = tp.nn.BatchNorm2d(8).to(device)
    bn.running_mean.uniform_(-1, 1)
    bn.running_var.uniform_(0.5, 2)
    bn.eval()

    def fn(x):
        return F.adaptive_avg_pool2d(F.relu(bn(x)), 1).flatten(1)

    x = tp.randn(4, 8, 6, 6, device=device)
    with tp.no_grad():
        out = tp.compile(fn)(x)
        expected = fn(x)
    assert tp.allclose(out, expected, atol=1e-5)
    assert not call_outs, dict(call_outs)


@pytest.mark.parametrize("device", DEVICES)
def test_batch_norm_training_gradients(device):
    tp.manual_seed(0)
    mine = tp.nn.Sequential(tp.nn.Conv2d(3, 8, 3, padding=1), tp.nn.BatchNorm2d(8), tp.nn.ReLU()).to(device)
    ref = tp.nn.Sequential(tp.nn.Conv2d(3, 8, 3, padding=1), tp.nn.BatchNorm2d(8), tp.nn.ReLU()).to(device)
    ref.load_state_dict(mine.state_dict())
    x = tp.randn(4, 3, 10, 10, device=device)
    out = tp.compile(mine)(x)
    expected = ref(x)
    out.pow(2).mean().backward()
    expected.pow(2).mean().backward()
    assert tp.allclose(out, expected, atol=1e-4)
    for p, q in zip(mine.parameters(), ref.parameters()):
        assert tp.allclose(p.grad, q.grad, atol=1e-4, rtol=1e-4)
    assert tp.allclose(mine[1].running_mean, ref[1].running_mean, atol=1e-5)
    assert tp.allclose(mine[1].running_var, ref[1].running_var, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_windows_named_by_indexing(device, call_outs):
    tp.manual_seed(0)

    def fn(x):
        return (
            (x.conj() * 2).select(2, 3).detach().sum()
            + x[:, 1:3, ::2].sum() * 3
            + x.select(1, -1).pow(2).sum()
            + tp.stack(x.split(2, dim=3)[:2], dim=0).sum()
        )

    _check(fn, [tp.randn(2, 4, 6, 5, device=device)], train=True, atol=1e-4)
    assert not [n for n in call_outs if "getitem" in n or "select" in n], dict(call_outs)


@pytest.mark.parametrize("device", DEVICES)
def test_var_mean_with_a_correction(device):
    tp.manual_seed(0)
    x = tp.randn(6, 10, device=device)
    for kwargs in ({"dim": 1}, {"dim": 1, "correction": 0}, {"dim": (0, 1), "keepdim": True}, {}):
        var, mean = tp.compile(lambda t: tp.var_mean(t, **kwargs))(x)
        rvar, rmean = tp.var_mean(x, **kwargs)
        assert tp.allclose(var, rvar, atol=1e-5) and tp.allclose(mean, rmean, atol=1e-5)
    var, _ = tp.var_mean(x, dim=1, correction=0)
    centered = x - x.mean(1, keepdim=True)
    assert tp.allclose(var, (centered * centered).mean(1), atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_einsum_is_differentiated_when_compiled(device):
    # einsum has no derivative of its own: it is differentiated through the
    # products it is made of, which tracing has to see one by one.
    tp.manual_seed(0)
    a = tp.randn(3, 4, 5, device=device)
    b = tp.randn(3, 5, 6, device=device)
    _check(lambda a, b: tp.einsum("bij,bjk->bik", a, b).tanh(), [a, b], train=True, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_inputs_the_program_never_reads_get_no_gradient(device):
    a = tp.randn(4, 4, device=device, requires_grad=True)
    b = tp.randn(4, 4, device=device, requires_grad=True)
    c = tp.randn(4, 4, device=device, requires_grad=True)
    out = tp.compile(lambda a, b, c: a.sin() * c.cos() + b.detach())(a, b, c)
    out.sum().backward()
    assert b.grad is None
    assert tp.allclose(a.grad, a.detach().cos() * c.detach().cos(), atol=1e-6)
    assert tp.allclose(c.grad, -a.detach().sin() * c.detach().sin(), atol=1e-6)


@needs_cuda
def test_extern_reads_views_of_pooled_buffers_at_their_offset():
    # q, k and v are windows of one projection.  Laid down in a pooled
    # allocation, each is read by the attention call at its own offset; an
    # offset counted from the pool's start instead reads another window.
    tp.manual_seed(0)
    x = tp.randn(2, 64, 384, device="cuda")
    cos = tp.randn(64, 16, device="cuda")
    sin = tp.randn(64, 16, device="cuda")

    def heads(t):
        return t.view(2, 64, 4, 32).transpose(1, 2)

    def rope(t):
        t1, t2 = t[..., ::2], t[..., 1::2]
        return tp.stack((t1 * cos - t2 * sin, t1 * sin + t2 * cos), dim=-1).flatten(-2)

    def fn(x):
        q = rope(heads(x[..., :128]))
        k = rope(heads(x[..., 128:256]))
        v = heads(x[..., 256:])
        return F.scaled_dot_product_attention(q, k, v)

    assert tp.allclose(tp.compile(fn)(x), fn(x), atol=1e-4)


@needs_cuda
def test_compiled_inputs_that_start_inside_their_storage():
    base = tp.randn(10, 32, device="cuda")
    x = base[3:]
    fn = lambda t: F.scaled_dot_product_attention(t[None, None], t[None, None], t[None, None]) * 2
    assert x.storage_offset() != 0
    assert tp.allclose(tp.compile(fn)(x), fn(x), atol=1e-4)
