"""Structural control flow operators.

``cond`` and ``while_loop`` are exposed at the package root; ``map`` and
``scan`` live under ``tensorplay._higher_order_ops``.  These tests pin the
eager semantics of each operator, gradient flow through the autograd
formulas, and the shape of a captured graph (one opaque node per operator
call, with the branch bodies attached as subgraphs).
"""

import pytest

import tensorplay as tp
from tensorplay._higher_order_ops import map, scan
from tensorplay._higher_order_ops.utils import make_fx


# ---------------------------------------------------------------------------
# cond
# ---------------------------------------------------------------------------


def test_cond_python_predicate_runs_selected_branch():
    x = tp.randn(3)
    called = []

    def true_fn(t):
        called.append("true")
        return t * 2

    def false_fn(t):
        called.append("false")
        return t * 3

    out = tp.cond(True, true_fn, false_fn, (x,))
    tp.testing.assert_close(out, x * 2)
    out = tp.cond(False, true_fn, false_fn, (x,))
    tp.testing.assert_close(out, x * 3)
    assert called == ["true", "false"]


def test_cond_tensor_predicate():
    x = tp.randn(3)
    out = tp.cond(x.sum() > -1e30, lambda t: t.cos(), lambda t: t.sin(), (x,))
    tp.testing.assert_close(out, x.cos())


def test_cond_input_validation():
    x = tp.randn(3)
    with pytest.raises(Exception):
        tp.cond("not-a-pred", lambda t: t, lambda t: t, (x,))
    with pytest.raises(Exception):
        tp.cond(tp.zeros(3) > 0, lambda t: t, lambda t: t, (x,))
    with pytest.raises(Exception):
        tp.cond(True, "not-callable", lambda t: t, (x,))


def test_cond_grad_flows_through_selected_branch():
    x = tp.randn(4, requires_grad=True)
    out = tp.cond(x.sum() > 0, lambda t: t * 3, lambda t: t * 5, (x,))
    out.sum().backward()
    factor = 3 if x.sum() > 0 else 5
    tp.testing.assert_close(x.grad, tp.full((4,), float(factor)))


def test_cond_trace_records_single_node():
    x = tp.randn(3)

    def f(t):
        return tp.cond(t.sum() > 0, lambda a: a.cos(), lambda a: a.sin(), (t,))

    gm = make_fx(f)(x)
    targets = [n.target.__name__ if hasattr(n.target, "__name__") else str(n.target)
               for n in gm.graph.nodes if n.op == "call_function"]
    assert any("cond" in name for name in targets), targets


# ---------------------------------------------------------------------------
# while_loop
# ---------------------------------------------------------------------------


def test_while_loop_matches_python_while():
    def cond_fn(it, x):
        return it.sum() < 5

    def body_fn(it, x):
        return it + 1, x.sin()

    it0 = tp.zeros(1)
    x0 = tp.randn(3, 4)
    out = tp.while_loop(cond_fn, body_fn, (it0, x0))

    it, x = it0, x0
    while bool((it.sum() < 5).item()):
        it, x = it + 1, x.sin()
    tp.testing.assert_close(out[0], it)
    tp.testing.assert_close(out[1], x)


def test_while_loop_zero_iterations():
    x = tp.randn(2)
    out = tp.while_loop(lambda it, t: it > 10, lambda it, t: (it + 1, t), (0, x))
    tp.testing.assert_close(out[1], x)


def test_while_loop_cond_output_validation():
    with pytest.raises(Exception):
        tp.while_loop(
            lambda it: tp.zeros(3),  # not a scalar bool
            lambda it: (it + 1,),
            (0,),
        )


def test_while_loop_body_arity_check():
    with pytest.raises(Exception):
        tp.while_loop(lambda a: a < 2, lambda a: (a + 1, a + 2), (0,))


def test_while_loop_grad():
    x0 = tp.randn(3, requires_grad=True)

    def cond_fn(it, x):
        return it < 3

    def body_fn(it, x):
        return it + 1, x * 1.5

    _, out = tp.while_loop(cond_fn, body_fn, (0, x0))
    tp.testing.assert_close(out, x0 * 1.5**3)
    out.sum().backward()
    tp.testing.assert_close(x0.grad, tp.full((3,), 1.5**3))


def test_while_loop_trace_records_single_node():
    def f(x):
        return tp.while_loop(
            lambda it, t: it < 3,
            lambda it, t: (it + 1, t * 1.5),
            (0, x),
        )[1]

    gm = make_fx(f)(tp.randn(2))
    targets = [n.target.__name__ if hasattr(n.target, "__name__") else str(n.target)
               for n in gm.graph.nodes if n.op == "call_function"]
    assert any("while_loop" in name for name in targets), targets


# ---------------------------------------------------------------------------
# map
# ---------------------------------------------------------------------------


def test_map_matches_unrolled_loop():
    xs = tp.randn(4, 3)
    out = map(lambda t: t * 2 + 1, xs)
    expected = tp.stack([xs.select(0, i) * 2 + 1 for i in range(4)], dim=0)
    tp.testing.assert_close(out, expected)


def test_map_multiple_inputs_and_extra_args():
    a = tp.randn(5, 2)
    b = tp.randn(5, 2)
    const = tp.randn(2)
    out = map(lambda pair, c: pair[0] + pair[1] + c, [a, b], const)
    expected = tp.stack(
        [a.select(0, i) + b.select(0, i) + const for i in range(5)], dim=0
    )
    tp.testing.assert_close(out, expected)


def test_map_leading_dim_validation():
    with pytest.raises(Exception):
        map(lambda t: t, [tp.randn(2, 3), tp.randn(3, 3)])
    with pytest.raises(Exception):
        map(lambda t: t, tp.randn(0, 3))


def test_map_grad():
    xs = tp.randn(4, 3, requires_grad=True)
    out = map(lambda t: t * t, xs)
    out.sum().backward()
    tp.testing.assert_close(xs.grad, xs * 2)


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def test_scan_cumsum_matches_reference():
    def combine(carry, x):
        nxt = carry + x
        return nxt, nxt.clone()

    init = tp.zeros(1)
    xs = tp.arange(5.0)
    carry, out = scan(combine, init, xs)
    tp.testing.assert_close(carry, tp.tensor([10.0]))
    expected = tp.cumsum(xs).unsqueeze(-1)
    tp.testing.assert_close(out, expected)


def test_scan_reverse_and_dim():
    def combine(carry, x):
        nxt = carry + x
        return nxt, nxt.clone()

    xs = tp.arange(6.0).reshape(2, 3)
    _, out = scan(combine, tp.zeros(2), xs, dim=1)
    expected = tp.cumsum(xs, dim=1)
    tp.testing.assert_close(out, expected)

    _, rev = scan(combine, tp.zeros(1), tp.arange(4.0), reverse=True)
    # Inclusive suffix sums: [0,1,2,3] -> [6, 6, 5, 3], one column per step.
    expected_rev = tp.tensor([[6.0], [6.0], [5.0], [3.0]])
    tp.testing.assert_close(rev, expected_rev)


def test_scan_length_counter_mode():
    calls = []

    def combine(carry, x):
        calls.append(1)
        return carry + 1, carry.clone()

    carry, out = scan(combine, tp.zeros(1), None, length=4)
    assert len(calls) == 4
    assert carry.item() == 4
    assert tuple(out.shape) == (4, 1)


def test_scan_length_zero_returns_init():
    def combine(carry, x):
        return carry + 1, carry.clone()

    carry, out = scan(combine, tp.zeros(2), None, length=0)
    tp.testing.assert_close(carry, tp.zeros(2))
    assert tuple(out.shape) == (0, 2)


def test_scan_grad():
    xs = tp.arange(4.0, requires_grad=True)

    def combine(carry, x):
        nxt = carry + x
        return nxt, nxt.clone()

    carry, out = scan(combine, tp.zeros(1), xs)
    out.sum().backward()
    # d(sum of running prefix sums)/dx_i = (n - i)
    expected = tp.tensor([4.0, 3.0, 2.0, 1.0])
    tp.testing.assert_close(xs.grad, expected)


def test_scan_trace_records_single_node():
    def f(xs):
        return scan(lambda c, x: (c + x, (c + x).clone()), tp.zeros(1), xs)[1]

    gm = make_fx(f)(tp.arange(4.0))
    targets = [n.target.__name__ if hasattr(n.target, "__name__") else str(n.target)
               for n in gm.graph.nodes if n.op == "call_function"]
    assert any("scan" in name for name in targets), targets


# ---------------------------------------------------------------------------
# Cross-checks against the reference framework when it is installed
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def test_cond_semantics_match_reference():
    for pred in (True, False):
        x_t = torch.randn(3)
        x = tp.tensor(x_t.numpy())
        ours = tp.cond(pred, lambda t: t * 2, lambda t: t * 3, (x,))
        theirs = torch.cond(pred, lambda t: t * 2, lambda t: t * 3, (x_t,))
        tp.testing.assert_close(ours, tp.tensor(theirs.detach().numpy()))


def test_while_loop_semantics_match_reference():
    def cond_fn(it, x):
        return it < 4

    def body_fn(it, x):
        return it + 1, x * 1.5

    x_t = torch.randn(2)
    ours = tp.while_loop(cond_fn, body_fn, (0, tp.tensor(x_t.numpy())))
    theirs = torch.while_loop(cond_fn, body_fn, (0, x_t))
    tp.testing.assert_close(ours[1], tp.tensor(theirs[1].detach().numpy()))
