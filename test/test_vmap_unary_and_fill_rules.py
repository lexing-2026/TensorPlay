"""vmap batches the unary pointwise ops, conjugations and fills.

Each rule keeps the operand's batch dimension, wherever it sits, and
matches the operation applied slice by slice -- including ops that record
no gradient (sign, the fills) and ops with scalar arguments.
"""

import pytest

import tensorplay as tp


def per_slice(fn, x, in_dim):
    return tp.stack([fn(s) for s in x.unbind(in_dim)])


UNARY = {
    "conj": lambda t: t.conj(),
    "resolve_conj": lambda t: t.conj().resolve_conj(),
    "conj_physical": lambda t: tp.conj_physical(t),
    "sgn": lambda t: t.sgn(),
    "sign": lambda t: t.real.sign(),
    "reciprocal": lambda t: t.reciprocal(),
    "square": lambda t: t.square(),
    "real": lambda t: t.real * 2,
    "imag": lambda t: t.imag * 2,
    "angle": lambda t: t.angle(),
    "view_as_real": lambda t: tp.view_as_real(t),
}

# Real-valued ops, on inputs inside (0, 1) where every one is defined.
REAL = {
    name: (lambda op: lambda t: getattr(tp, op)(t))(name)
    for name in ["acos", "asin", "atan", "atanh", "asinh", "tan", "deg2rad", "rad2deg",
                 "digamma", "lgamma", "erfinv", "exp2", "log2", "log10", "frac", "sinc",
                 "i0", "signbit", "isnan", "isinf", "isposinf", "isneginf"]
}
REAL.update({
    "acosh": lambda t: tp.acosh(t + 1),
    "hardsigmoid": lambda t: tp.nn.functional.hardsigmoid(t),
    "hardswish": lambda t: tp.nn.functional.hardswish(t),
    "mish": lambda t: tp.nn.functional.mish(t),
    "silu": lambda t: tp.nn.functional.silu(t),
    "celu": lambda t: tp.nn.functional.celu(t - 0.5, alpha=0.7),
    "elu": lambda t: tp.nn.functional.elu(t - 0.5, alpha=0.3),
    "gelu": lambda t: tp.nn.functional.gelu(t, approximate="tanh"),
    "hardshrink": lambda t: tp.nn.functional.hardshrink(t, lambd=0.4),
    "softshrink": lambda t: tp.nn.functional.softshrink(t, lambd=0.4),
    "hardtanh": lambda t: tp.nn.functional.hardtanh(t, -0.2, 0.6),
    "leaky_relu": lambda t: tp.nn.functional.leaky_relu(t - 0.5, 0.1),
    "softplus": lambda t: tp.nn.functional.softplus(t, beta=2.0, threshold=1.0),
    "threshold": lambda t: tp.nn.functional.threshold(t, 0.5, -1.0),
    "logit": lambda t: tp.logit(t, eps=0.2),
    "mvlgamma": lambda t: tp.mvlgamma(t + 2, p=2),
    "round_decimals": lambda t: tp.round(t, decimals=2),
    "nan_to_num": lambda t: tp.nan_to_num(tp.log(t - 0.5), nan=7.0, neginf=-3.0),
    "view_as_complex": lambda t: tp.view_as_complex(t.unsqueeze(-1).expand(*t.shape, 2)
                                                    .contiguous()),
})


@pytest.mark.parametrize("name", sorted(UNARY))
@pytest.mark.parametrize("in_dim", [0, 1])
def test_unary_rules_match_slice_by_slice(name, in_dim):
    tp.manual_seed(0)
    x = tp.randn(3, 4, dtype=tp.complex128) + 0.5
    fn = UNARY[name]
    got = tp.vmap(fn, in_dims=in_dim)(x)
    assert tp.allclose(got, per_slice(fn, x, in_dim))


@pytest.mark.parametrize("name", sorted(REAL))
@pytest.mark.parametrize("in_dim", [0, 1])
def test_real_unary_rules_match_slice_by_slice(name, in_dim):
    tp.manual_seed(1)
    x = tp.rand(3, 4, dtype=tp.float64) * 0.9 + 0.05
    fn = REAL[name]
    got = tp.vmap(fn, in_dims=in_dim)(x)
    assert tp.allclose(got, per_slice(fn, x, in_dim), equal_nan=True)


def test_view_as_complex_puts_the_batch_dimension_first():
    x = tp.randn(4, 3, 2, dtype=tp.float64)
    got = tp.vmap(tp.view_as_complex, in_dims=1)(x)
    assert got.shape == (3, 4)
    assert tp.allclose(got, tp.complex(x[..., 0], x[..., 1]).t())
    with pytest.raises(RuntimeError, match="one or more dimensions"):
        tp.vmap(tp.view_as_complex)(tp.randn(2, dtype=tp.float64))


@pytest.mark.parametrize("fill, value", [(tp.zeros_like, 0.0), (tp.ones_like, 1.0)])
def test_fills_keep_the_batch_dimension(fill, value):
    x = tp.randn(2, 5, dtype=tp.float64)
    got = tp.vmap(fill, in_dims=1)(x)
    assert got.shape == (5, 2)
    assert tp.equal(got, tp.full((5, 2), value, dtype=tp.float64))
    assert tp.vmap(lambda t: tp.empty_like(t, dtype=tp.float32))(x).shape == (2, 5)
    assert tp.vmap(lambda t: tp.zeros_like(t, dtype=tp.int64))(x).dtype == tp.int64


def test_ops_without_a_gradient_run_on_captured_tensors():
    x = tp.randn(2, 5, dtype=tp.float64)
    captured = tp.tensor([-1.0, 2.0], dtype=tp.float64)
    got = tp.vmap(lambda t: t.sum() * tp.ones_like(captured) + tp.sign(captured))(x)
    want = x.sum(1, keepdim=True) * tp.ones(2, dtype=tp.float64) + tp.sign(captured)
    assert tp.allclose(got, want)
