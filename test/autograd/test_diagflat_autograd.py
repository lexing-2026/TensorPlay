import tensorplay as tp
from tensorplay.autograd import gradcheck, gradgradcheck


def test_diagflat_preserves_leaf_and_backpropagates():
    x = tp.rand(3, 4, dtype=tp.float64).requires_grad_(True)
    out = tp.diagflat(x)

    assert x.is_leaf
    assert x.grad_fn is None
    out.sum().backward()
    assert tp.equal(x.grad, tp.ones_like(x))


def test_diagflat_gradcheck_and_gradgradcheck():
    x = tp.rand(2, 3, dtype=tp.float64).requires_grad_(True)
    fn = lambda value: tp.diagflat(value, offset=1)

    assert gradcheck(fn, (x,), eps=1e-6)
    assert gradgradcheck(fn, (x,), eps=1e-6)


def test_implicit_autograd_siblings_preserve_leaves():
    cases = (
        (lambda x: tp.concat([x, x]), (3,)),
        (lambda x: tp.concatenate([x, x]), (3,)),
        (lambda x: tp.ger(x, x), (3,)),
        (lambda x: tp.kron(x, x), (3,)),
        (lambda x: tp.matrix_power(x, 2), (2, 2)),
        (lambda x: tp.vander(x), (3,)),
    )

    for fn, shape in cases:
        x = tp.rand(*shape, dtype=tp.float64).requires_grad_(True)
        fn(x)
        assert x.is_leaf
        assert x.grad_fn is None
