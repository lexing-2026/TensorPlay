import pytest

import tensorplay as tp
from tensorplay._function_schema import parse_schema
from tensorplay._subclasses import SchemaCheckMode


class _FakeOp:
    """An operator whose implementation disagrees with its schema."""

    def __init__(self, text, impl):
        self._schema = parse_schema(text, namespace="test")
        self._impl = impl

    def __call__(self, *args, **kwargs):
        return self._impl(*args, **kwargs)


def test_records_declared_mutation_and_aliasing():
    x = tp.randn(3)
    with SchemaCheckMode() as mode:
        x.add_(1)
        x.view(3, 1)
    assert ("add_", "self") in [tuple(m) for m in mode.mutated]
    assert any(a.op_name == "view" and a.arg_name == "self" for a in mode.aliasing)
    assert "add_" in mode.ops


def test_plain_operators_pass():
    a, b = tp.randn(4), tp.randn(4)
    with SchemaCheckMode():
        (a * b).sin().sum()


def test_undeclared_mutation_raises():
    op = _FakeOp("bad_neg(Tensor self) -> Tensor", lambda t: t.neg_().clone())
    mode = SchemaCheckMode()
    with pytest.raises(RuntimeError, match="not defined as mutable"):
        mode.__tensorplay_dispatch__(op, (), (tp.randn(3),), {})


def test_undeclared_aliasing_raises():
    op = _FakeOp("bad_view(Tensor self) -> Tensor", lambda t: t.view(-1))
    mode = SchemaCheckMode()
    with pytest.raises(RuntimeError, match="not defined to alias output"):
        mode.__tensorplay_dispatch__(op, (), (tp.randn(3),), {})


def test_returning_an_input_from_a_functional_operator_raises():
    op = _FakeOp("bad_identity(Tensor(a) self) -> Tensor(a)", lambda t: t)
    mode = SchemaCheckMode()
    with pytest.raises(RuntimeError, match="not allowed to directly return inputs"):
        mode.__tensorplay_dispatch__(op, (), (tp.randn(3),), {})


def test_training_only_mutation_of_running_stats():
    x = tp.randn(4, 3)
    mean, var = tp.zeros(3), tp.ones(3)
    with SchemaCheckMode() as mode:
        tp.nn.functional.batch_norm(x, mean, var, training=True)
    assert any(m.arg_name in ("running_mean", "running_var") for m in mode.mutated)


def test_eager_debug_backend_runs_under_the_check():
    def fn(a):
        return (a + 1).relu()

    compiled = tp.compile(fn, backend="eager_debug")
    a = tp.randn(5)
    assert tp.allclose(compiled(a), fn(a))
