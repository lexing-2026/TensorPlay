"""Dynamic layouts preserve expressions and reuse generated kernels."""

import pytest
import sympy

import tensorplay as tp
from tensorplay.graph.experimental.symbolic_shapes import ShapeEnv


def _expr(value):
    return getattr(value, "expr", sympy.sympify(value))


def test_shape_hints_do_not_prove_static_facts():
    env = ShapeEnv()
    symbol = env.create_symbol(7, "x.size[0]")
    assert not env.guards
    assert env._maybe_evaluate_static(sympy.Eq(symbol, 7)) is None
    assert env.optimization_hint(symbol * 3) == 21
    assert not env.guards
    assert env.guarding_hint_or_throw(symbol * 3) == 21
    assert env.guards[-1].expr == sympy.Eq(symbol * 3, 21)


@pytest.mark.parametrize("transpose", [False, True])
def test_symbolic_strides_follow_dense_layout(transpose):
    x = tp.empty(3, 7)
    if transpose:
        x = x.t()
    env = ShapeEnv()
    sizes, strides, _ = env.create_symbolic_sizes_strides_storage_offset(x, "x")
    sizes, strides = list(map(_expr, sizes)), list(map(_expr, strides))
    assert strides == ([1, sizes[0]] if transpose else [sizes[1], 1])
    assert not env.guards


