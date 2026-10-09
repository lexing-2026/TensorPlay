"""Dynamic layouts preserve expressions and reuse generated kernels."""

import pytest

import tensorplay as tp
from tensorplay.compiler.backends.stax.backend import stax


def _counting_backend(artifacts):
    def backend(gm, inputs, **kwargs):
        artifact = stax(gm, inputs, **kwargs)
        artifacts.append(artifact)
        return artifact

    backend.lowers_operator_graphs = True
    backend.trace_decompositions = stax.trace_decompositions
    return backend


@pytest.mark.parametrize("fn", [lambda x: x * 2 + 1, lambda x: x.sum(-1), lambda x: x.mean(0)])
def test_native_dynamic_kernels_reuse_across_sizes(fn):
    artifacts = []

    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for rows, cols in ((3, 7), (5, 11), (2, 4)):
        x = tp.randn(rows, cols)
        actual, expected = compiled(x), fn(x)
        assert actual.shape == expected.shape
        assert tp.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert len(artifacts) == 1
    assert artifacts[0]._tensorplay_codegen == "stax-cpu"


def test_zero_and_one_sizes_have_guarded_specializations():
    artifacts = []
    compiled = tp.compile(lambda x: x * 2 + 1, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for size in (3, 1, 0, 7, 1, 0, 11):
        x = tp.arange(size, dtype=tp.float32)
        assert tp.equal(compiled(x), x * 2 + 1)
    assert len(artifacts) == 3


def test_layout_changes_select_a_valid_cached_artifact():
    artifacts = []
    compiled = tp.compile(lambda x: x * 2, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for rows, cols in ((3, 7), (5, 11)):
        x = tp.randn(rows, cols)
        for value in (x, x.t(), x):
            assert tp.equal(compiled(value), value * 2)
    assert len(artifacts) == 2


def test_duck_shaped_inputs_do_not_assume_unchecked_equalities():
    artifacts = []
    compiled = tp.compile(lambda x, y: x + y, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for first, second in ((3, 3), (5, 5), (7, 1), (11, 11), (13, 1)):
        x, y = tp.randn(first, 4), tp.randn(second, 4)
        assert tp.allclose(compiled(x, y), x + y)
    assert len(artifacts) == 2
