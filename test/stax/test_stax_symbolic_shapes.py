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
    backend.supports_symbolic_shapes = True
    backend.trace_decompositions = stax.trace_decompositions
    return backend


DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn", [lambda x: x * 2 + 1, lambda x: x.sum(-1), lambda x: x.mean(0)])
def test_native_dynamic_kernels_reuse_across_sizes(fn, device):
    artifacts = []

    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for rows, cols in ((3, 7), (5, 11), (2, 4)):
        x = tp.randn(rows, cols, device=device)
        actual, expected = compiled(x), fn(x)
        assert actual.shape == expected.shape
        assert tp.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert len(artifacts) == 1
    assert artifacts[0]._tensorplay_codegen == ("stax-cpu" if device == "cpu" else "triton")


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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "fn",
    [
        lambda x: x.reshape(x.shape[0], -1) * 2,
        lambda x: x.reshape(x.numel()) + 1,
        lambda x: x.reshape(x.size(0) * x.size(1), -1),
        lambda x: x.flatten() + 2,
        lambda x: (x * 2).reshape((x * 2).shape[0], -1),
        lambda x: x.sum(-1, keepdim=True).expand(x.shape) + 1,
    ],
)
def test_shape_arithmetic_stays_dynamic(fn, device):
    artifacts = []
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for rows, cols in ((3, 7), (5, 11), (2, 4)):
        x = tp.randn(rows, cols, device=device)
        result = compiled(x)
        assert result.shape == fn(x).shape
        assert tp.allclose(result, fn(x), atol=1e-5, rtol=1e-5)
    assert len(artifacts) == 1


@pytest.mark.parametrize("device", DEVICES)
def test_inferred_reshape_dimension_uses_runtime_integer_division(device):
    artifacts = []
    fn = lambda x: x.reshape(-1, 2)
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for rows, cols in ((3, 8), (5, 10), (2, 4)):
        x = tp.randn(rows, cols, device=device)
        assert tp.equal(compiled(x), fn(x))
    assert len(artifacts) == 1
    with pytest.raises(RuntimeError):
        compiled(tp.randn(3, 7, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_call_out_consumers_recheck_example_layouts(device):
    artifacts = []
    fn = lambda x: x[:, 1:] * 2
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for rows, cols in ((3, 7), (5, 11), (3, 7)):
        x = tp.randn(rows, cols, device=device)
        actual, expected = compiled(x), fn(x)
        assert actual.shape == expected.shape
        assert tp.equal(actual, expected)
    assert len(artifacts) == 2


def test_interpreted_fallback_still_checks_captured_shape_branches(monkeypatch):
    from tensorplay.compiler.backends.stax.graph_lowering import GraphLowering

    def cannot_generate(self):
        raise NotImplementedError("generated entry unavailable")

    monkeypatch.setattr(GraphLowering, "compile_to_module", cannot_generate)

    def fn(x):
        return x + 1 if x.shape[0] > 4 else x * 2

    artifacts = []
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True)
    for size in (3, 7, 2, 9):
        x = tp.arange(size, dtype=tp.float32)
        assert tp.equal(compiled(x), fn(x))
    assert len(artifacts) == 2


def test_dynamic_capture_binds_keywords_and_default_arguments():
    def fn(x, *, scale=2):
        return x.flatten() * scale

    artifacts = []
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for size in (3, 7):
        x = tp.randn(size, 4)
        assert tp.equal(compiled(x=x), fn(x))
        assert tp.equal(compiled(x=x, scale=3), fn(x, scale=3))
    assert len(artifacts) == 2


def test_shape_branch_guards_reuse_both_sides():
    def fn(x):
        return x + 1 if x.shape[0] > 4 else x * 2

    artifacts = []
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for size in (3, 7, 2, 9, 4, 11):
        x = tp.arange(size, dtype=tp.float32)
        assert tp.equal(compiled(x), fn(x))
    assert len(artifacts) == 2


@pytest.mark.parametrize("device", DEVICES)
def test_training_recaptures_saved_shapes_before_backward(device):
    artifacts = []
    fn = lambda x: (x * x).mean()
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for size in (3, 7, 3):
        x = tp.randn(size, device=device).requires_grad_()
        ref = x.detach().clone().requires_grad_()
        actual, expected = compiled(x), fn(ref)
        actual.backward()
        expected.backward()
        assert tp.allclose(actual, expected)
        assert tp.allclose(x.grad, ref.grad)


def test_training_fallback_recaptures_saved_shapes(monkeypatch):
    from tensorplay.compiler.backends.stax.graph_lowering import GraphLowering

    def cannot_generate(self):
        raise NotImplementedError("generated entry unavailable")

    monkeypatch.setattr(GraphLowering, "compile_to_module", cannot_generate)
    fn = lambda x: (x * x).mean()
    compiled = tp.compile(fn, dynamic=True)
    for size in (3, 7, 3):
        x = tp.randn(size).requires_grad_()
        ref = x.detach().clone().requires_grad_()
        compiled(x).backward()
        fn(ref).backward()
        assert tp.allclose(x.grad, ref.grad)


def test_nested_inputs_specialize_concrete_captured_metadata():
    fn = lambda inputs: inputs[0].reshape(inputs[0].shape[0], -1)
    compiled = tp.compile(fn, dynamic=True)
    for size in (3, 7, 3):
        x = [tp.randn(size, 4)]
        actual, expected = compiled(x), fn(x)
        assert actual.shape == expected.shape
        assert tp.equal(actual, expected)


def test_backend_variants_respect_recompile_limit():
    compiled = tp.compile(lambda x: x * 2, dynamic=True, strict_native=True, fullgraph=True, recompile_limit=1)
    compiled(tp.randn(3, 7))
    with pytest.raises(RuntimeError, match="specialization limit"):
        compiled(tp.randn(3, 7).t())
