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
def test_slice_consumers_reuse_symbolic_layouts(device):
    artifacts = []
    fn = lambda x: x[:, 1:] * 2
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for rows, cols in ((3, 7), (5, 11), (3, 7)):
        x = tp.randn(rows, cols, device=device)
        actual, expected = compiled(x), fn(x)
        assert actual.shape == expected.shape
        assert tp.equal(actual, expected)
    assert len(artifacts) == 1


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "fn",
    [
        lambda x: x[1:, ::2] + 1,
        lambda x: (x * 2)[:, 1:-1:2] + 3,
        lambda x: x[..., None, 1:-1] * 2,
        lambda x: x[:, -3:] * 2,
        lambda x: x[1] + 2,
        lambda x: x[-1] + 2,
        lambda x: x.select(-1, -1) * 3,
        lambda x: x[:, x.shape[1] // 2 :] - 1,
        lambda x: x[:, : x.shape[1] - 1 : 2] + 3,
        lambda x: x[x.shape[0] - 1] * 2,
        lambda x: tp.ops.tp.slice.Tensor(x, 1, 1, None, 2) + 1,
    ],
)
@pytest.mark.parametrize("transpose", [False, True])
def test_dynamic_indexing_reuses_generated_kernels(fn, device, transpose):
    artifacts = []
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for rows, cols in ((3, 7), (5, 11), (4, 8)):
        x = tp.randn(cols, rows, device=device).t() if transpose else tp.randn(rows, cols, device=device)
        actual, expected = compiled(x), fn(x)
        assert actual.shape == expected.shape
        assert tp.allclose(actual, expected)
    assert len(artifacts) == 1


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn", [lambda x: x[:, 1::2], lambda x: x[-1], lambda x: x[:, -2:]])
def test_dynamic_indexing_outputs_preserve_storage(fn, device):
    artifacts = []
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for rows, cols in ((3, 7), (5, 11), (2, 4)):
        x = tp.randn(rows, cols, device=device)
        actual, expected = compiled(x), fn(x)
        assert tp.equal(actual, expected)
        assert actual.stride() == expected.stride()
        assert actual.storage_offset() == expected.storage_offset()
        assert actual.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert len(artifacts) == 1


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn", [lambda x: x[:, 2:6] * 2, lambda x: x[:, -6:-2:2] + 1])
def test_dynamic_slice_clamping_keeps_valid_cached_artifacts(fn, device):
    artifacts = []
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for cols in (9, 11, 5, 7, 9, 5):
        x = tp.randn(3, cols, device=device)
        actual, expected = compiled(x), fn(x)
        assert actual.shape == expected.shape
        assert tp.equal(actual, expected)
    assert len(artifacts) == 2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn", [lambda x: x[:, 2:6] * 2, lambda x: x[:, -6:-2:2] + 1])
def test_dynamic_slices_cover_empty_and_singleton_results(fn, device):
    artifacts = []
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for cols in (9, 4, 3, 2, 1, 0):
        x = tp.randn(3, cols, device=device)
        actual, expected = compiled(x), fn(x)
        assert actual.shape == expected.shape
        assert tp.equal(actual, expected)
    count = len(artifacts)
    x = tp.randn(3, 9, device=device)
    assert tp.equal(compiled(x), fn(x))
    assert len(artifacts) == count


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("index", [3, -4])
def test_dynamic_selection_guards_out_of_range_indices(device, index):
    artifacts = []
    compiled = tp.compile(lambda x: x[index] * 2, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for rows in (5, 7):
        x = tp.randn(rows, 4, device=device)
        assert tp.equal(compiled(x), x[index] * 2)
    assert len(artifacts) == 1
    with pytest.raises((IndexError, RuntimeError)):
        compiled(tp.randn(3, 4, device=device))


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
def test_training_reuses_forward_and_backward_across_sizes(device):
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
    assert len(artifacts) == 2


def test_training_fallback_reuses_saved_symbolic_shapes(monkeypatch):
    from tensorplay.compiler.backends.stax.graph_lowering import GraphLowering

    def cannot_generate(self):
        raise NotImplementedError("generated entry unavailable")

    monkeypatch.setattr(GraphLowering, "compile_to_module", cannot_generate)
    fn = lambda x: (x * x).mean()
    artifacts = []
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True)
    for size in (3, 7, 3):
        x = tp.randn(size).requires_grad_()
        ref = x.detach().clone().requires_grad_()
        compiled(x).backward()
        fn(ref).backward()
        assert tp.allclose(x.grad, ref.grad)
    assert len(artifacts) == 2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "fn",
    [
        lambda x: (x * x).sum(),
        lambda x: (x * x).sum(-1),
        lambda x: (x * x).mean(0),
        lambda x: (x * x).mean(-1, keepdim=True),
        lambda x: (x * x).reshape(x.shape[0], -1),
        lambda x: (x * x).reshape(x.numel()),
        lambda x: (x * x).flatten(),
        lambda x: tp.reshape(x * x, (x.numel(),)),
        lambda x: x.sum(-1, keepdim=True).expand(x.shape),
    ],
)
@pytest.mark.parametrize("expanded_tangent", [False, True])
def test_training_shape_formulas_reuse_artifacts(fn, device, expanded_tangent):
    artifacts = []
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    pending = []
    for rows, cols in ((3, 7), (5, 11), (2, 4)):
        x = tp.randn(rows, cols, device=device).requires_grad_()
        ref = x.detach().clone().requires_grad_()
        actual, expected = compiled(x), fn(ref)
        assert actual.shape == expected.shape
        assert tp.allclose(actual, expected, atol=1e-5, rtol=1e-5)
        pending.append((x, ref, actual, expected))
    for x, ref, actual, expected in reversed(pending):
        tangent = tp.ones((), device=device).expand(actual.shape) if expanded_tangent else tp.randn_like(actual)
        actual.backward(tangent)
        expected.backward(tangent)
        assert tp.allclose(x.grad, ref.grad, atol=1e-5, rtol=1e-5)
    assert len(artifacts) == 2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("transpose", [False, True])
def test_training_broadcast_and_layout_reuse(device, transpose):
    artifacts = []
    fn = lambda x, y: ((x + y) * (x + y)).mean()
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for rows, cols in ((3, 7), (5, 11), (2, 4)):
        base = tp.randn(cols, rows, device=device).t() if transpose else tp.randn(rows, cols, device=device)
        x, y = base.requires_grad_(), tp.randn(cols, device=device).requires_grad_()
        refs = [v.detach().clone().requires_grad_() for v in (x, y)]
        actual, expected = compiled(x, y), fn(*refs)
        actual.backward()
        expected.backward()
        assert tp.allclose(actual, expected, atol=1e-5, rtol=1e-5)
        for value, ref in zip((x, y), refs):
            assert tp.allclose(value.grad, ref.grad, atol=1e-5, rtol=1e-5)
    assert len(artifacts) == 2


@pytest.mark.parametrize("fn", [lambda x: x.var(), lambda x: x.var(-1).sum(), lambda x: (x / x.shape[0]).sum()])
def test_training_specializes_concrete_saved_metadata(fn):
    artifacts = []
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True)
    for size in (3, 7, 3):
        x = tp.randn(size, 4).requires_grad_()
        ref = x.detach().clone().requires_grad_()
        compiled(x).backward()
        fn(ref).backward()
        assert tp.allclose(x.grad, ref.grad, atol=1e-5, rtol=1e-5)
    assert len(artifacts) == 4


@pytest.mark.parametrize("device", DEVICES)
def test_symbolic_scalar_results_reuse_artifacts(device):
    artifacts = []
    fn = lambda x: (x / x.shape[0], x.shape[0] * x.shape[1])
    compiled = tp.compile(fn, backend=_counting_backend(artifacts), dynamic=True, strict_native=True)
    for rows, cols in ((3, 7), (5, 11), (2, 4)):
        x = tp.randn(rows, cols, device=device)
        actual, count = compiled(x)
        assert tp.allclose(actual, fn(x)[0], atol=1e-5, rtol=1e-5)
        assert count == x.numel()
    assert len(artifacts) == 1


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
