import pytest
import tensorplay as tp
from tensorplay.graph import Tracer
from tensorplay.graph.passes import (
    DeadCodeElimination,
    DecomposePass,
    NormalizeOperators,
    PassManager,
)

from tensorplay.compiler.backends.stax.backend import (
    _lower_cuda_fused_pointwise,
    stax as stax_backend,
)
from tensorplay.compiler.backends.stax.codegen.cuda import generate_cuda_source


GPU = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")


def _lowered(fn, *inputs):
    graph_module = Tracer(execute=True).trace(
        fn,
        sample_inputs={f"arg{index}": value for index, value in enumerate(inputs)},
    )
    PassManager([NormalizeOperators(), DecomposePass(), DeadCodeElimination()])(graph_module)
    return _lower_cuda_fused_pointwise(graph_module, list(inputs))


def test_source_is_straight_line_and_embeds_constants():
    source = generate_cuda_source(
        [3, 0, -1, 1, 1, -2],
        [2.0, 1.0],
        [2],
        1,
    )
    assert "T t0 = ((x0) * (T(2.0)));" in source
    assert "T t1 = ((t0) + (T(1.0)));" in source
    assert "return t1;" in source
    assert "for (" not in source
    assert "switch (" not in source


def test_source_emits_multiple_stores():
    source = generate_cuda_source(
        [9, 0, 0, 15, 0, 0],
        [],
        [1, 2],
        1,
    )
    assert "T& out0, T& out1" in source
    assert "out0 = t0;" in source
    assert "out1 = t1;" in source


def test_source_rejects_invalid_references_and_casts():
    with pytest.raises(ValueError, match="value reference"):
        generate_cuda_source([1, 1, -1], [0.0], [1], 1)
    with pytest.raises(ValueError, match="unsupported"):
        generate_cuda_source([35, 0, 1], [], [1], 1)
    with pytest.raises(ValueError, match="where_rest"):
        generate_cuda_source([27, 0, 0], [], [1], 1)


@GPU
@pytest.mark.parametrize(
    ("fn", "rtol", "atol"),
    [
        (lambda x: tp.where(x > 0, x.sin(), x.cos()), 1e-6, 1e-6),
        (lambda x: tp.minimum(x, x + 0.25), 1e-6, 1e-6),
        (lambda x: x.clamp_min(-0.25).clamp_max(0.5), 1e-6, 1e-6),
        (lambda x: tp.rsqrt(x.abs() + 1.0) + tp.erf(x), 1e-6, 1e-6),
    ],
)
def test_native_cuda_pointwise_matches_eager(fn, rtol, atol):
    device = tp.device("cuda", 0)
    x = tp.linspace(-2.0, 2.0, 257, device=device)
    compiled = _lowered(fn, x)
    assert compiled is not None
    assert compiled._tensorplay_codegen == "stax-cuda"
    got = compiled(x)
    expected = fn(x)
    assert tp.allclose(got, expected, rtol=rtol, atol=atol)


@GPU
def test_native_cuda_pointwise_normalizes_broadcast_and_strides():
    device = tp.device("cuda", 0)
    base = tp.linspace(-1.0, 1.0, 48, device=device).reshape(4, 12)
    a = base[:, ::2]
    b = tp.linspace(0.25, 1.25, 6, device=device).reshape(1, 6)
    fn = lambda x, y: tp.where(x < y, x * y, x - y)
    compiled = _lowered(fn, a, b)
    assert compiled is not None
    assert tp.allclose(compiled(a, b), fn(a, b), rtol=1e-6, atol=1e-6)


@GPU
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (tp.float16, 2e-3, 2e-3),
        (tp.bfloat16, 3e-2, 3e-2),
        (tp.float32, 1e-6, 1e-6),
        (tp.float64, 1e-12, 1e-12),
    ],
)
def test_native_cuda_pointwise_dtype_surface(dtype, rtol, atol):
    device = tp.device("cuda", 0)
    x = tp.linspace(0.1, 0.9, 65, device=device, dtype=dtype)
    fn = lambda value: tp.erf(value) + value.exp() - value.rsqrt()
    compiled = _lowered(fn, x)
    assert compiled is not None
    assert tp.allclose(compiled(x), fn(x), rtol=rtol, atol=atol)


@GPU
def test_stax_region_prefers_native_cuda_codegen():
    device = tp.device("cuda", 0)
    x = tp.randn(8, 9, device=device)
    fn = lambda value: tp.where(value > 0, value * 2.0, value / 3.0)
    graph_module = Tracer(execute=True).trace(fn, sample_inputs={"x": x})
    PassManager([NormalizeOperators(), DecomposePass(), DeadCodeElimination()])(graph_module)
    compiled = stax_backend(
        graph_module,
        [x],
        options={"stax.cuda_codegen": True},
        strict_native=True,
    )
    assert compiled._tensorplay_codegen == "stax-cuda"
    assert tp.allclose(compiled(x), fn(x), rtol=1e-6, atol=1e-6)


@GPU
def test_disabling_triton_selects_native_cuda_codegen():
    device = tp.device("cuda", 0)
    x = tp.randn(6, 10, device=device)
    fn = lambda value: value.sigmoid() + value.tanh()
    graph_module = Tracer(execute=True).trace(fn, sample_inputs={"x": x})
    PassManager([NormalizeOperators(), DecomposePass(), DeadCodeElimination()])(graph_module)
    compiled = stax_backend(
        graph_module,
        [x],
        options={"stax.triton": False},
        strict_native=True,
    )
    assert compiled._tensorplay_codegen == "stax-cuda"
    assert tp.allclose(compiled(x), fn(x), rtol=1e-6, atol=1e-6)
