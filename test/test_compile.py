import subprocess
import sys
from pathlib import Path

import pytest

import tensorplay as tp


def test_standalone_import_loads_cuda_runtime_without_torch():
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux ELF loader test")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tensorplay; assert 'torch' not in sys.modules; print(tensorplay._stax.get_default_backend())",
        ],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "stax"


def test_compile_uses_single_public_entrypoint_and_stax_backend():
    def fn(x, y):
        return x * y + y

    x = tp.tensor([1.0, 2.0, 3.0])
    y = tp.tensor([4.0, 5.0, 6.0])
    expected = fn(x, y)
    compiled = tp.compile(fn)
    actual = compiled(x, y)

    assert tp._stax.get_default_backend() == "stax"
    assert tp.hub.__name__ == "tensorplay.hub"
    assert tp.types.__name__ == "tensorplay.types"
    assert not hasattr(tp, "not_a_tensorplay_name")
    assert actual.tolist() == expected.tolist()
    assert compiled._tensorplay_backend == "stax"
    assert tp._stax.list_backends() == ["stax", "tvm"]


def test_custom_backend_receives_graph_module_and_caches_specializations():
    calls = []

    @tp._stax.register_backend(name="test_capture_backend")
    def test_capture_backend(graph_module, example_inputs, **kwargs):
        calls.append((graph_module, tuple(example_inputs), kwargs))
        return graph_module.forward

    def fn(x):
        return x * 2

    compiled = tp.compile(fn, backend="test_capture_backend", name="test")
    x = tp.tensor([1.0, 2.0])
    assert compiled(x).tolist() == [2.0, 4.0]
    assert compiled(x).tolist() == [2.0, 4.0]
    assert len(calls) == 1
    assert calls[0][0].graph.placeholders[0].name == "x"
    assert calls[0][2] == {"name": "test"}


def test_dynamic_mode_reuses_one_specialization_across_sizes():
    calls = []

    def backend(graph_module, example_inputs, **kwargs):
        calls.append((graph_module, kwargs))
        return graph_module.recompile()

    def fn(x):
        return x + 1

    compiled = tp.compile(fn, backend=backend, dynamic=True)
    assert compiled(tp.tensor([1.0])).tolist() == [2.0]
    assert compiled(tp.tensor([1.0, 2.0])).tolist() == [2.0, 3.0]
    assert len(calls) == 1
    assert calls[0][1]["dynamic"] is True


def test_stax_native_lowering_handles_scalar_and_unary_pointwise_ops():
    def fn(x):
        return (x * 2 + 3).relu().sin()

    x = tp.tensor([-1.0, 0.5, 2.0])
    compiled = tp.compile(fn, backend="stax")
    assert compiled(x).tolist() == pytest.approx(fn(x).tolist())


def test_stax_strict_native_never_reports_python_graph_executor_as_compiled():
    compiled = tp.compile(
        lambda value: tp.zeros(value.shape),
        backend="stax",
        fullgraph=True,
        strict_native=True,
    )
    with pytest.raises(RuntimeError, match="strict_native|native Stax"):
        compiled(tp.tensor([[1.0, 2.0]]))


def test_stax_fusion_lowers_to_p10_and_keeps_autograd():
    def fn(x):
        return x * 2 + 3

    x = tp.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    compiled = tp.compile(fn, backend="stax")
    output = compiled(x)

    lowering = next(iter(compiled._tensorplay_cache.values()))
    assert [node.op_type for node in lowering.graph.nodes] == ["fused_pointwise"]
    assert lowering._gradient_plan is not None

    output.sum().backward()
    assert x.grad.tolist() == pytest.approx([2.0, 2.0, 2.0])


def test_stax_fusion_accepts_programs_beyond_64_instructions():
    # Regression: the fused pointwise kernel used to reject programs with
    # more than 64 instructions; long pointwise chains must fuse natively.
    def fn(x, y):
        for _ in range(100):
            x = x - y
        return x

    x = tp.tensor([1.0, 2.0, 3.0])
    y = tp.tensor([0.5, 0.5, 0.5])
    compiled = tp.compile(fn, backend="stax", strict_native=True)
    output = compiled(x, y)

    lowering = next(iter(compiled._tensorplay_cache.values()))
    assert [node.op_type for node in lowering.graph.nodes] == ["fused_pointwise"]
    assert output.tolist() == pytest.approx(fn(x, y).tolist())


def test_stax_fused_pointwise_extended_autograd_matches_eager():
    def fn(left, right):
        return ((left.abs() + right.sigmoid()).tanh() / (left.cos() + 2.0)).relu()

    left0 = tp.tensor([-2.0, -0.3, 0.4, 1.7])
    right0 = tp.tensor([0.2, 1.1, 2.0, 3.0])
    eager_left = left0.clone().requires_grad_()
    eager_right = right0.clone().requires_grad_()
    compiled_left = left0.clone().requires_grad_()
    compiled_right = right0.clone().requires_grad_()

    eager = fn(eager_left, eager_right)
    compiled_fn = tp.compile(fn, backend="stax", fullgraph=True)
    compiled = compiled_fn(compiled_left, compiled_right)
    eager.sum().backward()
    compiled.sum().backward()

    assert tp.allclose(compiled, eager, rtol=1e-5, atol=1e-5)
    assert tp.allclose(compiled_left.grad, eager_left.grad, rtol=1e-5, atol=1e-5)
    assert tp.allclose(compiled_right.grad, eager_right.grad, rtol=1e-5, atol=1e-5)


def test_mode_and_options_are_mutually_exclusive():
    with pytest.raises(RuntimeError, match="Either mode or options"):
        tp.compile(lambda x: x + 1, mode="default", options={})


def test_fullgraph_specializes_data_dependent_control_flow():
    """D1: execute-mode capture specializes tensor-data branches."""

    def fn(x):
        if bool((x > 0).all()):
            return x + 1
        return x - 1

    compiled = tp.compile(fn, fullgraph=True)
    assert compiled(tp.tensor([1.0, 2.0])).tolist() == [2.0, 3.0]
    # Branch flip: gate guards force a fresh specialization instead of
    # silently reusing the wrong side.  Keys hold gate OUTCOMES (not input
    # bytes), so different data taking the same branch shares one entry.
    assert compiled(tp.tensor([-3.0])).tolist() == [-4.0]
    assert compiled(tp.tensor([7.0, 8.0])).tolist() == [8.0, 9.0]
    assert len(compiled._tensorplay_cache) == 2
    chains = compiled._tensorplay_guard_chains
    assert any(chain._data_component for chain in chains.values())
    assert any(
        chain._data_component[0] == "gates" for chain in chains.values()
    )
    chains = compiled._tensorplay_guard_chains
    assert any(chain._data_component for chain in chains.values())


def test_data_guards_survive_in_place_mutation():
    """Identity reuse is unsound once a data-guarded input mutates."""

    def fn(x):
        if x.sum().item() > 0:
            return x * 10
        return x * -1

    compiled = tp.compile(fn, fullgraph=True)
    x = tp.tensor([1.0, 2.0])
    assert compiled(x).tolist() == [10.0, 20.0]
    x.sub_(10)  # [-9.0, -8.0]: identity unchanged, branch-deciding bytes flipped
    assert compiled(x).tolist() == [9.0, 8.0]


def test_scalar_and_numeric_gates_specialize_from_samples():
    def scalar_fn(x):
        if x.sum().item() > 2:
            return x * 100
        return x

    compiled = tp.compile(scalar_fn, fullgraph=True)
    assert compiled(tp.tensor([4.0, 5.0])).tolist() == [400.0, 500.0]
    assert compiled(tp.tensor([1.0, 1.0])).tolist() == [1.0, 1.0]

    def int_fn(x):
        return x + int(x.sum().item())

    int_compiled = tp.compile(int_fn, fullgraph=True)
    assert int_compiled(tp.tensor([1.0, 2.0])).tolist() == [4.0, 5.0]


def test_symbolic_tracer_still_rejects_data_dependent_control_flow():
    from tensorplay.graph import GraphCaptureError, Tracer

    def fn(x):
        if bool((x > 0).all()):
            return x + 1
        return x

    tracer = Tracer()
    with pytest.raises(GraphCaptureError):
        tracer.trace(fn, sample_inputs={"x": tp.tensor([1.0])})


def test_fullgraph_specializes_metadata_control_flow():
    def fn(x):
        if len(x) > 0:
            return x + 1
        return x

    compiled = tp.compile(fn, backend="stax", fullgraph=True)
    assert compiled(tp.tensor([1.0])).tolist() == [2.0]


def test_module_compile_uses_top_level_frontend():
    class AddOne(tp.nn.Module):
        def forward(self, x):
            return x + 1

    module = AddOne()
    module.compile(backend="stax")
    assert module(tp.tensor([1.0, 2.0])).tolist() == [2.0, 3.0]


def test_graph_capture_preserves_keyword_only_and_default_arguments():
    def fn(x, scale=2, *, bias=1):
        return x * scale + bias

    compiled = tp.compile(fn)
    x = tp.tensor([1.0, 2.0])

    assert compiled(x).tolist() == [3.0, 5.0]
    assert compiled(x, 3, bias=4).tolist() == [7.0, 10.0]
    # Scalar placeholders stay on the generated GraphModule path because the
    # native Stax ABI is intentionally Tensor-only.
    lowering = next(iter(compiled._tensorplay_cache.values()))
    assert getattr(lowering, "graph", None) is None


def test_generated_functional_wrappers_capture_into_stax():
    x = tp.tensor([-1.0, 0.5, 2.0])
    compiled = tp.compile(lambda value: tp.sin(value))
    result = compiled(x)
    lowering = next(iter(compiled._tensorplay_cache.values()))

    assert result.tolist() == pytest.approx(tp.sin(x).tolist())
    # CPU pointwise graphs are lowered to Stax's fused vector kernel.
    assert [node.op_type for node in lowering.graph.nodes] == ["fused_pointwise"]

    fused = tp.compile(lambda left, right: tp.add(left, right, alpha=2))
    assert fused(tp.tensor([1.0]), tp.tensor([3.0])).tolist() == [7.0]
    fused_lowering = next(iter(fused._tensorplay_cache.values()))
    assert [node.op_type for node in fused_lowering.graph.nodes] == ["fused_pointwise"]


def test_shape_dependent_factory_is_captured_without_proxy_pybind_calls():
    compiled = tp.compile(lambda value: tp.zeros(value.shape), fullgraph=True)
    result = compiled(tp.tensor([[1.0, 2.0], [3.0, 4.0]]))

    assert tuple(result.shape) == (2, 2)
    assert result.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_linear_lowering_keeps_live_parameters_and_autograd():
    module = tp.nn.Linear(3, 2)
    compiled = tp.compile(module)
    x = tp.randn(4, 3, requires_grad=True)
    result = compiled(x)
    lowering = next(iter(compiled._tensorplay_cache.values()))
    # (forward_graph + backward_graph); eval-mode uses the single native
    # forward graph.
    graph = getattr(lowering, "forward_graph", None) or lowering.graph

    # The runtime executes a fused "linear" node when available; otherwise
    # the lowering falls back to transpose + matmul + add.
    from tensorplay._stax.stax import _native_runs_linear
    expected = ["linear"] if _native_runs_linear() else ["t", "matmul", "add"]
    assert [node.op_type for node in graph.nodes] == expected
    result.sum().backward()
    assert x.grad is not None
    assert module.weight.grad is not None

    before = compiled(x.detach()).tolist()
    with tp.no_grad():
        module.weight.add_(1)
    after = compiled(x.detach()).tolist()
    assert before != after


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
def test_stax_triton_compiles_forward_and_backward_together():
    try:
        import triton  # noqa: F401
    except ImportError:
        pytest.skip("Triton is unavailable")
    from tensorplay._stax.codegen.triton import runtime_available

    if not runtime_available():
        pytest.skip("Triton runtime cannot target this device")

    def fn(left, right):
        return ((left.abs() + right.sigmoid()).tanh() / (left.cos() + 2.0)).relu()

    left0 = tp.randn((2048,), device="cuda")
    right0 = tp.randn((2048,), device="cuda")
    left = left0.clone().requires_grad_()
    right = right0.clone().requires_grad_()
    eager_left = left0.clone().requires_grad_()
    eager_right = right0.clone().requires_grad_()

    compiled = tp.compile(fn, fullgraph=True)
    result = compiled(left, right)
    eager_result = fn(eager_left, eager_right)
    result.sum().backward()
    eager_result.sum().backward()

    lowering = next(iter(compiled._tensorplay_cache.values()))
    assert lowering._tensorplay_codegen == "triton"
    assert lowering._tensorplay_backward_codegen == "triton"
    assert tp.allclose(result, eager_result, rtol=1e-5, atol=1e-5)
    assert tp.allclose(left.grad, eager_left.grad, rtol=1e-5, atol=1e-5)
    assert tp.allclose(right.grad, eager_right.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_relu_backward_materializes_expanded_sum_gradient():
    values = tp.tensor([-2.0, -0.3, 0.4, 1.7], device="cuda", requires_grad=True)
    values.relu().sum().backward()
    expected = tp.tensor([0.0, 0.0, 1.0, 1.0], device="cuda")
    assert tp.allclose(values.grad, expected, rtol=0.0, atol=0.0)
