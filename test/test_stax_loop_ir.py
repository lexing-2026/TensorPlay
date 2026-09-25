import pytest
import tensorplay as tp
from tensorplay.compiler.backends.stax.codegen.index_expr import (
    Const,
    ModularIndexing,
    Symbol,
    floordiv,
    modular_indexing,
)
from tensorplay.compiler.backends.stax.graph_lowering import GraphLowering
from tensorplay.compiler.backends.stax.kernel_scheduler import ExternNode, KernelScheduler
from tensorplay.compiler.backends.stax.loop_compile import compile_graph
from tensorplay.graph.experimental._dispatch_trace import dispatch_make_graph

GPU = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")


def _run(fn, inputs, tol=1e-4):
    graph_module = dispatch_make_graph(fn)(*inputs)
    program = compile_graph(graph_module, list(inputs))
    got = program(*inputs)
    want = fn(*inputs)
    if isinstance(want, (list, tuple)):
        for one_got, one_want in zip(got, want):
            assert one_got.shape == one_want.shape
            torch_free = (one_got - one_want).abs().max().item()
            assert torch_free <= tol * max(1.0, one_want.abs().max().item()), torch_free
    else:
        assert got[0].shape == want.shape
        error = (got[0] - want).abs().max().item()
        assert error <= tol * max(1.0, want.abs().max().item()), error
    return got


def _plan(fn, inputs):
    graph_module = dispatch_make_graph(fn)(*inputs)
    graph = GraphLowering(graph_module, list(inputs)).run()
    return KernelScheduler(graph).fuse()


def test_index_keys_sort_with_mixed_terms():
    """A sum of unlike terms must still canonicalize (keys are primitives)."""

    a, b, c = Symbol("a"), Symbol("b"), Const(3)
    # Building a sum orders its terms, which is where unlike key shapes would
    # otherwise meet: the canonical form has to survive it.
    mixed = a + modular_indexing(b, 2, 4) + floordiv(c, Const(2)) * a
    again = floordiv(c, Const(2)) * a + modular_indexing(b, 2, 4) + a
    assert mixed == again
    assert hash(mixed) == hash(again)
    assert len({mixed, again}) == 1
    assert mixed != ModularIndexing(b, 2, 4)


@GPU
def test_pointwise_fuses_into_one_kernel():
    x = tp.randn(64, 32, device="cuda")
    y = tp.randn(64, 32, device="cuda")

    def fn(a, b):
        return tp.nn.functional.silu(a) * b + 1.0

    groups = _plan(fn, [x, y])
    assert len(groups) == 1
    _run(fn, [x, y])


@GPU
def test_reduction_and_its_consumer_fuse():
    x = tp.randn(128, 64, device="cuda")

    def fn(a):
        return tp.sum(a, dim=1, keepdim=True) * 2.0 + 1.0

    _run(fn, [x])


@GPU
def test_long_reduction_over_inner_axis():
    """A row too long for one block is walked in steps."""

    x = tp.randn(8, 4096, device="cuda")

    def fn(a):
        return tp.sum(a, dim=1, keepdim=True)

    _run(fn, [x], tol=1e-3)


@GPU
def test_reduction_over_outer_axis():
    """Reducing the outer axis keeps the reduced elements strided."""

    x = tp.randn(64, 32, 8, device="cuda")

    def fn(a):
        return tp.sum(a, dim=2, keepdim=True)

    _run(fn, [x], tol=1e-3)


@GPU
def test_max_and_min_reductions():
    x = tp.randn(16, 40, device="cuda")

    _run(lambda a: tp.amax(a, dim=1, keepdim=True), [x])
    _run(lambda a: tp.amin(a, dim=1, keepdim=True), [x])


@GPU
def test_two_reductions_share_one_kernel():
    x = tp.randn(32, 24, device="cuda")

    def fn(a):
        return tp.sum(a, dim=1, keepdim=True) + tp.amax(a, dim=1, keepdim=True)

    groups = _plan(fn, [x])
    assert len(groups) == 1, [type(item).__name__ for item in groups]
    _run(fn, [x])


@GPU
def test_cat_and_view():
    x = tp.randn(4, 8, device="cuda")
    y = tp.randn(4, 8, device="cuda")

    def fn(a, b):
        joined = tp.cat([a, b], dim=1)
        return tp.reshape(joined, (4, 16)) * 2.0

    _run(fn, [x, y])


@GPU
def test_group_norm_forward():
    x = tp.randn(2, 8, 16, 16, device="cuda")
    w = tp.randn(8, device="cuda")
    b = tp.randn(8, device="cuda")

    def fn(a, weight, bias):
        return tp.nn.functional.group_norm(a, 4, weight, bias, 1e-5)

    _run(fn, [x, w, b])


@GPU
def test_group_norm_silu_is_one_pointwise_kernel():
    """Normalization moments reduce first; affine and the activation follow."""

    x = tp.randn(2, 8, 8, 8, device="cuda")
    w = tp.randn(8, device="cuda")
    b = tp.randn(8, device="cuda")

    def fn(a, weight, bias):
        return tp.nn.functional.silu(tp.nn.functional.group_norm(a, 4, weight, bias, 1e-5))

    groups = _plan(fn, [x, w, b])
    kernels = [item for item in groups if not isinstance(item, ExternNode)]
    assert len(kernels) == 2, [type(item).__name__ for item in groups]
    assert kernels[0].rnumel == 8 // 4 * 8 * 8
    assert kernels[1].rnumel == 1
    _run(fn, [x, w, b])


@GPU
def test_group_norm_silu_conv_block():
    tp.manual_seed(0)
    x = tp.randn(2, 8, 16, 16, device="cuda")
    w = tp.randn(8, device="cuda")
    b = tp.randn(8, device="cuda")
    weight = tp.randn(8, 8, 3, 3, device="cuda")
    bias = tp.randn(8, device="cuda")

    def fn(a, gamma, beta, kernel, shift):
        h = tp.nn.functional.silu(tp.nn.functional.group_norm(a, 4, gamma, beta, 1e-5))
        return tp.nn.functional.conv2d(h, kernel, shift, (1, 1))

    _run(fn, [x, w, b, weight, bias], tol=1e-4)


@GPU
def test_half_precision_inputs_round_trip():
    x = tp.randn(32, 64, device="cuda", dtype=tp.float16)
    y = tp.randn(32, 64, device="cuda", dtype=tp.float16)

    def fn(a, b):
        return tp.nn.functional.silu(a) * b

    graph_module = dispatch_make_graph(fn)(x, y)
    program = compile_graph(graph_module, [x, y])
    got = program(x, y)[0]
    assert got.dtype == tp.float16
    error = (got.float() - fn(x, y).float()).abs().max().item()
    assert error <= 1e-2 * max(1.0, fn(x, y).float().abs().max().item()), error
