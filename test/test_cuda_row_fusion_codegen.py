"""Row-staged CUDA fusion: kernel source, planning, and numerics.

Source tests run everywhere (pure string generation).  Numeric checks are
gated on ``runtime_available()`` because launching needs a real CUDA device.
"""

import numpy as np
import pytest

import tensorplay
import tensorplay.nn.functional as F
from tensorplay._stax import stax as stax_mod
from tensorplay._stax.codegen import triton_rowfusion as row
from tensorplay._stax.codegen.triton import runtime_available


def _plan(fn, in_shape, input_shapes=None):
    from tensorplay.graph import symbolic_trace

    module = symbolic_trace(fn)
    if input_shapes is None:
        input_shapes = (tuple(in_shape),)
    return stax_mod._plan_row_fusion(module, tuple(in_shape), input_shapes)


def softmax(x):
    m = x.amax(dim=[1], keepdim=True)
    e = (x - m).exp()
    return e / e.sum(dim=1, keepdim=True)


def rmsnorm(x, w):
    v = (x * x).mean(dim=-1, keepdim=True)
    return (x * (v + 1e-5).rsqrt()) * w


def layernorm(x, w, b):
    mu = x.mean(dim=-1, keepdim=True)
    d = x - mu
    var = (d * d).mean(dim=-1, keepdim=True)
    return (d * (var + 1e-5).rsqrt()) * w + b


def rownorm(x):
    return (x * x).sum(dim=1).sqrt()


# ---------------------------------------------------------------------------
# kernel source


def _render(fn, in_shape, input_shapes=None, input_strides=None):
    fusion = _plan(fn, in_shape, input_shapes)
    assert fusion is not None
    if input_shapes is None:
        input_shapes = (tuple(in_shape),)
    if input_strides is None:
        input_strides = ((in_shape[-1], 1),)
    from tensorplay._stax.codegen.cpp import analyze_input_modes

    modes = analyze_input_modes(input_shapes, input_strides, tuple(in_shape), 1)
    assert modes is not None
    return fusion, row.render_row_fusion_kernel(fusion, "probe", modes)


def test_kernel_keeps_the_row_resident_across_stages():
    fusion, source = _render(softmax, (1024, 256))
    # One load of the row, and every stage reads that register.
    assert source.count("tl.load(in_ptr0") == 1
    assert "xoffset = tl.program_id(0) * XBLOCK" in source
    assert "base = rows[:, None] * 256" in source
    assert "cols1 = tl.arange(0, BLOCK)" in source
    # Two reductions, one store, one launch.
    assert source.count("tl.reduce(") + source.count("tl.sum(") == 2
    assert source.count("tl.store") == 1
    assert "triton.cdiv(1024, meta['XBLOCK'])" in source


def test_order_reductions_carry_nan_through():
    _fusion, source = _render(softmax, (64, 128))
    # The built-in maximum drops NaN, so the fold goes through an explicit
    # combine; the sum in the same kernel needs no such treatment.
    assert "tl.reduce(" in source and "_tp_max" in source
    assert source.count("propagate_nan=tl.PropagateNan.ALL") == 1
    assert "tl.sum(" in source


def test_only_the_needed_combine_helpers_are_emitted():
    _fusion, source = _render(rownorm, (16, 32))
    assert "_tp_max" not in source and "_tp_min" not in source


def test_row_wide_input_is_addressed_inside_the_row():
    _fusion, source = _render(
        rmsnorm, (32, 64), ((32, 64), (64,)), ((64, 1), (1,))
    )
    assert "tl.load(in_ptr1 + cols1, mask=cmask, other=0.0)[None, :]" in source
    assert "tl.load(in_ptr0 + base + cols" in source


def test_row_valued_output_stores_one_element_per_row():
    _fusion, source = _render(rownorm, (16, 32))
    assert "tl.store(out_ptr + rows[:, None]," in source


def test_mean_divides_by_the_row_extent():
    _fusion, source = _render(rmsnorm, (8, 40), ((8, 40), (40,)), ((40, 1), (1,)))
    assert "/ 40.0" in source


def test_block_is_the_power_of_two_above_the_extent():
    _fusion, source = _render(softmax, (4, 129))
    assert "BLOCK=256)" in source
    assert "cmask = cols1 < 129" in source


def test_wide_rows_keep_their_existing_route():
    # A row that fills the tile budget still compiles; one wider than it
    # would only spill, so it keeps whatever route it had.
    assert row.supported(_plan(softmax, (4, 8192)))
    assert not row.supported(_plan(softmax, (4, 1 << 14)))


# ---------------------------------------------------------------------------
# numerics


def _close(got, ref, rel=5e-5):
    got = np.asarray(got.cpu().tolist(), dtype=np.float64)
    ref = np.asarray(ref.cpu().tolist(), dtype=np.float64)
    assert got.shape == ref.shape
    scale = max(1e-6, float(np.max(np.abs(ref))) if ref.size else 1.0)
    assert float(np.max(np.abs(got - ref))) <= rel * scale


cuda_only = pytest.mark.skipif(
    not runtime_available(), reason="needs a Triton-capable CUDA device"
)

CASES = [
    ("softmax-manual", softmax, 1),
    ("softmax-api", lambda v: F.softmax(v, dim=-1), 1),
    ("log-softmax", lambda v: F.log_softmax(v, dim=-1), 1),
    ("rmsnorm", rmsnorm, 2),
    ("layernorm", layernorm, 3),
    ("rownorm", rownorm, 1),
]


@cuda_only
@pytest.mark.parametrize("name,fn,arity", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("shape", [(64, 128), (1024, 256), (37, 129)])
def test_row_fusion_matches_the_reference(name, fn, arity, shape):
    tensorplay.manual_seed(0)
    value = tensorplay.randn(*shape, device="cuda")
    args = [value] + [
        tensorplay.randn(shape[-1], device="cuda") for _ in range(arity - 1)
    ]
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(*args), fn(*args))


@cuda_only
def test_compile_routes_the_region_to_the_row_staged_kernel():
    tensorplay.manual_seed(1)
    value = tensorplay.randn(256, 512, device="cuda")
    compiled = tensorplay.compile(softmax, backend="stax")
    _close(compiled(value), softmax(value))
    lowering = next(iter(compiled._tensorplay_cache.values()))
    assert lowering._tensorplay_codegen == "stax-fused-cuda-rowfuse"


@cuda_only
def test_widths_around_the_block_boundary():
    for width in (1, 3, 31, 32, 33, 127, 128, 129, 1024, 1025):
        tensorplay.manual_seed(width)
        value = tensorplay.randn(7, width, device="cuda")
        _close(tensorplay.compile(softmax, backend="stax")(value), softmax(value))


@cuda_only
def test_order_reduction_propagates_nan():
    import math

    value = tensorplay.tensor(
        [[1.0, float("nan"), 3.0], [4.0, 5.0, 6.0]], dtype=tensorplay.float32
    ).cuda()
    fn = lambda v: v - v.amax(dim=[1], keepdim=True)  # noqa: E731
    got = tensorplay.compile(fn, backend="stax")(value).cpu().tolist()
    ref = fn(value).cpu().tolist()
    for got_row, ref_row in zip(got, ref):
        assert [math.isnan(item) for item in got_row] == [
            math.isnan(item) for item in ref_row
        ]


@cuda_only
def test_grad_inputs_keep_the_uncompiled_route():
    value = tensorplay.randn(4, 8, device="cuda", requires_grad=True)
    result = tensorplay.compile(softmax, backend="stax")(value)
    result.sum().backward()
    assert value.grad is not None
