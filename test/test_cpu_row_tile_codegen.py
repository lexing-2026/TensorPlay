"""Row-structured CPU kernels: layouts flat addressing rejects still compile.

Column broadcasts, per-row scalars, strided rows, and row-broadcast widths
no vector peel can align used to lose every generated route (or, on the
segmented path, read past the input's storage through flat addressing).
They now compile as an outer row loop carrying per-row base pointers and
hoisted per-row scalars.
"""

import numpy as np
import pytest

import tensorplay
import tensorplay.compiler.config as config
from tensorplay.compiler.backends.stax.codegen import cpp as cpu_cpp


def _close(got, ref, rel=2e-5):
    got = np.asarray(got.tolist(), dtype=np.float64)
    ref = np.asarray(ref.tolist(), dtype=np.float64)
    assert got.shape == ref.shape
    scale = max(1e-6, float(np.max(np.abs(ref))) if ref.size else 1.0)
    assert float(np.max(np.abs(got - ref))) <= rel * scale


def _route(compiled):
    lowering = next(iter(compiled._tensorplay_cache.values()))
    return getattr(lowering, "_tensorplay_codegen", None)


# ---------------------------------------------------------------------------
# address classification


ROW_CLASSIFY = [
    ("dense", (8, 16), (16, 1), (8, 16), ("rowstrided", 16)),
    ("row-bias", (16,), (1,), (8, 16), ("rowstrided", 0)),
    ("expanded-bias", (8, 16), (0, 1), (8, 16), ("rowstrided", 0)),
    ("col-broadcast", (8, 1), (1, 1), (8, 16), ("rowscalar", 1)),
    ("expanded-col", (8, 16), (1, 0), (8, 16), ("rowscalar", 1)),
    ("strided-rows", (8, 16), (32, 1), (8, 16), ("rowstrided", 32)),
    ("rank3-dense", (2, 4, 16), (64, 16, 1), (2, 4, 16), ("rowstrided", 16)),
    ("rank3-col", (2, 4, 1), (4, 1, 1), (2, 4, 16), ("rowscalar", 1)),
    ("rank3-bias", (16,), (1,), (2, 4, 16), ("rowstrided", 0)),
    ("scalar", (1,), (1,), (8, 16), ("splat", 0)),
]


@pytest.mark.parametrize(
    "name,shape,strides,out_shape,expected",
    ROW_CLASSIFY,
    ids=[case[0] for case in ROW_CLASSIFY],
)
def test_row_mode_classification(name, shape, strides, out_shape, expected):
    modes = cpu_cpp.analyze_input_rows((shape,), (strides,), out_shape)
    assert modes == (expected,)


ROW_REJECTS = [
    ("transposed", (64, 48), (1, 64), (64, 48)),
    ("block-broadcast", (1, 4, 16), (64, 16, 1), (2, 4, 16)),
    ("rank-1-output", (16,), (1,), (16,)),
    ("inner-stride-2", (8, 16), (32, 2), (8, 16)),
    ("uneven-row-strides", (2, 4, 16), (70, 16, 1), (2, 4, 16)),
]


@pytest.mark.parametrize(
    "name,shape,strides,out_shape",
    ROW_REJECTS,
    ids=[case[0] for case in ROW_REJECTS],
)
def test_row_mode_rejects_non_affine_layouts(name, shape, strides, out_shape):
    assert cpu_cpp.analyze_input_rows((shape,), (strides,), out_shape) is None


def test_classification_rejects_mismatched_operand_counts():
    assert (
        cpu_cpp.analyze_input_rows(((8, 16),), ((16, 1), (16, 1)), (8, 16))
        is None
    )
    assert cpu_cpp.analyze_input_rows((), (), (8, 16)) is None


# ---------------------------------------------------------------------------
# rendering


def _render(out_shape, input_layouts, instructions=None):
    if instructions is None:
        instructions = [("add", 0, 1, 2), ("mul", 2, -1, 3), ("tanh", 3, -1, 4)]
    return cpu_cpp.render_kernel_source(
        instructions,
        [2.0],
        len(input_layouts),
        instructions[-1][3],
        "tp_test_entry",
        out_shape=out_shape,
        out_device=(0, -1),
        input_shapes=tuple(layout[0] for layout in input_layouts),
        input_strides=tuple(layout[1] for layout in input_layouts),
        lane_count=16,
    )


def test_row_kernel_renders_row_loop_and_hoists_row_scalars():
    source = _render((64, 48), [((64, 48), (48, 1)), ((64, 1), (1, 1))])
    assert "for (long long r = rb; r < re; ++r)" in source
    assert "const V s1 = V(in1[r * 1LL]);" in source
    assert "const float* __restrict__ p0 = in0 + r * 48LL;" in source
    assert "% " not in source
    # The parallel range counts rows with an element-grain chunk size.
    assert "tp_parallel_for_c(0, 64LL, 10LL, tp_body, &ctx);" in source


def test_row_kernel_skips_the_unroll_for_long_programs():
    steps = [("add", 0, 1, 2 + i) for i in range(20)]
    source = _render(
        (64, 48), [((64, 48), (48, 1)), ((64, 1), (1, 1))], instructions=steps
    )
    assert "col + 4 * W" not in source
    assert "for (; col + W <= 48LL; col += W)" in source


def test_flat_layouts_keep_the_flat_kernel():
    # A colmod-expressible bias stays on the flat kernel with its peel.
    source = _render((64, 48), [((64, 48), (48, 1)), ((48,), (1,))])
    assert "for (long long r = rb" not in source
    assert "i % W != 0" in source


def test_unaddressable_layouts_raise_instead_of_flat_fallback():
    with pytest.raises(cpu_cpp._ProgramError):
        _render((64, 48), [((64, 48), (48, 1)), ((64, 48), (1, 64))])


def test_build_declines_unaddressable_layouts():
    probe = tensorplay.randn(1)
    built = cpu_cpp.build_cpu_native_kernel(
        [("add", 0, 1, 2)],
        [],
        2,
        2,
        shape=(64, 48),
        device=probe.device,
        input_shapes=((64, 48), (64, 48)),
        input_strides=((48, 1), (1, 64)),
    )
    assert built is None


# ---------------------------------------------------------------------------
# end to end


def test_column_broadcast_region_compiles_and_matches():
    tensorplay.manual_seed(0)
    v = tensorplay.randn(64, 48)
    b = tensorplay.randn(64, 1)
    fn = lambda v, b: ((v + b) * 2.0).tanh()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, b), fn(v, b))
    assert _route(compiled) == "stax-fused-cpu"


def test_odd_width_bias_region_compiles_and_matches():
    tensorplay.manual_seed(1)
    v = tensorplay.randn(64, 45)
    b = tensorplay.randn(45)
    fn = lambda v, b: ((v + b) * 2.0).tanh()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, b), fn(v, b))
    assert _route(compiled) == "stax-fused-cpu"


def test_row_strided_input_region_compiles_and_matches():
    tensorplay.manual_seed(2)
    v = tensorplay.randn(64, 48)
    s = tensorplay.randn(128, 48)[::2]
    fn = lambda v, s: ((v + s) * 2.0).tanh()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, s), fn(v, s))
    assert _route(compiled) == "stax-fused-cpu"


def test_expanded_column_input_region_compiles_and_matches():
    tensorplay.manual_seed(3)
    v = tensorplay.randn(64, 48)
    e = tensorplay.randn(64, 1).expand(64, 48)
    fn = lambda v, e: ((v + e) * 2.0).tanh()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, e), fn(v, e))
    assert _route(compiled) == "stax-fused-cpu"


def test_rank3_column_broadcast_compiles_and_matches():
    tensorplay.manual_seed(4)
    v = tensorplay.randn(4, 8, 32)
    b = tensorplay.randn(4, 8, 1)
    fn = lambda v, b: ((v + b) * 2.0).tanh()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, b), fn(v, b))
    assert _route(compiled) == "stax-fused-cpu"


def test_narrow_columns_stay_correct():
    tensorplay.manual_seed(5)
    v = tensorplay.randn(4096, 8)
    b = tensorplay.randn(4096, 1)
    fn = lambda v, b: ((v + b) * 2.0).tanh()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, b), fn(v, b))
    assert _route(compiled) == "stax-fused-cpu"


def test_where_pair_survives_row_addressing():
    tensorplay.manual_seed(6)
    v = tensorplay.randn(64, 48)
    b = tensorplay.randn(64, 1)
    fn = lambda v, b: tensorplay.where(  # noqa: E731
        (v * b) > 0.5, v * 2.0, b + 1.0
    )
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, b), fn(v, b))
    assert _route(compiled) == "stax-fused-cpu"


def test_runtime_layout_guard_relowers_on_shape_change():
    tensorplay.manual_seed(7)
    v = tensorplay.randn(64, 48)
    b = tensorplay.randn(64, 1)
    fn = lambda v, b: ((v + b) * 2.0).tanh()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, b), fn(v, b))
    v2 = tensorplay.randn(32, 48)
    b2 = tensorplay.randn(32, 1)
    _close(compiled(v2, b2), fn(v2, b2))


# ---------------------------------------------------------------------------
# the segmented-path regression


def test_segmented_broadcast_region_matches_reference():
    # The reshape keeps the region out of the whole-region routes; the
    # (R, 1) input used to be addressed flat past its storage here.
    tensorplay.manual_seed(8)
    v = tensorplay.randn(64, 48)
    b = tensorplay.randn(64, 1)
    fn = lambda v, b: ((v + b) * 2.0).tanh().reshape(-1).exp()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, b), fn(v, b))
    assert _route(compiled) == "stax-fused-cpu-segments"


def test_segmented_odd_width_bias_matches_reference():
    tensorplay.manual_seed(9)
    v = tensorplay.randn(64, 45)
    b = tensorplay.randn(45)
    fn = lambda v, b: ((v + b) * 2.0).tanh().reshape(-1).exp()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, b), fn(v, b))
    assert _route(compiled) == "stax-fused-cpu-segments"


# ---------------------------------------------------------------------------
# the switch


def test_switch_off_restores_the_legacy_surface(monkeypatch):
    monkeypatch.setattr(config, "cpu_row_tiling", False)
    tensorplay.manual_seed(10)
    v = tensorplay.randn(64, 48)
    b = tensorplay.randn(64, 1)
    fn = lambda v, b: ((v + b) * 2.0).tanh()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, b), fn(v, b))
    assert _route(compiled) != "stax-fused-cpu"


def test_switch_off_keeps_mixed_regions_correct(monkeypatch):
    monkeypatch.setattr(config, "cpu_row_tiling", False)
    tensorplay.manual_seed(11)
    v = tensorplay.randn(64, 48)
    b = tensorplay.randn(64, 1)
    fn = lambda v, b: ((v + b) * 2.0).tanh().reshape(-1).exp()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, b), fn(v, b))
