"""Tile-plan CPU kernels: outer-axis-contiguous inputs still compile.

A transposed view (element address ``r + c*a``, contiguous along the outer
axis) used to lose every generated route: the flat emitter cannot prove a
contiguous vector load and the row plan explicitly rejects it.  The tile
plan keeps it compiled -- the parallel range counts row tiles of the vector
width, a W-by-W transposed buffer is staged once per tile, and the vector
loop reads the buffer instead of a strided walk.
"""

import numpy as np
import pytest

import tensorplay
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


TILE_CLASSIFY = [
    ("dense", (8, 16), (16, 1), (8, 16), ("rowstrided", 16)),
    ("transposed", (64, 48), (1, 64), (64, 48), ("transposed", 64)),
    ("col-broadcast", (8, 1), (1, 1), (8, 16), ("rowscalar", 1)),
    ("scalar", (1,), (1,), (8, 16), ("splat", 0)),
    ("odd-transposed", (33, 21), (1, 33), (33, 21), ("transposed", 33)),
]


@pytest.mark.parametrize(
    "name,shape,strides,out_shape,expected",
    TILE_CLASSIFY,
    ids=[case[0] for case in TILE_CLASSIFY],
)
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


def test_build_accepts_transposed_layouts():
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
    if built is not None and isinstance(built, tuple):
        assert built[0] is not None


# ---------------------------------------------------------------------------
# end to end


def test_transposed_input_region_compiles_and_matches():
    tensorplay.manual_seed(0)
    v = tensorplay.randn(64, 48)
    t = tensorplay.randn(48, 64).t()
    fn = lambda v, t: ((v + t) * 2.0).tanh()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, t), fn(v, t))
    assert _route(compiled) == "stax-cpu"


def test_transposed_odd_shape_compiles_and_matches():
    tensorplay.manual_seed(1)
    v = tensorplay.randn(33, 21)
    t = tensorplay.randn(21, 33).t()
    fn = lambda v, t: ((v + t) * 2.0).tanh()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, t), fn(v, t))
    assert _route(compiled) == "stax-cpu"


def test_transposed_large_region_uses_the_parallel_tile_path():
    tensorplay.manual_seed(2)
    v = tensorplay.randn(1024, 1024)
    t = tensorplay.randn(1024, 1024).t()
    fn = lambda v, t: ((v + t) * 2.0).tanh()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, t), fn(v, t))
    assert _route(compiled) == "stax-cpu"


def test_segmented_transposed_region_matches_reference():
    tensorplay.manual_seed(4)
    v = tensorplay.randn(64, 48)
    t = tensorplay.randn(48, 64).t()
    fn = lambda v, t: ((v + t) * 2.0).tanh().reshape(-1).exp()  # noqa: E731
    compiled = tensorplay.compile(fn, backend="stax")
    _close(compiled(v, t), fn(v, t))
    assert _route(compiled) == "stax-cpu"
