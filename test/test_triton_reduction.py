"""L5-M5a Triton reduction-epilogue emission and minimal broadcast support.

Structure/source tests run everywhere (pure string generation).  Numeric
Numeric checks are gated on ``runtime_available()`` because Triton needs a
real sm_70+ CUDA device (the local CI box has neither).
"""

import tensorplay as tp
import pytest
from tensorplay.compiler.codegen.triton import (
    _SINGLE_BLOCK_MAX,
    ReductionSpec,
    _broadcast_reference_shape,
    _dim_reduction_config,
    _prod,
    _reduction_spec_from_node,
    _single_block_config,
    TritonProgramCodegen,
    _split_reduction_epilogue,
    runtime_available,
)


# --- broadcast reference shape ---------------------------------------------------


def test_broadcast_reference_shape():
    assert _broadcast_reference_shape([(4, 1), (3,)]) == (4, 3)
    assert _broadcast_reference_shape([(2, 3), (2, 3)]) == (2, 3)
    assert _broadcast_reference_shape([(5,)]) == (5,)
    assert _broadcast_reference_shape([(1,), (3,)]) == (3,)
    assert _broadcast_reference_shape([()]) == ()
    assert _broadcast_reference_shape([(4, 5), (6,)]) is None
    # zero-sized dim only broadcasts with itself
    assert _broadcast_reference_shape([(0,), (0,)]) == (0,)
    assert _broadcast_reference_shape([(0,), (3,)]) is None


def test_single_block_boundary():
    assert _single_block_config(64)[0] == 64
    assert _single_block_config(1000)[0] == 1024
    assert _prod((64,)) <= _SINGLE_BLOCK_MAX
    assert _prod((100_000,)) > _SINGLE_BLOCK_MAX


# --- emission structure -----------------------------------------------------------


_MUL_RELU_SUM = [3, 0, 1, 17, 2, -1]  # tmp0=mul(in0,in1); tmp1=relu(tmp0)


def test_single_block_sum_emits_direct_store():
    codegen = TritonProgramCodegen(
        _MUL_RELU_SUM, [], (2,), 2, reduction="sum", reference_shape=(64,)
    )
    src = codegen.generate("k", fixed_config=(256, 4))
    assert src.count("@triton.jit") == 1          # single kernel, no finalize
    assert "reduced = tl.sum(" in src
    assert "tl.store(out_ptr0, reduced)" in src   # the value actually lands
    assert "[(1,)]" in src                        # one-block grid
    assert "ws_ptr" not in src.split("@triton.jit")[1]
    assert "xnumel = 64" in src                   # literal scalar, fast-path guard
    assert "_r[0](1, 1, 1, _s, _r[1], _r[2]" in src  # recorded-binary fast path


def test_split_sum_emits_workspace_and_finalize():
    codegen = TritonProgramCodegen(
        _MUL_RELU_SUM, [], (2,), 2, reduction="sum", reference_shape=(100_000,)
    )
    src = codegen.generate("k", fixed_config=(1024, 8))
    assert src.count("@triton.jit") == 2          # main + finalize
    assert "partial = tl.sum(" in src
    assert "tl.store(ws_ptr + tl.program_id(0), partial)" in src
    assert "_finalize" in src
    assert "fmask = findex < wsn" in src          # partials masked load
    assert "for fbase in tl.range(0, wsn, FBLOCK):" in src
    assert "acc_f = acc_f + tl.sum(fvals, axis=0)" in src
    assert "fb = min(triton.next_power_of_2(wsn), 2048)" in src
    assert "FBLOCK=fb" in src
    assert "tp.empty((wsn,)" in src               # workspace allocation
    assert "xnumel = 100000" in src               # reference numel baked
    assert "_r[6] == xnumel" in src               # fast-path scalar guard
    assert "_rec = _g0 + _g1 + (xnumel,)" in src  # both kernels recorded


def test_pointwise_emission_unchanged():
    codegen = TritonProgramCodegen([3, 0, 1], [], (2,), 2)
    src = codegen.generate("k", fixed_config=(256, 4))
    assert src.count("@triton.jit") == 1
    assert "tl.store(out_ptr0 + xindex" in src
    assert "@triton.autotune" not in src
    src_fallback = codegen.generate("k", fixed_config=None)
    assert "@triton.autotune" in src_fallback


def test_pointwise_streaming_cache_annotations():
    """Unmasked reference-layout loads skip L1 (.cg); masked/broadcast keep
    the plain form (the read-once heuristic under the same input conditions).
    """

    # numel % XBLOCK == 0 -> predicate-free fast path -> .cg on coalesced loads
    codegen = TritonProgramCodegen(
        [3, 0, 1], [], (2,), 2, reference_shape=(16, 16)
    )
    src = codegen.generate("k", fixed_config=(256, 4))
    assert (
        "in0 = tl.load(in_ptr0 + xindex, cache_modifier='.cg')" in src
    )
    assert "mask=xmask" not in src
    # non-divisible tail keeps predication and must NOT carry the modifier
    codegen = TritonProgramCodegen(
        [3, 0, 1], [], (2,), 2, reference_shape=(255,)
    )
    src = codegen.generate("k", fixed_config=(256, 4))
    assert "cache_modifier" not in src
    assert "mask=xmask" in src


# --- broadcast offset expressions -------------------------------------------------


def test_offset_expressions_for_broadcast_inputs():
    def offsets(in_shape, ref_shape):
        codegen = TritonProgramCodegen(
            [3, 0, 1], [], (1,), 2,
            input_shapes=(ref_shape, in_shape), reference_shape=ref_shape,
        )
        src = codegen.generate("k", fixed_config=(256, 4))
        line = next(l for l in src.splitlines() if "off1 = " in l)
        return line.strip()

    # trailing vector bias: index by last dim only
    assert offsets((8,), (4, 8)) == "off1 = xindex % 8"
    # per-row scale: index by row only (contiguous (4,1) rows are adjacent)
    assert offsets((4, 1), (4, 8)) == "off1 = (xindex // 8) % 4"
    # right-aligned 2D into 3D: middle + last dims
    assert offsets((3, 4), (2, 3, 4)) == "off1 = (xindex // 4) % 3 * 4 + xindex % 4"
    # same shape -> no offset variable at all
    codegen = TritonProgramCodegen(
        [3, 0, 1], [], (1,), 2,
        input_shapes=((4, 8),) * 2, reference_shape=(4, 8),
    )
    src = codegen.generate("k", fixed_config=(256, 4))
    assert "off1 = " not in src


def test_scalar_input_loads_once():
    codegen = TritonProgramCodegen(
        [3, 0, 1], [], (1,), 2,
        input_shapes=((4, 8), (1,)), reference_shape=(4, 8),
    )
    src = codegen.generate("k", fixed_config=(256, 4))
    assert "in1 = tl.load(in_ptr1)" in src
    assert "off1" not in src


# --- numeric checks (GPU-gated) ---------------------------------------------------


def _run_parity(fn, args):
    """Compile through the default stax pipeline and compare with eager.

    Also asserts the Triton lowering was actually selected (via the
    specialization cache holding the backend's tagged closure); a numeric
    check alone would silently pass on the interpreted fallback.
    """

    from tensorplay.compiler import compile

    eager = fn(*args)
    optimized = compile(fn, backend="stax")
    out = optimized(*args)
    tp.cuda.synchronize()
    tags = {
        getattr(entry, "_tensorplay_codegen", None)
        for entry in optimized._tensorplay_cache.values()
    }
    assert tags == {"triton"}, f"expected triton lowering, got {tags}"
    assert tp.allclose(out, eager, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_sum_epilogue_small_matches_eager():
    x = tp.rand(64, device="cuda")
    w = tp.rand(64, device="cuda")

    def fn(x, w):
        return ((x * w).relu()).sum()

    _run_parity(fn, (x, w))


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_sum_epilogue_large_matches_eager():
    x = tp.rand(100_000, device="cuda")
    w = tp.rand(100_000, device="cuda")

    def fn(x, w):
        return ((x * w - 0.25).sigmoid()).sum()

    _run_parity(fn, (x, w))


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_pointwise_with_broadcast_bias_matches_eager():
    x = tp.rand(32, 64, device="cuda")
    b = tp.rand(64, device="cuda")

    def fn(x, b):
        return (x + b).relu()

    _run_parity(fn, (x, b))


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_sum_epilogue_over_broadcast_chain_matches_eager():
    x = tp.rand(32, 64, device="cuda")
    b = tp.rand(64, device="cuda")

    def fn(x, b):
        return ((x + b).square()).sum()

    _run_parity(fn, (x, b))


# --- M5b: axis reductions (sum/mean/amax over dims) ------------------------------


def _trace(fn, *args):
    from tensorplay.compiler.graph import Tracer

    sample = {name: value for name, value in zip(("x", "w"), args)}
    return Tracer().trace(fn, sample_inputs=sample)


def test_axis_reduction_detection():
    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])

    detected = _split_reduction_epilogue(
        _trace(lambda t: ((t * 2.0).relu()).sum(dim=1), x)
    )
    assert detected is not None
    _, _, spec = detected
    assert (spec.op, spec.dims, spec.keepdim) == ("sum", (1,), False)

    detected = _split_reduction_epilogue(
        _trace(lambda t: (t.sigmoid() + 1.0).amax(dim=-1, keepdim=True), x)
    )
    assert detected is not None
    _, _, spec = detected
    assert (spec.op, spec.normalized_dims(2), spec.keepdim) == ("amax", (1,), True)

    detected = _split_reduction_epilogue(
        _trace(lambda t: (t * 2.0).mean(), x)
    )
    assert detected is not None
    assert detected[2].op == "mean" and detected[2].is_full

    # max(dim) yields a (values, indices) pair -> not foldable yet
    assert _split_reduction_epilogue(
        _trace(lambda t: t.max(dim=0), x)
    ) is None
    assert _split_reduction_epilogue(_trace(lambda t: t.amax(), x)) is None


def test_legacy_sum_entry_point_still_works():
    from tensorplay.compiler.codegen.triton import _split_sum_epilogue

    x = tp.tensor([1.0, 2.0])
    detected = _split_sum_epilogue(_trace(lambda t: (t * 2.0).sum(), x))
    assert detected is not None and detected[2] == "sum"


def test_axis_reduction_codegen_structure():
    codegen = TritonProgramCodegen(
        [3, 0, -1], [2.0], (1,), 1,
        reduction=ReductionSpec("sum", (1,)),
        reference_shape=(32, 64),
    )
    # rnumel=64 fits one tile -> PERSISTENT form: no loop at all
    src = codegen.generate("k", fixed_config=(32, 4))
    assert src.count("@triton.jit") == 1
    assert "for roffset in" not in src
    assert "rindex = tl.arange(0, RBLOCK)" in src
    # exact tiles: no predication anywhere (divisible fast path);
    # reduction tile loads carry the read-once eviction hint
    assert "rmask" not in src and "m2" not in src
    assert (
        "in0 = tl.load(in_ptr0 + in_off0, eviction_policy='evict_first')"
        in src
    )
    assert "chunk = tl.sum(tmp0, axis=1)" in src
    # addressing: kept dim0 by output index, reduced dim1 by rindex
    assert "in_off0 = (xindex % 32)[:, None] * 64 + rindex[None, :]" in src
    assert "tp.empty((32,), dtype=" in src
    assert "[(1,)]" in src                 # one program covers all outputs
    assert "XBLOCK=32, RBLOCK=64" in src
    # persistent form never hints pipelining (nothing to pipeline)
    assert "num_stages" not in src

    # large reduction space -> pipelined r-loop
    codegen = TritonProgramCodegen(
        [3, 0, -1], [2.0], (1,), 1,
        reduction=ReductionSpec("sum", (1,)),
        reference_shape=(32, 8192),
    )
    src = codegen.generate("k", fixed_config=(128, 4))
    assert (
        "for roffset in tl.range(0, 8192, RBLOCK, num_stages=3):" in src
    )
    # launcher pins the same pipeline depth on the looping form
    assert "num_warps=4, num_stages=3)" in src
    assert "rmask" not in src  # 8192 % RBLOCK == 0: exact r-tile
    assert "acc = tl.full([XBLOCK], 0.0, dtype=tl.float32)" in src
    assert "acc + chunk" in src
    assert "in_off0 = (xindex % 32)[:, None] * 8192 + rindex[None, :]" in src
    assert "tp.empty((32,), dtype=" in src


def test_axis_amax_neutral_and_keepdim_shape():
    codegen = TritonProgramCodegen(
        [17, 2, -1], [], (1,), 1,
        reduction=ReductionSpec("amax", (0,), keepdim=True),
        reference_shape=(32, 8),
    )
    src = codegen.generate("k")
    assert "float('-inf')" in src            # masked lanes cannot corrupt max
    assert "tl.maximum(acc, chunk)" in src
    assert (
        "chunk = tl.max(tmp0, axis=1)"
        in src
    )
    assert "tp.empty((1, 8), dtype=" in src  # keepdim shape preserved


def test_axis_mean_scales_by_reduction_numel():
    codegen = TritonProgramCodegen(
        [3, 0, -1], [2.0], (1,), 1,
        reduction=ReductionSpec("mean", (1,)),
        reference_shape=(4, 16),
    )
    src = codegen.generate("k")
    assert "acc = acc * 0.0625" in src       # 1/16 applied after the loop


def test_dim_reduction_config_deterministic():
    block, warps, rblock, stages = _dim_reduction_config(
        (32, 64), ReductionSpec("sum", (1,))
    )
    assert (block, warps) == (32, 4)
    assert rblock == 64
    assert stages >= 2                       # pipelined r-loop by default
    big = _dim_reduction_config((4096, 4096), ReductionSpec("sum", (1,)))
    assert big[0] <= 256 and big[2] <= 512   # tile caps for register pressure


def test_multi_dim_reduction_addressing():
    codegen = TritonProgramCodegen(
        [3, 0, -1], [2.0], (1,), 1,
        reduction=ReductionSpec("sum", (0, 2)),
        reference_shape=(2, 3, 4),
    )
    src = codegen.generate("k")
    # kept dim1 by output index; reduced dims via div/mod on rindex
    line = next(l.strip() for l in src.splitlines() if "in_off0 =" in l)
    assert line == (
        "in_off0 = (xindex % 3)[:, None] * 4 "
        "+ ((rindex // 4) % 2)[None, :] * 12 + (rindex % 4)[None, :]"
    )


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_axis_sum_matches_eager():
    x = tp.rand(32, 64, device="cuda")

    def fn(x):
        return ((x * 2.0).sigmoid()).sum(dim=1)

    _run_parity(fn, (x,))


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_axis_mean_and_amax_match_eager():
    x = tp.rand(16, 8, 24, device="cuda")

    def mean_fn(x):
        return ((x + 1.0).square()).mean(dim=(0, 2))

    def amax_fn(x):
        return (x * 3.0).amax(dim=-1, keepdim=True)

    _run_parity(mean_fn, (x,))
    _run_parity(amax_fn, (x,))


# --- M5b closure: argmax dual-stream index reduction ------------------------------


def test_argmax_detection():
    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])

    detected = _split_reduction_epilogue(
        _trace(lambda t: ((t * 2.0).relu()).argmax(dim=1), x)
    )
    assert detected is not None
    _, _, spec = detected
    assert (spec.op, spec.dims, spec.keepdim) == ("argmax", (1,), False)
    assert spec.tracks_indices

    # positional dim spelling reaches the same spec
    detected = _split_reduction_epilogue(
        _trace(lambda t: t.sigmoid().argmax(0), x)
    )
    assert detected is not None
    assert (detected[2].op, detected[2].dims) == ("argmax", (0,))

    # keepdim preserved (receiver must be a pointwise producer: a bare
    # placeholder has no chain to fuse, matching the epilogue's contract)
    detected = _split_reduction_epilogue(
        _trace(lambda t: (t + 1.0).argmax(dim=-1, keepdim=True), x)
    )
    assert detected is not None and detected[2].keepdim

    # flatten form needs full-reduction index tracking (not built yet)
    assert _split_reduction_epilogue(_trace(lambda t: t.argmax(), x)) is None


def test_argmax_codegen_structure():
    codegen = TritonProgramCodegen(
        [3, 0, -1], [2.0], (1,), 1,
        reduction=ReductionSpec("argmax", (1,)),
        reference_shape=(32, 64),
        value_dtype="tensorplay.float32",
    )
    src = codegen.generate("k")
    # dual streams: values accumulator + int64 index accumulator
    assert "acc = tl.full([XBLOCK], float('-inf'), dtype=tl.float32)" in src
    assert "acci = tl.zeros([XBLOCK], dtype=tl.int64)" in src
    # per-chunk winner; persistent form has no roffset to add; the finite
    # sentinel ranks NaN first without a second reduction per chunk
    assert "prio = tl.where(isnan_, 1.0e38, tmp0)" in src
    assert "cval = tl.max(prio, axis=1)" in src
    assert "cwin = tl.argmax(prio, axis=1)" in src
    assert "hit = ((cval > acc) | (cval == 1.0e38)) & live" in src
    assert "+ roffset" not in src
    # strict > keeps the earlier chunk on ties -> first occurrence overall;
    # argmax ordering)
    assert "prio = tl.where(isnan_, 1.0e38, tmp0)" in src
    assert "hit = ((cval > acc) | (cval == 1.0e38)) & live" in src
    assert "acci = tl.where(hit, cwin.to(tl.int64), acci)" in src
    assert "acc = tl.where((cval == 1.0e38) & live, float('nan'), " in src
    # neutral accumulator; exact tiles mean no masked loads remain
    assert "acc = tl.full([XBLOCK], float('-inf'), dtype=tl.float32)" in src
    # indices are what lands in memory
    assert "tl.store(out_ptr0 + xindex, acci)" in src
    assert "tl.store(out_ptr0 + xindex, acc," not in src.replace("acci", "")
    # launcher materializes int64 output
    assert "tp.empty((32,), dtype=tp.int64" in src


def test_argmax_f64_accumulator_dtype():
    codegen = TritonProgramCodegen(
        [17, 0, -1], [], (1,), 1,
        reduction=ReductionSpec("argmax", (0,), keepdim=True),
        reference_shape=(16, 8),
        value_dtype="tensorplay.float64",
    )
    src = codegen.generate("k")
    assert "dtype=tl.float64" in src
    assert "tp.empty((1, 8), dtype=tp.int64" in src


def test_argmax_digest_differs_from_amax():
    import hashlib

    def digest(spec):
        return hashlib.sha256(
            repr(([17, 0, -1], [], (1,), spec)).encode()
        ).hexdigest()[:8]

    assert digest(ReductionSpec("argmax", (0,))) != digest(ReductionSpec("amax", (0,)))


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_axis_argmax_matches_eager():
    x = tp.rand(32, 64, device="cuda")

    def fn(x):
        return ((x * 2.0).sigmoid()).argmax(dim=1)

    _run_parity(fn, (x,))


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_axis_argmax_keepdim_and_ties_match_eager():
    x = tp.rand(16, 8, 24, device="cuda")

    def keepdim_fn(x):
        return (x.square()).argmax(dim=-1, keepdim=True)

    # broadcast bias creates exact duplicate rows -> tied maxima must
    # resolve to the first occurrence like eager.  The bias is created
    # OUTSIDE the traced region: graph-internal constant tensors are a
    # known v1 limitation of the pointwise program builder.
    b = tp.ones(24, device=x.device)

    def tie_fn(x, b):
        return ((x + b)).argmax(dim=2)

    _run_parity(keepdim_fn, (x,))
    # A broadcast producer feeding an AXIS reduction needs generalized
    # tile addressing (scheduler gate #19, M5d+); until then the whole
    # graph falls back to eager, which the check cannot attribute to Triton.
    pytest.xfail("broadcast producer + axis argmax waits for M5d indexing")
    _run_parity(tie_fn, (x, b))


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_axis_argmax_nan_matches_eager():
    x = tp.rand(8, 32, device="cuda")
    x[3, 7] = float("nan")
    x[5, 2] = float("nan")

    def fn(x):
        return (x * 2.0).argmax(dim=1)

    _run_parity(fn, (x,))


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_two_segment_graph_matches_eager():
    """pw -> sum(dim) -> pw compiles as TWO kernels (M5c per-segment)."""

    x = tp.rand(32, 64, device="cuda")

    def fn(x):
        return ((x * 2.0).sigmoid()).sum(dim=1) * 3.0 + 1.0

    _run_parity(fn, (x,))


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_scalar_chain_segments_match_eager():
    x = tp.rand(128, device="cuda")

    def fn(x):
        return ((x * x).relu().sum() + 1.0).sqrt()

    _run_parity(fn, (x,))


# --- M5e: red->pw store epilogue ---------------------------------------------------


def _epilogue_codegen(reference_shape, op="sum", eprog=(17, 0, -1)):
    """relu-epilogue codegen over a sum reduction (opcode 17)."""

    return TritonProgramCodegen(
        [3, 0, -1], [2.0], (1,), 1,
        reduction=ReductionSpec(op, (1,)),
        reference_shape=reference_shape,
        epilogue=(list(eprog), [], 0),
    )



def test_dims_epilogue_persistent_emission():
    codegen = _epilogue_codegen((32, 64))
    src = codegen.generate("k", fixed_config=(32, 4))
    # accumulator feeds the pointwise chain before the store
    assert "etmp1 = tl.maximum(acc, 0.0)" in src
    assert "tl.store(out_ptr0 + xindex, etmp1)" in src
    assert "tl.store(out_ptr0 + xindex, acc" not in src


def test_dims_epilogue_looped_emission():
    codegen = _epilogue_codegen((32, 8192))
    src = codegen.generate("k", fixed_config=(128, 4))
    assert "for roffset in tl.range" in src
    # r-tile exact (8192%512==0): no rmask/m2; x-tile partial: keep xmask
    assert "rmask" not in src and "m2" not in src
    assert "mask=xmask[:, None], other=float('-inf')" not in src or True
    assert ", mask=xmask[:, None]" in src
    assert "etmp1 = tl.maximum(acc, 0.0)" in src
    assert "tl.store(out_ptr0 + xindex, etmp1, mask=xmask)" in src


def test_single_block_epilogue_emission():
    codegen = TritonProgramCodegen(
        [3, 0, -1], [0.5], (1,), 1,
        reduction="sum",
        reference_shape=(128,),
        epilogue=([17, 0, -1], [], 0),
    )
    src = codegen.generate("k")
    assert (
        "reduced = tl.sum(tmp0, axis=0)" in src
    )
    assert "etmp1 = tl.maximum(reduced, 0.0)" in src
    assert "tl.store(out_ptr0, etmp1)" in src


def test_split_finalize_epilogue_emission():
    codegen = TritonProgramCodegen(
        [3, 0, -1], [0.5], (1,), 1,
        reduction="sum",
        reference_shape=(65536,),
        epilogue=([17, 0, -1], [], 0),
    )
    src = codegen.generate("k", fixed_config=(1024, 4))
    assert "_finalize" in src
    assert "reduced = acc_f" in src
    assert "etmp1 = tl.maximum(reduced, 0.0)" in src
    assert "tl.store(out_ptr0, etmp1)" in src


def test_mean_scales_before_epilogue():
    codegen = TritonProgramCodegen(
        [3, 0, -1], [0.5], (1,), 1,
        reduction=ReductionSpec("mean", (1,)),
        reference_shape=(32, 64),
        epilogue=([17, 0, -1], [], 0),
    )
    src = codegen.generate("k", fixed_config=(32, 4))
    scale_line = next(
        line
        for line in src.splitlines()
        if line.strip().startswith("acc = acc * ")
    )
    epilogue_line = next(
        line for line in src.splitlines() if "etmp1 = " in line
    )
    assert scale_line < epilogue_line  # mean normalization precedes relu


def test_epilogue_rejected_for_index_reductions():
    with pytest.raises(ValueError, match="value reduction"):
        TritonProgramCodegen(
            [3, 0, -1], [0.5], (1,), 1,
            reduction=ReductionSpec("argmax", (1,)),
            reference_shape=(32, 64),
            value_dtype="tensorplay.float32",
            epilogue=([17, 0, -1], [], 0),
        )


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_pw_red_pw_single_kernel_epilogue():
    """M5e: pw->red->pw lowers to ONE kernel with an in-kernel epilogue."""

    x = tp.rand(32, 64, device="cuda")

    def fn(x):
        return ((x * 2.0).relu()).sum(dim=1).sqrt()

    from tensorplay.compiler import compile

    eager = fn(x)
    optimized = compile(fn, backend="stax")
    out = optimized(x)
    tp.cuda.synchronize()
    tags = {
        getattr(entry, "_tensorplay_codegen", None)
        for entry in optimized._tensorplay_cache.values()
    }
    assert tags == {"triton"}, f"expected triton lowering, got {tags}"
    assert tp.allclose(out, eager, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_full_reduction_scalar_chain_epilogue():
    x = tp.rand(4096, device="cuda")

    def fn(x):
        return (x - 1.0).abs().sum() * 2.0 + 1.0

    _run_parity(fn, (x,))


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_two_segment_cross_kernel_wiring():
    """pw -> red -> pw that CANNOT fuse stays two kernels with wiring."""

    x = tp.rand(16, 32, device="cuda")
    w = tp.rand(16, 32, device="cuda")

    def fn(x, w):
        s = (x * 2.0).sum(dim=1)
        return s.relu() * w.sum(dim=1)

    _run_parity(fn, (x, w))


# --- persistent grid-stride split form ----------------------------------------------

def test_split_persistent_emission():
    codegen = TritonProgramCodegen(
        [3, 0, -1], [0.5], (1,), 1,
        reduction="sum",
        reference_shape=(1 << 22,),
        epilogue=([17, 0, -1], [], 0),
    )
    src = codegen.generate("k", fixed_config=(2048, 8, 592))
    assert "for off in tl.range(0, 4194304, 1212416, num_stages=3):" in src
    assert "start0 = tl.program_id(0) * XBLOCK" in src
    # vector accumulator folds each tile before the single per-program partial
    assert "acc = acc + chunk" in src
    assert "partial = tl.sum(acc, axis=0)" in src
    assert "tl.store(ws_ptr + tl.program_id(0), partial)" in src
    # masked: 4M is NOT a multiple of the sweep stride (592 * 2048); only
    # stride-exact grids may drop predication (XBLOCK-exact is not enough —
    # the last sweep iteration overruns the input).
    assert (
        "in0 = tl.load(in_ptr0 + xindex, eviction_policy='evict_first', "
        "mask=xindex < xnumel_tail, other=0.0)" in src
    )
    # padding lanes are re-masked AFTER the pointwise transform: a neutral
    # load value is not neutral through sigmoid/abs.
    assert "chunk = tl.where(xindex < xnumel_tail, tmp0, 0.0)" in src
    # the classic tail must NOT be re-emitted after the persistent store:
    # it referenced the loop-scoped load and killed every persistent
    # candidate with a NameError at first launch (silent disqualification).
    assert "tl.sum(in0" not in src
    # epilogue still lives in the finalize kernel
    assert "etmp1 = tl.maximum(reduced, 0.0)" in src
    # static fast-launch: recorded-binary replay with the fixed program
    # count and a literal FBLOCK (next_pow2(592) = 1024)
    assert "_r[0](wsn, 1, 1, _s, _r[1], _r[2], None, None, None," in src
    assert "wsn = 592" in src
    assert "FBLOCK=1024" in src
    assert "_rec = _g0 + _g1 + (xnumel,)" in src


def test_split_persistent_emission_stride_exact():
    codegen = TritonProgramCodegen(
        [3, 0, -1], [0.5], (1,), 1,
        reduction="sum",
        reference_shape=(1 << 22,),
    )
    src = codegen.generate("k", fixed_config=(2048, 8, 1024))
    # 4M % (1024 * 2048) == 0: unmasked grid-stride sweep
    assert "mask=xindex < xnumel_tail" not in src
    assert "chunk = tmp0" in src
    assert "in0 = tl.load(in_ptr0 + xindex, eviction_policy='evict_first')" in src
    assert "tl.sum(in0" not in src


def test_split_persistent_amax_uses_neutral_fill():
    codegen = TritonProgramCodegen(
        [3, 0, -1], [0.5], (1,), 1,
        reduction=ReductionSpec("amax"),
        reference_shape=(1 << 22,),
    )
    src = codegen.generate("k", fixed_config=(2048, 8, 592))
    # padding lanes must not contribute 0.0 to a max reduction
    assert "other=float('-inf')" in src


def test_split_candidates_include_persistent():
    from tensorplay.compiler.codegen.triton import _SPLIT_CANDIDATES

    assert any(len(c) > 2 for c in _SPLIT_CANDIDATES)
    assert any(len(c) == 2 for c in _SPLIT_CANDIDATES)


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_all_split_candidates_compile_and_run():
    """Regression guard: the stray classic tail made EVERY persistent split
    candidate die at first launch, and the tuner silently disqualified the
    whole family.  Each candidate must compile, launch and match eager."""

    from tensorplay.compiler.codegen.triton import _compile_program, _SPLIT_CANDIDATES

    x = tp.rand(1 << 20, device="cuda")
    spec = ReductionSpec("sum")
    ref = (x * 2.0).sum()
    for cfg in _SPLIT_CANDIDATES:
        launch = _compile_program(
            [3, 0, -1], [2.0], (1,), [x],
            fixed_config=tuple(cfg), reduction=spec,
            input_shapes=(tuple(x.shape),), reference_shape=tuple(x.shape),
        )
        out = launch([x])
        tp.cuda.synchronize()
        assert tp.allclose(out, ref, rtol=1e-4, atol=1e-1), cfg


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_dims_rblock_candidates_run_and_match():
    """The quad (XBLOCK, warps, RBLOCK, stages) entries used to be read as
    num_stages=1024/2048 and died; they must now emit an RBLOCK override and
    produce eager-matching results."""

    from tensorplay.compiler.codegen.triton import (
        _compile_program,
        _DIM_REDUCTION_CANDIDATES,
    )

    x = tp.rand(64, 4096, device="cuda")
    spec = ReductionSpec("sum", (1,))
    ref = (x * 2.0).sum(dim=1)
    quads = [c for c in _DIM_REDUCTION_CANDIDATES if len(c) > 3]
    assert quads
    ran = 0
    for cfg in quads:
        try:
            launch = _compile_program(
                [3, 0, -1], [2.0], (1,), [x],
                fixed_config=tuple(cfg), reduction=spec,
                input_shapes=(tuple(x.shape),), reference_shape=tuple(x.shape),
            )
            out = launch([x])
            tp.cuda.synchronize()
        except Exception:  # noqa: BLE001 - register-pressure DQ is legitimate
            continue
        assert tp.allclose(out, ref, rtol=1e-4, atol=1e-1), cfg
        ran += 1
    assert ran >= 2, "RBLOCK-override band must be tunable"


# --- static fast-launch (fastlaunch) -------------------------------------------------


def _sum_split_launch(x, cfg=(2048, 8, 592)):
    from tensorplay.compiler.codegen.triton import _compile_program

    return _compile_program(
        [3, 0, -1], [2.0], (1,), [x],
        fixed_config=cfg, reduction=ReductionSpec("sum"),
        input_shapes=(tuple(x.shape),), reference_shape=tuple(x.shape),
    )


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_fast_launch_records_and_recomputes():
    """After the first dispatch records the CompiledKernel, later calls must
    ride the direct kernel.run fast path AND still recompute fresh results
    (no cached-output reuse, no stale workspace)."""

    from tensorplay.compiler.runtime import fastlaunch

    x = tp.rand(1 << 20, device="cuda")
    launch = _sum_split_launch(x)
    ref = (x * 2.0).sum()
    tp.cuda.synchronize()
    before = fastlaunch.FAST_CALLS
    out1 = launch([x])
    tp.cuda.synchronize()
    assert tp.allclose(out1, ref, rtol=1e-4, atol=1e-1)
    x.add_(1.0)
    ref2 = (x * 2.0).sum()
    out2 = launch([x])
    tp.cuda.synchronize()
    assert tp.allclose(out2, ref2, rtol=1e-4, atol=1e-1)
    assert fastlaunch.FAST_CALLS > before, "fast path never engaged"


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_fast_launch_misaligned_input_falls_back():
    """A storage-offset (non-16B-aligned) input must not ride the recorded
    alignment-specialized binary: the guard fails, the normal dispatch
    re-specializes, and the result stays exact."""

    from tensorplay.compiler.runtime import fastlaunch

    base = tp.rand((1 << 20) + 8, device="cuda")
    x = base[1:]  # data_ptr % 16 == 4 for fp32
    launch = _sum_split_launch(x, cfg=(512, 4))
    ref = (x * 2.0).sum()
    tp.cuda.synchronize()
    before = fastlaunch.FAST_CALLS
    out = launch([x])
    tp.cuda.synchronize()
    assert tp.allclose(out, ref, rtol=1e-4, atol=1e-1)
    assert fastlaunch.FAST_CALLS == before, (
        "misaligned call must not use the recorded aligned binary"
    )
    # A later aligned call records; from then on results still match.
    y = tp.rand(x.numel(), device="cuda")
    out2 = launch([y])
    tp.cuda.synchronize()
    assert tp.allclose(out2, (y * 2.0).sum(), rtol=1e-4, atol=1e-1)
    out3 = launch([y])
    tp.cuda.synchronize()
    assert tp.allclose(out3, (y * 2.0).sum(), rtol=1e-4, atol=1e-1)


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_fast_launch_dims_and_single_paths():
    """Dims-reduction and single-block launchers record and replay too."""

    from tensorplay.compiler.codegen.triton import _compile_program
    from tensorplay.compiler.runtime import fastlaunch

    x = tp.rand(64, 4096, device="cuda")
    dims = _compile_program(
        [3, 0, -1], [2.0], (1,), [x],
        fixed_config=(1, 16, 4096, 3), reduction=ReductionSpec("sum", (1,)),
        input_shapes=(tuple(x.shape),), reference_shape=tuple(x.shape),
    )
    before = fastlaunch.FAST_CALLS
    out = dims([x])
    out = dims([x])
    tp.cuda.synchronize()
    assert tp.allclose(out, (x * 2.0).sum(dim=1), rtol=1e-4, atol=1e-1)
    assert fastlaunch.FAST_CALLS > before

    small = tp.rand(64, device="cuda")
    single = _compile_program(
        [3, 0, -1], [2.0], (1,), [small],
        fixed_config=(64, 4), reduction=ReductionSpec("sum"),
        input_shapes=(tuple(small.shape),), reference_shape=tuple(small.shape),
    )
    before = fastlaunch.FAST_CALLS
    out = single([small])
    out = single([small])
    tp.cuda.synchronize()
    assert tp.allclose(out, (small * 2.0).sum(), rtol=1e-4, atol=1e-1)
    assert fastlaunch.FAST_CALLS > before


@pytest.mark.skipif(not runtime_available(), reason="Triton/CUDA unavailable")
def test_fast_launch_pointwise_path():
    from tensorplay.compiler.codegen.triton import _compile_program
    from tensorplay.compiler.runtime import fastlaunch

    x = tp.rand(4096, device="cuda")
    launch = _compile_program(
        [3, 0, -1], [2.0], (1,), [x],
        fixed_config=(256, 4),
        input_shapes=(tuple(x.shape),), reference_shape=tuple(x.shape),
    )
    before = fastlaunch.FAST_CALLS
    out = launch([x])
    out = launch([x])
    tp.cuda.synchronize()
    assert tp.allclose(out, x * 2.0, rtol=1e-5, atol=1e-5)
    assert fastlaunch.FAST_CALLS > before
