"""GPU numerics and behavior for the half-precision GEMV kernels and the
vectorized two-byte 2-D transpose copy.

Covers the memory-bound shapes the dedicated kernels exist for: matrix @
vector (N == 1), the linear-layer pattern x @ W.t() with small activation
batches, reduction lengths that do not pack into 16-byte packets, and the
beta == 0 epilogue contract where stale output bits must be ignored.
"""

import numpy as np
import pytest
import tensorplay as tp

CUDA_SHAPES_GEMV = [
    (1, 4096, 1),      # plain matrix @ vector
    (7, 4096, 1),
    (256, 512, 1),
    (4096, 1103, 1),   # K not a multiple of 8: scalar rows schedule
    (511, 12, 1),      # K below one packet
    (1, 4096, 512),    # x @ W.t(), batch 1
    (4, 4096, 1024),   # small batch, transposed weight
    (8, 1103, 512),    # small batch, odd K, transposed weight
    (3, 2048, 512),    # small batch, contiguous weight (cols kernel)
    (2, 15, 7),        # tiny odd K through the generic schedule
]

HALF_DTYPES = [tp.float16, tp.bfloat16]


def _available():
    return tp.cuda.is_available()


def _ref_mm(a32, b32):
    return (a32.to(tp.float32) @ b32.to(tp.float32)).to("cpu")


def _max_rel(out, ref):
    worst = 0.0
    nans = 0
    M, N = out.shape[0], out.shape[1]
    for i in range(M):
        for j in range(N):
            g = out[i][j].item()
            if g != g:
                nans += 1
                continue
            worst = max(worst, abs(g - ref[i][j].item()))
    return worst, nans


def _gen32(shape, seed):
    tp.manual_seed(seed)
    return tp.randn(shape, device="cuda", dtype=tp.float32)


@pytest.mark.parametrize("dt", HALF_DTYPES)
@pytest.mark.parametrize("M,K,N", CUDA_SHAPES_GEMV)
@pytest.mark.parametrize("transposed", [False, True])
def test_half_gemv_matches_fp32_reference(dt, M, K, N, transposed):
    if not _available():
        pytest.skip("no cuda")
    if transposed and N == 1:
        pytest.skip("transpose flag meaningless for N == 1")

    af = _gen32([M, K], M * 31 + K)
    bf = _gen32([K, N], K * 17 + N)
    a = af.to(dt)
    b = bf.to(dt)
    if transposed:
        b = b.t().contiguous().t()

    out = (a @ b).to(tp.float32).to("cpu")
    ref = _ref_mm(af, bf)

    # Output quantization of the result plus fp32-accumulated reduced-precision
    # products; K grows the random-walk spread of the element error.
    tol = 6e-2 * (K ** 0.5) if dt == tp.float16 else 4e-1 * (K ** 0.5) / 8.0
    worst, nans = _max_rel(out, ref)
    assert nans == 0, f"unexpected NaNs for {(M, K, N, transposed)}"
    assert worst <= max(tol, 1.0), f"worst abs err {worst} for {(M, K, N, transposed)}"


@pytest.mark.parametrize("dt", HALF_DTYPES)
def test_half_gemv_beta_zero_ignores_stale_output(dt):
    if not _available():
        pytest.skip("no cuda")
    # addmm with beta=0 must not fold in the uninitialized destination bits.
    M, K, N = 256, 512, 1
    af = _gen32([M, K], 11)
    bf = _gen32([K, N], 13)
    a = af.to(dt)
    b = bf.to(dt)
    out = tp.empty([M, N], device="cuda", dtype=dt)
    res = tp.addmm(out, a, b, beta=0.0, alpha=1.0)
    vals = res.to(tp.float32).to("cpu")
    nans = sum(
        1 for i in range(M) if vals[i][0].item() != vals[i][0].item()
    )
    assert nans == 0, "beta=0 epilogue read stale output bits"


@pytest.mark.parametrize("dt", HALF_DTYPES)
@pytest.mark.parametrize("rows,cols", [(256, 256), (1024, 512), (2048, 4096),
                                       (512, 300), (4096, 1024)])
def test_two_byte_transpose_copy_bit_exact(dt, rows, cols):
    if not _available():
        pytest.skip("no cuda")
    src = _gen32([cols, rows], rows + cols * 3).to(dt)
    out = src.t().contiguous()
    ref = src.to(tp.float32).t().to("cpu")
    got = out.to(tp.float32).to("cpu")
    for i in range(rows):
        for j in range(cols):
            assert got[i][j].item() == ref[i][j].item(), (
                f"mismatch at ({i},{j}) for {rows}x{cols}"
            )


@pytest.mark.parametrize("dt", HALF_DTYPES)
def test_two_byte_transpose_untiled_sizes(dt):
    if not _available():
        pytest.skip("no cuda")
    # Not multiples of the tile edge: exercises the guarded scalar edges of
    # the packet kernel and the fallback scalar tile.
    for rows, cols in [(64, 64), (130, 200), (254, 260), (1024, 1023)]:
        src = _gen32([cols, rows], rows * 7 + cols).to(dt)
        out = src.t().contiguous()
        ref = src.to(tp.float32).t().to("cpu")
        got = out.to(tp.float32).to("cpu")
        assert (got - ref).abs().max().item() == 0.0
