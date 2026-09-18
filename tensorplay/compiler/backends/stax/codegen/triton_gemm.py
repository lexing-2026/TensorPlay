"""Compile-time GEMM candidate selection for extern matmul segments.

Under the max-autotune mode, a two-dimensional float32 matmul that would run
as an extern (native) operator is benched against tiled Triton GEMM kernels
once per shape, and the winner is baked into the segment's launch.  The
native operator is always one of the candidates, so the selected path can
never lose against the plain extern plan; a Triton winner that disagrees
with the native result on a deterministic probe feed is rejected outright.

The chosen path is persisted in the autotune decision cache keyed by
(operand shape, dtype, device, kernel source), so later processes skip
benchmarking entirely.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Callable, Optional, Sequence, Tuple

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised on CPU-only installs
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages).  Curated tile set:
# tl.dot requires at least 16 lanes per block dimension, and the deeper
# BLOCK_K tiles trade shared-memory staging for fewer k-loop trips on
# bandwidth-bound skinny shapes where the native GEMM is weakest.
GEMM_CANDIDATE_CONFIGS: Tuple[Tuple[int, int, int, int, int], ...] = (
    (16, 64, 64, 4, 4),
    (32, 64, 64, 4, 4),
    (64, 64, 32, 4, 3),
    (64, 64, 64, 4, 4),
    (64, 128, 32, 4, 3),
    (128, 64, 32, 4, 3),
    (128, 128, 32, 8, 3),
    (128, 128, 64, 8, 3),
)

# Salt for the persisted decision: bump when the kernel body or the
# candidate table changes so old decisions cannot pin stale geometry.
GEMM_TUNING_VERSION = "gemm-v1"

_DECISION_NAMESPACE = "triton-autotune"

if HAS_TRITON:

    @triton.jit
    def _gemm_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        EVEN_K: tl.constexpr,
    ):
        """C = A(M,K) @ B(K,N), full-fp32 accumulation (no TF32).

        Output rows/cols wrap with ``% M``/``% N`` so the store needs no
        separate bounds mask; loads guard the k tail unless EVEN_K.
        """

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        off_m = (pid_m * BM + tl.arange(0, BM)) % M
        off_n = (pid_n * BN + tl.arange(0, BN)) % N
        off_k = tl.arange(0, BK)
        a_ptrs = a_ptr + off_m[:, None] * stride_am + off_k[None, :] * stride_ak
        b_ptrs = b_ptr + off_k[:, None] * stride_bk + off_n[None, :] * stride_bn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BK)):
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                k_tail = K - k * BK
                a = tl.load(a_ptrs, mask=off_k[None, :] < k_tail, other=0.0)
                b = tl.load(b_ptrs, mask=off_k[:, None] < k_tail, other=0.0)
            acc = tl.dot(a, b, acc, input_precision="ieee")
            a_ptrs += BK * stride_ak
            b_ptrs += BK * stride_bk
        c_ptrs = c_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
        tl.store(c_ptrs, acc)


def _kernel_source_digest() -> str:
    body = inspect.getsource(_gemm_kernel.fn if hasattr(_gemm_kernel, "fn")
                             else _gemm_kernel)
    return hashlib.sha256(
        f"{GEMM_TUNING_VERSION}|{body}".encode()
    ).hexdigest()[:16]


def _standard_2d(shape: Tuple[int, ...], stride: Tuple[int, ...]) -> bool:
    """Row- or column-major 2-D layout: one unit-stride axis, no overlap.

    Both layouts run through the same stride-parameterized kernel, which
    covers transposed weights without materializing a copy.
    """

    if len(shape) != 2 or len(stride) != 2:
        return False
    if shape[0] == 0 or shape[1] == 0:
        return False
    unit_row = stride[1] == 1 and stride[0] >= shape[1]
    unit_col = stride[0] == 1 and stride[1] >= shape[0]
    return unit_row or unit_col


def _decision_key(M: int, N: int, K: int, dtype: str, device: str) -> str:
    source = (
        f"gemm|{_kernel_source_digest()}|{M}|{N}|{K}|{dtype}|{device}"
    )
    return hashlib.sha256(source.encode()).hexdigest()[:24]


def _probe_feed(M: int, K: int, N: int, dtype: Any, device: Any) -> list:
    """Deterministic operand pair exercising every load/store lane.

    A linear ramp keeps values in a small band so the fp32 accumulation of
    the reference and the candidate stay comparable, while still making any
    stride or masking bug produce a grossly wrong product.
    """

    import tensorplay as tp

    def ramp(rows: int, cols: int) -> Any:
        flat = tp.arange(rows * cols, dtype=dtype, device=device)
        return flat * 3.17e-4 - 0.5

    return [ramp(M, K).reshape(M, K), ramp(K, N).reshape(K, N)]


def _triton_launch_factory(
    a_spec: Tuple[Optional[int], Any],
    b_spec: Tuple[Optional[int], Any],
    M: int, N: int, K: int,
    config: Tuple[int, int, int, int, int],
    base_launch: Callable[[list], Any],
):
    """Build a launch closure running one fixed GEMM tile configuration.

    Each operand spec is ``(feed position, literal tensor)`` — exactly one
    side is set: graph operands come from the feed, constants (module
    weights captured as literals) are closed over.
    """

    block_m, block_n, block_k, num_warps, num_stages = config

    def operand(feed: list, spec: Tuple[Optional[int], Any]) -> Any:
        position, literal = spec
        return feed[position] if position is not None else literal

    def launch(feed: list) -> Any:
        import tensorplay as tp

        a = operand(feed, a_spec)
        b = operand(feed, b_spec)
        if not (
            _standard_2d(tuple(a.shape), tuple(a.stride()))
            and _standard_2d(tuple(b.shape), tuple(b.stride()))
        ):
            # The baked kernel assumes standard 2-D layouts; a differently
            # strayed call takes the native operator instead.
            return base_launch(feed)
        out = tp.empty((M, N), dtype=a.dtype, device=a.device)
        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
        _gemm_kernel[grid](
            a, b, out, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
            BM=block_m, BN=block_n, BK=block_k,
            EVEN_K=(K % block_k == 0),
            num_warps=num_warps, num_stages=num_stages,
        )
        return out

    return launch


def tuned_matmul_launch(
    base_launch: Callable[[list], Any],
    sample_feed: Sequence[Any],
    operand_specs: Tuple[Tuple[Optional[int], Any], Tuple[Optional[int], Any]],
    out_shape: Tuple[int, ...],
) -> Optional[Callable[[list], Any]]:
    """Benchmark native vs Triton GEMM for one matmul extern segment.

    Each operand spec is ``(feed position, literal tensor)`` with exactly
    one side set; ``sample_feed`` supplies the stand-in tensor for every
    feed position.  Returns the winner-baked launch, or ``None`` when this
    segment does not qualify (non-2D, non-fp32, non-CUDA, exotic layouts)
    or the native operator wins.  ``base_launch`` remains the floor: it is
    one of the benched candidates and the fallback for unbenchable inputs.
    """

    import tensorplay as tp

    if not HAS_TRITON or len(out_shape) != 2:
        return None
    if not tp.cuda.is_available():
        return None

    def operand_tensor(spec: Tuple[Optional[int], Any]) -> Any:
        position, literal = spec
        return literal if position is None else sample_feed[position]

    a = operand_tensor(operand_specs[0])
    b = operand_tensor(operand_specs[1])
    if a is None or b is None:
        return None
    if str(a.dtype) != "tensorplay.float32" or str(b.dtype) != "tensorplay.float32":
        return None
    if not a.device.is_cuda():
        return None
    shape_a, shape_b = tuple(a.shape), tuple(b.shape)
    if len(shape_a) != 2 or len(shape_b) != 2 or shape_a[1] != shape_b[0]:
        return None
    if not (
        _standard_2d(shape_a, tuple(a.stride()))
        and _standard_2d(shape_b, tuple(b.stride()))
    ):
        return None

    M, K = shape_a
    N = shape_b[1]
    device_key = repr(a.device)
    cache_key = _decision_key(M, N, K, str(a.dtype), device_key)

    try:
        from ..codecache import default_cache

        cache = default_cache(_DECISION_NAMESPACE)
    except Exception:  # noqa: BLE001 - tuning is an optimization only
        return None

    def launch_for(choice: dict) -> Callable[[list], Any]:
        if choice.get("choice") != "triton":
            return base_launch
        config = (
            int(choice["bm"]), int(choice["bn"]), int(choice["bk"]),
            int(choice["warps"]), int(choice["stages"]),
        )
        return _triton_launch_factory(
            operand_specs[0], operand_specs[1],
            M, N, K, config, base_launch,
        )

    payload = cache.load(cache_key, ext="json")
    if payload is not None:
        try:
            return launch_for(json.loads(payload.decode()))
        except (ValueError, KeyError, TypeError):
            pass

    candidates: list = [("native",)]
    candidates.extend(("triton", cfg) for cfg in GEMM_CANDIDATE_CONFIGS)

    probe = _probe_feed(M, K, N, a.dtype, a.device)

    def build(candidate):
        if candidate[0] == "native":
            return base_launch
        return _triton_launch_factory(
            operand_specs[0], operand_specs[1],
            M, N, K, candidate[1], base_launch,
        )

    def bench(launch: Any, args: list) -> float:
        from ..runtime.stax_autotune import bench_launch

        return bench_launch(launch, args)

    try:
        best, best_launch, _ = _bench_candidates(build, candidates, probe, bench)
        if best is None or best[0] == "native":
            record = {"choice": "native"}
        else:
            reference = base_launch(probe)
            produced = best_launch(probe)
            if not tp.allclose(produced, reference, rtol=1e-4, atol=1e-3):
                record = {"choice": "native"}
                best_launch = base_launch
            else:
                cfg = best[1]
                record = {
                    "choice": "triton",
                    "bm": cfg[0], "bn": cfg[1], "bk": cfg[2],
                    "warps": cfg[3], "stages": cfg[4],
                }
        cache.store(cache_key, json.dumps(record).encode(), ext="json")
        return launch_for(record)
    except Exception:  # noqa: BLE001 - tuning is an optimization only
        return base_launch


def _bench_candidates(build, candidates, args, bench):
    """Interleaved-round benchmarking reusing the shared autotune harness."""

    from ..runtime.stax_autotune import bench_candidates

    return bench_candidates(build, candidates, args, bench_fn=bench)
