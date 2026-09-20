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
GEMM_TUNING_VERSION = "gemm-v3"

_DECISION_NAMESPACE = "triton-autotune"

# Head of the bias/epilogue-fused tile kernel: identical arithmetic to
# ``_gemm_kernel`` up to the accumulator; an optional row-broadcast bias
# load, the chain's instruction lines and the final store splice in after
# the k-loop.  The tail work runs on the accumulator registers, so the
# unfused product never touches memory.
_GEMM_EPI_TEMPLATE_HEAD = '''\
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _gemm_epi_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    EVEN_K: tl.constexpr, ALLOW_TF32: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Bias-and-epilogue fused C = chain(A @ B) with fp32 accumulation.

    TF32 shortens the multiplier datapath when ``ALLOW_TF32`` is set (the
    caller's global matmul switch); the accumulated sum stays fp32 either
    way.  Output rows/cols wrap with ``% M``/``% N`` so the store needs no
    separate bounds mask (wrapped lanes recompute identical values); loads
    guard the k tail unless ``EVEN_K``.  ``HAS_BIAS`` adds the length-N
    column bias to every row of the accumulator tile.
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
        if ALLOW_TF32:
            acc = tl.dot(a, b, acc, input_precision="tf32")
        else:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    if HAS_BIAS:
        acc = acc + tl.load(bias_ptr + off_n)[None, :]
'''

_GEMM_EPI_TEMPLATE_STORE = '''\
    c_ptrs = c_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
    tl.store(c_ptrs, {store_source})
'''

_EPI_KERNEL_MEMO: dict = {}

if HAS_TRITON:

    @triton.jit
    def _gemm_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        EVEN_K: tl.constexpr, ALLOW_TF32: tl.constexpr,
    ):
        """C = A(M,K) @ B(K,N) with fp32 accumulation.

        TF32 shortens the multiplier datapath when ``ALLOW_TF32`` is set:
        the accumulated sum stays fp32 either way, and the reduced mantissa
        precision is the trade the caller opted into through the global
        matmul switch.  Output rows/cols wrap with ``% M``/``% N`` so the
        store needs no separate bounds mask; loads guard the k tail unless
        EVEN_K.
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
            if ALLOW_TF32:
                acc = tl.dot(a, b, acc, input_precision="tf32")
            else:
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


def _matmul_allow_tf32() -> bool:
    """The global fp32-matmul precision switch, gated on hardware support.

    TF32 tiles only enter the candidate table when the device executes them
    natively; the same switch governs the native cuBLAS floor, so candidate
    and floor stay on one precision footing either way.
    """

    import tensorplay as tp

    try:
        if not tp.cuda.is_tf32_supported():
            return False
        from tensorplay.backends import cuda as cuda_backends

        return bool(cuda_backends.matmul.allow_tf32)
    except Exception:  # noqa: BLE001 - precision is an opt-in only
        return False


def _decision_key(
    M: int,
    N: int,
    K: int,
    dtype: str,
    device: str,
    allow_tf32: bool,
    epilogue: Optional[Tuple[list[int], list[float], int]] = None,
    bias: bool = False,
    b_transposed: bool = False,
) -> str:
    source = (
        f"gemm|{_kernel_source_digest()}|{M}|{N}|{K}|{dtype}|{device}"
        f"|tf32={int(allow_tf32)}"
    )
    if epilogue is not None:
        program, constants, esrc = epilogue
        source += "|epi=" + hashlib.sha256(
            repr((tuple(program), tuple(constants), esrc)).encode()
        ).hexdigest()[:12]
    if bias or b_transposed:
        # the linear form: a transposed weight operand plus an optional
        # length-N bias — never share a record with the plain matmul
        source += f"|lin={int(bias)}{int(b_transposed)}"
    return hashlib.sha256(source.encode()).hexdigest()[:24]


def _probe_feed(
    M: int, K: int, N: int, dtype: Any, device: Any, bias: bool = False
) -> list:
    """Deterministic operand set exercising every load/store lane.

    A linear ramp keeps values in a small band so the fp32 accumulation of
    the reference and the candidate stay comparable, while still making any
    stride or masking bug produce a grossly wrong product.  With ``bias``
    the feed carries the linear form: the (N, K) weight stand-in (the
    launch transposes it) and a length-N bias ramp.
    """

    import tensorplay as tp

    def ramp(rows: int, cols: int) -> Any:
        flat = tp.arange(rows * cols, dtype=dtype, device=device)
        return flat * 3.17e-4 - 0.5

    if bias:
        return [
            ramp(M, K).reshape(M, K),
            ramp(N, K).reshape(N, K),
            tp.arange(N, dtype=dtype, device=device) * 3.17e-4 - 0.5,
        ]
    return [ramp(M, K).reshape(M, K), ramp(K, N).reshape(K, N)]


def _fused_gemm_kernel(
    has_bias: bool,
    epilogue: Optional[Tuple[list[int], list[float], int]],
    config: Tuple[int, int, int, int, int],
    allow_tf32: bool,
):
    """JIT the bias/epilogue-fused tile kernel for one form and config.

    The optional bias load, the chain's instruction lines and the final
    store are spliced after the k-loop; the memo keeps one JITFunction per
    (form, config, precision) so repeated candidate launches reuse the
    compiled binary.
    """

    key = hashlib.sha256(
        (
            GEMM_TUNING_VERSION
            + f"|bias={int(has_bias)}"
            + (repr((tuple(epilogue[0]), tuple(epilogue[1]), epilogue[2]))
               if epilogue is not None else "-")
            + repr(config)
            + repr(allow_tf32)
        ).encode()
    ).hexdigest()[:16]
    cached = _EPI_KERNEL_MEMO.get(key)
    if cached is not None:
        return cached
    if epilogue is None:
        lines, store_source = [], "acc"
    else:
        from .triton import emit_tile_epilogue_lines

        lines, store_source = emit_tile_epilogue_lines(
            list(epilogue[0]), list(epilogue[1]), epilogue[2], "acc"
        )
    source = (
        _GEMM_EPI_TEMPLATE_HEAD
        + "\n".join("    " + line for line in lines)
        + "\n"
        + _GEMM_EPI_TEMPLATE_STORE.format(store_source=store_source)
    )
    import linecache

    fake_file = f"<tensorplay-stax-gemm-epi-{key}>"
    # The jit decorator reads the decorated function's source through the
    # linecache, so the generated text must be registered under the
    # compile-time filename before the exec that defines it.
    linecache.cache[fake_file] = (
        len(source),
        None,
        source.splitlines(True),
        fake_file,
    )
    namespace: dict[str, Any] = {"triton": triton, "tl": tl}
    exec(compile(source, fake_file, "exec"), namespace, namespace)
    kernel = namespace["_gemm_epi_kernel"]
    _EPI_KERNEL_MEMO[key] = kernel
    return kernel


def _triton_launch_factory(
    a_spec: Tuple[Optional[int], Any],
    b_spec: Tuple[Optional[int], Any],
    M: int, N: int, K: int,
    config: Tuple[int, int, int, int, int],
    base_launch: Callable[[list], Any],
    allow_tf32: bool = False,
    epilogue: Optional[Tuple[list[int], list[float], int]] = None,
    bias_spec: Optional[Tuple[Optional[int], Any]] = None,
    b_transposed: bool = False,
):
    """Build a launch closure running one fixed GEMM tile configuration.

    Each operand spec is ``(feed position, literal tensor)`` — exactly one
    side is set: graph operands come from the feed, constants (module
    weights captured as literals) are closed over.  ``b_transposed`` marks
    the ``linear`` layout (B arrives as the (N, K) weight; the kernel
    consumes its zero-copy transposed view).  With ``bias_spec`` or
    ``epilogue`` the generated tile adds the row-broadcast bias and applies
    the pointwise chain to the accumulator before the store;
    ``base_launch`` (the full region fallback) still covers non-standard
    operand layouts at call time.
    """

    block_m, block_n, block_k, num_warps, num_stages = config
    has_bias = bias_spec is not None
    fused = has_bias or epilogue is not None
    if not fused:
        kernel = _gemm_kernel
    else:
        kernel = _fused_gemm_kernel(has_bias, epilogue, config, allow_tf32)

    def operand(feed: list, spec: Tuple[Optional[int], Any]) -> Any:
        position, literal = spec
        return feed[position] if position is not None else literal

    def launch(feed: list) -> Any:
        import tensorplay as tp

        a = operand(feed, a_spec)
        b = operand(feed, b_spec)
        if b_transposed:
            b = b.t()
        bias = operand(feed, bias_spec) if has_bias else a
        if not (
            _standard_2d(tuple(a.shape), tuple(a.stride()))
            and _standard_2d(tuple(b.shape), tuple(b.stride()))
            and (
                not has_bias
                or (
                    bias.dim() == 1
                    and int(bias.shape[0]) == N
                    and bias.is_contiguous()
                )
            )
        ):
            # The baked kernel assumes standard 2-D layouts (and a plain
            # length-N bias); a differently strayed call takes the
            # fallback launch instead.
            return base_launch(feed)
        out = tp.empty((M, N), dtype=a.dtype, device=a.device)
        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
        if fused:
            kernel[grid](
                a, b, bias, out, M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                out.stride(0), out.stride(1),
                BM=block_m, BN=block_n, BK=block_k,
                EVEN_K=(K % block_k == 0),
                ALLOW_TF32=allow_tf32,
                HAS_BIAS=has_bias,
                num_warps=num_warps, num_stages=num_stages,
            )
        else:
            kernel[grid](
                a, b, out, M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                out.stride(0), out.stride(1),
                BM=block_m, BN=block_n, BK=block_k,
                EVEN_K=(K % block_k == 0),
                ALLOW_TF32=allow_tf32,
                num_warps=num_warps, num_stages=num_stages,
            )
        return out

    return launch


def tuned_matmul_launch(
    base_launch: Callable[[list], Any],
    sample_feed: Sequence[Any],
    operand_specs: Tuple[Tuple[Optional[int], Any], Tuple[Optional[int], Any]],
    out_shape: Tuple[int, ...],
    *,
    epilogue: Optional[Tuple[list[int], list[float], int]] = None,
    epilogue_launch: Optional[Callable[[list], Any]] = None,
    bias_spec: Optional[Tuple[Optional[int], Any]] = None,
    b_transposed: bool = False,
) -> Optional[Callable[[list], Any]]:
    """Benchmark native vs Triton GEMM for one matmul-family extern segment.

    Each operand spec is ``(feed position, literal tensor)`` with exactly
    one side set; ``sample_feed`` supplies the stand-in tensor for every
    feed position.  Returns the winner-baked launch, or ``None`` when this
    segment does not qualify (non-2D, non-fp32, non-CUDA, exotic layouts)
    or the native operator wins.  ``base_launch`` remains the floor: it is
    one of the benched candidates and the fallback for unbenchable inputs.

    With ``epilogue`` (a ``(program, constants, esrc)`` chain replaying on
    the matmul output), the Triton candidates fuse the chain into the tile
    kernel and the native side composes ``base_launch`` (the bare operator)
    with ``epilogue_launch`` (the chain as its own kernel) — every benched
    candidate runs the full region, so the comparison stays fair.
    ``bias_spec``/``b_transposed`` select the ``linear`` form: the B operand
    arrives as the (N, K) weight (the launch consumes its transposed view)
    and an optional length-N bias joins the accumulator before the chain.
    """

    import tensorplay as tp

    if epilogue is not None and epilogue_launch is None:
        # No composed chain kernel: the caller keeps its own launch.
        return None
    if not HAS_TRITON or len(out_shape) != 2:
        return None
    if not tp.cuda.is_available():
        return None

    if epilogue is None:
        native_launch = base_launch
    else:

        def native_launch(
            feed: list,
            _base: Callable[[list], Any] = base_launch,
            _epi: Callable[[list], Any] = epilogue_launch,
        ):
            return _epi([_base(feed)])

    def operand_tensor(spec: Tuple[Optional[int], Any]) -> Any:
        position, literal = spec
        return literal if position is None else sample_feed[position]

    a = operand_tensor(operand_specs[0])
    b = operand_tensor(operand_specs[1])
    bias = operand_tensor(bias_spec) if bias_spec is not None else None
    if a is None or b is None:
        return None
    if str(a.dtype) != "tensorplay.float32" or str(b.dtype) != "tensorplay.float32":
        return None
    if bias is not None and str(bias.dtype) != "tensorplay.float32":
        return None
    if not a.device.is_cuda():
        return None
    shape_a, shape_b = tuple(a.shape), tuple(b.shape)
    if b_transposed:
        # the linear layout: B is the (N, K) weight; the kernel consumes
        # its zero-copy transposed view
        if len(shape_a) != 2 or len(shape_b) != 2 or shape_a[1] != shape_b[1]:
            return None
        M, K = shape_a[0], shape_a[1]
        N = shape_b[0]
        if not _standard_2d(
            (shape_b[1], shape_b[0]), (b.stride(1), b.stride(0))
        ):
            return None
    else:
        if len(shape_a) != 2 or len(shape_b) != 2 or shape_a[1] != shape_b[0]:
            return None
        M, K = shape_a
        N = shape_b[1]
        if not _standard_2d(shape_b, tuple(b.stride())):
            return None
    if not _standard_2d(shape_a, tuple(a.stride())):
        return None
    if bias is not None and (
        bias.dim() != 1
        or int(bias.shape[0]) != N
        or not bias.is_contiguous()
    ):
        return None

    device_key = repr(a.device)
    allow_tf32 = _matmul_allow_tf32()
    cache_key = _decision_key(
        M, N, K, str(a.dtype), device_key, allow_tf32,
        epilogue, bias_spec is not None, b_transposed,
    )

    try:
        from ..codecache import default_cache

        cache = default_cache(_DECISION_NAMESPACE)
    except Exception:  # noqa: BLE001 - tuning is an optimization only
        return None

    def launch_for(choice: dict) -> Callable[[list], Any]:
        if choice.get("choice") != "triton":
            return native_launch
        config = (
            int(choice["bm"]), int(choice["bn"]), int(choice["bk"]),
            int(choice["warps"]), int(choice["stages"]),
        )
        return _triton_launch_factory(
            operand_specs[0], operand_specs[1],
            M, N, K, config, native_launch,
            allow_tf32=bool(choice.get("tf32", False)),
            epilogue=epilogue,
            bias_spec=bias_spec,
            b_transposed=b_transposed,
        )

    payload = cache.load(cache_key, ext="json")
    if payload is not None:
        try:
            return launch_for(json.loads(payload.decode()))
        except (ValueError, KeyError, TypeError):
            pass

    candidates: list = [("native",)]
    candidates.extend(("triton", cfg) for cfg in GEMM_CANDIDATE_CONFIGS)

    probe = _probe_feed(
        M, K, N, a.dtype, a.device, bias=bias_spec is not None
    )

    def build(candidate):
        if candidate[0] == "native":
            return native_launch
        return _triton_launch_factory(
            operand_specs[0], operand_specs[1],
            M, N, K, candidate[1], native_launch,
            allow_tf32=allow_tf32,
            epilogue=epilogue,
            bias_spec=bias_spec,
            b_transposed=b_transposed,
        )

    def bench(launch: Any, args: list) -> float:
        from ..runtime.stax_autotune import bench_launch

        return bench_launch(launch, args)

    # TF32 tiles trade mantissa precision for tensor-core throughput, so the
    # gate against the native floor widens accordingly; anything grossly
    # wrong still loses to the native operator.
    tolerance = (2e-2, 2e-2) if allow_tf32 else (1e-4, 1e-3)

    try:
        best, best_launch, _ = _bench_candidates(build, candidates, probe, bench)
        if best is None or best[0] == "native":
            record = {"choice": "native", "tf32": allow_tf32}
        else:
            reference = native_launch(probe)
            produced = best_launch(probe)
            if not tp.allclose(produced, reference, rtol=tolerance[0], atol=tolerance[1]):
                record = {"choice": "native", "tf32": allow_tf32}
                best_launch = native_launch
            else:
                cfg = best[1]
                record = {
                    "choice": "triton",
                    "bm": cfg[0], "bn": cfg[1], "bk": cfg[2],
                    "warps": cfg[3], "stages": cfg[4],
                    "tf32": allow_tf32,
                }
        cache.store(cache_key, json.dumps(record).encode(), ext="json")
        return launch_for(record)
    except Exception:  # noqa: BLE001 - tuning is an optimization only
        return native_launch


def _bench_candidates(build, candidates, args, bench):
    """Interleaved-round benchmarking reusing the shared autotune harness."""

    from ..runtime.stax_autotune import bench_candidates

    return bench_candidates(build, candidates, args, bench_fn=bench)
