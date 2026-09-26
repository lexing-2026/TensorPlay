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


def _standard_3d(shape: Tuple[int, ...], stride: Tuple[int, ...]) -> bool:
    """Row- or column-major batched layout: one unit-stride axis, no overlap.

    The batch axis has to be a real stride, not a broadcast of one matrix over
    the batch: a shared leading operand is worth a kernel of its own, and a
    launcher that accepted it here would compute a product of one batch and
    store it as if it were all of them.
    """

    if len(shape) != 3 or len(stride) != 3:
        return False
    if shape[0] == 0 or shape[1] == 0 or shape[2] == 0:
        return False
    plane = shape[1] * shape[2]
    unit_row = stride[2] == 1 and stride[1] >= shape[2] and stride[0] >= plane
    unit_col = stride[1] == 1 and stride[0] >= shape[1] and stride[2] >= plane
    return unit_row or unit_col


def _broadcast_3d(shape: Tuple[int, ...], stride: Tuple[int, ...]) -> bool:
    """Batched layout whose batch axis is one matrix spread over the batch.

    A zero stride on the batch axis is what makes the matrix shared, and it is
    also what makes the two layouts here different from the general batched
    one: a caller that does not mean it gets the general kernel, because a
    shared matrix read once per batch is a different amount of traffic than the
    caller asked for.
    """

    if len(shape) != 3 or len(stride) != 3:
        return False
    if shape[0] == 0 or shape[1] == 0 or shape[2] == 0:
        return False
    if stride[0] != 0:
        return False
    unit_row = stride[2] == 1 and stride[1] >= shape[2]
    unit_col = stride[1] == 1 and stride[2] >= shape[1]
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
    M: int, K: int, N: int, dtype: Any, device: Any, bias: bool = False,
    batch: int = 1,
) -> list:
    """Deterministic operand set exercising every load/store lane.

    A linear ramp keeps values in a small band so the fp32 accumulation of
    the reference and the candidate stay comparable, while still making any
    stride or masking bug produce a grossly wrong product.  With ``bias``
    the feed carries the linear form: the (N, K) weight stand-in (the
    launch transposes it) and a length-N bias ramp.  With ``batch`` above one
    the same ramp covers the whole batch, so a batched product measures on a
    feed whose every matrix is distinct and a kernel that reads one of them
    for all of them is caught.
    """

    import tensorplay as tp

    # The element type arrives as the name the graph uses, and the tensor
    # interface wants the type itself: a ramp built at the default width would
    # have to be converted afterwards, which is a pass the probe does not need.
    element = getattr(tp, dtype) if isinstance(dtype, str) else dtype

    def ramp(rows: int, cols: int) -> Any:
        flat = tp.arange(rows * cols, device=device).to(element)
        return flat * 3.17e-4 - 0.5

    if bias:
        return [
            ramp(M, K).reshape(M, K),
            ramp(N, K).reshape(N, K),
            tp.arange(N, device=device).to(element) * 3.17e-4 - 0.5,
        ]
    if int(batch) != 1:
        return [
            ramp(int(batch) * M, K).reshape(int(batch), M, K),
            ramp(int(batch) * K, N).reshape(int(batch), K, N),
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
    # A tile epilogue arrives as the IR's own expression, not as instructions:
    # What the store writes is a value, not a branch: the accumulator as it
    # stands.  A chain fused into the tile would be printed by the group's own
    # emitter and substituted here the same way, which is why the kernel lives
    # in a file and this is a rendering rather than a concatenation.
    store_source = "acc"

    from ..templates.select_algorithm import TritonTemplate

    source = TritonTemplate.from_file(
        "mm", file="triton_mm", symbol="_gemm_epi_kernel"
    ).render(
        allow_tf32=bool(allow_tf32),
        has_bias=bool(has_bias),
        store_source=store_source,
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


def _as_tile_config(config) -> Optional[Tuple[int, int, int, int, int]]:
    """A tile shape, whether it was written as a tuple or as block names.

    A template names a tile by the blocks it is made of, because that is how a
    configuration is written down; the kernel wants them in the order it walks
    them.
    """

    if config is None:
        return None
    if isinstance(config, dict):
        try:
            return (
                int(config["BLOCK_M"]), int(config["BLOCK_N"]), int(config["BLOCK_K"]),
                int(config["num_warps"]), int(config["num_stages"]),
            )
        except KeyError:
            return None
    values = tuple(int(v) for v in config)
    return values if len(values) == 5 else None


def _triton_launch_factory(
    a_spec: Tuple[Optional[int], Any],
    b_spec: Tuple[Optional[int], Any],
    M: int, N: int, K: int,
    config: Tuple[int, int, int, int, int],
    base_launch: Callable[[list], Any],
    allow_tf32: bool = False,
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
    if not has_bias:
        kernel = _gemm_kernel
    else:
        kernel = _fused_gemm_kernel(has_bias, None, config, allow_tf32)

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
        if has_bias:
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
    epilogue_launch: Optional[Callable[[list], Any]] = None,
    bias_spec: Optional[Tuple[Optional[int], Any]] = None,
    b_transposed: bool = False,
    config=None,
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

    if not HAS_TRITON or len(out_shape) != 2:
        return None
    if not tp.cuda.is_available():
        return None

    if epilogue_launch is None:
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
    # A caller that has already chosen names the tile it wants.  Building it as
    # it stands is the whole job then: there is nothing left to measure here,
    # because the comparison against the operator is what the caller's own
    # candidates are for.
    tile = _as_tile_config(config)
    if tile is not None:
        return _triton_launch_factory(
            operand_specs[0], operand_specs[1],
            M, N, K, tile, native_launch,
            allow_tf32=allow_tf32,
            bias_spec=bias_spec,
            b_transposed=b_transposed,
        )
    cache_key = _decision_key(
        M, N, K, str(a.dtype), device_key, allow_tf32,
        None, bias_spec is not None, b_transposed,
    )

    try:
        from ..kernel_cache import default_cache

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


#: Salt for a persisted decision: bumped when the kernel body changes, so a
#: stored choice cannot outlive the kernel it named.
PERSISTENT_MM_TUNING_VERSION = "persistent-mm-1"

_PERSISTENT_MM_MEMO: dict[str, Any] = {}


def persistent_mm_kernel(block_m: int, block_n: int, block_k: int,
                         group_m: int, even_k: bool, precision: str):
    """The swept body for one form, built once and remembered.

    The body is a file rather than a string because it is long enough that
    keeping it inline makes the code and the parts that vary hard to tell apart.
    """

    key = hashlib.sha256(
        "|".join(str(v) for v in (
            PERSISTENT_MM_TUNING_VERSION, block_m, block_n, block_k, group_m,
            even_k, precision,
        )).encode()
    ).hexdigest()[:16]
    cached = _PERSISTENT_MM_MEMO.get(key)
    if cached is not None:
        return cached
    from ..templates.select_algorithm import KernelArgs, TritonTemplate

    plane = (0, 0)
    source = TritonTemplate.from_file(
        "persistent_mm", file="triton_persistent_mm", symbol="_gemm_persistent_kernel",
    ).render_with(
        KernelArgs(
            {"A": {"shape": plane, "stride": plane},
             "B": {"shape": plane, "stride": plane}},
            {"C": {"shape": plane, "stride": plane}},
            {
                "NUM_SMS": 1, "GROUP_M": int(group_m),
                "BLOCK_M": int(block_m), "BLOCK_N": int(block_n),
                "BLOCK_K": int(block_k),
                "EVEN_K": bool(even_k),
                "USE_FAST_ACCUM": True,
            },
        ),
        precision=precision,
    )
    fake_file = f"<tensorplay-stax-persistent-mm-{key}>"
    linecache.cache[fake_file] = (
        len(source), None, source.splitlines(True), fake_file,
    )
    namespace: dict[str, Any] = {"triton": triton, "tl": tl}
    exec(compile(source, fake_file, "exec"), namespace, namespace)
    kernel = namespace["_gemm_persistent_kernel"]
    _PERSISTENT_MM_MEMO[key] = kernel
    return kernel


def persistent_matmul_launch(
    base_launch: Callable[[list], Any],
    operand_specs: Tuple[Tuple[Optional[int], Any], Tuple[Optional[int], Any]],
    out_shape: Tuple[int, ...],
    config: dict,
    *,
    num_sms: int = 1,
    group_m: int = 8,
    bias_spec: Optional[Tuple[Optional[int], Any]] = None,
    b_transposed: bool = False,
    allow_tf32: Optional[bool] = None,
):
    """Build a launch running the swept form, or the plain one when it will not do.

    A call whose real layout is not the one the kernel bakes in takes
    ``base_launch`` instead, which is what keeps a launcher from being a promise
    about a call it has not seen.
    """

    import tensorplay as tp

    block_m = int(config["BLOCK_M"])
    block_n = int(config["BLOCK_N"])
    block_k = int(config["BLOCK_K"])
    num_warps = int(config["num_warps"])
    num_stages = int(config["num_stages"])
    tf32 = _matmul_allow_tf32() if allow_tf32 is None else bool(allow_tf32)

    def operand(feed: list, spec: Tuple[Optional[int], Any]) -> Any:
        position, literal = spec
        return feed[position] if position is not None else literal

    def launch(feed: list) -> Any:
        a = operand(feed, operand_specs[0])
        b = operand(feed, operand_specs[1])
        if b_transposed:
            b = b.t()
        if len(out_shape) != 2 or not (
            _standard_2d(tuple(a.shape), tuple(a.stride()))
            and _standard_2d(tuple(b.shape), tuple(b.stride()))
        ):
            return base_launch(feed)
        m, n = int(a.shape[0]), int(b.shape[1])
        k = int(a.shape[1])
        if int(b.shape[0]) != k:
            return base_launch(feed)
        out = tp.empty((m, n), dtype=a.dtype, device=a.device)
        tiles = -(-m // block_m) * -(-n // block_n)
        grid = (min(int(num_sms), tiles), 1, 1)
        kernel = persistent_mm_kernel(
            block_m, block_n, block_k, int(group_m), k % block_k == 0,
            "tf32" if tf32 else "ieee",
        )
        kernel[grid](
            a, b, out,
            *(int(v) for v in a.shape), *(int(v) for v in b.shape),
            *(int(v) for v in out.shape),
            *(int(v) for v in a.stride()),
            *(int(v) for v in b.stride()),
            *(int(v) for v in out.stride()),
            int(num_sms), int(group_m), block_m, block_n, block_k,
            k % block_k == 0,
            num_warps=num_warps, num_stages=num_stages,
        )
        return out

    return launch


def _batched_product_launch(
    base_launch: Callable[[list], Any],
    operand_specs: Tuple[Tuple[Optional[int], Any], Tuple[Optional[int], Any]],
    out_shape: Tuple[int, ...],
    kernel: Any,
    grid: Callable[..., Tuple[int, int, int]],
    block: Tuple[Tuple[str, Any], ...],
    left_layout: Callable[[Tuple[int, ...], Tuple[int, ...]], bool],
    *,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Callable[[list], Any]:
    """The launch both batched forms share, given what each one calls a layout.

    ``block`` is the list of block extents in the order the kernel's signature
    declares them, and the argument list below is that same list: the extents
    and strides of every operand come first, in declaration order, and the
    block extents last.  Reading both off one list is what keeps a body that
    asks for an extent and the arguments it is handed from drifting apart.

    ``base_launch`` is the operator, and it is what a call the kernel does not
    cover runs on: an extents mismatch, a layout with an axis overlapping
    another, a batch the two operands disagree about.  A launcher that ran
    anyway would be a promise about a call it has not seen.
    """

    def operand(feed: list, spec: Tuple[Optional[int], Any]) -> Any:
        position, literal = spec
        return feed[position] if position is not None else literal

    def launch(feed: list) -> Any:
        import tensorplay as tp

        a = operand(feed, operand_specs[0])
        b = operand(feed, operand_specs[1])
        if len(out_shape) != 3:
            return base_launch(feed)
        shape_a = tuple(int(s) for s in a.shape)
        shape_b = tuple(int(s) for s in b.shape)
        if len(shape_a) != 3 or len(shape_b) != 3:
            return base_launch(feed)
        batch, m, k = shape_a
        inner, n = shape_b[1], shape_b[2]
        if (
            inner != k
            or shape_b[0] != batch
            or (batch, m, n) != tuple(int(v) for v in out_shape)
        ):
            return base_launch(feed)
        stride_a = tuple(int(s) for s in a.stride())
        stride_b = tuple(int(s) for s in b.stride())
        if not (
            left_layout(shape_a, stride_a) and _standard_3d(shape_b, stride_b)
        ):
            return base_launch(feed)
        out = tp.empty((batch, m, n), dtype=a.dtype, device=a.device)
        out_size = (batch, m, n)
        kernel[grid(batch, m, n, dict(block), cdiv=triton.cdiv)](
            a, b, out,
            *shape_a, *shape_b, *out_size,
            *stride_a, *stride_b, *tuple(int(s) for s in out.stride()),
            *(value for _name, value in block),
            num_warps=int(num_warps), num_stages=int(num_stages),
        )
        return out

    return launch


def batched_matmul_launch(
    base_launch: Callable[[list], Any],
    operand_specs: Tuple[Tuple[Optional[int], Any], Tuple[Optional[int], Any]],
    out_shape: Tuple[int, ...],
    kernel: Any,
    grid: Callable[..., Tuple[int, int, int]],
    block: Tuple[Tuple[str, Any], ...],
    *,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Callable[[list], Any]:
    """Build a launch running one fixed tile of a batched product.

    The batch is a third grid axis rather than a loop and rather than a factor
    in the tile count: a batch is a set of independent products, so the
    programs that would have run one after another run side by side instead,
    and the tiles are exactly the tiles the plain product would have had.
    """

    return _batched_product_launch(
        base_launch, operand_specs, out_shape, kernel, grid, block, _standard_3d,
        num_warps=num_warps, num_stages=num_stages,
    )


def shared_a_matmul_launch(
    base_launch: Callable[[list], Any],
    operand_specs: Tuple[Tuple[Optional[int], Any], Tuple[Optional[int], Any]],
    out_shape: Tuple[int, ...],
    kernel: Any,
    grid: Callable[..., Tuple[int, int, int]],
    block: Tuple[Tuple[str, Any], ...],
    *,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Callable[[list], Any]:
    """Build a launch running one fixed tile of the shared-left batched product.

    The batch is grouped rather than merely parallel: ``block``'s group extent
    is how many batches one loaded left tile serves, so the left operand is
    fetched once for the group instead of once for each batch in it.  The price
    is an accumulator that many times wider, which is what the table of tilings
    is bounded by.
    """

    return _batched_product_launch(
        base_launch, operand_specs, out_shape, kernel, grid, block, _broadcast_3d,
        num_warps=num_warps, num_stages=num_stages,
    )


def mm_plus_mm_launch(
    base_launch: Callable[[list], Any],
    operand_specs: Tuple[Tuple[Optional[int], Any], ...],
    out_shape: Tuple[int, ...],
    kernel: Any,
    grid: Callable[..., Tuple[int, int, int]],
    block: Tuple[Tuple[str, Any], ...],
    *,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Callable[[list], Any]:
    """Build a launch running one fixed tile of two products into one result.

    Both contractions are walked by the same program into the same accumulator
    and the result is stored once, so the two products are measured as one
    choice against being two launches of the plain product -- which is the only
    thing that makes the form worth having: it can lose, and it cannot be wrong.

    A configuration that says the contraction divides the tile is checked
    against the call rather than trusted: the kernel reads it to decide whether
    to guard its loads, and a guard that is missing is a number that is wrong in
    a way nothing downstream would notice.
    """

    named = dict(block)

    def operand(feed: list, spec: Tuple[Optional[int], Any]) -> Any:
        position, literal = spec
        return feed[position] if position is not None else literal

    def launch(feed: list) -> Any:
        import tensorplay as tp

        if len(operand_specs) != 4 or len(out_shape) != 2:
            return base_launch(feed)
        first, second, third, fourth = (
            operand(feed, spec) for spec in operand_specs
        )
        shape_a = tuple(int(s) for s in first.shape)
        shape_b = tuple(int(s) for s in second.shape)
        shape_c = tuple(int(s) for s in third.shape)
        shape_d = tuple(int(s) for s in fourth.shape)
        if not (
            len(shape_a) == 2 and len(shape_b) == 2
            and len(shape_c) == 2 and len(shape_d) == 2
        ):
            return base_launch(feed)
        m, inner_a = shape_a
        inner_b, n = shape_b
        rows_c, inner_c = shape_c
        inner_d, cols_d = shape_d
        # The two products have to be the same shape: the kernel indexes both
        # with one tile and one accumulator, so a pair that differs is two
        # products rather than this one.
        if (
            inner_a != inner_b
            or (rows_c, inner_c) != (m, inner_a)
            or (inner_d, cols_d) != (inner_a, n)
            or (m, n) != tuple(int(v) for v in out_shape)
        ):
            return base_launch(feed)
        strides = tuple(
            tuple(int(s) for s in tensor.stride())
            for tensor in (first, second, third, fourth)
        )
        if not all(
            _standard_2d(shape, stride)
            for shape, stride in zip(
                (shape_a, shape_b, shape_c, shape_d), strides
            )
        ):
            return base_launch(feed)
        if named["EVEN_K"] and inner_a % int(named["BLOCK_K"]):
            return base_launch(feed)
        out = tp.empty((m, n), dtype=first.dtype, device=first.device)
        out_stride = tuple(int(s) for s in out.stride())
        kernel[grid(m, n, dict(block), cdiv=triton.cdiv)](
            first, second, third, fourth, out,
            *shape_a, *shape_b, *shape_c, *shape_d, m, n,
            *strides[0], *strides[1], *strides[2], *strides[3], *out_stride,
            *(value for _name, value in block),
            num_warps=int(num_warps), num_stages=int(num_stages),
        )
        return out

    return launch


def grouped_matmul_launch(
    base_launch: Callable[[list], Any],
    operand_specs: Tuple[Tuple[Optional[int], Any], ...],
    out_shape: Tuple[int, ...],
    kernel: Any,
    grid: Callable[..., Tuple[int, int, int]],
    block: Tuple[Tuple[str, Any], ...],
    fetch: dict,
    *,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Callable[[list], Any]:
    """Build a launch running one fixed tile over a number of grouped products.

    The operands arrive as the kernel's signature declares them, which is the
    two matrices, then the factors if the call has them, then the group
    boundaries if the groups are cut out of one matrix -- so the argument list
    below reads the values off ``operand_specs`` in order instead of naming
    them, and a form that takes more operands takes them in the order it
    declares.

    ``fetch`` says how the body was built to read a tile: by descriptor or by
    address, and which way round each matrix's two axes are.  A descriptor is a
    mapping stated once, so a matrix handed over in the other layout is not
    read a little differently -- it is read wrong -- so the layout the body was
    built for is checked here rather than remembered.
    """

    from ..templates.mm_common import (
        descriptor_extents_fit, descriptor_offset_fits,
    )

    def operand(feed: list, spec: Tuple[Optional[int], Any]) -> Any:
        position, literal = spec
        return feed[position] if position is not None else literal

    def launch(feed: list) -> Any:
        import tensorplay as tp

        if len(operand_specs) < 2:
            return base_launch(feed)
        values = [operand(feed, spec) for spec in operand_specs]
        # What came after the two matrices is read off how many there are: the
        # factors are a pair, the boundaries are a single vector, and the order
        # they were declared in is the order they are in.
        if len(values) not in (2, 3, 4, 5):
            return base_launch(feed)
        scaled = len(values) >= 4
        boundaries = len(values) in (3, 5)
        first, second = values[0], values[1]
        shape_a = tuple(int(s) for s in first.shape)
        shape_b = tuple(int(s) for s in second.shape)
        if len(shape_a) not in (2, 3) or len(shape_b) not in (2, 3):
            return base_launch(feed)
        a_is_2d = len(shape_a) == 2
        b_is_2d = len(shape_b) == 2
        if a_is_2d:
            m, k = shape_a
        else:
            _groups, m, k = shape_a
        if b_is_2d:
            inner, n = shape_b
        else:
            _groups_b, inner, n = shape_b
        if inner != k:
            return base_launch(feed)
        if boundaries:
            # One operand is one matrix the groups are cut out of, so the
            # boundaries say where the cuts fall and how many there are.  They
            # have to be a vector of whole numbers: the kernel adds and
            # subtracts them to find a group's extents, and a fractional
            # boundary would index the operand with it.
            bounds = values[-1]
            if len(tuple(bounds.shape)) != 1 or "int" not in str(bounds.dtype):
                return base_launch(feed)
            groups = int(bounds.shape[0])
            if not a_is_2d and int(first.shape[0]) != groups:
                return base_launch(feed)
            if not b_is_2d and int(second.shape[0]) != groups:
                return base_launch(feed)
        else:
            if not a_is_2d and not b_is_2d and shape_a[0] != shape_b[0]:
                return base_launch(feed)
            groups = shape_a[0] if not a_is_2d else shape_b[0]
        size = tuple(int(v) for v in out_shape)
        # The result is batched exactly when the two operands agree about being
        # batched: a group boundary then indexes the result itself.
        expected = (m, n) if a_is_2d != b_is_2d else (groups, m, n)
        if size != expected:
            return base_launch(feed)
        # A factor is one value per row, and how many rows it covers follows
        # from which axis the groups are cut out along.  Cut along the rows (or
        # the columns) the whole matrix is walked once, so the factor is a run
        # over the whole of it.  Cut along the contraction instead, the rows are
        # numbered afresh per group and a group is addressed whole, so the run is
        # over every group.  A matrix given per group keeps the group as an axis
        # of its own factor.
        if scaled:
            for factor, is_2d, other_is_2d, rows in (
                (values[2], a_is_2d, b_is_2d, m),
                (values[3], b_is_2d, a_is_2d, n),
            ):
                if not is_2d:
                    want = (groups, rows)
                elif not other_is_2d:
                    want = (rows,)
                else:
                    want = (groups * rows,)
                if tuple(int(v) for v in factor.shape) != want:
                    return base_launch(feed)
        stride_a = tuple(int(v) for v in first.stride())
        stride_b = tuple(int(v) for v in second.stride())
        if not (
            _standard_2d(shape_a[-2:], stride_a[-2:])
            and _standard_2d(shape_b[-2:], stride_b[-2:])
            and (a_is_2d or stride_a[0] >= m * k)
            and (b_is_2d or stride_b[0] >= k * n)
        ):
            return base_launch(feed)
        if fetch.get("USE_TMA_LOAD") and not (
            (int(stride_a[-1]) == 1) == bool(fetch.get("A_IS_K_MAJOR"))
            and (int(stride_b[-2]) == 1) == bool(fetch.get("B_IS_K_MAJOR"))
        ):
            # The body built a mapping for the layout it was rendered against,
            # and this call's matrices are laid out the other way round: the
            # descriptor would name the wrong elements rather than read them
            # slowly.
            return base_launch(feed)
        if fetch.get("USE_TMA_LOAD") and not all(
            descriptor_extents_fit(shape) and descriptor_offset_fits(shape, stride)
            for shape, stride in ((shape_a, stride_a), (shape_b, stride_b))
        ):
            # A descriptor addresses in 32 bits, so an operand it cannot name is
            # one it cannot describe.
            return base_launch(feed)
        out = tp.empty(size, dtype=first.dtype, device=first.device)
        kernel[grid(dict(block))](
            *values, out,
            *(int(v) for tensor in values for v in tensor.shape),
            *size,
            *(int(v) for tensor in values for v in tensor.stride()),
            *(int(v) for v in out.stride()),
            *(value for _name, value in block),
            num_warps=int(num_warps), num_stages=int(num_stages),
        )
        return out

    return launch

