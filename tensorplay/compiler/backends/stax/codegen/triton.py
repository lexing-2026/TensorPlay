"""Internal Triton code generation selected by the Stax backend.

Triton is deliberately not a second public TensorPlay compiler backend.  The
frontend and backend contract remain ``tensorplay.compile(..., backend='stax')``;
Stax selects this code generator for eligible CUDA pointwise groups.  Training
uses the local ahead-of-time split: Stax receives a forward program and a
separately compiled reverse-mode program, then the runtime Function only
assembles those two compiled artifacts for autograd.
"""

from __future__ import annotations

import hashlib
import json
import linecache
import operator
import textwrap
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised on CPU-only installs
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False

# Index reductions emit ``tl.argmax``; older Triton releases lack it, so the
# folding detector treats absence as "op not available" instead of failing at
# kernel-compile time (backend failures are hard compiler errors).
HAS_TL_ARGMAX = HAS_TRITON and hasattr(tl, "argmax")
# Pair reductions over the lowest value (min(dim)) carry their index stream
# with ``tl.argmin``; same availability contract as argmax.
HAS_TL_ARGMIN = HAS_TRITON and hasattr(tl, "argmin")

from tensorplay.graph import Graph, GraphModule, Node
from tensorplay.graph._utils import _map_arg
from ..scheduler import annotate as scheduler_annotate
from ..scheduler import segment_graph

# Salt for every content-addressed kernel cache key.  Bump whenever the
# EMITTER changes semantics (masking, NaN handling, launcher allocation,
# load cache annotations, new fused opcodes) so stale generated sources
# cannot be replayed against a new compiler.
_CODEGEN_VERSION = "m10-2026-09-20-loop-pass-pair"
from ..backend import (
    _CPU_FUSED_AUTOGRAD_OPS,
    _CPU_FUSED_OPS,
    _CAST_DTYPE_IDS,
    _TRITON_OPCODES,
    _build_fused_gradient_graphs,
    _build_pointwise_program,
    _nodes,
    _normalize_pointwise_grad_output,
    _target_name,
)
from ..runtime.stax_autotune import disabled as disabled_autotune

# Process-level memo of exec'd launch callables (L5-M1), keyed by
# "<digest>:<fixed_config>".
_launch_memo: dict[str, Any] = {}


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float))


def _scalar_source(value: Any) -> str:
    if not _is_scalar(value):
        raise TypeError(f"Triton pointwise scalar must be numeric, got {type(value)!r}")
    if isinstance(value, bool):
        return "1.0" if value else "0.0"
    return repr(float(value))


def _prod(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


# Fused-cast dtype ids (shared with stax._CAST_DTYPE_IDS) to Triton types.
_CAST_SOURCE = {
    1: "tl.float16",
    2: "tl.bfloat16",
    3: "tl.float32",
    4: "tl.float64",
}


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def _broadcast_reference_shape(
    shapes: list[tuple[int, ...]],
) -> tuple[int, ...] | None:

    rank = max(len(shape) for shape in shapes)
    reference = [1] * rank
    for shape in shapes:
        pad = rank - len(shape)
        for index, dim in enumerate(shape):
            pos = pad + index
            if dim == 1 or reference[pos] == dim:
                continue
            if reference[pos] != 1:
                return None
            reference[pos] = dim
    return tuple(reference)


# A full-sum epilogue whose input fits one block is emitted as a single
# kernel writing the scalar directly; larger inputs take the two-stage split
# (partial sums to a workspace, then a tiny finalize kernel), matching
# Multilayer split-reduction semantics in miniature.
_SINGLE_BLOCK_MAX = 1024

# Deterministic config used when autotuning is off/unavailable and the split
# path still needs a pinned XBLOCK (the workspace size is baked per config).
_STATIC_REDUCTION_CONFIG = (256, 4)
# Split-reduction candidates: classic (XBLOCK, warps) two-kernel form and
# persistent (XBLOCK, warps, NPROG) grid-stride triples.  The autotuner
# benches both families and keeps the winner per shape bucket.  The
# persistent geometries below are the ones that hold L2-resident bandwidth
# on Ada (16M fp32: ~21us vs ~40us for the classic form); 4096-wide tiles
# with 8 warps spill registers and 296-program/2048-lane grids underfill
# the SMs, so they are deliberately absent.
_SPLIT_CANDIDATES = (
    (256, 4),
    (512, 4),
    (1024, 8),
    (2048, 8),
    (1024, 8, 1184),
    (2048, 8, 592),
    (2048, 4, 592),
    (4096, 4, 296),
)
_STATIC_SPLIT_PERSISTENT = (2048, 8, 592)


def _single_block_config(numel: int) -> tuple[int, int]:
    block = max(_next_power_of_two(numel), 16)
    warps = max(1, min(4, block // 256))
    return block, warps


# Per-chunk accumulator update emitted inside the axis-reduction r-loop
# (M5b): ``chunk`` is the RBLOCK-folded partial for the current tile row.
_ACC_UPDATE = {
    "sum": "acc + chunk",
    "mean": "acc + chunk",
    "amax": "tl.maximum(acc, chunk)",
    "max": "tl.maximum(acc, chunk)",
    "amin": "tl.minimum(acc, chunk)",
    "min": "tl.minimum(acc, chunk)",
}

# Value-stream dtype for index reductions (M5b dual-stream skeleton).  The
# accumulator must match the loaded tile dtype or tl.where/tl.argmax promote
# unpredictably; only types with verified numerics are foldable.  Keys are
# ``str(tp_dtype)`` spellings.
_VALUE_TYPES = {
    "tensorplay.float32": "tl.float32",
    "tensorplay.float64": "tl.float64",
}


def _dim_reduction_config(
    reference_shape: tuple[int, ...], spec: "ReductionSpec"
) -> tuple[int, int, int, int]:
    """Deterministic (XBLOCK, num_warps, RBLOCK, num_stages) for an axis reduction.

    The static default sits mid-table; ``_autotune_dims_launch`` benchmarks
    the full candidate set when tuning is enabled (M5d).
    """

    rank = len(reference_shape)
    reduced = {dim % rank for dim in spec.dims}
    out_sizes = [
        size for index, size in enumerate(reference_shape) if index not in reduced
    ]
    onumel = max(1, _prod(out_sizes))
    rnumel = max(1, spec.reduction_numel(reference_shape))
    xblock = min(max(_next_power_of_two(onumel), 16), 256)
    rblock = min(_next_power_of_two(rnumel), _PERSISTENT_RNUMEL_MAX)
    return xblock, 4, rblock, _DIM_NUM_STAGES


# Software-pipelining depth for the reduction r-loop: keeps the next chunk's
# loads in flight while the current one reduces (memory-bound kernels are
# latency-bound without it — the generated kernel exposes this depth).
_DIM_NUM_STAGES = 3

# Candidate table for axis-reduction autotuning.  Triples are
# (XBLOCK, num_warps, stages) with the shape-derived RBLOCK; quads are
# (XBLOCK, num_warps, RBLOCK, stages) and override RBLOCK for the
# few-output-lane / wide-reduction band where the derived 512 cap leaves
# the r-loop too shallow.  The XBLOCK*RBLOCK product is what bounds
# register pressure, so the quads trade grid parallelism for r-tile depth
# at a constant footprint.  The 16-warp quads use an inner contiguous band
# (XBLOCK 1-2, RBLOCK min(rnumel, 2048),
# num_warps = tile/128 — 4-8 elements/thread; see
# low-pressure band our 3-tuple geometries cannot reach because 512-deep
# tiles with <=8 warps spill.
_DIM_REDUCTION_CANDIDATES: tuple[tuple[int, ...], ...] = (
    (16, 4, 2),
    (32, 4, 2),
    (64, 4, 2),
    (128, 4, 3),
    (256, 4, 3),
    (256, 8, 3),
    (128, 8, 4),
    # bandwidth-bound shapes: few output lanes, wide pipelined r-tile
    (8, 4, 1024, 3),
    (16, 4, 1024, 3),
    (16, 8, 1024, 3),
    (4, 4, 2048, 2),
    (16, 8, 2048, 2),
    # Inner band: one/two output lanes per program, deep r-tile,
    # warps scaled to keep ~4-8 elements/thread
    (1, 16, 2048, 3),
    (2, 16, 2048, 3),
    (1, 16, 4096, 3),
)

_STATIC_DIM_TRIPLE = (128, 4, 3)

# Reductions whose entire space fits one tile skip the r-loop entirely
# (persistent-reduction shape): no loop-carried acc, one reduce.
_PERSISTENT_RNUMEL_MAX = 512


class ReductionSpec:
    """Structured description of a reduction epilogue (L5-M5b).

    ``op``    : "sum" | "mean" | "amax" | "amin" | "max" | "min" | "argmax"
    ``dims``  : reduction axes, ascending; empty tuple = full reduction
    ``keepdim``: whether reduced axes stay as size-1 dimensions

    ``argmax`` is an *index* reduction: the kernel carries a value stream and
    an index stream side by side and stores the index stream only; it
    requires explicit dims and float32/float64 inputs.  ``max``/``min`` WITH
    an axis are *pair* reductions (``is_pair``): the same dual-stream kernel
    stores one output per consumed projection — "values", "indices" — as the
    segment's exports.
    """

    __slots__ = ("op", "dims", "keepdim")

    # kernel-side combine/finalize/neutral per op
    _FINAL = {
        "sum": "tl.sum",
        "mean": "tl.sum",
        "amax": "tl.max",
        "max": "tl.max",
        "amin": "tl.min",
        "min": "tl.min",
    }
    _COMBINE = {
        "sum": "acc + {value}",
        "mean": "acc + {value}",
        "amax": "tl.maximum(acc, {value})",
        "max": "tl.maximum(acc, {value})",
        "amin": "tl.minimum(acc, {value})",
        "min": "tl.minimum(acc, {value})",
    }
    _NEUTRAL = {
        "sum": "0.0",
        "mean": "0.0",
        "amax": "float('-inf')",
        "max": "float('-inf')",
        "amin": "float('inf')",
        "min": "float('inf')",
        "argmax": "float('-inf')",
    }

    def __init__(self, op: str, dims: tuple[int, ...] = (), *, keepdim: bool = False) -> None:
        if op not in self._FINAL and op != "argmax":
            raise ValueError(f"unsupported reduction op: {op}")
        if op == "argmax":
            if not dims:
                # full-reduction machinery to track indices, which v1 does not
                # implement (single/split paths are value-only).
                raise ValueError("argmax folding requires a dim argument")
            if keepdim is False and len(dims) > 1:
                pass  # multi-dim compaction handled by output_shape
        self.op = op
        self.dims = tuple(sorted(int(dim) for dim in dims))
        self.keepdim = bool(keepdim)

    @property
    def tracks_indices(self) -> bool:
        return self.op == "argmax"

    @property
    def is_pair(self) -> bool:
        """True for ``max(dim)``/``min(dim)``: value+index pair output."""

        return self.op in ("max", "min") and bool(self.dims)

    @property
    def is_full(self) -> bool:
        return not self.dims

    def normalized_dims(self, rank: int) -> tuple[int, ...]:
        """Dims wrapped into ``[0, rank)`` ascending."""

        return tuple(sorted(dim % rank for dim in self.dims))

    def finalize_call(self, value: str) -> str:
        return f"{self._FINAL[self.op]}({value})"

    def combine_expr(self, acc: str, value: str) -> str:
        """Fold ``value`` into the running accumulator ``acc``."""

        return self._COMBINE[self.op].format(value=value).replace("acc", acc, 1)

    def neutral(self) -> str:
        return self._NEUTRAL[self.op]

    def output_shape(self, reference_shape: tuple[int, ...]) -> tuple[int, ...]:
        if self.is_full:
            return ()
        rank = len(reference_shape)
        normalized = tuple(dim % rank for dim in self.dims)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"duplicate reduction dims: {self.dims}")
        shape = list(reference_shape)
        for dim in normalized:
            shape[dim] = 1
        if self.keepdim:
            return tuple(shape)
        return tuple(
            size for index, size in enumerate(shape) if index not in normalized
        )

    def reduction_numel(self, reference_shape: tuple[int, ...]) -> int:
        rank = len(reference_shape)
        return _prod(
            reference_shape[dim % rank] for dim in self.dims
        ) if self.dims else _prod(reference_shape)

    def __repr__(self) -> str:
        return f"ReductionSpec({self.op!r}, {self.dims!r}, keepdim={self.keepdim})"

    def digest_key(self) -> tuple[Any, ...]:
        return ("reduction", self.op, self.dims, self.keepdim)


_runtime_probe_done = False
_runtime_probe_ok = False


def runtime_available() -> bool:
    """True when Triton can actually compile AND launch here.

    Importing ``triton`` is not enough: its CUDA driver needs a compatible
    Probed once per process by compiling and launching a trivial kernel;
    callers should treat False as "use another lowering".
    """

    global _runtime_probe_done, _runtime_probe_ok
    if _runtime_probe_done:
        return _runtime_probe_ok
    _runtime_probe_done = True
    try:
        args = None
        import tensorplay as _tp

        _tp.cuda.init()
        args = [_tp.rand(16, device=_tp.device("cuda", 0))]
        launch = _compile_program(
            program=[3, 0, -1], constants=[1.0], output_refs=(1,),
            example_inputs=args,
        )
        launch(args)
        _tp.cuda.synchronize()
        _runtime_probe_ok = True
    except Exception:  # noqa: BLE001 - any failure means "not available"
        _runtime_probe_ok = False
    return _runtime_probe_ok


_TENSOR_TYPE: Any = None


def _supports_runtime_inputs(
    example_inputs: list[Any],
    *,
    allow_grad: bool = False,
    reference_shape: tuple[int, ...] | None = None,
) -> bool:
    # Per-call dispatch guard on the compiled wrapper: must stay cheap.
    if not example_inputs:
        return False
    global _TENSOR_TYPE
    if _TENSOR_TYPE is None:
        try:
            import tensorplay

            _TENSOR_TYPE = tensorplay.Tensor
        except (AttributeError, ImportError):
            return False
    tensor_type = _TENSOR_TYPE
    for value in example_inputs:
        if not isinstance(value, tensor_type):
            return False
        if (
            not value.device.is_cuda()
            or not value.is_contiguous()
            or (value.requires_grad and not allow_grad)
        ):
            return False
    first = example_inputs[0]
    shapes = [tuple([int(dim) for dim in value.shape]) for value in example_inputs]
    # Without a compiled-in reference the historic contract applies: every
    # input must share one shape.  With one, inputs may broadcast to it.
    # The exact-match case (every compiled shape) skips the broadcast math.
    if reference_shape is None:
        if any(shape != shapes[0] for shape in shapes[1:]):
            return False
    elif any(shape == reference_shape for shape in shapes):
        if any(shape != reference_shape for shape in shapes):
            if _broadcast_reference_shape(shapes) != tuple(reference_shape):
                return False
    else:
        if _broadcast_reference_shape(shapes) != tuple(reference_shape):
            return False
    return all(
        value.dtype == first.dtype and value.device == first.device
        for value in example_inputs[1:]
    )


class TritonProgramCodegen:
    """Generate explicit Triton source for Stax's shared postfix program.

    The same representation is used by the CPU fused path and by the CUDA
    forward/backward kernels.  Instructions are expanded into source at
    compile time, so the kernel has no per-element opcode dispatch loop.
    """

    _OP_NAMES = {
        code: name for name, code in _TRITON_OPCODES.items()
    }

    def __init__(
        self,
        program: list[int],
        constants: list[float],
        output_refs: tuple[int, ...],
        input_count: int,
        *,
        reduction: str | None = None,
        input_shapes: tuple[tuple[int, ...], ...] | None = None,
        reference_shape: tuple[int, ...] | None = None,
        value_dtype: str | None = None,
        epilogue: tuple[list[int], list[float], int] | None = None,
        reduction_outputs: tuple[str, ...] | None = None,
    ) -> None:
        if len(program) % 3:
            raise ValueError("Triton Stax program must contain triples")
        self.program = program
        self.constants = constants
        self.output_refs = output_refs
        self.input_count = input_count
        if epilogue is not None:
            eprogram, _, _ = epilogue
            if len(eprogram) % 3:
                raise ValueError("epilogue program must contain triples")
            if reduction is None or (
                isinstance(reduction, ReductionSpec)
                and (reduction.tracks_indices or reduction.is_pair)
            ):
                # Index-carrying reductions emit an int64 stream; float
                # pointwise on top of it is meaningless, and plain pw needs
                # no acc.
                raise ValueError(
                    "reduction epilogues require a value reduction"
                )
        self.epilogue = epilogue
        if reduction_outputs is not None and not all(
            kind in ("values", "indices") for kind in reduction_outputs
        ):
            raise ValueError(
                f"unsupported reduction output kinds: {reduction_outputs!r}"
            )
        self.reduction_outputs = reduction_outputs
        if isinstance(reduction, ReductionSpec):
            self.reduction_spec: ReductionSpec | None = reduction
        elif reduction == "sum":
            self.reduction_spec = ReductionSpec("sum")
        else:
            self.reduction_spec = None
        self.reduction = (
            self.reduction_spec.op if self.reduction_spec is not None else None
        )
        self.value_type = _VALUE_TYPES.get(value_dtype or "", "tl.float32")
        self.reference_shape = (
            tuple(int(dim) for dim in reference_shape)
            if reference_shape is not None
            else None
        )
        if input_shapes is not None:
            self.input_shapes = tuple(
                tuple(int(dim) for dim in shape) for shape in input_shapes
            )
            if len(self.input_shapes) != input_count:
                raise ValueError("input_shapes must match input_count")
            if self.reference_shape is None:
                # Historic same-shape contract: first input is the reference.
                self.reference_shape = self.input_shapes[0]
        elif self.reference_shape is not None:
            self.input_shapes = (self.reference_shape,) * input_count
        else:
            self.input_shapes = None

    @property
    def _reduction_single_block(self) -> bool:
        """True when a full-reduction epilogue fits ONE block and stores directly."""

        spec = self.reduction_spec
        if spec is None or not spec.is_full or self.reference_shape is None:
            return False
        return _prod(self.reference_shape) <= _SINGLE_BLOCK_MAX

    def _offset_expression(self, index: int) -> str | None:
        """Flat-index offset expression for input ``index``.

        Returns ``None`` when the input shares the reference layout (plain
        ``xindex`` addressing).  Broadcast inputs get compile-time div/mod
        chains with zero strides folded away; numel-1 inputs load once.
        """

        if self.input_shapes is None or self.reference_shape is None:
            return None
        shape = self.input_shapes[index]
        if shape == self.reference_shape:
            return None
        rank = len(self.reference_shape)
        aligned = (1,) * (rank - len(shape)) + shape
        terms: list[str] = []
        for dim in range(rank):
            if aligned[dim] == 1:
                continue  # broadcast dimension contributes stride 0
            stride = _prod(aligned[dim + 1 :])
            div = _prod(self.reference_shape[dim + 1 :])
            size = self.reference_shape[dim]
            index_expr = f"xindex % {size}" if div == 1 else f"(xindex // {div}) % {size}"
            terms.append(index_expr if stride == 1 else f"{index_expr} * {stride}")
        if not terms:
            return "0"  # scalar tensor: single unmasked load below
        return " + ".join(terms)

    def _ref(self, ref: int) -> str:
        if ref < 0:
            index = -ref - 1
            if index < 0 or index >= len(self.constants):
                raise ValueError(f"invalid Triton Stax constant reference: {ref}")
            return _scalar_source(self.constants[index])
        if ref < self.input_count:
            return f"in{ref}"
        return f"tmp{ref - self.input_count}"

    def _epilogue_lines(self, source_reg: str) -> tuple[list[str], str]:
        """Emit the post-reduction pointwise chain; return its final register.

        Ref space of the epilogue program: ``esrc`` is the reduction result
        (mapped to ``source_reg``), negatives index epilogue constants, and
        positive refs are temporaries numbered from 1 — the program's only
        external input is the reduction result itself.
        """

        assert self.epilogue is not None
        eprogram, econstants, esrc = self.epilogue

        def resolve(ref: int) -> str:
            if ref < 0:
                index = -ref - 1
                if index >= len(econstants):
                    raise ValueError("invalid epilogue constant reference")
                return _scalar_source(econstants[index])
            if ref == esrc:
                return source_reg
            return f"etmp{ref}"

        lines = self._program_lines(
            eprogram, resolve, "etmp", first_index=1
        )
        return lines, f"etmp{len(eprogram) // 3}"

    def _expression(
        self,
        opcode: int,
        lhs_ref: int,
        rhs_ref: int,
        resolver: Callable[[int], str] | None = None,
    ) -> str:
        resolve = resolver if resolver is not None else self._ref
        try:
            name = self._OP_NAMES[opcode]
        except KeyError as exc:
            raise ValueError(f"unsupported Triton Stax opcode: {opcode}") from exc
        lhs = resolve(lhs_ref)
        if name in {"add", "sub", "mul", "div", "pow"}:
            rhs = resolve(rhs_ref)
            return {
                "add": f"{lhs} + {rhs}",
                "sub": f"{lhs} - {rhs}",
                "mul": f"{lhs} * {rhs}",
                "div": f"{lhs} / {rhs}",
                "pow": f"{lhs} ** {rhs}",
            }[name]
        if name in {"lt", "le", "gt", "ge", "eq", "ne"}:
            rhs = resolve(rhs_ref)
            # Comparisons normalize to 0.0/1.0 floats: the program's value
            # space is float, so a bare int1 would mis-type every consumer
            # (arithmetic, stores) that is not ``where``.
            comparison = {
                "lt": f"{lhs} < {rhs}",
                "le": f"{lhs} <= {rhs}",
                "gt": f"{lhs} > {rhs}",
                "ge": f"{lhs} >= {rhs}",
                "eq": f"{lhs} == {rhs}",
                "ne": f"{lhs} != {rhs}",
            }[name]
            return f"({comparison}).to(tl.float32)"
        if name in {"minimum", "maximum", "clamp_min", "clamp_max"}:
            rhs = resolve(rhs_ref)
            return {
                "minimum": f"tl.minimum({lhs}, {rhs})",
                "maximum": f"tl.maximum({lhs}, {rhs})",
                "clamp_min": f"tl.maximum({lhs}, {rhs})",
                "clamp_max": f"tl.minimum({lhs}, {rhs})",
            }[name]
        return {
            "neg": f"-{lhs}",
            "pos": lhs,
            "abs": f"tl.abs({lhs})",
            "sin": f"tl.sin({lhs})",
            "cos": f"tl.cos({lhs})",
            "exp": f"tl.exp({lhs})",
            "log": f"tl.log({lhs})",
            "sigmoid": f"(1.0 / (1.0 + tl.exp(-{lhs})))",
            "sqrt": f"tl.sqrt({lhs})",
            "square": f"{lhs} * {lhs}",
            "tanh": f"libdevice.tanh({lhs})",
            "relu": f"tl.maximum({lhs}, 0.0)",
            "rsqrt": f"libdevice.rsqrt({lhs})",
            "exp2": f"libdevice.exp2({lhs})",
            "erf": f"libdevice.erf({lhs})",
            "relu_grad": f"tl.where({lhs} > 0.0, 1.0, 0.0)",
            "abs_grad": (
                f"tl.where({lhs} > 0.0, 1.0, "
                f"tl.where({lhs} < 0.0, -1.0, 0.0))"
            ),
        }[name]

    def _program_lines(
        self,
        program: list[int],
        resolver: Callable[[int], str],
        tmp_prefix: str,
        *,
        first_index: int = 0,
    ) -> list[str]:
        """Emit one source line per program instruction.

        Shared by every kernel form.  Two opcodes are resolved here instead
        of in ``_expression``: the ``where``/``where_rest`` pair (ternary
        select encoded across two adjacent instructions — ``where`` carries
        the condition and the true-branch, emits nothing, and ``where_rest``
        completes the select with the false-branch), and ``cast`` (whose rhs
        operand slot carries a dtype id, not a value reference).
        """

        lines: list[str] = []
        pending_where: tuple[str, str] | None = None
        for offset in range(0, len(program), 3):
            index = first_index + offset // 3
            opcode, lhs_ref, rhs_ref = program[offset : offset + 3]
            name = self._OP_NAMES[opcode]
            if name == "where":
                pending_where = (resolver(lhs_ref), resolver(rhs_ref))
                continue
            if name == "where_rest":
                if pending_where is None:
                    raise ValueError("where_rest without a preceding where")
                cond_source, then_source = pending_where
                pending_where = None
                lines.append(
                    f"{tmp_prefix}{index} = tl.where("
                    f"({cond_source}) != 0.0, {then_source}, "
                    f"{resolver(rhs_ref)})"
                )
                continue
            if name == "cast":
                dtype = _CAST_SOURCE.get(rhs_ref)
                if dtype is None:
                    raise ValueError(f"unsupported Triton Stax cast id: {rhs_ref}")
                lines.append(
                    f"{tmp_prefix}{index} = {resolver(lhs_ref)}.to({dtype})"
                )
                continue
            lines.append(
                f"{tmp_prefix}{index} = "
                f"{self._expression(opcode, lhs_ref, rhs_ref, resolver)}"
            )
        return lines

    def _load_lines(self, use_mask: bool = True) -> list[str]:
        """Per-input load lines honouring broadcast offsets.

        ``use_mask=False`` (every lane valid: numel divides XBLOCK) drops
        predication entirely and marks
        reference-layout loads with ``cache_modifier='.cg'``: a read-once,
        coalesced stream has no reuse for L1, so bypassing it keeps the
        resident working set (the skip-L1 heuristic, under these input
        conditions: not broadcasted, not inside a reduction, single use).
        Broadcast/offset inputs and any predicated load keep the plain form.
        """

        lines: list[str] = []
        for index in range(self.input_count):
            offset = self._offset_expression(index)
            if offset == "0":
                lines.append(f"in{index} = tl.load(in_ptr{index})")
            elif offset is None:
                if use_mask:
                    lines.append(
                        f"in{index} = tl.load(in_ptr{index} + xindex, "
                        "mask=xmask, other=0.0)"
                    )
                else:
                    lines.append(
                        f"in{index} = tl.load(in_ptr{index} + xindex, "
                        "cache_modifier='.cg')"
                    )
            else:
                lines.append(f"off{index} = {offset}")
                if use_mask:
                    lines.append(
                        f"in{index} = tl.load(in_ptr{index} + off{index}, "
                        "mask=xmask, other=0.0)"
                    )
                else:
                    lines.append(f"in{index} = tl.load(in_ptr{index} + off{index})")
        return lines

    def generate(
        self,
        kernel_name: str,
        *,
        fixed_config: tuple[int, int] | None = None,
    ) -> str:
        """Emit kernel source.

        Pointwise programs keep the historic shapes: ``fixed_config=None``
        emits the runtime ``@triton.autotune`` decorator (fallback behaviour),
        while a ``(xblock, num_warps)`` pair drops the decorator and pins the
        winning autotuned config explicitly (L5-M2).

        Full-reduction epilogues are always pinned: a reference input within
        one block takes the single-kernel direct-store form, anything larger
        takes the two-stage split — per-program partial results into a
        workspace plus a tiny finalize kernel — using multilayer split
        reduction in miniature.  Axis reductions (the
        ``sum(dim)`` family, M5b) emit an output-space kernel whose inner
        ``tl.range`` loop folds RBLOCK-sized chunks of the reduction space;
        their configs are deterministic in v1 (tuning lands with M5d).
        """

        spec = self.reduction_spec
        single_reduction = self._reduction_single_block
        full_reduction = spec is not None and spec.is_full
        split_reduction = full_reduction and not single_reduction
        dims_reduction = spec is not None and not spec.is_full

        numel_total = (
            _prod(self.reference_shape)
            if self.reference_shape is not None
            else None
        )
        # Loop-pass vectorize width (pointwise kernels only — the reduction
        # families carry their own iteration spaces).  The row-uniform mask
        # contract needs the element count to divide the width.
        vec = 1
        if (
            fixed_config is not None
            and len(fixed_config) > 2
            and spec is None
            and numel_total is not None
            and numel_total % int(fixed_config[2]) == 0
        ):
            vec = int(fixed_config[2])

        body: list[str]
        if vec > 1:
            # Packed iteration space: each lane owns VEC consecutive
            # elements, so one load/store covers one contiguous segment and
            # the backend emits vector memory instructions.  The mask is
            # row-uniform (xnumel % VEC == 0 makes a row full whenever its
            # base is in-bounds), which keeps the vector access intact in
            # the last partial block.
            body = [
                f"xoffset = tl.program_id(0) * XBLOCK * {vec}",
                # AxisInfo hint: proves contiguity+alignment of every xindex
                # derived access, unlocking vectorized ld/st through the
                # multiple_of annotation.
                f"xoffset = tl.multiple_of(xoffset, XBLOCK * {vec})",
                "xrow = tl.arange(0, XBLOCK)",
                "xlane = tl.arange(0, VEC)",
                "xindex = xoffset + xrow[:, None] * VEC + xlane[None, :]",
                f"xmask = (xoffset + xrow * {vec})[:, None] < xnumel",
            ]
        else:
            body = [
                "xoffset = tl.program_id(0) * XBLOCK",
                # AxisInfo hint: proves contiguity+alignment of every xindex
                # derived access, unlocking vectorized ld/st through the
                # multiple_of annotation.
                "xoffset = tl.multiple_of(xoffset, XBLOCK)",
                "xindex = xoffset + tl.arange(0, XBLOCK)",
                "xmask = xindex < xnumel",
            ]
        if dims_reduction:
            assert self.reference_shape is not None and spec is not None
            block, warps, rblock, stages_default = _dim_reduction_config(
                self.reference_shape, spec
            )
            stages = stages_default
            # Loop-pass knobs (extended dims-config tail): reduction-r-loop
            # unroll factor and main/tail split flag.  Both default to the
            # loop-neutral emission.
            unroll = 1
            tail_split = False
            if fixed_config is not None:
                block, warps = fixed_config[0], fixed_config[1]
                if len(fixed_config) > 3:
                    rblock, stages = fixed_config[2], fixed_config[3]
                elif len(fixed_config) > 2:
                    stages = fixed_config[2]
                if len(fixed_config) > 5:
                    unroll = int(fixed_config[4])
                    tail_split = bool(fixed_config[5])
            reference = self.reference_shape
            rank = len(reference)
            reduced_dims = tuple(dim % rank for dim in spec.dims)
            kept_dims = tuple(
                dim for dim in range(rank) if dim not in reduced_dims
            )
            out_sizes = tuple(reference[dim] for dim in kept_dims)
            onumel = _prod(out_sizes)
            rnumel = spec.reduction_numel(reference)

            def _stride(dim: int) -> int:
                return _prod(reference[dim + 1 :])

            terms: list[str] = []
            # Non-reduced axes decompose the flat OUTPUT index.
            for position, dim in enumerate(kept_dims):
                divisor = _prod(out_sizes[position + 1 :])
                size = reference[dim]
                coord = (
                    f"xindex % {size}"
                    if divisor == 1
                    else f"(xindex // {divisor}) % {size}"
                )
                stride = _stride(dim)
                terms.append(
                    f"({coord})[:, None]"
                    if stride == 1
                    else f"({coord})[:, None] * {stride}"
                )
            # Reduced axes decompose the flat REDUCTION index.
            ordered_reduced = sorted(reduced_dims)
            for position, dim in enumerate(ordered_reduced):
                tail_sizes = [
                    reference[other]
                    for other in ordered_reduced[position + 1 :]
                ]
                divisor = _prod(tail_sizes)
                size = reference[dim]
                stride = _stride(dim)
                if divisor == 1 and not tail_sizes and len(ordered_reduced) == 1:
                    # Sole reduced axis: rmask bounds rindex by its size,
                    # so the modulo would be redundant.
                    terms.append(
                        "rindex[None, :]"
                        if stride == 1
                        else f"rindex[None, :] * {stride}"
                    )
                    continue
                coord = (
                    f"rindex % {size}"
                    if divisor == 1
                    else f"(rindex // {divisor}) % {size}"
                )
                terms.append(
                    f"({coord})[None, :]"
                    if stride == 1
                    else f"({coord})[None, :] * {stride}"
                )
            # Per-input addressing: a broadcast input (e.g. a bias over the
            # reduced axis) must NOT be addressed with the full-shape
            # formula, or its lanes read unrelated memory.  Each input's
            # terms drop every dimension whose aligned size is 1.
            div_x = onumel % block == 0
            div_r = rnumel % rblock == 0
            def _input_offset(index: int) -> str:
                if self.input_shapes is None:
                    return " + ".join(terms) if terms else "0"
                shape = self.input_shapes[index]
                if shape == reference:
                    return " + ".join(terms) if terms else "0"
                rank_i = len(reference)
                aligned = (1,) * (rank_i - len(shape)) + tuple(shape)
                own: list[str] = []

                def add_term(is_reduced: bool, position: int, dim: int) -> None:
                    if aligned[dim] == 1:
                        return
                    tail_sizes = (
                        [reference[o] for o in ordered_reduced[position + 1:]]
                        if is_reduced
                        else []
                    )
                    divisor = _prod(tail_sizes)
                    size = reference[dim]
                    stride = _stride(dim)
                    if is_reduced and divisor == 1 and not tail_sizes and len(ordered_reduced) == 1:
                        expr = "rindex[None, :]" if stride == 1 else f"rindex[None, :] * {stride}"
                        own.append(expr)
                        return
                    coord_src = "rindex" if is_reduced else "xcoord"
                    coord = (
                        f"{coord_src} % {size}"
                        if divisor == 1
                        else f"({coord_src} // {divisor}) % {size}"
                    )
                    suffix = "[None, :]" if is_reduced else "[:, None]"
                    own.append(
                        f"{coord}{suffix}" if stride == 1 else f"{coord}{suffix} * {stride}"
                    )

                for position, dim in enumerate(kept_dims):
                    divisor = _prod(out_sizes[position + 1 :])
                    size = reference[dim]
                    stride = _stride(dim)
                    if aligned[dim] != 1:
                        coord = (
                            f"xindex % {size}"
                            if divisor == 1
                            else f"(xindex // {divisor}) % {size}"
                        )
                        own.append(
                            f"{coord}[:, None]"
                            if stride == 1
                            else f"{coord}[:, None] * {stride}"
                        )
                for position, dim in enumerate(ordered_reduced):
                    if aligned[dim] != 1:
                        tail_sizes = [
                            reference[o] for o in ordered_reduced[position + 1 :]
                        ]
                        divisor = _prod(tail_sizes)
                        size = reference[dim]
                        stride = _stride(dim)
                        if divisor == 1 and not tail_sizes and len(ordered_reduced) == 1:
                            own.append(
                                "rindex[None, :]"
                                if stride == 1
                                else f"rindex[None, :] * {stride}"
                            )
                            continue
                        coord = (
                            f"rindex % {size}"
                            if divisor == 1
                            else f"(rindex // {divisor}) % {size}"
                        )
                        own.append(
                            f"({coord})[None, :]"
                            if stride == 1
                            else f"({coord})[None, :] * {stride}"
                        )
                expr = " + ".join(own) if own else "0"
                if "[:, None]" not in expr:
                    # No kept-dim term: force the [XBLOCK, 1] shape so the
                    # pointer tensor broadcasts against the [XBLOCK, RBLOCK]
                    # mask inside tl.load.
                    expr = f"{expr} + xindex[:, None] * 0" if expr != "0" else "xindex[:, None] * 0"
                return expr

            dual_stream = spec.tracks_indices or spec.is_pair
            persistent = rnumel <= rblock
            # Split applicability: only a loop whose tile does not exactly
            # divide the reduction space has a masked tail to peel.
            split_tail = tail_split and not persistent and not div_r
            if dual_stream:
                body.append(
                    f"acc = tl.full([XBLOCK], {spec.neutral()}, dtype={self.value_type})"
                )
                # Index stream: the running winner in flat reduced-space
                # coordinates (same order the r-loop enumerates).
                body.append("acci = tl.zeros([XBLOCK], dtype=tl.int64)")
            else:
                body.append(
                    f"acc = tl.full([XBLOCK], {spec.neutral()}, dtype=tl.float32)"
                )
            pfx = "" if persistent else "    "

            if dual_stream:
                # Priority-stream polarity: max-family scans for the greatest
                # value, min-family for the lowest.  tl.argmax/tl.argmin
                # break ties toward the lower lane, and combining chunks with
                # a strict compare keeps the earlier chunk, so the global
                # winner is the first extremum — the value-selection tie
                # rule.  NaN ordering is explicit: several triton versions
                # ignore NaNs inside reductions.  A finite sentinel ranks NaN
                # beyond every real value (above for max, below for min),
                # letting cval double as the has-NaN flag without a second
                # reduction per chunk.
                if spec.op == "min":
                    sentinel, cmp, val_fn, win_fn = (
                        "-1.0e38", "<", "tl.min", "tl.argmin",
                    )
                else:
                    sentinel, cmp, val_fn, win_fn = (
                        "1.0e38", ">", "tl.max", "tl.argmax",
                    )

            def emit_r_tile(rbase: str, exact: bool, indent: str) -> None:
                """Emit loads + program + combine for ONE r-tile at ``rbase``.

                ``exact`` marks a tile whose lanes are all in-bounds (no
                rmask predication); ``indent`` prefixes every line (loop
                bodies are indented, a peeled tail is not).
                """

                inner = [f"rindex = {rbase} + tl.arange(0, RBLOCK)"]
                if not exact:
                    inner.append(f"rmask = rindex < {rnumel}")
                if div_x and exact:
                    mask_text = ""          # every lane valid
                elif exact:
                    mask_text = "xmask[:, None]"
                elif div_x:
                    mask_text = "rmask[None, :]"
                else:
                    mask_text = "m2"
                    inner.append("m2 = rmask[None, :] & xmask[:, None]")
                for index in range(self.input_count):
                    inner.append(f"in_off{index} = {_input_offset(index)}")
                    # Reduction tiles stream through L2 exactly once, so give the
                    # lines evict-first priority — the rule for every load
                    # inside a reduction loop; persistent single-tile reads keep
                    # it too (the tile is still read-once).
                    if mask_text:
                        inner.append(
                            f"in{index} = tl.load(in_ptr{index} + in_off{index}, "
                            f"eviction_policy='evict_first', "
                            f"mask={mask_text}, other={spec.neutral()})"
                        )
                    else:
                        inner.append(
                            f"in{index} = tl.load(in_ptr{index} + in_off{index}, "
                            "eviction_policy='evict_first')"
                        )
                inner.extend(self._program_lines(self.program, self._ref, "tmp"))
                body.extend(textwrap.indent(line, indent) for line in inner)
                last = self._ref(self.output_refs[0])
                # The pointwise program transforms padded lanes' neutral loads
                # into non-neutral values (sigmoid(0) = 0.5 and friends), so
                # the reduction must re-mask its INPUT — unless the r-tile is
                # exact.
                if exact:
                    last_masked = last
                else:
                    last_masked = (
                        f"tl.where(rmask[None, :], {last}, {spec.neutral()})"
                    )
                if dual_stream:
                    body.append(f"{indent}isnan_ = {last_masked} != {last_masked}")
                    body.append(
                        f"{indent}prio = tl.where(isnan_, {sentinel}, {last_masked})"
                    )
                    body.append(f"{indent}cval = {val_fn}(prio, axis=1)")
                    body.append(
                        f"{indent}cwin = {win_fn}(prio, axis=1) + {rbase}"
                    )
                    body.append(f"{indent}live = acc == acc")
                    body.append(
                        f"{indent}hit = ((cval {cmp} acc) | (cval == {sentinel})) & live"
                    )
                    body.append(
                        f"{indent}acci = tl.where(hit, cwin.to(tl.int64), acci)"
                    )
                    body.append(
                        f"{indent}acc = tl.where((cval == {sentinel}) & live, "
                        f"float('nan'), tl.where((cval {cmp} acc) & live, cval, acc))"
                    )
                else:
                    body.append(
                        f"{indent}chunk = {spec.finalize_call(last_masked + ', axis=1')}"
                    )
                    body.append(f"{indent}acc = {_ACC_UPDATE[spec.op]}")

            if persistent:
                emit_r_tile("0", div_r, "")
            else:
                range_kwargs = f"num_stages={stages}"
                if unroll > 1:
                    range_kwargs += f", loop_unroll_factor={unroll}"
                if split_tail:
                    # Main part: full tiles only — every lane in-bounds, so
                    # the loop body carries no r-side predication.  The
                    # remainder runs once after the loop, masked.
                    rmain = rnumel - rnumel % rblock
                    body.append(
                        f"for roffset in tl.range(0, {rmain}, RBLOCK, "
                        f"{range_kwargs}):"
                    )
                    emit_r_tile("roffset", True, "    ")
                    emit_r_tile(repr(rmain), False, "")
                else:
                    body.append(
                        f"for roffset in tl.range(0, {rnumel}, RBLOCK, "
                        f"{range_kwargs}):"
                    )
                    emit_r_tile("roffset", div_r, "    ")
            if spec.op == "mean":
                body.append(f"acc = acc * {repr(1.0 / rnumel)}")
            if dual_stream:
                out_kinds = self.reduction_outputs
                if spec.is_pair:
                    if not out_kinds:
                        raise ValueError(
                            "pair reduction without value/index projections"
                        )
                elif out_kinds is None:
                    out_kinds = ("indices",)
                for port, kind in enumerate(out_kinds):
                    store_source = "acci" if kind == "indices" else "acc"
                    if div_x:
                        body.append(
                            f"tl.store(out_ptr{port} + xindex, {store_source})"
                        )
                    else:
                        body.append(
                            f"tl.store(out_ptr{port} + xindex, "
                            f"{store_source}, mask=xmask)"
                        )
            else:
                store_source = "acc"
                if self.epilogue is not None:
                    epilogue_lines, epilogue_last = self._epilogue_lines("acc")
                    body.extend(epilogue_lines)
                    store_source = epilogue_last
                if div_x:
                    body.append(f"tl.store(out_ptr0 + xindex, {store_source})")
                else:
                    body.append(
                        f"tl.store(out_ptr0 + xindex, {store_source}, mask=xmask)"
                    )
            n_out_ports = (
                len(out_kinds) if dual_stream else 1
            )
            signature = [
                *(f"in_ptr{index}" for index in range(self.input_count)),
                *(f"out_ptr{port}" for port in range(n_out_ports)),
                "xnumel",
                "XBLOCK: tl.constexpr",
                "RBLOCK: tl.constexpr",
            ]
        elif split_reduction and fixed_config is not None and len(fixed_config) > 2:
            # Persistent grid-stride main kernel: a FIXED, SM-shaped program
            # count sweeps the whole input, vector-accumulating locally and
            # writing ONE partial per program into the workspace.  Kills the
            # 16k-program scheduling tax of the classic form; the tiny
            # finalize still combines ``nprog`` partials (and owns any
            # epilogue).
            block = fixed_config[0]
            warps = fixed_config[1]
            nprog = fixed_config[2]
            stride = nprog * block
            # The shared preamble (xoffset/xindex/xmask) is dead here: the
            # sweep recomputes xindex per iteration and the masked form names
            # its bound xnumel_tail, so keeping the preamble would reference
            # a nonexistent parameter.
            body.clear()
            # Unmasked loads are only safe when the grid-stride sweep tiles
            # the input EXACTLY: the last iteration still reaches up to
            # numel_total, so divisibility by the stride (not just XBLOCK)
            # is the bound.
            div = numel_total is not None and numel_total % stride == 0
            body.append("start0 = tl.program_id(0) * XBLOCK")
            if spec.op in ("sum", "mean"):
                body.append("acc = tl.zeros([XBLOCK], dtype=tl.float32)")
            else:
                body.append(
                    f"acc = tl.full([XBLOCK], {spec.neutral()}, dtype=tl.float32)"
                )
            body.append(
                f"for off in tl.range(0, {numel_total}, {stride}, num_stages=3):"
            )
            inner = ["xindex = start0 + off + tl.arange(0, XBLOCK)"]
            for index in range(self.input_count):
                offset = self._offset_expression(index)
                if offset == "0":
                    if div:
                        inner.append(f"in{index} = tl.load(in_ptr{index})")
                    else:
                        inner.append(
                            f"in{index} = tl.load(in_ptr{index}, "
                            f"mask=xindex < xnumel_tail, other={spec.neutral()})"
                        )
                    continue
                base = f"in_off{index}" if offset not in (None,) else "xindex"
                if offset not in (None,):
                    inner.append(f"{base} = {offset}")
                # Grid-stride sweep: each element crosses L2 exactly once per
                # launch — evict-first keeps it from displacing the resident
                # accumulator/next-tile lines.
                if div:
                    inner.append(
                        f"in{index} = tl.load(in_ptr{index} + {base}, "
                        "eviction_policy='evict_first')"
                    )
                else:
                    inner.append(
                        f"in{index} = tl.load(in_ptr{index} + {base}, "
                        "eviction_policy='evict_first', "
                        f"mask=xindex < xnumel_tail, other={spec.neutral()})"
                    )
            body.extend(textwrap.indent(l, "    ") for l in inner)
            body.extend(
                textwrap.indent(line, "    ")
                for line in self._program_lines(
                    self.program, self._ref, "tmp"
                )
            )
            last = self._ref(self.output_refs[0])
            # Re-mask program output: a neutral-filled padding lane is only
            # neutral BEFORE the pointwise transform (sigmoid(0)=0.5,
            # abs(0-1)=1).  Same contract as the classic split path.
            if not div:
                last = (
                    f"tl.where(xindex < xnumel_tail, {last}, {spec.neutral()})"
                )
            body.append(f"    chunk = {last}")
            body.append(f"    acc = {_ACC_UPDATE[spec.op]}")
            scale_p = (
                f" * {repr(1.0 / numel_total)}"
                if spec.op == "mean" and numel_total
                else ""
            )
            body.append(
                f"partial = {spec.finalize_call('acc, axis=0')}{scale_p}"
            )
            body.append("tl.store(ws_ptr + tl.program_id(0), partial)")
            signature = [
                *(f"in_ptr{index}" for index in range(self.input_count)),
                "ws_ptr",
                ("xnumel_tail" if not div else "xnumel"),
                "XBLOCK: tl.constexpr",
            ]
        else:
            pw_block = fixed_config[0] if fixed_config is not None else None
            if pw_block is None and single_reduction:
                pw_block = _single_block_config(numel_total)[0]
            use_xmask = not (
                pw_block is not None
                and numel_total is not None
                and numel_total % (pw_block * vec) == 0
            )
            body.extend(self._load_lines(use_mask=use_xmask))
            body.extend(self._program_lines(self.program, self._ref, "tmp"))
            last = self._ref(self.output_refs[0])
            # Re-mask program output: transformed padding lanes (e.g.
            # sigmoid(0) = 0.5) must not enter the reduction.  With the
            # no-mask fast path every lane is valid, so nothing to do.
            if use_xmask and spec is not None:
                last = f"tl.where(xmask, {last}, {spec.neutral()})"

        if single_reduction:
            assert self.reference_shape is not None and spec is not None
            numel = _prod(self.reference_shape)
            block, warps = _single_block_config(numel)
            xnumel_source = repr(numel)
            scale = f" * {repr(1.0 / numel)}" if spec.op == "mean" else ""
            body.append(f"reduced = {spec.finalize_call(last + ', axis=0')}{scale}")
            store_source = "reduced"
            if self.epilogue is not None:
                epilogue_lines, epilogue_last = self._epilogue_lines("reduced")
                body.extend(epilogue_lines)
                store_source = epilogue_last
            body.append(f"tl.store(out_ptr0, {store_source})")
            signature = [
                *(f"in_ptr{index}" for index in range(self.input_count)),
                "out_ptr0",
                "xnumel",
                "XBLOCK: tl.constexpr",
            ]
        elif split_reduction:
            assert spec is not None
            persistent_split = (
                fixed_config is not None and len(fixed_config) > 2
            )
            if fixed_config is not None:
                block, warps = fixed_config[0], fixed_config[1]
            else:
                block, warps = _STATIC_REDUCTION_CONFIG
            xnumel_source = (
                repr(_prod(self.reference_shape))
                if self.reference_shape is not None
                else "inputs[0].numel()"
            )
            if not persistent_split:
                # The persistent branch already folded the accumulator and
                # stored one partial per program; re-emitting the classic
                # tail here referenced the loop-scoped load and made every
                # persistent candidate fail to compile.
                body.append(
                    f"partial = {spec.finalize_call(last + ', axis=0')}"
                )
                body.append("tl.store(ws_ptr + tl.program_id(0), partial)")
                signature = [
                    *(f"in_ptr{index}" for index in range(self.input_count)),
                    "ws_ptr",
                    "xnumel",
                    "XBLOCK: tl.constexpr",
                ]
        elif not dims_reduction:
            for output_index, output_ref in enumerate(self.output_refs):
                if use_xmask:
                    body.append(
                        f"tl.store(out_ptr{output_index} + xindex, "
                        f"{self._ref(output_ref)}, mask=xmask)"
                    )
                else:
                    body.append(
                        f"tl.store(out_ptr{output_index} + xindex, "
                        f"{self._ref(output_ref)})"
                    )
            signature = [
                *(f"in_ptr{index}" for index in range(self.input_count)),
                *(f"out_ptr{index}" for index in range(len(self.output_refs))),
                "xnumel",
                "XBLOCK: tl.constexpr",
                *(["VEC: tl.constexpr"] if vec > 1 else []),
            ]

        source = (
            "import triton\n"
            "import triton.language as tl\n"
            "import triton.language.extra.cuda.libdevice as libdevice\n"
            "import tensorplay as tp\n"
            "from tensorplay._stax.runtime import fastlaunch as _fl\n\n"
        )
        if self.reduction_spec is None and fixed_config is None:
            source += "@triton.autotune(\n"
            source += "    configs=[\n"
            source += "        triton.Config({'XBLOCK': 128}, num_warps=4),\n"
            source += "        triton.Config({'XBLOCK': 256}, num_warps=4),\n"
            source += "        triton.Config({'XBLOCK': 512}, num_warps=8),\n"
            source += "        triton.Config({'XBLOCK': 1024}, num_warps=8),\n"
            source += "        triton.Config({'XBLOCK': 2048}, num_warps=8),\n"
            source += "    ],\n"
            source += "    key=['xnumel'],\n"
            source += ")\n"
        source += "@triton.jit\n"
        source += f"def {kernel_name}({', '.join(signature)}):\n"
        source += textwrap.indent("\n".join(body), "    ") + "\n\n"

        if single_reduction:
            call_args = [*(f"inputs[{index}]" for index in range(self.input_count)), "out"]
            args_txt = ", ".join(call_args)
            ptrs = " | ".join(
                [
                    *(f"inputs[{i}].data_ptr()" for i in range(self.input_count)),
                    "out.data_ptr()",
                ]
            )
            guard = f"({ptrs}) % 16 == 0"
            source += "_rec = None\n\n"
            source += "def kernel_launch(inputs):\n"
            source += "    global _rec\n"
            source += (
                "    out = tp.empty((), dtype=inputs[0].dtype, "
                "device=inputs[0].device)\n"
            )
            source += f"    xnumel = {xnumel_source}\n"
            # Fast path (fastlaunch): replay the recorded CompiledKernel.run
            # directly — same call shape JITFunction.run uses — once the
            # pointer alignment / scalar specialization matches the recorded
            # binary.  Any miss (or profiling hooks) drops to the dispatch
            # below, which re-specializes and can re-record.
            source += "    _r = _rec\n"
            source += (
                f"    if _r is not None and {guard} and _fl.hooks_clear() "
                "and _r[3] == xnumel:\n"
            )
            source += "        try:\n"
            source += "            _s = _fl.current_stream()\n"
            source += (
                f"            _r[0](1, 1, 1, _s, _r[1], _r[2], None, None, "
                f"None, {args_txt}, xnumel, {block})\n"
            )
            source += "            _fl.bump()\n"
            source += "            return out\n"
            source += "        except Exception:\n"
            source += "            _rec = None\n"
            source += "    _snap = -1\n"
            source += (
                f"    if _r is None and _fl.hooks_clear() and {guard}:\n"
            )
            source += f"        _snap = _fl.cache_size({kernel_name})\n"
            source += (
                f"    {kernel_name}[(1,)]({args_txt}, "
                f"xnumel, XBLOCK={block}, num_warps={warps})\n"
            )
            source += "    if _snap >= 0:\n"
            source += f"        _g = _fl.take_kernel({kernel_name}, _snap)\n"
            source += "        if _g is not None:\n"
            source += "            _rec = _g + (xnumel,)\n"
            source += "    return out\n"
        elif dims_reduction:
            assert self.reference_shape is not None and spec is not None
            out_shape = tuple(
                int(size) for size in spec.output_shape(self.reference_shape)
            )
            onumel = _prod(out_shape)
            grid_size = max(1, -(-onumel // block))
            # One output buffer per kernel port: an index stream always
            # materializes int64 regardless of the value-stream dtype; a
            # value port keeps the input dtype.
            out_kinds = self.reduction_outputs
            if out_kinds is None:
                out_kinds = ("indices",) if spec.tracks_indices else ("values",)
            if not out_kinds:
                raise ValueError("dims reduction needs at least one output")
            out_dtypes = tuple(
                "tp.int64" if kind == "indices" else "inputs[0].dtype"
                for kind in out_kinds
            )
            call_args = [
                *(f"inputs[{index}]" for index in range(self.input_count)),
                *(f"outs[{port}]" for port in range(len(out_kinds))),
            ]
            args_txt = ", ".join(call_args)
            ptrs = " | ".join(
                [
                    *(f"inputs[{i}].data_ptr()" for i in range(self.input_count)),
                    *(f"outs[{p}].data_ptr()" for p in range(len(out_kinds))),
                ]
            )
            guard = f"({ptrs}) % 16 == 0"
            source += "_rec = None\n\n"
            source += "def kernel_launch(inputs):\n"
            source += "    global _rec\n"
            dtype_tuple = (
                f"({out_dtypes[0]},)"
                if len(out_dtypes) == 1
                else "(" + ", ".join(out_dtypes) + ")"
            )
            source += (
                f"    outs = [tp.empty({out_shape!r}, dtype=dt, "
                f"device=inputs[0].device) for dt in {dtype_tuple}]\n"
            )
            # Fast path: see the single-reduction branch; the scalar arg is
            # the compile-time output numel, so the guard is exact.
            source += "    _r = _rec\n"
            source += (
                f"    if _r is not None and {guard} and _fl.hooks_clear() "
                f"and _r[3] == {onumel}:\n"
            )
            source += "        try:\n"
            source += "            _s = _fl.current_stream()\n"
            source += (
                f"            _r[0]({grid_size}, 1, 1, _s, _r[1], _r[2], "
                f"None, None, None, {args_txt}, {onumel}, {block}, {rblock})\n"
            )
            source += "            _fl.bump()\n"
            source += "            return outs[0] if len(outs) == 1 else outs\n"
            source += "        except Exception:\n"
            source += "            _rec = None\n"
            source += "    _snap = -1\n"
            source += (
                f"    if _r is None and _fl.hooks_clear() and {guard}:\n"
            )
            source += f"        _snap = _fl.cache_size({kernel_name})\n"
            stages_kw = (
                "" if persistent else f", num_stages={stages}"
            )
            source += (
                f"    {kernel_name}[({grid_size},)]({args_txt}, "
                f"{onumel}, XBLOCK={block}, RBLOCK={rblock}, "
                f"num_warps={warps}{stages_kw})\n"
            )
            source += "    if _snap >= 0:\n"
            source += f"        _g = _fl.take_kernel({kernel_name}, _snap)\n"
            source += "        if _g is not None:\n"
            source += f"            _rec = _g + ({onumel},)\n"
            source += (
                "    return outs[0] if len(outs) == 1 else outs\n"
            )
        elif split_reduction:
            assert spec is not None
            finalize_name = kernel_name + "_finalize"
            total = (
                _prod(self.reference_shape)
                if self.reference_shape is not None
                else None
            )
            scale = (
                f" * {repr(1.0 / total)}"
                if spec.op == "mean" and total is not None
                else ""
            )
            # Stream the partials in FBLOCK chunks: materializing
            # next_pow2(wsn) lanes in ONE vector spills registers once the
            # grid grows past a few thousand programs (16M inputs -> 64k
            # partials).
            finalize_body = [
                "acc_f = " + spec.neutral(),
                "_offs = tl.arange(0, FBLOCK)",
            ]
            if spec.op == "mean":
                finalize_body.append("total = 0")
            finalize_body.append("for fbase in tl.range(0, wsn, FBLOCK):")
            finalize_body.append("    findex = fbase + _offs")
            finalize_body.append("    fmask = findex < wsn")
            finalize_body.append(
                f"    fvals = tl.load(ws_ptr + findex, mask=fmask, "
                f"other={spec.neutral()})"
            )
            finalize_body.append(
                f"    acc_f = {spec.combine_expr('acc_f', spec.finalize_call('fvals, axis=0'))}"
            )
            if spec.op == "mean":
                finalize_body.append("    total += tl.sum((findex < wsn).to(tl.int32), axis=0)")
            reduced_expr = "acc_f"
            if spec.op == "mean":
                reduced_expr = "acc_f * (1.0 / total.to(tl.float32))"
            if self.epilogue is not None:
                epilogue_lines, epilogue_last = self._epilogue_lines("reduced")
                finalize_body.append(f"reduced = {reduced_expr}")
                finalize_body.extend(epilogue_lines)
                finalize_body.append(f"tl.store(out_ptr0, {epilogue_last})")
            else:
                finalize_body.append(f"tl.store(out_ptr0, {reduced_expr})")
            source += "@triton.jit\n"
            source += (
                f"def {finalize_name}(ws_ptr, out_ptr0, wsn, "
                "FBLOCK: tl.constexpr):\n"
            )
            source += textwrap.indent("\n".join(finalize_body), "    ") + "\n\n"
            persistent_split = (
                fixed_config is not None and len(fixed_config) > 2
            )
            # Static workspace (the preallocated-buffer pattern):
            # wsn is a per-kernel constant, so the partial buffer is
            # allocated once and reused.  Every launch overwrites all wsn
            # entries before the finalize reads them, and the returned
            # scalar `out` stays fresh.  Per-call allocation was ~10-20us of
            # CPU that also drowned the tuner's ranking of fast candidates.
            source += "_ws = None\n"
            source += "_rec = None\n\n"
            source += "def kernel_launch(inputs):\n"
            source += "    global _ws, _rec\n"
            source += "    xnumel = " + xnumel_source + "\n"
            if persistent_split:
                # main grid is the fixed program count, not cdiv
                source += f"    wsn = {fixed_config[2]}\n"
                fb = min(_next_power_of_two(fixed_config[2]), 2048)
            else:
                source += f"    wsn = triton.cdiv(xnumel, {block})\n"
                source += "    fb = min(triton.next_power_of_2(wsn), 2048)\n"
                fb = "fb"
            source += (
                "    out = tp.empty((), dtype=inputs[0].dtype, "
                "device=inputs[0].device)\n"
            )
            source += "    if _ws is None:\n"
            source += (
                "        _ws = tp.empty((wsn,), dtype=inputs[0].dtype, "
                "device=inputs[0].device)\n"
            )
            source += "    ws = _ws\n"
            call_args = [*(f"inputs[{index}]" for index in range(self.input_count)), "ws"]
            args_txt = ", ".join(call_args)
            ptrs = " | ".join(
                [
                    *(f"inputs[{i}].data_ptr()" for i in range(self.input_count)),
                    "ws.data_ptr()",
                    "out.data_ptr()",
                ]
            )
            guard = f"({ptrs}) % 16 == 0"
            # Fast path (fastlaunch): two direct CompiledKernel.run calls —
            # the main sweep and the finalize — with the recorded grid,
            # function handle and packed metadata, instead of two full
            # JITFunction dispatches (~20us of binder/spec-key Python each).
            # Guards: divisibility-16 alignment of every tensor arg (the
            # recorded binary is the aligned specialization), xnumel equal
            # to the recorded value (int ==1 / %16 specialization and the
            # literal loop bound are both pinned to it) and no profiling
            # hooks.  Any miss falls through to the dispatches below.
            source += "    _r = _rec\n"
            source += (
                f"    if _r is not None and {guard} and _fl.hooks_clear() "
                "and _r[6] == xnumel:\n"
            )
            source += "        try:\n"
            source += "            _s = _fl.current_stream()\n"
            source += (
                f"            _r[0](wsn, 1, 1, _s, _r[1], _r[2], None, None, "
                f"None, {args_txt}, xnumel, {block})\n"
            )
            source += (
                f"            _r[3](1, 1, 1, _s, _r[4], _r[5], None, None, "
                f"None, ws, out, wsn, {fb})\n"
            )
            source += "            _fl.bump()\n"
            source += "            return out\n"
            source += "        except Exception:\n"
            source += "            _rec = None\n"
            source += "    _s0 = _s1 = -1\n"
            source += (
                f"    if _r is None and _fl.hooks_clear() and {guard}:\n"
            )
            source += f"        _s0 = _fl.cache_size({kernel_name})\n"
            source += f"        _s1 = _fl.cache_size({finalize_name})\n"
            source += (
                f"    {kernel_name}[(wsn,)]({args_txt}, "
                f"xnumel, XBLOCK={block}, num_warps={warps})\n"
            )
            source += (
                f"    {finalize_name}[(1,)](ws, out, wsn, "
                f"FBLOCK={fb}, num_warps=4)\n"
            )
            source += "    if _s0 >= 0:\n"
            source += f"        _g0 = _fl.take_kernel({kernel_name}, _s0)\n"
            source += f"        _g1 = _fl.take_kernel({finalize_name}, _s1)\n"
            source += "        if _g0 is not None and _g1 is not None:\n"
            source += "            _rec = _g0 + _g1 + (xnumel,)\n"
            source += "    return out\n"
        else:
            source += "_rec = None\n\n"
            source += "def kernel_launch(inputs):\n"
            source += "    global _rec\n"
            if self.reference_shape is not None:
                # Compile-time output numel: the runtime feed order may put
                # a broadcast operand first, whose numel would truncate the
                # grid.
                source += f"    xnumel = {repr(_prod(self.reference_shape))}\n"
            else:
                source += "    xnumel = inputs[0].numel()\n"
            if self.reference_shape is not None:
                # Allocate by the COMPILED output shape: input order in the
                # runtime feed may differ from the example order, so
                # empty_like(inputs[0]) can pick up a broadcast operand's
                # shape and silently truncate the result.
                out_shape = repr(tuple(int(d) for d in self.reference_shape))
                if len(self.output_refs) == 1:
                    source += (
                        f"    outputs = [tp.empty({out_shape}, "
                        "dtype=inputs[0].dtype, device=inputs[0].device)]\n"
                    )
                else:
                    source += (
                        f"    outputs = [tp.empty({out_shape}, "
                        "dtype=inputs[0].dtype, device=inputs[0].device) "
                        f"for _ in range({len(self.output_refs)})]\n"
                    )
            else:
                source += "    outputs = [tp.empty_like(inputs[0], requires_grad=False) for _ in range(" \
                    f"{len(self.output_refs)})]\n"
            call_args = [
                *(f"inputs[{index}]" for index in range(self.input_count)),
                *(f"outputs[{index}]" for index in range(len(self.output_refs))),
                "xnumel",
            ]
            call_args_txt = ", ".join(call_args)
            # When a fixed config is given, always bake constexpr overrides
            # (XBLOCK, num_warps, and the vectorize width when packed) into
            # the kernel call so the triton dispatch does not have to resolve
            # them per-call through meta. This works whether the grid is
            # literal or via a lambda.
            constexpr_extra = f", VEC={vec}" if vec > 1 else ""
            constexpr_kw = ""
            if fixed_config is not None:
                constexpr_kw = (
                    f", XBLOCK={fixed_config[0]}, "
                    f"num_warps={fixed_config[1]}{constexpr_extra}"
                )
            if fixed_config is not None:
                ptrs = " | ".join(
                    [
                        *(
                            f"inputs[{i}].data_ptr()"
                            for i in range(self.input_count)
                        ),
                        *(
                            f"outputs[{i}].data_ptr()"
                            for i in range(len(self.output_refs))
                        ),
                    ]
                )
                guard = f"({ptrs}) % 16 == 0"
                # Fast path (fastlaunch): see the reduction branches.  The
                # grid is recomputed from the guarded xnumel when the shape
                # is not compile-time, so the recorded binary always sees
                # the geometry it was compiled for.  A packed iteration
                # space launches one program per XBLOCK*VEC elements.
                if self.reference_shape is not None:
                    grid_src = repr(
                        -(-_prod(self.reference_shape)
                          // (fixed_config[0] * vec))
                    )
                else:
                    grid_src = (
                        f"-(-xnumel // {fixed_config[0] * vec})"
                    )
                source += "    _r = _rec\n"
                source += (
                    f"    if _r is not None and {guard} and "
                    "_fl.hooks_clear() and _r[3] == xnumel:\n"
                )
                source += "        try:\n"
                source += "            _s = _fl.current_stream()\n"
                source += (
                    f"            _r[0]({grid_src}, 1, 1, _s, _r[1], _r[2], "
                    f"None, None, None, {call_args_txt}, {fixed_config[0]}"
                    f"{', ' + str(vec) if vec > 1 else ''})\n"
                )
                source += "            _fl.bump()\n"
                if len(self.output_refs) == 1:
                    source += "            return outputs[0]\n"
                else:
                    source += "            return outputs\n"
                source += "        except Exception:\n"
                source += "            _rec = None\n"
                source += "    _snap = -1\n"
                source += (
                    f"    if _r is None and _fl.hooks_clear() and {guard}:\n"
                )
                source += f"        _snap = _fl.cache_size({kernel_name})\n"
            if fixed_config is not None and self.reference_shape is not None:
                # Pinned config: the grid is a compile-time constant, so emit
                # it literally instead of paying triton's per-call
                # grid-lambda/meta resolution (the launch path is the pw
                # chain's bottleneck once kernels reach hardware throughput).
                grid_n = -(
                    -_prod(self.reference_shape) // (fixed_config[0] * vec)
                )
                source += (
                    f"    {kernel_name}[({grid_n},)]({call_args_txt}{constexpr_kw})\n"
                )
            else:
                source += (
                    "    grid = lambda meta: "
                    "(triton.cdiv(xnumel, meta['XBLOCK']),)\n"
                    f"    {kernel_name}[grid]({call_args_txt}{constexpr_kw})\n"
                )
            if fixed_config is not None:
                source += "    if _snap >= 0:\n"
                source += (
                    f"        _g = _fl.take_kernel({kernel_name}, _snap)\n"
                )
                source += "        if _g is not None:\n"
                source += "            _rec = _g + (xnumel,)\n"
            if len(self.output_refs) == 1:
                source += "    return outputs[0]\n"
            else:
                source += "    return outputs\n"
        return source


def emit_tile_epilogue_lines(
    program: list[int], constants: list[float], esrc: int, source_reg: str
) -> tuple[list[str], str]:
    """Emit a pointwise chain applied to one tile register.

    The store-time epilogue renderer shared by reduction tails and the GEMM
    tile: ``esrc`` names the chain's single tensor input (mapped onto
    ``source_reg``, the register already holding the pre-epilogue value);
    negatives index epilogue constants; other positive refs are temporaries
    numbered from 1.  Returns the source lines and the final register
    holding the chain result.
    """

    if len(program) % 3:
        raise ValueError("epilogue program must contain triples")
    # The instance only carries the payload the shared instruction renderer
    # reads (``_epilogue_lines`` touches nothing else); no kernel is
    # generated from it.
    emitter = object.__new__(TritonProgramCodegen)
    emitter.epilogue = (program, constants, esrc)
    return emitter._epilogue_lines(source_reg)


def _compile_program(
    program: list[int],
    constants: list[float],
    output_refs: tuple[int, ...],
    example_inputs: list[Any],
    *,
    fixed_config: tuple[int, int] | None = None,
    reduction: str | None = None,
    input_shapes: tuple[tuple[int, ...], ...] | None = None,
    reference_shape: tuple[int, ...] | None = None,
    value_dtype: str | None = None,
    epilogue: tuple[list[int], list[float], int] | None = None,
    reduction_outputs: tuple[str, ...] | None = None,
):
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed")
    if not _supports_runtime_inputs(
        example_inputs, allow_grad=True, reference_shape=reference_shape
    ):
        raise NotImplementedError("Triton requires matching contiguous CUDA tensors")
    digest = hashlib.sha256(
        (
            repr(
                (
                    _CODEGEN_VERSION,
                    program,
                    constants,
                    output_refs,
                    reduction,
                    epilogue,
                    reduction_outputs,
                )
            )
            + repr(
                [
                    (tuple(value.shape), repr(value.dtype), repr(value.device))
                    for value in example_inputs
                ]
            )
        ).encode()
    ).hexdigest()[:16]
    # is content-addressed and persisted; a process-level memo keeps the
    # exec'd launch callable so repeated compile() calls skip regeneration.
    memo_key = f"{digest}:{fixed_config}"
    cached_launch = _launch_memo.get(memo_key)
    if cached_launch is not None:
        return cached_launch
    kernel_name = f"stax_triton_program_{digest}"
    source = TritonProgramCodegen(
        program, constants, output_refs, len(example_inputs),
        reduction=reduction,
        input_shapes=input_shapes,
        reference_shape=reference_shape,
        value_dtype=value_dtype,
        epilogue=epilogue,
        reduction_outputs=reduction_outputs,
    ).generate(kernel_name, fixed_config=fixed_config)
    try:
        from ..codecache import default_cache

        cache = default_cache("triton")
        key = cache.cache_key(source)
        if cache.load(key, ext="py") is None:
            cache.store(key, source.encode(), ext="py")
    except Exception:  # noqa: BLE001 - cache is best-effort
        pass
    fake_file = f"<tensorplay-stax-triton-program-{digest}>"
    linecache.cache[fake_file] = (
        len(source),
        None,
        source.splitlines(True),
        fake_file,
    )
    namespace: dict[str, Any] = {"triton": triton, "tl": tl}
    exec(compile(source, fake_file, "exec"), namespace, namespace)
    _launch_memo[memo_key] = namespace["kernel_launch"]
    return namespace["kernel_launch"]


def _dims_decision_key(
    digest: str,
    reduction: "ReductionSpec",
    onumel: int,
    rnumel: int,
    device_repr: str,
    value_dtype: str | None,
    epilogue_repr: str,
    *,
    tier: str = "table",
    outputs_repr: str = "",
) -> str:
    """Persisted-decision key for the axis-reduction family (M5d).

    Covers codegen generation, tuning salt, program content, reduction spec,
    output-port kinds, shape buckets, device, value dtype and epilogue so a
    hit can never pin a decision from an older emitter or candidate table.
    The selection tier is part of the key: a coordinate-descent refinement
    and a baseline-table pick for the same program are separate records.
    """

    from ..runtime import stax_autotune

    digest_source = (
        _CODEGEN_VERSION
        + "|"
        + stax_autotune.TUNING_VERSION
        + f"|{tier}|"
        + digest
        + f"|{reduction.op}|{reduction.dims}|{int(reduction.keepdim)}"
        + f"|{stax_autotune.xnumel_bucket(onumel)}|{stax_autotune.xnumel_bucket(rnumel)}"
        + f"|{device_repr}|{value_dtype}|{epilogue_repr}|{outputs_repr}"
    )
    return hashlib.sha256(f"dimred|{digest_source}".encode()).hexdigest()[:24]


def _autotune_dims_program(
    role: str,
    program: list[int],
    constants: list[float],
    output_refs: tuple[int, ...],
    example_inputs: list[Any],
    *,
    reduction: "ReductionSpec",
    input_shapes: tuple[tuple[int, ...], ...] | None,
    reference_shape: tuple[int, ...] | None,
    value_dtype: str | None = None,
    epilogue: tuple[list[int], list[float], int] | None = None,
    reduction_outputs: tuple[str, ...] | None = None,
    max_autotune: bool = False,
    coordinate_descent_tuning: bool = False,
):
    """Benchmark the axis-reduction candidate table once; persist the decision.

    The decision cache key covers program content, reduction spec, output
    kinds, shape buckets and device, so a hit skips both benchmarking and
    recompiles.  The key also carries the selection tier (baseline table,
    exhaustive max-autotune table, coordinate-descent refinement) so
    policies never read each other's records.  The exhaustive tier widens the
    base geometries with the loop-pass space (r-loop unroll factors and
    main/tail split variants).
    """

    assert reference_shape is not None and isinstance(reduction, ReductionSpec)
    from ..runtime import stax_autotune
    from . import loop_pass

    tier = (
        "coordesc"
        if coordinate_descent_tuning
        else ("exhaustive" if max_autotune else "table")
    )

    def build(config: tuple[int, ...]):
        return _compile_program(
            program,
            constants,
            output_refs,
            example_inputs,
            fixed_config=config,
            reduction=reduction,
            input_shapes=input_shapes,
            reference_shape=reference_shape,
            value_dtype=value_dtype,
            epilogue=epilogue,
            reduction_outputs=reduction_outputs,
        )

    rank = len(reference_shape)
    reduced = {dim % rank for dim in reduction.dims}
    onumel = max(
        1,
        _prod(
            [
                size
                for index, size in enumerate(reference_shape)
                if index not in reduced
            ]
        ),
    )
    rnumel = max(1, reduction.reduction_numel(reference_shape))
    outputs_repr = ",".join(reduction_outputs or ())
    decision_key = _dims_decision_key(
        stax_autotune.program_digest(program, constants, output_refs),
        reduction,
        onumel,
        rnumel,
        repr(example_inputs[0].device),
        value_dtype,
        epilogue is not None and repr(epilogue) or "",
        tier=tier,
        outputs_repr=outputs_repr,
    )

    try:
        from ..codecache import default_cache

        cache = default_cache("triton-autotune")
        payload = cache.load(decision_key, ext="json")
    except Exception:  # noqa: BLE001 - cache is best-effort
        payload = None

    if payload is not None:
        try:
            record = json.loads(payload.decode())
            cached = _dims_record_config(record)
            if _dims_decision_acceptable(cached, tier):
                return build(cached)
        except (ValueError, KeyError, TypeError):
            pass

    if disabled_autotune():
        return build(_STATIC_DIM_TRIPLE)

    candidates: tuple[tuple[int, ...], ...] = _DIM_REDUCTION_CANDIDATES
    if max_autotune:
        candidates = loop_pass.dims_loop_candidates(
            rnumel, candidates
        )
    best_config, best_launch, best_time = stax_autotune.bench_candidates(
        build, candidates, list(example_inputs)
    )
    if best_config is None:
        return build(_STATIC_DIM_TRIPLE)
    if coordinate_descent_tuning:
        from ..runtime import coordinate_descent

        best_config, best_launch = coordinate_descent.refiner_for(
            coordinate_descent.dims_fields(len(best_config))
        )(build, best_config, list(example_inputs))
    try:
        record = {
            "xblock": best_config[0],
            "warps": best_config[1],
            "stages": best_config[3]
            if len(best_config) > 3
            else best_config[-1],
        }
        if len(best_config) > 3:
            record["rblock"] = best_config[2]
        if len(best_config) > 5:
            record["unroll"] = best_config[4]
            record["split"] = int(bool(best_config[5]))
        cache.store(decision_key, json.dumps(record).encode(), ext="json")
    except Exception:  # noqa: BLE001 - cache is best-effort
        pass
    return best_launch


def _dims_record_config(record: dict) -> tuple[int, ...]:
    """Rebuild a config tuple from a persisted decision record.

    ``unroll``/``split`` are written only by the loop-pass tiers; a record
    without them decodes to the base 3/4-tuple.
    """

    if record.get("rblock") is not None:
        config = (
            int(record["xblock"]),
            int(record["warps"]),
            int(record["rblock"]),
            int(record["stages"]),
        )
    else:
        config = (
            int(record["xblock"]),
            int(record["warps"]),
            int(record["stages"]),
        )
    if record.get("unroll") is not None:
        config = config + (
            int(record["unroll"]),
            int(record.get("split", 0)),
        )
    return config


def _dims_decision_acceptable(
    config: tuple[int, ...], tier: str
) -> bool:
    """Validate a loaded decision against the policy that produced it.

    The table tier only accepts members of the curated candidate table; the
    exhaustive tier also accepts loop-pass-extended six-tuples whose geometry
    prefix is a table member and whose loop knobs stay in the searched
    space; the coordinate-descent tier accepts any structurally valid config
    because the descent may leave the table.
    """

    if tier == "coordesc":
        return len(config) in (3, 4, 6) and all(
            isinstance(value, int) and value >= 1 for value in config
        )
    if tier == "exhaustive" and len(config) == 6:
        from . import loop_pass

        # The geometry prefix must be a table member (3-tuple geometries
        # compare by (xblock, warps) because the extension materializes
        # their derived RBLOCK), and the loop knobs must stay in the
        # searched space.
        prefixes = {
            (entry[0], entry[1], len(entry)) for entry in _DIM_REDUCTION_CANDIDATES
        }
        matched = (config[0], config[1], 4) in prefixes or (
            config[0],
            config[1],
            3,
        ) in prefixes
        return (
            matched
            and int(config[4]) in loop_pass.UNROLL_FACTORS
            and int(config[5]) in (0, 1)
        )
    return any(entry == config for entry in _DIM_REDUCTION_CANDIDATES)


def _autotune_split_program(
    role: str,
    program: list[int],
    constants: list[float],
    output_refs: tuple[int, ...],
    example_inputs: list[Any],
    *,
    reduction: "ReductionSpec",
    input_shapes,
    reference_shape,
    value_dtype=None,
    epilogue=None,
    max_autotune: bool = False,
    coordinate_descent_tuning: bool = False,
):
    """Bench classic vs persistent split-reduction forms once per bucket.

    The decision key carries the selection tier so baseline-table records
    and coordinate-descent refinements never shadow each other.
    """

    from ..runtime import stax_autotune

    tier = (
        "coordesc"
        if coordinate_descent_tuning
        else ("exhaustive" if max_autotune else "table")
    )

    def build(config):
        return _compile_program(
            program,
            constants,
            output_refs,
            example_inputs,
            fixed_config=config,
            reduction=reduction,
            input_shapes=input_shapes,
            reference_shape=reference_shape,
            value_dtype=value_dtype,
            epilogue=epilogue,
        )

    digest_source = (
        _CODEGEN_VERSION
        + "|split|"
        + stax_autotune.TUNING_VERSION
        + f"|{tier}|"
        + stax_autotune.program_digest(program, constants, output_refs)
        + f"|{reduction.op}|{stax_autotune.xnumel_bucket(_prod(reference_shape))}"
        + f"|{repr(example_inputs[0].device)}|{epilogue is not None}"
    )
    decision_key = hashlib.sha256(digest_source.encode()).hexdigest()[:24]

    try:
        from ..codecache import default_cache

        cache = default_cache("triton-autotune")
        payload = cache.load(decision_key, ext="json")
    except Exception:  # noqa: BLE001 - cache is best-effort
        payload = None

    def _valid(record):
        cfg = tuple(int(record[k]) for k in ("xblock", "warps"))
        if len(record) > 2:
            cfg = cfg + (int(record["nprog"]),)
        return cfg

    def _acceptable(cfg):
        # The coordinate-descent tier may leave the curated table; every
        # other tier only accepts table members.
        if tier == "coordesc":
            return len(cfg) in (2, 3) and all(value >= 1 for value in cfg)
        return any(tuple(c) == cfg for c in _SPLIT_CANDIDATES)

    if payload is not None:
        try:
            record = json.loads(payload.decode())
            cached = _valid(record)
            if _acceptable(cached):
                return build(cached)
        except (ValueError, KeyError, TypeError):
            pass

    if disabled_autotune():
        return build(_STATIC_REDUCTION_CONFIG)

    best_cfg, best_launch, best_time = stax_autotune.bench_candidates(
        build, _SPLIT_CANDIDATES, list(example_inputs)
    )
    if best_cfg is None:
        return build(
            _STATIC_SPLIT_PERSISTENT
            if any(len(c) > 2 for c in _SPLIT_CANDIDATES)
            else _STATIC_REDUCTION_CONFIG
        )
    if coordinate_descent_tuning:
        from ..runtime import coordinate_descent

        best_cfg, best_launch = coordinate_descent.refiner_for(
            coordinate_descent.split_fields(len(best_cfg))
        )(build, best_cfg, list(example_inputs))
    record = {"xblock": best_cfg[0], "warps": best_cfg[1]}
    if len(best_cfg) > 2:
        record["nprog"] = best_cfg[2]
    try:
        cache.store(decision_key, json.dumps(record).encode(), ext="json")
    except Exception:  # noqa: BLE001 - cache is best-effort
        pass
    return best_launch


def _autotune_launch(
    role: str,
    program: list[int],
    constants: list[float],
    output_refs: tuple[int, ...],
    example_inputs: list[Any],
    *,
    reduction: str | None = None,
    reduction_mode: str | None = None,
    input_shapes: tuple[tuple[int, ...], ...] | None = None,
    reference_shape: tuple[int, ...] | None = None,
    bucket_numel: int | None = None,
    value_dtype: str | None = None,
    epilogue: tuple[list[int], list[float], int] | None = None,
    reduction_outputs: tuple[str, ...] | None = None,
    max_autotune: bool = False,
    coordinate_descent_tuning: bool = False,
):
    """Compile a program, autotuning the launch config when possible (M2).

    Benchmark candidate configs once at compile time and emit a
    fixed-config kernel; persist the decision so later processes skip
    benchmarking.  The max-autotune knobs widen the search: an exhaustive
    candidate table for pointwise programs (extended with the loop-pass
    vectorize widths) and coordinate-descent refinement of the benchmark
    winner.  Any failure falls back to a static pinned config for reductions
    (the split workspace is baked per config) or the plain
    ``@triton.autotune`` emission for pointwise programs.
    """

    def build(config: tuple[int, int] | None):
        return _compile_program(
            program,
            constants,
            output_refs,
            example_inputs,
            fixed_config=config,
            reduction=reduction,
            input_shapes=input_shapes,
            reference_shape=reference_shape,
            value_dtype=value_dtype,
            epilogue=epilogue,
            reduction_outputs=reduction_outputs,
        )

    spec = (
        reduction
        if isinstance(reduction, ReductionSpec)
        else (ReductionSpec("sum") if reduction == "sum" else None)
    )
    if spec is not None and spec.is_full and reduction_mode == "single":
        # Deterministic geometry: one block covers the whole input; nothing
        # to tune.
        assert reference_shape is not None
        return build(_single_block_config(_prod(reference_shape)))
    if spec is not None and not spec.is_full:
        # Axis reductions (incl. argmax): benchmark the candidate table once,
        # persist the decision (M5d).
        return _autotune_dims_program(
            role,
            program,
            constants,
            output_refs,
            example_inputs,
            reduction=spec,
            input_shapes=input_shapes,
            reference_shape=reference_shape,
            value_dtype=value_dtype,
            epilogue=epilogue,
            reduction_outputs=reduction_outputs,
            max_autotune=max_autotune,
            coordinate_descent_tuning=coordinate_descent_tuning,
        )
    if spec is not None and spec.is_full and reduction_mode == "split":
        return _autotune_split_program(
            role,
            program,
            constants,
            output_refs,
            example_inputs,
            reduction=spec,
            input_shapes=input_shapes,
            reference_shape=reference_shape,
            value_dtype=value_dtype,
            epilogue=epilogue,
            max_autotune=max_autotune,
            coordinate_descent_tuning=coordinate_descent_tuning,
        )
    if disabled_autotune():
        if reduction:
            return build(_STATIC_REDUCTION_CONFIG)
        return _compile_program(
            program, constants, output_refs, example_inputs,
            reduction=reduction,
            input_shapes=input_shapes,
            reference_shape=reference_shape,
        )
    try:
        from ..runtime import coordinate_descent, stax_autotune

        digest = stax_autotune.program_digest(program, constants, output_refs)
        if bucket_numel is not None:
            xnumel = bucket_numel
        else:
            xnumel = int(example_inputs[0].numel())
        device_key = repr(example_inputs[0].device)

        def build_fixed(config: tuple[int, int]):
            return build(config)

        pointwise_candidates = (
            stax_autotune.EXHAUSTIVE_CANDIDATE_CONFIGS
            if max_autotune
            else None
        )
        if max_autotune:
            from . import loop_pass

            itemsize = int(
                getattr(example_inputs[0].dtype, "itemsize", 4) or 4
            )
            pointwise_candidates = loop_pass.pointwise_loop_candidates(
                xnumel,
                itemsize,
                stax_autotune.EXHAUSTIVE_CANDIDATE_CONFIGS,
            )

        # Key on the bare program digest: load_decision() consumers key the
        # same way, and role namespacing is redundant given bucket+device.
        config, launch = stax_autotune.pick_config(
            digest,
            xnumel,
            device_key,
            build_fixed,
            list(example_inputs),
            candidates=pointwise_candidates,
            refiner=(
                coordinate_descent.refiner_for(
                    coordinate_descent.POINTWISE_FIELDS
                )
                if coordinate_descent_tuning
                else None
            ),
            tier=(
                "coordesc"
                if coordinate_descent_tuning
                else ("exhaustive" if max_autotune else "table")
            ),
        )
        del config  # baked into the returned fixed-config launch
        return launch
    except Exception:  # noqa: BLE001 - autotuning is an optimization only
        if reduction:
            try:
                return build(_STATIC_REDUCTION_CONFIG)
            except Exception:  # noqa: BLE001 - fall through to legacy path
                pass
        return build(None)


# Reduction-tail op families: scalar value tails (no axes for min/max —
# with an axis they become pair tails), pair tails (min/max over one axis,
# values+indices), and the index reduction.
_REDUCTION_SCALAR_TAILS = frozenset({"sum", "mean", "amax", "amin", "max", "min"})
_REDUCTION_PAIR_TAILS = frozenset({"max", "min"})
_REDUCTION_INDEX_TAILS = frozenset({"argmax"})
_REDUCTION_TAIL_OPS = _REDUCTION_SCALAR_TAILS | _REDUCTION_PAIR_TAILS | _REDUCTION_INDEX_TAILS
# Extremum reductions: the tangent selects the positions attaining the
# extremum instead of distributing uniformly (M5f select-mask VJP).
_MASK_REDUCTION_OPS = frozenset({"amax", "amin", "max", "min"})


def _reduction_spec_from_node(node: Node) -> ReductionSpec | None:
    """Parse a ``call_method`` reduction node into a :class:`ReductionSpec`.

    Returns ``None`` for anything this backend cannot fold yet (unknown
    kwargs like ``dtype``, multi-axis value-index pairs, tensor ``dim``
    values, ``amax()`` without axes, ``argmax()`` without axes).
    """

    op = _target_name(node.target)
    if op not in _REDUCTION_TAIL_OPS:
        return None
    # call_method nodes carry the receiver as args[0]; parse only the rest.
    args = [
        value
        for value in node.args[1:]
        if isinstance(value, (int, float, str, bool, tuple, list))
    ]
    kwargs = dict(node.kwargs)

    keepdim: bool = False
    dim_value: Any = None
    if "keepdim" in kwargs:
        keepdim = kwargs.pop("keepdim")
        if not isinstance(keepdim, bool):
            return None
    if "dim" in kwargs:
        dim_value = kwargs.pop("dim")
    if kwargs:
        return None

    if dim_value is None and args:
        if len(args) == 1:
            dim_value = args[0]
        elif len(args) == 2:
            dim_value = args[0]
            keepdim_extra = args[1]
            if not isinstance(keepdim_extra, bool):
                return None
            keepdim = keepdim_extra
        else:
            return None

    if dim_value is None:
        dims: tuple[int, ...] = ()
    elif isinstance(dim_value, bool):
        return None
    elif isinstance(dim_value, int):
        dims = (dim_value,)
    elif isinstance(dim_value, (tuple, list)) and all(
        isinstance(item, int) and not isinstance(item, bool)
        for item in dim_value
    ):
        dims = tuple(dim_value)
    else:
        return None

    if op in ("amax", "amin") and not dims:
        return None
    if op in ("max", "min") and dims and len(dims) != 1:
        # The value-index pair form takes exactly one axis; multi-axis
        # spellings stay on the eager path.
        return None
    try:
        return ReductionSpec(op, dims, keepdim=keepdim)
    except ValueError:
        return None


def _split_reduction_epilogue(
    graph_module: GraphModule,
):
    """Detect a reduction tail over a pointwise chain (L5-M5b).

    Returns ``(tail_node, producer, ReductionSpec)`` when the graph's single
    output is a supported ``chain_result.sum()/mean()/amax()/max()`` and
    every other node is pointwise-fusible — the shape lowered to one
    kernel with a fused reduction epilogue.  Otherwise returns ``None``.
    """

    output_values = [
        value
        for out_node in graph_module.graph.outputs
        for value in _nodes(out_node.args)
    ]
    if len(output_values) != 1:
        return None
    tail = output_values[0]
    if not isinstance(tail, Node) or tail.op != "call_method":
        return None

    spec = _reduction_spec_from_node(tail)
    if spec is None:
        return None

    producer = tail.args[0]
    if not isinstance(producer, Node) or producer.op in {"placeholder", "output"}:
        return None

    # Every node other than the tail itself must be pointwise-fusible, using
    # the same constraints as _build_pointwise_program.
    for node in graph_module.graph.nodes:
        if node is tail:
            continue
        if node.op in {"placeholder", "output"}:
            continue
        if node.op not in {"call_function", "call_method"} or node.kwargs:
            return None
        if _target_name(node.target) not in _CPU_FUSED_OPS:
            return None
    return tail, producer, spec


def _split_sum_epilogue(
    graph_module: GraphModule,
):
    """Legacy full-sum entry point (kept for older callers/tests)."""

    detected = _split_reduction_epilogue(graph_module)
    if detected is None:
        return None
    tail, producer, spec = detected
    if spec.op != "sum" or not spec.is_full:
        return None
    return tail, producer, "sum"


@dataclass(frozen=True)
class _ExternSource:
    """Where a segment's runtime input comes from.

    Frozen because the wiring keys a lookup table by it: an extern segment
    resolves each operand by matching the source it was planned against.
    ``port`` selects one output buffer of the producing kernel (0 = main
    export; horizontal fusion assigns further ports to extra stores).
    """

    kind: str  # "arg" (graph placeholder position) | "seg" (segment index)
    index: int
    port: int = 0


@dataclass
class _ExternPlan:
    """A one-node segment executed eagerly between fused kernels.

    The scheduler turns every operator outside the fused surface into its
    own ``"extern"`` segment, so a captured graph can interleave compiled
    pointwise/reduction kernels with unsupported operators (softmax, matmul,
    casts the program cannot express, ...).  ``launch`` resolves the node's
    dependencies through ``extern_sources`` — the same wiring contract fused
    segments use — and calls the operator on real tensors.

    Training closes each extern segment with a tangent rule
    (``backward_launch``): a closed-form rule when one exists
    (``_build_extern_analytic_vjp``), otherwise the engine rule
    (``_build_extern_engine_vjp`` — recompute + nested ``grad``).
    ``vjp_ready`` marks whether a rule was found.
    """

    launch: Any
    extern_sources: tuple
    #: example output buffer (shape/dtype of this segment's export)
    example: Any = None
    output_shape: tuple = ()
    #: True when the operator has a closed-form tangent rule
    vjp_ready: bool = False
    #: fields mirroring ``_SegmentPlan`` so the training sweep treats both
    #: plan kinds uniformly
    spec: Any = None
    instructions: tuple = ()
    needs_broadcast: bool = False
    tangent_plan: tuple | None = None
    backward_launch: Any = None


def _build_extern_analytic_vjp(node: Any, position_of: Any):
    """Closed-form tangent rule for one eager operator.

    Analytic rules are preferred over the engine fallback because they
    recompute only what the rule needs with plain eager ops (no nested
    engine run, no leaf clones) — mixed-region training closes each extern
    segment with one of these when covered.  The rule receives the segment
    feed and the export tangent and returns one gradient per feed position.
    ``None`` when the operator has no covered rule — the engine rule
    (``_build_extern_engine_vjp``) then takes over.
    """

    name = str(getattr(node.target, "__name__", node.target))
    if not node.args or not isinstance(node.args[0], Node):
        return None
    x_pos = position_of(node.args[0])
    if x_pos is None:
        return None

    def recompute(feed: list, rule) -> Any:
        return rule(feed[x_pos])

    def single(rule):
        """Wrap a unary input rule ``(x, y_or_none) -> dx``."""

        def vjp(feed: list, tangent: Any) -> tuple:
            grads = [None] * len(feed)
            grads[x_pos] = rule(feed[x_pos], feed, tangent)
            return tuple(grads)

        return vjp

    def arg_or_kwarg(index: int, key: str, default: Any = None) -> Any:
        if len(node.args) > index + 1:
            return node.args[index + 1]
        return (node.kwargs or {}).get(key, default)

    if name == "softmax":
        dim = arg_or_kwarg(0, "dim", 0)

        def rule(x: Any, feed: list, tangent: Any) -> Any:
            y = x.softmax(dim=dim)
            inner = (tangent * y).sum(dim=dim)
            shape = [1 if axis == dim else extent
                     for axis, extent in enumerate(y.shape)]
            return y * (tangent - inner.reshape(shape))

        return single(rule)

    if name in ("reshape", "view", "flatten"):
        def rule(x: Any, feed: list, tangent: Any) -> Any:
            return tangent.reshape([int(extent) for extent in x.shape])

        return single(rule)

    if name == "transpose":
        d0 = arg_or_kwarg(0, "dim0", 0)
        d1 = arg_or_kwarg(1, "dim1", 1)

        def rule(x: Any, feed: list, tangent: Any) -> Any:
            return tangent.transpose(d0, d1)

        return single(rule)

    if name == "permute":
        dims = node.args[1:] or (node.kwargs or {}).get("dims", ())
        inverse = [0] * len(dims)
        for out_axis, in_axis in enumerate(dims):
            inverse[int(in_axis)] = out_axis

        def rule(x: Any, feed: list, tangent: Any) -> Any:
            return tangent.permute(inverse)

        return single(rule)

    if name == "contiguous":
        return single(lambda x, feed, tangent: tangent)

    if name == "neg":
        return single(lambda x, feed, tangent: -tangent)

    if name == "exp":
        def rule(x: Any, feed: list, tangent: Any) -> Any:
            return tangent * x.exp()

        return single(rule)

    if name == "sigmoid":
        def rule(x: Any, feed: list, tangent: Any) -> Any:
            y = x.sigmoid()
            return tangent * y * (1.0 - y)

        return single(rule)

    if name == "tanh":
        def rule(x: Any, feed: list, tangent: Any) -> Any:
            y = x.tanh()
            return tangent * (1.0 - y * y)

        return single(rule)

    if name == "sqrt":
        def rule(x: Any, feed: list, tangent: Any) -> Any:
            return 0.5 * tangent / x.sqrt()

        return single(rule)

    if name == "rsqrt":
        def rule(x: Any, feed: list, tangent: Any) -> Any:
            y = x.rsqrt()
            return -0.5 * tangent * y * y * y

        return single(rule)

    if name == "abs":
        def rule(x: Any, feed: list, tangent: Any) -> Any:
            return tangent * x.sign()

        return single(rule)

    if name == "relu":
        def rule(x: Any, feed: list, tangent: Any) -> Any:
            return tangent * (x > 0).to(x.dtype)

        return single(rule)

    if name == "to":
        def rule(x: Any, feed: list, tangent: Any) -> Any:
            if not x.dtype.is_floating_point:
                # An integer source (an index stream) has no differentiable
                # path: the cast carries no tangent back.
                return None
            return tangent.to(x.dtype)

        return single(rule)

    return None


def _build_extern_engine_vjp(node: Any, resolve: Any, position_of: Any):
    """Engine tangent rule for one eager operator: recompute + grad.

    Covers every differentiable operator the closed-form table misses —
    cumsum, matmul, reductions without a uniform rule, composites — by
    running the operator a second time on fresh leaf clones under grad mode
    and differentiating with a nested ``autograd.grad`` call.  The engine
    serves such re-entrant calls from any thread: a caller busy evaluating
    a node runs the nested graph on its own local queue.  Analytic rules
    stay preferred where they exist (no recompute, no re-entrancy); this is
    the general fallback.  ``None`` when no operand is a feed-resolvable
    tensor — nothing to differentiate.
    """

    import tensorplay as _tp

    specs = [("arg", i, value) for i, value in enumerate(node.args)]
    specs += [
        ("kwarg", name, value) for name, value in (node.kwargs or {}).items()
    ]
    # one differentiation slot per tensor operand that resolves to a feed
    # position; non-floating operands (index tensors, masks) never carry a
    # gradient and constants need none
    slots = [
        (kind, key, position_of(value))
        for kind, key, value in specs
        if isinstance(value, Node) and position_of(value) is not None
    ]
    if not slots:
        return None

    def vjp(feed: list, tangent: Any) -> tuple:
        args = [resolve(feed, value) for value in node.args]
        kwargs = {
            name: resolve(feed, value) for name, value in node.kwargs.items()
        }
        # under a create_graph backward the caller's grad mode is on; the
        # recomputed graph and the returned tangents must join that
        # higher-order graph exactly like the closed-form rules' ops do
        track_higher_order = _tp.autograd.is_grad_enabled()
        leaves = []
        leaf_positions = []
        with _tp.autograd.enable_grad():
            for kind, key, position in slots:
                value = feed[position]
                if not value.dtype.is_floating_point:
                    continue
                leaf = value.detach().clone()
                leaf.requires_grad_(True)
                leaves.append(leaf)
                leaf_positions.append(position)
                if kind == "arg":
                    args[key] = leaf
                else:
                    kwargs[key] = leaf
            if not leaves:
                return tuple(None for _ in feed)
            if node.op == "call_function":
                out = node.target(*args, **kwargs)
            else:
                out = getattr(args[0], node.target)(*args[1:], **kwargs)
            if not out.requires_grad:
                return tuple(None for _ in feed)
            grads = _tp.autograd.grad(
                out,
                leaves,
                grad_outputs=[tangent],
                allow_unused=True,
                create_graph=track_higher_order,
            )
        # one gradient per feed position; an operand resolved from the same
        # source twice contributes both partials to that position
        returned: list = [None] * len(feed)
        for position, grad in zip(leaf_positions, grads):
            if grad is not None:
                existing = returned[position]
                returned[position] = (
                    grad if existing is None else existing + grad
                )
        return tuple(returned)

    return vjp


def _extern_segment_plan(
    graph_module: GraphModule,
    seg: Any,
    sources: tuple,
    sample_dtype: Any,
    sample_device: Any,
    placeholder_positions: dict,
    export_ports: dict,
    *,
    sample_feed: list | None = None,
    max_autotune: bool = False,
    training: bool = False,
    attr_position: int | None = None,
    role: str = "fwd?",
    coordinate_descent_tuning: bool = False,
) -> _ExternPlan | None:
    """Build the eager executor for one extern segment; None when unsupported.

    With ``max_autotune`` (inference regions only), a two-dimensional fp32
    matmul segment additionally benches tiled Triton GEMM candidates against
    the native operator and bakes the winner into the launch.

    A ``get_attr`` segment serves a lifted module attribute (parameter or
    buffer).  Inference keeps the closure value; training wires the CURRENT
    attribute value as a trailing autograd input (``attr_position`` names
    it among the region's attributes) so the sweep routes its gradient back
    to the leaf — the identity tangent rule, since the segment is a pure
    pass-through of the attribute value.

    A pointwise chain the scheduler attached as the segment's store-time
    epilogue runs on the operator output: composed as its own tuned
    pointwise kernel after the eager call, and — for a qualifying matmul
    under ``max_autotune`` — baked into the benched GEMM tile instead.
    """

    import tensorplay as _tp

    node = seg.nodes[0]

    epilogue_payload = None
    epi_output_ref = None
    if seg.epilogue:
        # Inference only: training schedules carry no store-time epilogues.
        if training:
            return None
        epi_view, epi_mapping, epi_externals = _extract_segment_view(
            graph_module.graph, list(seg.epilogue), seg.epilogue[-1]
        )
        # v1 epilogue contract: the ONLY tensor input is the operator
        # output; everything else is a scalar constant (guaranteed by the
        # scheduler's single-user attach — re-checked here).
        if len(epi_externals) != 1 or node not in epi_externals:
            return None
        built_epi = _build_pointwise_program(
            epi_view, output_override=epi_mapping[seg.epilogue[-1]]
        )
        if built_epi is None:
            return None
        _, eprogram, econstants, _, epi_output_ref = built_epi
        esrc = next(
            index
            for index, placeholder in enumerate(epi_view.graph.placeholders)
            if placeholder.name == node.name
        )
        epilogue_payload = (eprogram, econstants, esrc)

    def _epilogue_compose(base_launch, out_shape, out_dtype):
        """Wrap ``base_launch`` with the chain as its own pointwise kernel.

        Returns ``(composed_launch, chain_launch)``; both ``None`` when the
        chain leaves the operator's shape or dtype — the plan's example
        feed and the fused tile store assume both.
        """

        epi_meta = seg.epilogue[-1].meta.get("tensor_meta")
        if (
            epi_meta is None
            or tuple(int(dim) for dim in getattr(epi_meta, "shape", ()))
            != out_shape
            or getattr(epi_meta, "dtype", None) != out_dtype
        ):
            return None, None
        eprogram, econstants, _ = epilogue_payload
        epi_example = _tp.empty(
            out_shape, dtype=out_dtype, device=sample_device
        )
        chain_launch = _autotune_launch(
            f"{role}ep",
            eprogram,
            econstants,
            (epi_output_ref,),
            [epi_example],
            input_shapes=(out_shape,),
            reference_shape=out_shape,
            bucket_numel=_prod(out_shape),
            max_autotune=max_autotune,
            coordinate_descent_tuning=coordinate_descent_tuning,
        )

        def composed(feed, _base=base_launch, _chain=chain_launch):
            return _chain([_base(feed)])

        return composed, chain_launch

    if node.op == "get_attr":
        value = graph_module._get_attr(node.target)
        if training:
            if (
                attr_position is None
                or not isinstance(value, _tp.Tensor)
                or value.dtype != sample_dtype
                or value.device != sample_device
                or not value.is_contiguous()
            ):
                # Training closes every extern segment with a tangent rule
                # over feed values; an attribute outside that contract (a
                # non-tensor constant, a foreign dtype downstream kernels
                # did not specialize for) keeps the native path.
                return None
            return _ExternPlan(
                launch=lambda feed: feed[0],
                extern_sources=(_ExternSource("attr", attr_position, 0),),
                example=value,
                output_shape=tuple(int(dim) for dim in value.shape),
                vjp_ready=True,
                backward_launch=(
                    lambda feed_and_tangent: (feed_and_tangent[-1],)
                ),
            )
        attr_launch = lambda feed: value  # noqa: E731
        if epilogue_payload is not None:
            composed, _ = _epilogue_compose(
                attr_launch, tuple(int(dim) for dim in value.shape), value.dtype
            )
            if composed is None:
                return None
            attr_launch = composed
        return _ExternPlan(
            launch=attr_launch,
            extern_sources=sources,
            example=value,
            output_shape=tuple(int(dim) for dim in value.shape),
            vjp_ready=False,
            backward_launch=None,
        )
    if node.op not in {"call_function", "call_method"}:
        # Attribute loads and friends stay on the fallback path.
        return None
    meta = node.meta.get("tensor_meta")
    if meta is None:
        # Shape metadata is advisory; without it downstream segments cannot
        # specialize against this segment's output.
        return None
    shape = getattr(meta, "shape", None)
    dtype = getattr(meta, "dtype", None)
    if shape is None or dtype is None:
        # A multi-output operator records a structure rather than one
        # tensor's metadata; a later segment cannot specialize against it.
        return None
    if dtype != sample_dtype:
        # Downstream fused kernels assume one dtype per region.
        return None
    key_positions = {
        source: position for position, source in enumerate(sources)
    }
    unplanned = object()

    def resolve(feed: list, value: Any) -> Any:
        if isinstance(value, Node):
            if value.op == "placeholder":
                key = _ExternSource("arg", placeholder_positions[value.name])
            else:
                located = export_ports.get(value)
                if located is None:
                    return unplanned
                key = _ExternSource("seg", located[0], located[1])
            position = key_positions.get(key)
            if position is None:
                return unplanned
            return feed[position]
        return value

    def position_of(value: Any) -> int | None:
        """Feed position of a Node operand; None when it is not a source."""

        if not isinstance(value, Node):
            return None
        if value.op == "placeholder":
            key = _ExternSource("arg", placeholder_positions[value.name])
        else:
            located = export_ports.get(value)
            if located is None:
                return None
            key = _ExternSource("seg", located[0], located[1])
        return key_positions.get(key)

    def launch(feed: list) -> Any:
        args = [resolve(feed, value) for value in node.args]
        kwargs = {
            name: resolve(feed, value) for name, value in node.kwargs.items()
        }
        if any(value is unplanned for value in args) or any(
            value is unplanned for value in kwargs.values()
        ):
            raise RuntimeError("extern segment resolved an unplanned operand")
        if node.op == "call_function":
            return node.target(*args, **kwargs)
        owner, rest = args[0], args[1:]
        return getattr(owner, node.target)(*rest, **kwargs)

    vjp = _build_extern_analytic_vjp(node, position_of)
    if vjp is None:
        vjp = _build_extern_engine_vjp(node, resolve, position_of)

    output_shape = tuple(int(dim) for dim in shape)
    bare_launch = launch
    epi_launch = None
    if epilogue_payload is not None:
        composed, epi_launch = _epilogue_compose(
            bare_launch, output_shape, dtype
        )
        if composed is None:
            return None
        launch = composed
    if max_autotune and not training and sample_feed is not None:
        from .triton_gemm import tuned_matmul_launch

        def operand_spec(value: Any) -> tuple | None:
            """``(feed position, literal)`` for one operand; None when the
            plan cannot resolve it (an unplanned graph dependency)."""

            if isinstance(value, Node):
                position = position_of(value)
                return (position, None) if position is not None else None
            return (None, value)

        is_matmul = (
            node.op == "call_function" and node.target is operator.matmul
        ) or (node.op == "call_method" and node.target == "matmul")
        is_linear = (
            node.op == "call_function"
            and getattr(node.target, "__name__", "") == "linear"
            and getattr(node.target, "__module__", "")
            == "tensorplay.nn.functional"
        )
        bias_spec = None
        if is_matmul and len(node.args) == 2:
            a_spec = operand_spec(node.args[0])
            b_spec = operand_spec(node.args[1])
            b_transposed = False
        elif is_linear and 2 <= len(node.args) <= 3 and not node.kwargs:
            # linear(input, weight, bias?): the tile consumes the weight's
            # transposed view and adds the length-N bias to the accumulator
            a_spec = operand_spec(node.args[0])
            b_spec = operand_spec(node.args[1])
            b_transposed = True
            if len(node.args) > 2 and node.args[2] is not None:
                bias_spec = operand_spec(node.args[2])
        else:
            a_spec = b_spec = None
            b_transposed = False
        if (
            a_spec is not None
            and b_spec is not None
            and (bias_spec is None or bias_spec[0] is not None
                 or bias_spec[1] is not None)
        ):
            try:
                tuned = tuned_matmul_launch(
                    bare_launch,
                    sample_feed,
                    (a_spec, b_spec),
                    output_shape,
                    epilogue=epilogue_payload,
                    epilogue_launch=epi_launch,
                    bias_spec=bias_spec,
                    b_transposed=b_transposed,
                )
            except Exception:  # noqa: BLE001 - tuning is an optimization only
                tuned = None
            if tuned is not None:
                launch = tuned
    example = _tp.empty(
        output_shape, dtype=dtype, device=sample_device
    )
    return _ExternPlan(
        launch=launch,
        extern_sources=sources,
        example=example,
        output_shape=output_shape,
        vjp_ready=vjp is not None,
        backward_launch=(
            (lambda feed_and_tangent, vjp=vjp: vjp(feed_and_tangent[:-1],
                                                   feed_and_tangent[-1]))
            if vjp is not None
            else None
        ),
    )


@dataclass
class _SegmentPlan:
    launch: Any
    extern_sources: tuple
    spec: ReductionSpec | None
    #: pointwise program artifacts needed to synthesize the segment's local
    #: VJP (training mode); ``None`` for segments that never built a plain
    #: pointwise program.
    program: list[int] | None = None
    constants: list[float] | None = None
    instructions: list[tuple[str, int, int, int]] | None = None
    output_ref: int = 0
    #: every kernel output ref — the main export plus horizontal-fusion
    #: extra stores; the launch returns one buffer per ref
    output_refs: tuple = ()
    #: example inputs feeding this segment (launch specialization shape)
    examples: tuple = ()
    #: True when some segment input broadcasts against the local reference —
    #: its VJP partial is summed to the operand shape before accumulation.
    needs_broadcast: bool = False
    #: output-shaped stand-in anchoring the backward launch's tangent input;
    #: broadcast inputs leave the iteration space to this example.
    tangent_example: Any = None
    #: fused local-VJP launch (training mode); None until built.
    backward_launch: Any = None
    #: compile-time tangent layout for reduction segments:
    #: ``(reshape_sizes, expand_shape, scale)`` — the export gradient is
    #: reshaped/expanded back to the reduction-input shape (and divided by
    #: rnumel for mean). ``None`` for pure-pointwise segments (same-shape
    #: tangent).
    tangent_plan: tuple | None = None


def _reduction_tangent_plan(
    spec: ReductionSpec, producer_shape: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], float] | None:
    """Local-VJP tangent layout for a trainable reduction segment.

    ``sum``/``mean`` distribute their export gradient uniformly back over
    the reduction input (expand + divide-by-rnumel), so the segment's
    existing elementwise VJP program can be seeded with the expanded
    tangent.  ``amax``/``max``/``argmax`` route gradients to extremum
    positions only and stay M5f.
    """

    if spec.op not in ("sum", "mean"):
        return None
    rank = len(producer_shape)
    reduced = (
        set(range(rank)) if spec.is_full else set(spec.normalized_dims(rank))
    )
    out_shape = spec.output_shape(producer_shape)
    if spec.keepdim:
        reshape_sizes = tuple(int(size) for size in out_shape)
    else:
        kept = iter(int(size) for size in out_shape)
        reshape_sizes = tuple(
            1 if dim in reduced else next(kept) for dim in range(rank)
        )
    scale = 1.0 if spec.op == "sum" else 1.0 / spec.reduction_numel(
        producer_shape
    )
    return (reshape_sizes, tuple(int(size) for size in producer_shape), scale)


def _sum_to_shape(grad: Any, target_shape: tuple[int, ...]) -> Any:
    """Reduce an expanded elementwise partial back to one operand's shape.

    Sums the leading extra dims plus every dim the target carries as size 1
    (keepdim), then views the kept leading ones away, so a gradient that
    lives at the fused iteration space lands exactly on the broadcast
    operand it belongs to.
    """

    if grad is None:
        return None
    if tuple(int(dim) for dim in grad.shape) == tuple(
        int(dim) for dim in target_shape
    ):
        return grad
    leading = grad.ndim - len(target_shape)
    dims = list(range(leading)) + [
        leading + index
        for index, size in enumerate(target_shape)
        if int(size) == 1 and int(grad.shape[leading + index]) != 1
    ]
    return grad.sum(dim=dims, keepdim=True).reshape(list(target_shape))


def _reduction_mask_vjp(spec: ReductionSpec, input_position: int):
    """Select-mask VJP for a bare extremum reduction (M5f).

    The export tangent flows to every position attaining the extremum,
    splitting ties evenly: ``expand(t) * (x == y) / tie_count`` per
    reduced slice.  The extremum value is recomputed from the reduction
    input, so the backward needs no extra forward state.  ``feed`` is the
    segment feed with the reduction input at ``input_position``; returns
    one gradient per feed position.
    """

    def vjp(feed: list, tangent: Any) -> list:
        x = feed[input_position]
        rank = len(x.shape)
        dims = (
            list(range(rank)) if spec.is_full else list(spec.normalized_dims(rank))
        )
        if spec.op == "max":
            y = x.max()
        elif spec.op == "min":
            y = x.min()
        else:
            y = getattr(x, spec.op)(dim=dims, keepdim=spec.keepdim)
        keep_shape = [
            1 if dim in dims else int(size) for dim, size in enumerate(x.shape)
        ]
        y_b = y.reshape(keep_shape)
        mask = (x == y_b).to(x.dtype)
        # every reduced slice attains its extremum at least once, so the
        # tie count is never zero where the mask selects
        ties = mask.sum(dim=dims, keepdim=True)
        grad_x = tangent.reshape(keep_shape) * mask / ties
        grads = [None] * len(feed)
        grads[input_position] = grad_x
        return grads

    return vjp


def _reduction_pair_vjp(spec: ReductionSpec, input_position: int):
    """Index-scatter VJP for a bare pair reduction (``max(dim)``/``min(dim)``).

    The tangent on the values output flows to the extremum position each
    reduced slice selected: a zero gradient buffer with the tangent
    scattered in at the recorded indices.  ``feed`` is the segment feed
    with the reduction input at ``input_position``; the indices are
    recomputed from that input, so the backward needs no extra forward
    state.  The indices output itself carries no tangent — an integer
    stream has no differentiable path.
    """

    def vjp(feed: list, tangent: Any) -> list:
        import tensorplay as _tp

        x = feed[input_position]
        dim = spec.normalized_dims(len(x.shape))[0]
        pair = getattr(x, spec.op)(dim=dim, keepdim=spec.keepdim)
        if spec.keepdim:
            grad, indices = tangent, pair.indices
        else:
            grad = tangent.unsqueeze(dim)
            indices = pair.indices.unsqueeze(dim)
        grad_x = _tp.zeros_like(x).scatter_(
            dim, indices, grad.to(x.dtype)
        )
        grads = [None] * len(feed)
        grads[input_position] = grad_x
        return grads

    return vjp


def _reduction_drop_vjp(source_count: int):
    """No-tangent rule for an index reduction segment.

    An argmax output is an integer stream: no gradient path exists through
    it, matching eager semantics where the index result never joins the
    autograd graph.  The rule accepts the sweep's combined feed+tangent and
    returns no gradient per feed position, so any spurious tangent dies
    here.
    """

    def backward(feed_and_tangent: list) -> list:
        del feed_and_tangent
        return [None] * source_count

    return backward


def _extract_segment_view(
    graph: Graph, nodes, export_node: Node, extra_exports: tuple = ()
):
    """Clone ``nodes`` into a standalone Graph with placeholder externals.

    Returns ``(view, mapping)`` where ``view.graph`` feeds the program
    builder and ``mapping`` translates original nodes to their clones.
    ``extra_exports`` names further nodes (clones of ``nodes``) stored
    alongside ``export_node`` — horizontal fusion's extra kernel outputs.
    """

    sub = Graph()
    mapping: dict = {}
    externals: dict = {}

    def resolve(value):
        if isinstance(value, Node):
            if value in mapping:
                return mapping[value]
            cached = externals.get(value)
            if cached is not None:
                return cached
            ph = sub.placeholder(value.name)
            externals[value] = ph
            return ph
        return value

    for node in nodes:
        new_args = _map_arg(node.args, resolve)
        new_kwargs = _map_arg(node.kwargs, resolve)
        mapping[node] = sub.create_node(
            node.op,
            node.target,
            tuple(new_args) if isinstance(new_args, list) else new_args,
            dict(new_kwargs) if isinstance(new_kwargs, dict) else {},
        )
    export_new = mapping.get(export_node)
    if export_new is None:
        export_new = externals[export_node]
    # Multi-value output node: one output carrying the main export plus the
    # extra horizontal-fusion stores (Graph.output holds a single result).
    sub.create_node(
        "output",
        "output",
        (export_new, *(mapping[node] for node in extra_exports)),
        {},
        "output",
    )
    return SimpleNamespace(graph=sub), mapping, externals



def _dbg(msg):
    import os as _os
    if _os.environ.get("TP_STAX_DEBUG"):
        print("[stax]", msg, flush=True)

def compile_graph_module(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    max_autotune: bool = False,
    coordinate_descent_tuning: bool = False,
    strict_native: bool = False,
    **kwargs: Any,
):
    del kwargs
    if not HAS_TRITON:
        _dbg('fallback gate #1')
        return None
    try:
        reference_shape = _broadcast_reference_shape(
            [tuple(int(dim) for dim in value.shape) for value in example_inputs]
        )
    except (AttributeError, TypeError, ValueError):
        _dbg('fallback gate #2')
        return None
    if reference_shape is None or not _supports_runtime_inputs(
        example_inputs, allow_grad=True, reference_shape=reference_shape
    ):
        _dbg('fallback gate #3')
        return None

    def _is_pointwise(node: Node) -> bool:
        return (
            node.op in {"call_function", "call_method"}
            and not node.kwargs
            and _target_name(node.target) in _CPU_FUSED_OPS
        )

    def _classify_reduction(node: Node):
        return (
            _reduction_spec_from_node(node)
            if node.op == "call_method"
            else None
        )

    # M5c/M5e: the scheduler is the single source of fusion truth.  Any
    # number of validated segments lowers through per-segment emission;
    # each segment's externals must be graph placeholders or the exported
    # tail of an earlier segment.
    any_grad = any(value.requires_grad for value in example_inputs)
    segments = segment_graph(
        graph_module,
        is_pointwise=_is_pointwise,
        classify_reduction=_classify_reduction,
        # training schedules keep store-time epilogues out: the split
        # epilogue becomes its own pointwise segment closed by a local VJP
        allow_epilogue=not any_grad,
    )
    if segments is None:
        _dbg('fallback gate #4')
        return None
    scheduler_annotate(graph_module, segments)

    needs_broadcast = any(
        tuple(int(dim) for dim in value.shape) != reference_shape
        for value in example_inputs
    )
    if any_grad and any(
        seg.kind == "pw" and len(seg.exports) > 1 for seg in segments
    ):
        # Horizontal fusion kernels carry extra stores; the local-VJP
        # training sweep routes gradients through the single export only.
        # Pair reductions are exempt: their multi-port exports ARE the
        # per-stream outputs the scatter VJP closes.
        _dbg('fallback gate #5d')
        return None
    if any_grad:
        # Training lowers through per-segment local VJPs.  Pointwise
        # segments take elementwise VJPs; sum/mean reduction segments take
        # an expanded tangent into their prologue's VJP program (the
        # forward program already exports the reduction input); bare
        # extremum reductions take an eager select-mask VJP and bare pair
        # reductions (max/min with dim) an eager index-scatter VJP; index
        # reductions (argmax) drop the tangent — an integer stream carries
        # no gradient path.  Extern segments take an engine VJP through one
        # recomputed eager call; extremum reductions behind a pointwise
        # chain fall back at the plan gate.
        for seg in segments:
            if seg.epilogue:
                _dbg('fallback gate #5a')
                return None
    # Broadcast operands train through the per-segment local VJPs: each
    # elementwise partial comes back at the fused iteration space and is
    # summed to the operand's own shape before accumulation.

    # --- acceptance gate for runtime wiring (M5c per-segment emission) ----
    output_values = [
        value
        for out_node in graph_module.graph.outputs
        for value in _nodes(out_node.args)
    ]
    if len(output_values) != 1 or not isinstance(output_values[0], Node):
        scheduler_annotate(graph_module, segments)
        _dbg('fallback gate #7')
        return None
    final_value = output_values[0]
    node_to_seg: dict = {}
    for index, seg in enumerate(segments):
        # exports included: a pair reduction's projection nodes wire like
        # any other producer output though they are not emission nodes
        for node in [*seg.nodes, *seg.epilogue, *seg.exports]:
            node_to_seg[node] = index

    def _extern_sources(seg_index: int, seg):
        """Validate cross-segment wiring; None when unsupported."""

        inside = set(seg.nodes) | set(seg.epilogue) | set(seg.exports)
        placeholder_positions = {
            node.name: position
            for position, node in enumerate(graph_module.graph.placeholders)
        }
        sources = []
        seen = set()
        for node in [*seg.nodes, *seg.epilogue]:
            outside_users = [
                user for user in node.users if user not in inside
            ]
            is_final = node is final_value
            if (outside_users or is_final) and node not in seg.exports:
                # a consumed interior value the producer cannot store
                _dbg('fallback gate #8')
                return None
            # Deterministic encounter order — args before kwargs, segment
            # node order — MUST match the sub-view's placeholder numbering:
            # the launch feeds positionally by ref, and the VJP program's
            # gradient outputs are ordered by the same refs.
            for dep in (*_nodes(node.args), *_nodes(node.kwargs or {})):
                if dep in inside:
                    continue
                if dep.op == "placeholder":
                    position = placeholder_positions.get(dep.name)
                    if position is None:
                        _dbg('fallback gate #9')
                        return None
                    key = ("arg", position, 0)
                else:
                    producer_seg = node_to_seg.get(dep)
                    if producer_seg is None or producer_seg >= seg_index:
                        _dbg('fallback gate #10')
                        return None
                    producer_exports = segments[producer_seg].exports
                    if dep not in producer_exports:
                        _dbg('fallback gate #11')
                        return None
                    key = ("seg", producer_seg, producer_exports.index(dep))
                if key not in seen:
                    seen.add(key)
                    sources.append(_ExternSource(key[0], key[1], key[2]))
        return tuple(sources)

    segment_plans: list[_SegmentPlan] = []
    import tensorplay as _tp

    sample_dtype = example_inputs[0].dtype
    sample_device = example_inputs[0].device
    placeholder_positions_all = {
        node.name: position
        for position, node in enumerate(graph_module.graph.placeholders)
    }
    export_ports = {
        node: (seg_index, port)
        for seg_index, seg in enumerate(segments)
        for port, node in enumerate(seg.exports)
    }
    #: extern segments have no scheduler-side shape; their eager plan
    #: records the output shape when it is built (segment order guarantees
    #: it exists before any later consumer asks).
    extern_shapes: dict[int, tuple] = {}

    def _sample_feed(sources_list: tuple) -> list:
        """Sample tensors for one segment's extern sources, in feed order.

        Arg sources reuse the region's own example tensors; producer
        sources get empty stand-ins shaped like the producer's export
        (extern_shapes entries filled by earlier loop iterations).
        """

        feed = []
        for source in sources_list:
            if source.kind == "arg":
                feed.append(example_inputs[source.index])
            else:
                producer_seg = segments[source.index]
                if producer_seg.kind == "pw+red":
                    # Reduction output (scalar for full, kept-dims for axis
                    # reductions); an epilogue tail has the same shape.
                    shape: tuple[int, ...] = producer_seg.reduction.output_shape(
                        reference_shape
                    )
                elif producer_seg.kind == "extern":
                    shape = extern_shapes[source.index]
                else:
                    shape = reference_shape
                feed.append(
                    _tp.empty(shape, dtype=sample_dtype, device=sample_device)
                )
        return feed

    #: lifted module attributes (get_attr segments) in encounter order —
    #: training passes their CURRENT values as trailing autograd inputs
    #: and returns their gradients alongside the placeholder gradients.
    attr_targets: list[str] = []

    def _attr_value(target: str):
        return graph_module._get_attr(target)

    for seg_index, seg in enumerate(segments):
        sources = _extern_sources(seg_index, seg)
        if sources is None:
            scheduler_annotate(graph_module, segments)
            _dbg('fallback gate #12')
            return None
        if seg.kind == "extern":
            attr_position = None
            if (
                any_grad
                and seg.nodes[0].op == "get_attr"
                and seg.nodes[0].target not in attr_targets
            ):
                attr_targets.append(seg.nodes[0].target)
                attr_position = len(attr_targets) - 1
            elif any_grad and seg.nodes[0].op == "get_attr":
                attr_position = attr_targets.index(seg.nodes[0].target)
            extern_plan = _extern_segment_plan(
                graph_module,
                seg,
                sources,
                sample_dtype,
                sample_device,
                placeholder_positions_all,
                export_ports,
                sample_feed=_sample_feed(sources),
                max_autotune=max_autotune,
                training=any_grad,
                attr_position=attr_position,
                role=f"fwd{seg_index}",
                coordinate_descent_tuning=coordinate_descent_tuning,
            )
            if extern_plan is None:
                scheduler_annotate(graph_module, segments)
                _dbg('fallback gate #12b')
                return None
            if any_grad and not extern_plan.vjp_ready:
                # no tangent rule of either kind: the operator has no
                # differentiable eager form the VJP can close
                scheduler_annotate(graph_module, segments)
                _dbg('fallback gate #12c')
                return None
            extern_shapes[seg_index] = extern_plan.output_shape
            segment_plans.append(extern_plan)
            continue
        # Horizontal-fusion extra stores are a pointwise-only concept; a
        # pair reduction's exports are projection nodes the view must not
        # clone (the kernel's dual streams carry them).
        extra_exports = (
            tuple(seg.exports[1:]) if seg.kind == "pw" else ()
        )
        sub_view, mapping, externals = _extract_segment_view(
            graph_module.graph, seg.nodes, seg.tail, extra_exports
        )
        reduction = None
        reduction_mode_local = None
        reduction_outputs = None
        if seg.kind == "pw+red":
            reduction = seg.reduction
            if reduction.is_pair:
                # The pair kernel needs both a projection set to export
                # and the dual-stream runtime (tl.argmax/tl.argmin with
                # float32/float64 value streams).
                if not seg.export_kinds:
                    _dbg('fallback gate #13b')
                    return None
                if not (HAS_TL_ARGMAX and HAS_TL_ARGMIN):
                    _dbg('fallback gate #13')
                    return None
                if sample_dtype not in (_tp.float32, _tp.float64):
                    _dbg('fallback gate #14')
                    return None
                reduction_outputs = tuple(seg.export_kinds)
            elif reduction.tracks_indices:
                # v1 index reductions: float32/float64 only + tl.argmax.
                if not HAS_TL_ARGMAX:
                    _dbg('fallback gate #13')
                    return None
                if sample_dtype not in (_tp.float32, _tp.float64):
                    _dbg('fallback gate #14')
                    return None
                reduction_outputs = ("indices",)
            if seg.epilogue and (reduction.tracks_indices or reduction.is_pair):
                # A pointwise chain living on an index-carrying reduction's
                # registers would have to run on the int64 index stream —
                # the store-time epilogue programs are float-value-space
                # only.  The region falls back instead.
                _dbg('fallback gate #13c')
                return None
        if seg.kind == "pw+red":
            producer_new = mapping.get(seg.producer)
            if producer_new is None:
                producer_new = externals[seg.producer]
            # A bare reduction has an empty pointwise prologue; its
            # reduction reads the external input directly.
            pointwise = _build_pointwise_program(
                sub_view,
                skip_node=mapping[seg.tail],
                output_override=producer_new,
                allow_empty=True,
            )
        else:
            pointwise = _build_pointwise_program(
                sub_view,
                extra_outputs=(
                    [mapping[node] for node in extra_exports]
                    if extra_exports
                    else None
                ),
            )
        if pointwise is None:
            _dbg('fallback gate #15')
            return None
        if seg.kind == "pw+red" or not extra_exports:
            placeholders_s, forward_program, forward_constants, instructions, output_ref = (
                pointwise
            )
            extra_refs: tuple[int, ...] = ()
        else:
            placeholders_s, forward_program, forward_constants, instructions, output_ref, extra_refs = (
                pointwise
            )
        output_refs = (output_ref, *extra_refs)
        # M5e: red→pw store epilogue — the post-reduction pointwise chain
        # runs on the accumulator registers inside the same kernel.
        epilogue_payload = None
        if seg.epilogue:
            assert seg.kind == "pw+red" and not seg.reduction.tracks_indices
            # Clone ONLY the epilogue nodes; the reduction tail resolves to
            # the view's single external placeholder.
            epi_view, epi_mapping, epi_externals = _extract_segment_view(
                graph_module.graph, list(seg.epilogue), seg.epilogue[-1]
            )
            # v1 epilogue contract: the ONLY tensor input is the reduction
            # result; everything else is a scalar constant (guaranteed by
            # the scheduler's join rule — re-checked here).
            if len(epi_externals) != 1 or seg.tail not in epi_externals:
                _dbg('fallback gate #16')
                return None
            built_epi = _build_pointwise_program(
                epi_view,
                output_override=epi_mapping[seg.epilogue[-1]],
            )
            if built_epi is None:
                _dbg('fallback gate #17')
                return None
            _, eprogram, econstants, _, _ = built_epi
            esrc = next(
                index
                for index, node in enumerate(epi_view.graph.placeholders)
                if node.name == seg.tail.name
            )
            # ``output_ref`` stays the MAIN program's reduction source; the
            # epilogue tail replaces only the STORE value inside codegen.
            epilogue_payload = (eprogram, econstants, esrc)
        seg_examples = _sample_feed(sources)
        # Segments compose: each lowers against the broadcast shape of ITS
        # OWN inputs, which for later segments is an intermediate shape, not
        # the graph-wide reference.
        local_ref = _broadcast_reference_shape(
            [tuple(int(dim) for dim in value.shape) for value in seg_examples]
        )
        if local_ref is None:
            _dbg('fallback gate #18')
            return None
        local_needs_broadcast = any(
            tuple(int(dim) for dim in value.shape) != local_ref
            for value in seg_examples
        )
        if seg.kind == "pw+red":
            if seg.reduction.is_full:
                reduction_mode_local = (
                    "single"
                    if _prod(local_ref) <= _SINGLE_BLOCK_MAX
                    else "split"
                )
            else:
                # Per-input offsets generalize to broadcast operands.
                reduction_mode_local = "dims"
        seg_launch = _autotune_launch(
            f"fwd{seg_index}",
            forward_program,
            forward_constants,
            output_refs,
            seg_examples,
            reduction=reduction,
            reduction_mode=reduction_mode_local,
            input_shapes=tuple(
                tuple(int(dim) for dim in value.shape) for value in seg_examples
            ),
            reference_shape=local_ref,
            bucket_numel=_prod(local_ref),
            value_dtype=str(sample_dtype) if reduction is not None else None,
            epilogue=epilogue_payload,
            reduction_outputs=reduction_outputs,
            max_autotune=max_autotune,
            coordinate_descent_tuning=coordinate_descent_tuning,
        )
        # Plan-time eager VJPs (training): no backward kernel is built for
        # these segments.
        #   bare extremum reduction (input a direct source, no pointwise
        #     producer) — the select-mask VJP;
        #   bare pair reduction — the index-scatter VJP;
        #   index reduction — the no-tangent rule (integer stream).
        mask_launch = None
        if reduction is not None and any_grad:
            bare = (
                len(sources) == 1
                and seg.producer is not None
                and seg.producer not in seg.nodes
            )
            if (
                reduction.op in _MASK_REDUCTION_OPS
                and not reduction.is_pair
                and bare
            ):
                mask_vjp = _reduction_mask_vjp(reduction, 0)
                mask_launch = lambda feed_and_tangent, vjp=mask_vjp: vjp(  # noqa: E731
                    feed_and_tangent[:-1], feed_and_tangent[-1]
                )
            elif reduction.is_pair:
                if not bare:
                    # A pair reduction behind an in-segment producer chain
                    # has no scatter VJP wiring yet.
                    scheduler_annotate(graph_module, segments)
                    _dbg('fallback gate #20b')
                    return None
                pair_vjp = _reduction_pair_vjp(reduction, 0)
                mask_launch = lambda feed_and_tangent, vjp=pair_vjp: vjp(  # noqa: E731
                    feed_and_tangent[:-1], feed_and_tangent[-1]
                )
            elif reduction.tracks_indices:
                mask_launch = _reduction_drop_vjp(len(sources))
        segment_plans.append(
            _SegmentPlan(
                seg_launch,
                sources,
                reduction,
                program=list(forward_program),
                constants=list(forward_constants),
                instructions=list(instructions),
                output_ref=output_ref,
                output_refs=output_refs,
                examples=tuple(seg_examples),
                needs_broadcast=bool(local_needs_broadcast),
                tangent_example=_tp.empty(
                    local_ref, dtype=sample_dtype, device=sample_device
                ),
                tangent_plan=(
                    None
                    if reduction is None
                    else _reduction_tangent_plan(reduction, local_ref)
                ),
                backward_launch=mask_launch,
            )
        )

    placeholders = graph_module.graph.placeholders
    backward_launch = None
    autograd_function: Any | None = None

    def _run_segment(plan, feed):
        """Run one segment; normalize the launch result to an export tuple."""

        values = plan.launch(feed)
        if isinstance(values, list):
            return tuple(values)
        return (values,)

    # The graph output is one of the last kernel's exports (the scheduler
    # promotes an interior final value to an extra store).
    last_exports = segments[-1].exports
    if final_value not in last_exports:
        scheduler_annotate(graph_module, segments)
        _dbg('fallback gate #22')
        return None
    final_port = last_exports.index(final_value)
    if any(value.requires_grad for value in example_inputs):
        # M5c training: chain one local VJP program per segment.  The
        # reverse sweep feeds each segment's export-gradient through its
        # own fused backward kernel and accumulates contributions into
        # segment boundaries / placeholders (fan-out sums).  Gradients are
        # keyed by the segment's extern sources, whose order follows the
        # kernel's input-ref order — not the placeholder order — so the
        # sweep maps every contribution back through the source indices.
        for plan in segment_plans:
            if (
                plan.tangent_plan is None
                and plan.spec is not None
                and plan.backward_launch is None
                or any(
                    op_name not in _CPU_FUSED_AUTOGRAD_OPS
                    for op_name, *_ in plan.instructions
                )
            ):
                _dbg('fallback gate #20')
                return None
        for seg_index, plan in enumerate(segment_plans):
            if plan.backward_launch is not None:
                # extern plans carry their engine VJP from plan time
                continue
            gradient_plan = _build_fused_gradient_graphs(
                len(plan.extern_sources),
                plan.instructions,
                plan.program,
                plan.constants,
                len(plan.program) // 3,
                plan.output_ref,
            )
            if gradient_plan is None:
                _dbg('fallback gate #21')
                return None
            bwd_program, bwd_constants, bwd_outputs = gradient_plan
            # Same input-count convention as forward: the final external
            # input is this segment's export tangent, expanded to the
            # segment's iteration space (the anchor against broadcast
            # operands); per-input shapes keep broadcast operands on
            # their own offsets inside the VJP kernel.
            bwd_examples = [*plan.examples, plan.tangent_example]
            plan.backward_launch = _autotune_launch(
                f"bwd{seg_index}",
                bwd_program,
                bwd_constants,
                bwd_outputs,
                bwd_examples,
                input_shapes=tuple(
                    tuple(int(dim) for dim in value.shape)
                    for value in bwd_examples
                ),
                reference_shape=tuple(
                    int(dim) for dim in plan.tangent_example.shape
                ),
                bucket_numel=_prod(plan.tangent_example.shape),
                max_autotune=max_autotune,
                coordinate_descent_tuning=coordinate_descent_tuning,
            )

        from .....autograd import Function

        attr_count = len(attr_targets)

        def _feed_of(forward_inputs, intermediates, plan):
            """One segment's runtime inputs from args, attrs and exports."""

            return [
                forward_inputs[source.index]
                if source.kind == "arg"
                else (
                    forward_inputs[len(placeholders) + source.index]
                    if source.kind == "attr"
                    else intermediates[source.index][source.port]
                )
                for source in plan.extern_sources
            ]

        def _grads_of(plan, feed_and_tangent):
            """One gradient per extern source, always a flat sequence.

            A single-output launch returns one tensor, not a list — zip
            would otherwise iterate its leading dimension as if each row
            were a separate source gradient.
            """
            values = plan.backward_launch(feed_and_tangent)
            if isinstance(values, (list, tuple)):
                return values
            return [values]

        class _StaxTritonAutograd(Function):
            @staticmethod
            def forward(ctx: Any, *forward_inputs: Any) -> Any:
                intermediates: dict[int, tuple] = {}
                feed_all = []
                for index, plan in enumerate(segment_plans):
                    feed = _feed_of(forward_inputs, intermediates, plan)
                    intermediates[index] = _run_segment(plan, feed)
                    feed_all.append(feed)
                ctx.stax_feed_all = feed_all
                ctx.save_for_backward(
                    *forward_inputs,
                    *(value for values in intermediates.values() for value in values),
                )
                return intermediates[len(segment_plans) - 1][final_port]

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
                grad_output = grad_outputs[0] if grad_outputs else None
                saved = ctx.saved_tensors
                inputs_count = len(placeholders)
                if grad_output is None:
                    return (None,) * (inputs_count + attr_count)
                # normalize once against the final output's operand shape;
                # every downstream tangent already has its producer shape.
                seg_grads: dict[int, Any] = {
                    len(segment_plans) - 1: _normalize_pointwise_grad_output(
                        grad_output, saved[-1]
                    )
                }
                arg_grads: dict[int, Any] = {}
                attr_grads: dict[int, Any] = {}

                def accumulate(bucket: dict, key: int, value: Any) -> None:
                    if value is None:
                        return
                    existing = bucket.get(key)
                    bucket[key] = value if existing is None else existing + value

                for index in reversed(range(len(segment_plans))):
                    tangent = seg_grads.pop(index, None)
                    if tangent is None:
                        continue
                    plan = segment_plans[index]
                    if plan.tangent_plan is not None:
                        # sum/mean: uniform distribution back over the
                        # reduction input (expand + mean scale), matching
                        # backward kernels read dense buffers (same contract
                        # as _normalize_pointwise_grad_output), so the
                        # stride-0 expansion is materialized here.
                        reshape_sizes, expand_shape, scale = plan.tangent_plan
                        tangent = _tp.reshape(tangent, list(reshape_sizes))
                        tangent = _tp.expand(tangent, list(expand_shape))
                        if scale != 1.0:
                            tangent = tangent * scale
                        else:
                            tangent = tangent.contiguous()
                    grads = _grads_of(
                        plan, [*ctx.stax_feed_all[index], tangent]
                    )
                    for position, (source, grad) in enumerate(
                        zip(plan.extern_sources, grads)
                    ):
                        # an elementwise partial lives at the fused iteration
                        # space; broadcast operands receive their own shape.
                        # extern plans resolve operands at call time and
                        # their VJPs return operand-shaped gradients.
                        examples = getattr(plan, "examples", ())
                        if examples:
                            grad = _sum_to_shape(
                                grad, tuple(examples[position].shape)
                            )
                        if source.kind == "arg":
                            accumulate(arg_grads, source.index, grad)
                        elif source.kind == "attr":
                            accumulate(attr_grads, source.index, grad)
                        else:
                            accumulate(seg_grads, source.index, grad)
                return (
                    tuple(
                        arg_grads.get(position)
                        for position in range(inputs_count)
                    )
                    + tuple(
                        attr_grads.get(position)
                        for position in range(attr_count)
                    )
                )

        autograd_function = _StaxTritonAutograd

    placeholders = graph_module.graph.placeholders
    fallback = None if strict_native else graph_module.recompile()

    def compiled(*args: Any, **call_kwargs: Any) -> Any:
        if not call_kwargs and len(args) == len(placeholders):
            inputs = list(args)
        else:
            bound = graph_module.signature.bind_partial(*args, **call_kwargs)
            bound.apply_defaults()
            inputs = [
                bound.arguments[node.target if isinstance(node.target, str) else node.name]
                for node in placeholders
            ]
        if not _supports_runtime_inputs(
            inputs, allow_grad=True, reference_shape=reference_shape
        ):
            if strict_native:
                raise RuntimeError(
                    "Stax strict_native Triton lowering received inputs outside "
                    "its compiled specialization"
                )
            assert fallback is not None
            return fallback(*args, **call_kwargs)
        # lifted attributes re-fetch per call: an optimizer step that
        # rebinds the module's parameter must be seen by the region
        attr_values = [_attr_value(target) for target in attr_targets]
        if autograd_function is not None and any(
            value.requires_grad for value in inputs
        ):
            return autograd_function.apply(*inputs, *attr_values)
        intermediates: dict[int, tuple] = {}
        for index, plan in enumerate(segment_plans):
            feed = [
                inputs[source.index]
                if source.kind == "arg"
                else intermediates[source.index][source.port]
                for source in plan.extern_sources
            ]
            intermediates[index] = _run_segment(plan, feed)
        return intermediates[len(segment_plans) - 1][final_port]

    compiled._tensorplay_codegen = "triton"  # type: ignore[attr-defined]
    compiled._tensorplay_backward_codegen = (  # type: ignore[attr-defined]
        "triton"
        if (backward_launch is not None or autograd_function is not None)
        else None
    )
    return compiled
