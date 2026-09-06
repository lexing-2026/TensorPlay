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

from tensorplay.graph import Graph, GraphModule, Node
from tensorplay.graph._utils import _map_arg
from ..scheduler import annotate as scheduler_annotate
from ..scheduler import segment_graph

# Salt for every content-addressed kernel cache key.  Bump whenever the
# EMITTER changes semantics (masking, NaN handling, launcher allocation,
# load cache annotations, new fused opcodes) so stale generated sources
# cannot be replayed against a new compiler.
_CODEGEN_VERSION = "m9-2026-08-30-pw-surface"
from ..stax import (
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

    ``op``    : "sum" | "mean" | "amax" | "max" | "argmax"
    ``dims``  : reduction axes, ascending; empty tuple = full reduction
    ``keepdim``: whether reduced axes stay as size-1 dimensions

    ``argmax`` is an *index* reduction: the kernel carries a value stream and
    an index stream side by side (the dual-output skeleton) but v1 stores only
    It requires explicit dims and float32/float64 inputs.
    """

    __slots__ = ("op", "dims", "keepdim")

    # kernel-side combine/finalize/neutral per op
    _FINAL = {"sum": "tl.sum", "mean": "tl.sum", "amax": "tl.max", "max": "tl.max"}
    _COMBINE = {
        "sum": "acc + {value}",
        "mean": "acc + {value}",
        "amax": "tl.maximum(acc, {value})",
        "max": "tl.maximum(acc, {value})",
    }
    _NEUTRAL = {
        "sum": "0.0",
        "mean": "0.0",
        "amax": "float('-inf')",
        "max": "float('-inf')",
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
                isinstance(reduction, ReductionSpec) and reduction.tracks_indices
            ):
                # Index reductions emit an int64 stream; float pointwise on
                # top of it is meaningless, and plain pw needs no acc.
                raise ValueError(
                    "reduction epilogues require a value reduction"
                )
        self.epilogue = epilogue
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

        body: list[str] = [
            "xoffset = tl.program_id(0) * XBLOCK",
            # AxisInfo hint: proves contiguity+alignment of every xindex
            # derived access, unlocking vectorized ld/st through the
            # multiple_of annotation.
            "xoffset = tl.multiple_of(xoffset, XBLOCK)",
            "xindex = xoffset + tl.arange(0, XBLOCK)",
            "xmask = xindex < xnumel",
        ]
        numel_total = (
            _prod(self.reference_shape)
            if self.reference_shape is not None
            else None
        )
        if dims_reduction:
            assert self.reference_shape is not None and spec is not None
            block, warps, rblock, stages_default = _dim_reduction_config(
                self.reference_shape, spec
            )
            stages = stages_default
            if fixed_config is not None:
                block, warps = fixed_config[0], fixed_config[1]
                if len(fixed_config) > 3:
                    rblock, stages = fixed_config[2], fixed_config[3]
                elif len(fixed_config) > 2:
                    stages = fixed_config[2]
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

            tracks_indices = spec.tracks_indices
            persistent = rnumel <= rblock
            if tracks_indices:
                body.append(
                    f"acc = tl.full([XBLOCK], {spec.neutral()}, dtype={self.value_type})"
                )
                # Index stream: the running argmax in flat reduced-space
                # coordinates (same order the r-loop enumerates).
                body.append("acci = tl.zeros([XBLOCK], dtype=tl.int64)")
            else:
                body.append(
                    f"acc = tl.full([XBLOCK], {spec.neutral()}, dtype=tl.float32)"
                )
            pfx = "" if persistent else "    "
            if not persistent:
                body.append(
                    f"for roffset in tl.range(0, {rnumel}, RBLOCK, "
                    f"num_stages={stages}):"
                )
            inner = [
                (
                    "rindex = tl.arange(0, RBLOCK)"
                    if persistent
                    else "rindex = roffset + tl.arange(0, RBLOCK)"
                ),
            ]
            if not div_r:
                inner.append("rmask = rindex < %d" % rnumel)
            if div_x and div_r:
                mask_text = ""          # every lane valid
            elif div_r:
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
            body.extend(textwrap.indent(line, pfx) for line in inner)
            body.extend(
                textwrap.indent(line, pfx)
                for line in self._program_lines(
                    self.program, self._ref, "tmp"
                )
            )
            last = self._ref(self.output_refs[0])
            # The pointwise program transforms padded lanes' neutral loads
            # into non-neutral values (sigmoid(0) = 0.5 and friends), so the
            # reduction must re-mask its INPUT — unless the r-tile is exact.
            if div_r:
                last_masked = last
            else:
                last_masked = (
                    f"tl.where(rmask[None, :], {last}, {spec.neutral()})"
                )
            chunk_offset_add = "" if persistent else " + roffset"
            if tracks_indices:
                # Per-chunk winners over the priority stream.  tl.argmax
                # breaks ties toward the lower lane, and combining chunks
                # with a strict ``>`` keeps the earlier chunk, so the global
                #
                # NaN ordering is explicit: several triton versions ignore
                # greatest value (first NaN wins).  A finite sentinel ranks
                # NaN above every real value, letting cval double as the
                # has-NaN flag without a second reduction per chunk.
                body.append(f"{pfx}isnan_ = {last_masked} != {last_masked}")
                body.append(
                    f"{pfx}prio = tl.where(isnan_, 1.0e38, {last_masked})"
                )
                body.append(f"{pfx}cval = tl.max(prio, axis=1)")
                body.append(
                    f"{pfx}cwin = tl.argmax(prio, axis=1){chunk_offset_add}"
                )
                body.append(f"{pfx}live = acc == acc")
                body.append(
                    f"{pfx}hit = ((cval > acc) | (cval == 1.0e38)) & live"
                )
                body.append(f"{pfx}acci = tl.where(hit, cwin.to(tl.int64), acci)")
                body.append(
                    f"{pfx}acc = tl.where((cval == 1.0e38) & live, float('nan'), "
                    f"tl.where((cval > acc) & live, cval, acc))"
                )
            else:
                body.append(
                    f"{pfx}chunk = {spec.finalize_call(last_masked + ', axis=1')}"
                )
                body.append(f"{pfx}acc = {_ACC_UPDATE[spec.op]}")
            if spec.op == "mean":
                body.append(f"acc = acc * {repr(1.0 / rnumel)}")
            store_source = "acci" if tracks_indices else "acc"
            if self.epilogue is not None and not tracks_indices:
                epilogue_lines, epilogue_last = self._epilogue_lines("acc")
                body.extend(epilogue_lines)
                store_source = epilogue_last
            if div_x:
                body.append(f"tl.store(out_ptr0 + xindex, {store_source})")
            else:
                body.append(
                    f"tl.store(out_ptr0 + xindex, {store_source}, mask=xmask)"
                )
            signature = [
                *(f"in_ptr{index}" for index in range(self.input_count)),
                "out_ptr0",
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
                and numel_total % pw_block == 0
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
            # Index reductions always materialize int64 indices regardless of
            # the value-stream dtype.
            out_dtype = "tp.int64" if spec.tracks_indices else "inputs[0].dtype"
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
                f"    out = tp.empty({out_shape!r}, dtype={out_dtype}, "
                "device=inputs[0].device)\n"
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
            source += "            return out\n"
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
            source += "    return out\n"
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
            # (XBLOCK, num_warps) into the kernel call so the triton dispatch
            # does not have to resolve them per-call through meta. This works
            # whether the grid is literal or via a lambda.
            constexpr_kw = ""
            if fixed_config is not None:
                constexpr_kw = (
                    f", XBLOCK={fixed_config[0]}, "
                    f"num_warps={fixed_config[1]}"
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
                # the geometry it was compiled for.
                if self.reference_shape is not None:
                    grid_src = repr(
                        -(-_prod(self.reference_shape) // fixed_config[0])
                    )
                else:
                    grid_src = f"-(-xnumel // {fixed_config[0]})"
                source += "    _r = _rec\n"
                source += (
                    f"    if _r is not None and {guard} and "
                    "_fl.hooks_clear() and _r[3] == xnumel:\n"
                )
                source += "        try:\n"
                source += "            _s = _fl.current_stream()\n"
                source += (
                    f"            _r[0]({grid_src}, 1, 1, _s, _r[1], _r[2], "
                    f"None, None, None, {call_args_txt}, "
                    f"{fixed_config[0]})\n"
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
                grid_n = -(-_prod(self.reference_shape) // fixed_config[0])
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
):
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed")
    if not _supports_runtime_inputs(
        example_inputs, allow_grad=True, reference_shape=reference_shape
    ):
        raise NotImplementedError("Triton requires matching contiguous CUDA tensors")
    digest = hashlib.sha256(
        (
            repr((_CODEGEN_VERSION, program, constants, output_refs, reduction, epilogue))
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
) -> str:
    """Persisted-decision key for the axis-reduction family (M5d).

    Covers codegen generation, tuning salt, program content, reduction spec,
    shape buckets, device, value dtype and epilogue so a hit can never pin a
    decision from an older emitter or candidate table.
    """

    from ..runtime import stax_autotune

    digest_source = (
        _CODEGEN_VERSION
        + "|"
        + stax_autotune.TUNING_VERSION
        + "|"
        + digest
        + f"|{reduction.op}|{reduction.dims}|{int(reduction.keepdim)}"
        + f"|{stax_autotune.xnumel_bucket(onumel)}|{stax_autotune.xnumel_bucket(rnumel)}"
        + f"|{device_repr}|{value_dtype}|{epilogue_repr}"
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
):
    """Benchmark ``_DIM_REDUCTION_CANDIDATES`` once; persist the decision.

    Uses the CachingAutotuner flow for the axis-reduction kernel family:
    the decision cache key covers program content, reduction spec, shape
    buckets and device, so a hit skips both benchmarking and recompiles.
    """

    assert reference_shape is not None and isinstance(reduction, ReductionSpec)
    from ..runtime import stax_autotune

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
    decision_key = _dims_decision_key(
        stax_autotune.program_digest(program, constants, output_refs),
        reduction,
        onumel,
        rnumel,
        repr(example_inputs[0].device),
        value_dtype,
        epilogue is not None and repr(epilogue) or "",
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
            if record.get("rblock") is not None:
                cached: tuple[int, ...] = (
                    int(record["xblock"]),
                    int(record["warps"]),
                    int(record["rblock"]),
                    int(record["stages"]),
                )
            else:
                cached = (
                    int(record["xblock"]),
                    int(record["warps"]),
                    int(record["stages"]),
                )
            if any(entry == cached for entry in _DIM_REDUCTION_CANDIDATES):
                return build(cached)
        except (ValueError, KeyError, TypeError):
            pass

    if disabled_autotune():
        return build(_STATIC_DIM_TRIPLE)

    best_config, best_launch, best_time = stax_autotune.bench_candidates(
        build, _DIM_REDUCTION_CANDIDATES, list(example_inputs)
    )
    if best_config is None:
        return build(_STATIC_DIM_TRIPLE)
    try:
        record = {
            "xblock": best_config[0],
            "warps": best_config[1],
            "stages": best_config[-1],
        }
        if len(best_config) > 3:
            record["rblock"] = best_config[2]
        cache.store(decision_key, json.dumps(record).encode(), ext="json")
    except Exception:  # noqa: BLE001 - cache is best-effort
        pass
    return best_launch


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
):
    """Bench classic vs persistent split-reduction forms once per bucket."""

    from ..runtime import stax_autotune

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
        + "|"
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

    if payload is not None:
        try:
            record = json.loads(payload.decode())
            cached = _valid(record)
            if any(tuple(c) == cached for c in _SPLIT_CANDIDATES):
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
):
    """Compile a program, autotuning the launch config when possible (M2).

    Uses the CachingAutotuner flow: benchmark candidate configs once at
    compile time and emit a fixed-config kernel; persist the decision so
    later processes skip benchmarking.  Any failure falls back to a static
    pinned config for reductions (the split workspace is baked per config)
    or the plain ``@triton.autotune`` emission for pointwise programs.
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
        from ..runtime import stax_autotune

        digest = stax_autotune.program_digest(program, constants, output_refs)
        if bucket_numel is not None:
            xnumel = bucket_numel
        else:
            xnumel = int(example_inputs[0].numel())
        device_key = repr(example_inputs[0].device)

        def build_fixed(config: tuple[int, int]):
            return build(config)

        # Key on the bare program digest: load_decision() consumers key the
        # same way, and role namespacing is redundant given bucket+device.
        config, launch = stax_autotune.pick_config(
            digest,
            xnumel,
            device_key,
            build_fixed,
            list(example_inputs),
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


_REDUCTION_TAIL_OPS = {"sum", "mean", "amax", "max", "argmax"}


def _reduction_spec_from_node(node: Node) -> ReductionSpec | None:
    """Parse a ``call_method`` reduction node into a :class:`ReductionSpec`.

    Returns ``None`` for anything this backend cannot fold yet (unknown
    kwargs like ``dtype``, ``min``/``max`` value-index pairs, tensor ``dim``
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

    if op == "amax" and not dims:
        return None
    if op == "max" and dims:
        return None  # max(dim) yields a (values, indices) pair
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
    """

    kind: str  # "arg" (graph placeholder position) | "seg" (segment index)
    index: int


@dataclass
class _ExternPlan:
    """A one-node segment executed eagerly between fused kernels.

    The scheduler turns every operator outside the fused surface into its
    own ``"extern"`` segment, so a captured graph can interleave compiled
    pointwise/reduction kernels with unsupported operators (softmax, matmul,
    casts the program cannot express, ...).  ``launch`` resolves the node's
    dependencies through ``extern_sources`` — the same wiring contract fused
    segments use — and calls the operator on real tensors.
    """

    launch: Any
    extern_sources: tuple
    #: example output buffer (shape/dtype of this segment's export)
    example: Any = None
    output_shape: tuple = ()


def _extern_segment_plan(
    graph_module: GraphModule,
    seg: Any,
    sources: tuple,
    sample_dtype: Any,
    sample_device: Any,
    placeholder_positions: dict,
    node_to_seg: dict,
) -> _ExternPlan | None:
    """Build the eager executor for one extern segment; None when unsupported."""

    import tensorplay as _tp

    node = seg.nodes[0]
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
        (_ExternSource(source.kind, source.index)): position
        for position, source in enumerate(sources)
    }

    def resolve(feed: list, value: Any) -> Any:
        if isinstance(value, Node):
            if value.op == "placeholder":
                key = _ExternSource("arg", placeholder_positions[value.name])
            else:
                key = _ExternSource("seg", node_to_seg[value])
            return feed[key_positions[key]]
        return value

    def launch(feed: list) -> Any:
        args = [resolve(feed, value) for value in node.args]
        kwargs = {
            name: resolve(feed, value) for name, value in node.kwargs.items()
        }
        if node.op == "call_function":
            return node.target(*args, **kwargs)
        owner, rest = args[0], args[1:]
        return getattr(owner, node.target)(*rest, **kwargs)

    output_shape = tuple(int(dim) for dim in shape)
    example = _tp.empty(
        output_shape, dtype=dtype, device=sample_device
    )
    return _ExternPlan(
        launch=launch,
        extern_sources=sources,
        example=example,
        output_shape=output_shape,
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
    #: example inputs feeding this segment (launch specialization shape)
    examples: tuple = ()
    #: True when some segment input broadcasts against the local reference —
    #: broadcast VJPs need sum-to-shape (M5f), so training rejects these.
    needs_broadcast: bool = False
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


def _extract_segment_view(
    graph: Graph, nodes, export_node: Node
):
    """Clone ``nodes`` into a standalone Graph with placeholder externals.

    Returns ``(view, mapping)`` where ``view.graph`` feeds the program
    builder and ``mapping`` translates original nodes to their clones.
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
    sub.output(export_new)
    return SimpleNamespace(graph=sub), mapping, externals



def _dbg(msg):
    import os as _os
    if _os.environ.get("TP_STAX_DEBUG"):
        print("[stax]", msg, flush=True)

def compile_graph_module(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    mode: str | None = None,
    strict_native: bool = False,
    **kwargs: Any,
):
    del mode, kwargs
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
    segments = segment_graph(
        graph_module,
        is_pointwise=_is_pointwise,
        classify_reduction=_classify_reduction,
    )
    if segments is None:
        _dbg('fallback gate #4')
        return None
    scheduler_annotate(graph_module, segments)

    needs_broadcast = any(
        tuple(int(dim) for dim in value.shape) != reference_shape
        for value in example_inputs
    )
    any_grad = any(value.requires_grad for value in example_inputs)
    if any_grad:
        # Training lowers through per-segment local VJPs.  Pointwise
        # segments take elementwise VJPs; sum/mean reduction segments take
        # an expanded tangent into their prologue's VJP program (the
        # forward program already exports the reduction input).  Store-time
        # epilogues, index reductions, extremum reductions and extern
        # segments still need their own gradient paths (M5f).
        for seg in segments:
            if seg.epilogue:
                _dbg('fallback gate #5a')
                return None
            if seg.kind == "extern":
                _dbg('fallback gate #5c')
                return None
            if seg.kind == "pw+red" and (
                seg.reduction.tracks_indices
                or seg.reduction.op not in ("sum", "mean")
            ):
                _dbg('fallback gate #5b')
                return None
    if any_grad and needs_broadcast:
        # broadcast gradients need sum-to-shape (M5f).
        _dbg('fallback gate #6')
        return None

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
        for node in [*seg.nodes, *seg.epilogue]:
            node_to_seg[node] = index

    def _extern_sources(seg_index: int, seg):
        """Validate cross-segment wiring; None when unsupported."""

        inside = set(seg.nodes) | set(seg.epilogue)
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
            if (outside_users or is_final) and node is not seg.export_node:
                # interior skip connections need multi-output segments (v2)
                _dbg('fallback gate #8')
                return None
            deps = set(_nodes(node.args)) | set(
                _nodes(node.kwargs)
            )
            for dep in deps:
                if dep in inside:
                    continue
                if dep.op == "placeholder":
                    position = placeholder_positions.get(dep.name)
                    if position is None:
                        _dbg('fallback gate #9')
                        return None
                    key = ("arg", position)
                else:
                    producer_seg = node_to_seg.get(dep)
                    if producer_seg is None or producer_seg >= seg_index:
                        _dbg('fallback gate #10')
                        return None
                    if dep is not segments[producer_seg].export_node:
                        _dbg('fallback gate #11')
                        return None
                    key = ("seg", producer_seg)
                if key not in seen:
                    seen.add(key)
                    sources.append(_ExternSource(key[0], key[1]))
        return tuple(sources)

    segment_plans: list[_SegmentPlan] = []
    import tensorplay as _tp

    sample_dtype = example_inputs[0].dtype
    sample_device = example_inputs[0].device
    placeholder_positions_all = {
        node.name: position
        for position, node in enumerate(graph_module.graph.placeholders)
    }
    for seg_index, seg in enumerate(segments):
        sources = _extern_sources(seg_index, seg)
        if sources is None:
            scheduler_annotate(graph_module, segments)
            _dbg('fallback gate #12')
            return None
        if seg.kind == "extern":
            extern_plan = _extern_segment_plan(
                graph_module,
                seg,
                sources,
                sample_dtype,
                sample_device,
                placeholder_positions_all,
                node_to_seg,
            )
            if extern_plan is None:
                scheduler_annotate(graph_module, segments)
                _dbg('fallback gate #12b')
                return None
            segment_plans.append(extern_plan)
            continue
        sub_view, mapping, externals = _extract_segment_view(
            graph_module.graph, seg.nodes, seg.tail
        )
        reduction = None
        reduction_mode_local = None
        if seg.kind == "pw+red":
            reduction = seg.reduction
            if reduction.tracks_indices:
                # v1 index reductions: float32/float64 only + tl.argmax.
                if not HAS_TL_ARGMAX:
                    _dbg('fallback gate #13')
                    return None
                if sample_dtype not in (_tp.float32, _tp.float64):
                    _dbg('fallback gate #14')
                    return None
            if seg.epilogue:
                assert not reduction.tracks_indices, (
                    "scheduler never joins an epilogue onto argmax"
                )
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
            pointwise = _build_pointwise_program(sub_view)
        if pointwise is None:
            _dbg('fallback gate #15')
            return None
        placeholders_s, forward_program, forward_constants, instructions, output_ref = (
            pointwise
        )
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
        seg_examples = []
        for source in sources:
            if source.kind == "arg":
                seg_examples.append(example_inputs[source.index])
            else:
                producer_seg = segments[source.index]
                if producer_seg.kind == "pw+red":
                    # Reduction output (scalar for full, kept-dims for axis
                    # reductions); an epilogue tail has the same shape.
                    shape: tuple[int, ...] = producer_seg.reduction.output_shape(
                        reference_shape
                    )
                elif producer_seg.kind == "extern":
                    shape = producer_seg.output_shape
                else:
                    shape = reference_shape
                seg_examples.append(
                    _tp.empty(shape, dtype=sample_dtype, device=sample_device)
                )
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
            (output_ref,),
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
        )
        segment_plans.append(
            _SegmentPlan(
                seg_launch,
                sources,
                reduction,
                program=list(forward_program),
                constants=list(forward_constants),
                instructions=list(instructions),
                output_ref=output_ref,
                examples=tuple(seg_examples),
                needs_broadcast=bool(local_needs_broadcast),
                tangent_plan=(
                    None
                    if reduction is None
                    else _reduction_tangent_plan(reduction, local_ref)
                ),
            )
        )

    placeholders = graph_module.graph.placeholders
    backward_launch = None
    autograd_function: Any | None = None
    if any(value.requires_grad for value in example_inputs):
        if len(segment_plans) == 1:
            plan = segment_plans[0]
            if any(
                op_name not in _CPU_FUSED_AUTOGRAD_OPS
                for op_name, *_ in plan.instructions
            ):
                return None
            gradient_plan = _build_fused_gradient_graphs(
                len(placeholders),
                plan.instructions,
                plan.program,
                plan.constants,
                len(plan.program) // 3,
                plan.output_ref,
            )
            if gradient_plan is None:
                return None
            backward_program, backward_constants, backward_outputs = gradient_plan
            # The extra input is the tangent/grad-output supplied by autograd.
            backward_launch = _autotune_launch(
                "bwd",
                backward_program,
                backward_constants,
                backward_outputs,
                [*example_inputs, example_inputs[0]],
            )
        else:
            # M5c training: chain one local VJP program per segment.  The
            # reverse sweep feeds each segment's export-gradient through its
            # own fused backward kernel and accumulates contributions into
            # segment boundaries / placeholders (fan-out sums).
            for plan in segment_plans:
                if (
                    plan.needs_broadcast
                    or plan.tangent_plan is None and plan.spec is not None
                    or any(
                        op_name not in _CPU_FUSED_AUTOGRAD_OPS
                        for op_name, *_ in plan.instructions
                    )
                ):
                    _dbg('fallback gate #20')
                    return None
            for seg_index, plan in enumerate(segment_plans):
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
                # input is this segment's export tangent.
                plan.backward_launch = _autotune_launch(
                    f"bwd{seg_index}",
                    bwd_program,
                    bwd_constants,
                    bwd_outputs,
                    [*plan.examples, plan.examples[0]],
                )

        from ...autograd import Function

        multi_segment_training = len(segment_plans) > 1

        class _StaxTritonAutograd(Function):
            @staticmethod
            def forward(ctx: Any, *forward_inputs: Any) -> Any:
                intermediates = {}
                feed_all = []
                for index, plan in enumerate(segment_plans):
                    feed = [
                        forward_inputs[source.index]
                        if source.kind == "arg"
                        else intermediates[source.index]
                        for source in plan.extern_sources
                    ]
                    intermediates[index] = plan.launch(feed)
                    feed_all.append(feed)
                ctx.stax_feed_all = feed_all
                ctx.save_for_backward(*forward_inputs, *intermediates.values())
                return intermediates[len(segment_plans) - 1]

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
                grad_output = grad_outputs[0] if grad_outputs else None
                saved = ctx.saved_tensors
                if grad_output is None:
                    return (None,) * len(saved)
                inputs_count = len(placeholders)
                if not multi_segment_training:
                    grad_output = _normalize_pointwise_grad_output(
                        grad_output, saved[0]
                    )
                    return tuple(backward_launch([*saved[:inputs_count], grad_output]))
                # normalize once against the final output's operand shape;
                # every downstream tangent already has its producer shape.
                seg_grads: dict[int, Any] = {
                    len(segment_plans) - 1: _normalize_pointwise_grad_output(
                        grad_output, saved[-1]
                    )
                }
                arg_grads: dict[int, Any] = {}

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
                    grads = plan.backward_launch(
                        [*ctx.stax_feed_all[index], tangent]
                    )
                    for source, grad in zip(plan.extern_sources, grads):
                        if source.kind == "arg":
                            accumulate(arg_grads, source.index, grad)
                        else:
                            accumulate(seg_grads, source.index, grad)
                return tuple(
                    arg_grads.get(position) for position in range(inputs_count)
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
        if autograd_function is not None and any(
            value.requires_grad for value in inputs
        ):
            return autograd_function.apply(*inputs)
        intermediates: dict[int, Any] = {}
        for index, plan in enumerate(segment_plans):
            feed = [
                inputs[source.index]
                if source.kind == "arg"
                else intermediates[source.index]
                for source in plan.extern_sources
            ]
            intermediates[index] = plan.launch(feed)
        return intermediates[len(segment_plans) - 1]

    compiled._tensorplay_codegen = "triton"  # type: ignore[attr-defined]
    compiled._tensorplay_backward_codegen = (  # type: ignore[attr-defined]
        "triton"
        if (backward_launch is not None or autograd_function is not None)
        else None
    )
    return compiled
