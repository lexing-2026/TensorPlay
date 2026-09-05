"""Runtime C++ code generation for fused CPU pointwise programs.

The captured program — a flat, topologically ordered instruction list —
is rendered as straight-line C++ that runs explicit ``Vectorized`` SIMD
operations (SLEEF-backed transcendentals included) inside the in-tree
OpenMP worksharing bridge, compiled once by the system compiler, cached
by content hash, and loaded as a Python callable.

Code structure per kernel:

* a four-vector unrolled main loop (independent chains give the scheduler
  instructions to interleave) used when the program is short enough to keep
  register pressure sane;
* a single-vector loop covering the mid body;
* a scalar tail handling the final partial vector through the partial-width
  ``loadu``/``store`` overloads.

Every data pointer is ``__restrict__``-qualified and the hot loops carry
``#pragma GCC ivdep`` so the compiler can reorder loads and stores freely.
Constants are materialized once per body, outside all loops.

The supported operation surface covers the pointwise fused set plus the
comparison/``where``/order-relation extension: comparisons yield 0.0f/1.0f
lanes (the same boolean value domain the program interpreter uses), and
``where`` selects through ``blendv`` driven by the raw comparison mask.

Compilation uses the same multi-tier capability scheme as the in-tree
kernels: a dry-compile probe selects the SIMD tier, and the tier's
compiler definitions and architecture flags are applied verbatim.  All
vector math therefore matches the interpreter path bit-for-bit.

The module degrades gracefully: without a system compiler, without the
development headers, on unsupported programs, or on any build/load failure
it returns ``None`` and callers keep the generic kernel path.
``TP_STAX_CPU_NATIVE=0`` disables the path outright.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from typing import Any, Callable, Optional

from ..codecache import default_cache, file_lock
from ..cpp_builder import (
    CppBuilder,
    CppOptions,
    get_compiler_version_info,
    get_cpp_compiler,
    package_paths,
)
from ..cpu_vec_isa import VecISA, pick_vec_isa
from .index_expr import (
    Const as _IxC,
    Symbol as _IxS,
    affine_coeff as _ix_affine,
    modular_indexing as _ix_mod,
    render as _ix_render,
)

# The four-way unrolled loop is emitted only when replicating the program
# four times keeps live temporaries within reason; longer programs use the
# single-vector loop (the compiler still schedules within one vector).
_UNROLL_MAX_STEPS = 16

_MAX_INPUTS = 16

# Instruction surface, keyed by the program's op names.  ``{a}``/``{b}``
# are the operand placeholders; every expression is a single C++ rvalue.
_BINARY_EXPR: dict[str, str] = {
    "add": "({a} + {b})",
    "sub": "({a} - {b})",
    "mul": "({a} * {b})",
    "div": "({a} / {b})",
    "pow": "{a}.pow({b})",
}
_COMPARE_EXPR: dict[str, str] = {
    "lt": "{a}.lt({b})",
    "le": "{a}.le({b})",
    "gt": "{a}.gt({b})",
    "ge": "{a}.ge({b})",
    "eq": "{a}.eq({b})",
    "ne": "{a}.ne({b})",
}
_ORDER_EXPR: dict[str, str] = {
    "minimum": "tensorplay::vec::minimum({a}, {b})",
    "maximum": "tensorplay::vec::maximum({a}, {b})",
    "clamp_min": "tensorplay::vec::maximum({a}, {b})",
    "clamp_max": "tensorplay::vec::minimum({a}, {b})",
}
_UNARY_EXPR: dict[str, str] = {
    "neg": "(-{a})",
    "pos": "{a}",
    "abs": "{a}.abs()",
    "sin": "{a}.sin()",
    "cos": "{a}.cos()",
    "exp": "{a}.exp()",
    "log": "{a}.log()",
    "sigmoid": "(V(1.0f) / (V(1.0f) + (-{a}).exp()))",
    "sqrt": "{a}.sqrt()",
    "square": "({a} * {a})",
    "tanh": "{a}.tanh()",
    "relu": "tensorplay::vec::maximum({a}, V(0.0f))",
    "relu_grad": "{a}.gt(V(0.0f))",
    "abs_grad": "({a}.gt(V(0.0f)) - {a}.lt(V(0.0f)))",
    "rsqrt": "{a}.rsqrt()",
    "exp2": "{a}.exp2()",
    "erf": "{a}.erf()",
}
_BINARY_OPS = frozenset(_BINARY_EXPR) | frozenset(_COMPARE_EXPR) | frozenset(_ORDER_EXPR)
_UNARY_OPS = frozenset(_UNARY_EXPR)

# float32 identity is the only cast the float-domain kernel can express;
# other targets keep the graph on the interpreter/Triton paths.
_F32_CAST_ID = 3


class _ProgramError(Exception):
    """Raised when an instruction list cannot be rendered."""


def _const_text(value: float) -> str:
    text = f"{value:.17g}"
    if not any(ch in text for ch in ".eE"):
        text += ".0"
    return f"V({text}f)"


_I_i = _IxS("i")
_I_W = _IxS("W")


def _lane_offset(lane: int) -> str:
    """Address of one unroll lane, derived and verified in the index algebra.

    Lane ``k`` reads element ``i + k*W``; the algebra confirms the address is
    unit-stride in the induction variable (any ``W``-bearing terms are
    loop-invariant constants at kernel time), which is the precondition for
    the contiguous ``V::loadu`` path.
    """

    expr = _I_i if lane == 0 else _I_i + _IxC(lane) * _I_W
    coeff = _ix_affine(expr, _I_i)
    if coeff != 1:
        raise _ProgramError(f"non-unit-stride lane addressing (stride {coeff})")
    return _ix_render(expr)


# Per-input addressing modes, one per program input.
#
# flat    -- address is the flat output index; contiguous ``loadu`` per lane.
# splat   -- the input has one element (any broadcast scalar); a single
#            ``V(in[0])`` splat is hoisted out of every loop.
# colmod  -- the input's only non-unit dim is the output's last dim of width
#            ``S`` (a row broadcast, e.g. a bias vector): address ``i % S``.
#            Every vector stays inside one row because ``W`` (the lane count)
#            divides ``S`` and the induction variable is peeled to a ``W``
#            boundary first, so each ``loadu`` is still contiguous.
InputMode = tuple[str, int]


def analyze_input_modes(
    input_shapes: tuple[tuple[int, ...], ...],
    input_strides: tuple[tuple[int, ...], ...],
    out_shape: tuple[int, ...],
    lane_count: int,
) -> tuple[InputMode, ...] | None:
    """Classify how each input is addressed from the flat output index.

    Returns one mode per input, or ``None`` when any input uses an addressing
    pattern the emitter cannot prove contiguous within a vector (strided
    walks, column broadcasts, nested rearrangements): those keep the generic
    fallback instead of risking wrong addresses.
    """

    if len(input_shapes) != len(input_strides) or not input_shapes:
        return None
    rank = len(out_shape)
    modes: list[InputMode] = []
    for shape, strides in zip(input_shapes, input_strides):
        if len(shape) != len(strides) or len(shape) > rank:
            return None
        # Left-align broadcast dims: leading dims the input lacks are size-1.
        pad = rank - len(shape)
        aligned_shape = (1,) * pad + tuple(int(d) for d in shape)
        aligned_strides = (0,) * pad + tuple(int(s) for s in strides)
        # Dims that contribute nothing to the address: size-1 (broadcast) or
        # stride-0 (``expand`` of any extent -- every element along the dim
        # aliases the same address).
        live = [
            (d, aligned_strides[d])
            for d in range(rank)
            if aligned_shape[d] != 1 and aligned_strides[d] != 0
        ]
        if not live:
            modes.append(("splat", 0))
            continue
        if len(live) == 1:
            dim, stride = live[0]
            if (
                dim == rank - 1
                and stride == 1
                and aligned_shape[dim] == out_shape[dim]
                and rank >= 2
                and out_shape[dim] > 0
                and out_shape[dim] % lane_count == 0
            ):
                modes.append(("colmod", int(out_shape[dim])))
                continue
            if (
                rank == 1
                and dim == 0
                and aligned_shape[0] == out_shape[0]
            ):
                # Rank-1 output: a matching trailing dim is the flat index
                # itself; the vector path needs unit stride.
                if stride == 1:
                    modes.append(("flat", 0))
                    continue
                return None
            return None
        if len(live) == rank:
            # Full-rank input: only the unit-stride flat layout is provably
            # row-contiguous; any stride permutation is rejected.
            expected = 1
            flat_contiguous = True
            for d in range(rank - 1, -1, -1):
                if aligned_strides[d] != expected:
                    flat_contiguous = False
                    break
                expected *= aligned_shape[d]
            if flat_contiguous:
                modes.append(("flat", 0))
                continue
        return None
    return tuple(modes)


def _check_ref(
    ref: int, constants: list[float], input_count: int, temp_count: int
) -> None:
    if ref >= 0:
        if ref >= input_count + temp_count:
            raise _ProgramError("reference beyond program surface")
        return
    index = -ref - 1
    if index < 0 or index >= len(constants):
        raise _ProgramError("constant reference out of range")


def _analyze_instructions(
    instructions: list[tuple[str, int, int, int]],
    constants: list[float],
    input_count: int,
    output_ref: int | None = None,
    *,
    allow_empty: bool = False,
) -> set[int]:
    """Validate the instruction list and return the referenced input set.

    Ref layout matches the program encoding: ``0..input_count-1`` are
    inputs, ``input_count + i`` is the result of instruction ``i``, and
    negative refs index ``constants``.  ``where``/``where_rest`` must
    appear as an adjacent pair sharing the condition ref.
    """

    if input_count <= 0 or input_count > _MAX_INPUTS:
        raise _ProgramError("degenerate program")
    if not instructions and not allow_empty:
        # A pointwise kernel with no instruction would only copy an input;
        # reduction kernels legitimately reduce a raw input, so they opt in.
        raise _ProgramError("degenerate program")
    temp_count = len(instructions)
    used: set[int] = set()
    skip_where_rest = False
    for i, (op, lhs, rhs, result) in enumerate(instructions):
        if skip_where_rest:
            if op != "where_rest" or result != input_count + i:
                raise _ProgramError("where/where_rest pairing broken")
            _check_ref(rhs, constants, input_count, temp_count)
            if 0 <= rhs < input_count:
                used.add(rhs)
            skip_where_rest = False
            continue
        if result != input_count + i:
            raise _ProgramError("instruction result refs must be sequential")
        if op == "where":
            if i + 1 >= temp_count:
                raise _ProgramError("where without where_rest")
            nxt = instructions[i + 1]
            if nxt[0] != "where_rest" or nxt[1] != lhs:
                raise _ProgramError("where/where_rest pairing broken")
            _check_ref(lhs, constants, input_count, temp_count)
            _check_ref(rhs, constants, input_count, temp_count)
            used.update(ref for ref in (lhs, rhs) if 0 <= ref < input_count)
            skip_where_rest = True
            continue
        if op == "where_rest":
            raise _ProgramError("where_rest without where")
        if op in _BINARY_OPS:
            _check_ref(lhs, constants, input_count, temp_count)
            _check_ref(rhs, constants, input_count, temp_count)
            used.update(ref for ref in (lhs, rhs) if 0 <= ref < input_count)
        elif op in _UNARY_OPS:
            _check_ref(lhs, constants, input_count, temp_count)
            if rhs != -1:
                raise _ProgramError("unary op with rhs operand")
            if 0 <= lhs < input_count:
                used.add(lhs)
        elif op == "cast":
            _check_ref(lhs, constants, input_count, temp_count)
            if rhs != _F32_CAST_ID:
                raise _ProgramError("unsupported cast target")
            if 0 <= lhs < input_count:
                used.add(lhs)
        else:
            raise _ProgramError(f"unsupported op: {op}")
    if output_ref is not None and 0 <= output_ref < input_count:
        used.add(output_ref)
    return used


def _operand_expr(
    ref: int,
    constants: list[float],
    input_count: int,
    names: Callable[[int], str],
) -> str:
    if ref >= 0:
        return names(ref)
    index = -ref - 1
    return _const_text(constants[index])


def _expr_for(
    op: str,
    lhs: int,
    rhs: int,
    constants: list[float],
    input_count: int,
    names: Callable[[int], str],
) -> str:
    if op in _BINARY_OPS:
        template = (
            _BINARY_EXPR.get(op)
            or _COMPARE_EXPR.get(op)
            or _ORDER_EXPR.get(op)
        )
        return template.format(
            a=_operand_expr(lhs, constants, input_count, names),
            b=_operand_expr(rhs, constants, input_count, names),
        )
    if op in _UNARY_OPS:
        return _UNARY_EXPR[op].format(
            a=_operand_expr(lhs, constants, input_count, names)
        )
    if op == "cast":
        return _operand_expr(lhs, constants, input_count, names)
    raise _ProgramError(f"unsupported op: {op}")  # pragma: no cover


def _emit_body(
    instructions: list[tuple[str, int, int, int]],
    constants: list[float],
    input_count: int,
    output_ref: int,
    used_inputs: set[int],
    indent: str,
    *,
    unrolled: bool,
    partial: bool,
    input_modes: tuple[InputMode, ...] | None = None,
) -> str:
    """Emit one loop body for a given variable-naming scheme.

    ``unrolled`` suffixes every variable with the unroll lane index and
    loads/stores four vectors; ``partial`` switches loads/stores to the
    partial-width ``count`` overloads (scalar tail).  ``input_modes`` maps
    each input to its addressing mode: ``flat`` keeps the flat index, and
    ``colmod`` addresses through ``i % S`` (the peel in the rendered kernel
    guarantees the lane never straddles a row).  ``splat`` inputs are hoisted
    before all loops and referenced without a lane suffix.
    """

    if input_modes is None:
        input_modes = (("flat", 0),) * input_count

    def name(ref: int, lane: int = 0) -> str:
        if ref < input_count:
            if input_modes[ref][0] == "splat":
                return f"h{ref}"
            base = f"x{ref}"
        else:
            base = f"t{ref - input_count}"
        return f"{base}{lane}" if unrolled else base

    count_expr = "count" if partial else "W"
    lines: list[str] = []
    used_temps = {
        ref
        for op, lhs, rhs, _ in instructions
        for ref in ((lhs,) if (op in _UNARY_OPS or op == "cast") else (lhs, rhs))
        if ref >= input_count
    }

    def lane_offset(mode: str, width: int, lane: int) -> str:
        if mode == "colmod":
            expr = _I_i if lane == 0 else _I_i + _IxC(lane) * _I_W
            return _ix_render(_ix_mod(expr, 1, width))
        return "i" if lane == 0 else _lane_offset(lane)

    def emit_loads(lane: int) -> None:
        for ref in sorted(used_inputs):
            if input_modes[ref][0] == "splat":
                continue
            var = name(ref, lane)
            mode, width = input_modes[ref]
            lines.append(
                f"{indent}V {var} = V::loadu(in{ref} + {lane_offset(mode, width, lane)}, {count_expr});"
            )

    def emit_steps(lane: int) -> None:
        pending_where: tuple[int, str] | None = None
        for op, lhs, rhs, result in instructions:
            if op == "where":
                pending_where = (
                    result,
                    _operand_expr(
                        rhs, constants, input_count, lambda r: name(r, lane)
                    ),
                )
                continue
            if op == "where_rest":
                assert pending_where is not None
                where_result, a_expr = pending_where
                cond_expr = _operand_expr(
                    lhs, constants, input_count, lambda r: name(r, lane)
                )
                b_expr = _operand_expr(
                    rhs, constants, input_count, lambda r: name(r, lane)
                )
                lines.append(f"{indent}V {name(where_result, lane)} = {a_expr};")
                lines.append(
                    f"{indent}V {name(result, lane)} = V::blendv("
                    f"{b_expr}, {a_expr}, ({cond_expr} > V(0.0f)));"
                )
                pending_where = None
                continue
            expr = _expr_for(
                op, lhs, rhs, constants, input_count, lambda r: name(r, lane)
            )
            lines.append(f"{indent}V {name(result, lane)} = {expr};")
        if pending_where is not None:  # pragma: no cover - guarded by analysis
            raise _ProgramError("where without where_rest")

    def emit_store(lane: int) -> None:
        var = name(output_ref, lane)
        lines.append(f"{indent}{var}.store(out + {lane_offset('flat', 0, lane)}, {count_expr});")

    if unrolled:
        for lane in range(4):
            emit_loads(lane)
        for lane in range(4):
            emit_steps(lane)
        for lane in range(4):
            emit_store(lane)
    else:
        emit_loads(0)
        emit_steps(0)
        emit_store(0)
    return "\n".join(lines)


def emit_value_program(
    instructions: list[tuple[str, int, int, int]],
    constants: list[float],
    input_count: int,
    output_ref: int,
    used_inputs: set[int],
    *,
    indent: str,
    offset: str,
    count_expr: str,
    suffix: str,
    input_modes: tuple[InputMode, ...],
) -> tuple[list[str], str]:
    """Emit one vector evaluation of the program and name its result.

    ``offset`` is a C expression for the flat element index addressed by lane
    0 of this vector; ``count_expr`` is the active lane count (``W`` for a
    full vector, a runtime ``count`` for a masked tail).  ``suffix``
    disambiguates the temporaries of independent evaluations that coexist in
    one scope, which is how the unrolled accumulator groups keep separate
    dependency chains.  ``splat`` inputs resolve to the hoisted broadcast and
    are not reloaded.

    Returns the emitted lines and the identifier holding the program result,
    so a caller can feed it straight into an accumulator combine instead of
    storing it.
    """

    lines: list[str] = []

    def name(ref: int) -> str:
        if ref < input_count:
            if input_modes[ref][0] == "splat":
                return f"h{ref}"
            return f"x{ref}{suffix}"
        return f"t{ref - input_count}{suffix}"

    for ref in sorted(used_inputs):
        mode, width = input_modes[ref]
        if mode == "splat":
            continue
        if mode == "colmod":
            address = f"in{ref} + (({offset}) % {width})"
        else:
            address = f"in{ref} + ({offset})"
        lines.append(f"{indent}V {name(ref)} = V::loadu({address}, {count_expr});")

    pending_where: tuple[int, str] | None = None
    for op, lhs, rhs, result in instructions:
        if op == "where":
            pending_where = (
                result,
                _operand_expr(rhs, constants, input_count, name),
            )
            continue
        if op == "where_rest":
            if pending_where is None:  # pragma: no cover - guarded by analysis
                raise _ProgramError("where_rest without where")
            where_result, a_expr = pending_where
            cond_expr = _operand_expr(lhs, constants, input_count, name)
            b_expr = _operand_expr(rhs, constants, input_count, name)
            lines.append(f"{indent}V {name(where_result)} = {a_expr};")
            lines.append(
                f"{indent}V {name(result)} = V::blendv("
                f"{b_expr}, {a_expr}, ({cond_expr} > V(0.0f)));"
            )
            pending_where = None
            continue
        expr = _expr_for(op, lhs, rhs, constants, input_count, name)
        lines.append(f"{indent}V {name(result)} = {expr};")
    if pending_where is not None:  # pragma: no cover - guarded by analysis
        raise _ProgramError("where without where_rest")
    return lines, name(output_ref)


def render_kernel_source(
    instructions: list[tuple[str, int, int, int]],
    constants: list[float],
    input_count: int,
    output_ref: int,
    entry: str,
    *,
    out_shape: tuple[int, ...] | None = None,
    out_device: tuple[int, int] | None = None,
    input_shapes: tuple[tuple[int, ...], ...] | None = None,
    input_strides: tuple[tuple[int, ...], ...] | None = None,
    lane_count: int | None = None,
) -> str:
    """Render the full translation unit for one fused CPU kernel.

    ``out_shape``/``out_device`` (DeviceType ordinal, device index) pin the
    specialization; when both are given the unit also emits a METH_FASTCALL
    runner that receives the input tensor list, extracts the data pointers
    in C, allocates the output in C, calls the kernel, and wraps the result
    — the steady-state call never re-enters Python.

    ``input_shapes``/``input_strides`` (when given) select per-input
    addressing modes: broadcast scalars become hoisted splats and row
    broadcasts become ``i % S`` addresses under an alignment peel.  Passing
    them without ``out_shape`` has no effect -- the modes are only valid for
    a pinned specialization.
    """

    used_inputs = _analyze_instructions(
        instructions, constants, input_count, output_ref
    )
    if output_ref < 0 or output_ref >= input_count + len(instructions):
        raise _ProgramError("output reference out of range")

    const_decls = (
        "\n".join(
            f"    const V c{index} = {_const_text(value)};"
            for index, value in enumerate(constants)
        )
        or "    (void)0;"
    )

    # ``lane_count`` comes from the picked SIMD tier (the same one that will
    # compile the unit).  Without it the analysis falls back to the widest
    # tier width, which is conservative in the safe direction: a ``colmod``
    # width divisible by 16 is divisible by every smaller lane count.
    tier_width = lane_count if lane_count is not None else 16
    input_modes: tuple[InputMode, ...] | None = None
    if out_shape is not None and input_shapes is not None and input_strides is not None:
        try:
            input_modes = analyze_input_modes(
                input_shapes, input_strides, out_shape, tier_width
            )
        except (TypeError, ValueError):
            input_modes = None
    if input_modes is None:
        input_modes = (("flat", 0),) * input_count
    splat_refs = [
        ref for ref in sorted(used_inputs) if input_modes[ref][0] == "splat"
    ]
    splat_decls = "\n".join(
        f"    const V h{ref} = V(in{ref}[0]);" for ref in splat_refs
    )

    unrolled_body = ""
    if len(instructions) <= _UNROLL_MAX_STEPS:
        unrolled_body = _emit_body(
            instructions,
            constants,
            input_count,
            output_ref,
            used_inputs,
            "        ",
            unrolled=True,
            partial=False,
            input_modes=input_modes,
        )
    single_body = _emit_body(
        instructions,
        constants,
        input_count,
        output_ref,
        used_inputs,
        "        ",
        unrolled=False,
        partial=False,
        input_modes=input_modes,
    )
    tail_body = _emit_body(
        instructions,
        constants,
        input_count,
        output_ref,
        used_inputs,
        "        ",
        unrolled=False,
        partial=True,
        input_modes=input_modes,
    )

    input_params = ", ".join(
        f"const float* __restrict__ in{i}" for i in range(input_count)
    )
    ctx_fields = "".join(f"    const float* in{i};\n" for i in range(input_count))
    ctx_init = ", ".join(f"in{i}" for i in range(input_count)) + ", out"
    ctx_loads = "".join(
        f"    const float* __restrict__ in{i} = c->in{i};\n"
        for i in range(input_count)
    )

    # Alignment peel for ``colmod`` inputs: every vector must start at a
    # flat index congruent to 0 mod the vector width so, together with the
    # row width being a multiple of the widest tier width, the whole vector
    # stays inside one row.  Advance from the chunk start to the next
    # boundary with scalar steps; interior iterations are then fully
    # vectorized row-interior loads.
    peel_needed = any(mode == "colmod" for mode, _ in input_modes)
    peel_loop = ""
    if peel_needed:
        peel_loop = (
            "    for (; i % W != 0 && i < e; ++i) {\n"
            "        const long count = 1;\n"
            f"{tail_body}\n"
            "    }\n"
        )

    splat_hoists = ""
    if splat_refs:
        splat_hoists = f"{splat_decls}\n"

    unrolled_loop = ""
    if unrolled_body:
        unrolled_loop = (
            "    #pragma GCC ivdep\n"
            "    for (; i + 4 * W <= e; i += 4 * W) {\n"
            f"{unrolled_body}\n"
            "    }\n"
        )

    # Worksharing policy follows the parallel-depth decision of the
    # CPU kernels: a region runs on the shared pool only when each
    # thread would still receive at least one minimum chunk (otherwise the
    # body runs inline, serially), and the chunk size is an even static
    # split of the trip count.  ``min_chunk`` matches the in-tree kernel
    # grain floor.
    min_chunk = 512
    threads = _pool_threads()
    serial_cutoff = threads * min_chunk

    entry_call_tail = ""
    runner_section = ""
    direct_section = ""
    extra_includes = ""
    if out_shape is not None and out_device is not None:
        shape_init = ", ".join(f"{int(d)}LL" for d in out_shape)
        dev_ordinal, dev_index = out_device
        dev_name = {0: "CPU", 1: "CUDA"}[dev_ordinal]
        ptr_args = "".join(f", in[{i}]" for i in range(input_count))
        numel = 1
        for d in out_shape:
            numel *= int(d)
        runner_section = (
            "\n"
            "static PyObject* tp_runner(PyObject*, PyObject* const* args, "
            "Py_ssize_t nargs) {\n"
            "    try {\n"
            "        if (nargs != 1)\n"
            "            throw std::runtime_error(\"runner expects the input list\");\n"
            "        PyObject* inputs = args[0];\n"
            f"        if (!PyList_CheckExact(inputs) ||\n"
            f"            PyList_GET_SIZE(inputs) != {input_count})\n"
            "            throw std::runtime_error(\"runner input list mismatch\");\n"
            f"        const float* in[{input_count}];\n"
            f"        for (long i = 0; i < {input_count}; ++i) {{\n"
            "            in[i] = static_cast<const float*>(\n"
            "                tensorplay::python_c::tpx_py_tensor_cref(\n"
            "                    PyList_GET_ITEM(inputs, i)).data_ptr());\n"
            "        }\n"
            "        tensorplay::Tensor out = tensorplay::Tensor::empty(\n"
            f"            std::vector<int64_t>{{{shape_init}}},\n"
            "            tensorplay::ScalarType::Float32,\n"
            f"            tensorplay::Device(tensorplay::DeviceType::{dev_name}, "
            f"{int(dev_index)}LL), false);\n"
            f"        {entry}({numel}LL{ptr_args}, out.data_ptr<float>());\n"
            "        return tensorplay::python_c::tpx_py_wrap(out);\n"
            "    } catch (const std::exception& e) {\n"
            "        PyErr_SetString(PyExc_RuntimeError, e.what());\n"
            "        return nullptr;\n"
            "    } catch (...) {\n"
            "        PyErr_SetString(PyExc_RuntimeError, \"unhandled kernel runner error\");\n"
            "        return nullptr;\n"
            "    }\n"
            "}\n"
            "\n"
            "static PyMethodDef tp_runner_def = {\n"
            "    \"runner\", (PyCFunction)(void (*)(void))tp_runner,\n"
            "    METH_FASTCALL, nullptr};\n"
            "\n"
            'extern "C" PyObject* tp_make_runner(void) {\n'
            "// The factory may be invoked through a GIL-releasing foreign-call\n"
            "// bridge; object creation needs the interpreter held.\n"
            "    PyGILState_STATE gil = PyGILState_Ensure();\n"
            "    PyObject* runner = PyCFunction_New(&tp_runner_def, nullptr);\n"
            "    PyGILState_Release(gil);\n"
            "    return runner;\n"
            "}\n"
        )
        # Direct entry for the compiled-call trampoline: takes the C array
        # of data pointers the steady-state guard has already validated,
        # allocates the pinned output, runs the kernel, and returns the
        # wrapped result.  No Python containers cross this boundary in
        # either direction.
        direct_section = (
            'extern "C" PyObject* tp_direct(const void* const* ins) {\n'
            "    try {\n"
            f"        const float* in[{input_count}];\n"
            f"        for (int i = 0; i < {input_count}; ++i)\n"
            "            in[i] = static_cast<const float*>(ins[i]);\n"
            "        tensorplay::Tensor out = tensorplay::Tensor::empty(\n"
            f"            std::vector<int64_t>{{{shape_init}}},\n"
            "            tensorplay::ScalarType::Float32,\n"
            f"            tensorplay::Device(tensorplay::DeviceType::{dev_name}, "
            f"{int(dev_index)}LL), false);\n"
            f"        {entry}({numel}LL"
            + "".join(f", in[{i}]" for i in range(input_count))
            + ", out.data_ptr<float>());\n"
            "        return tensorplay::python_c::tpx_py_wrap(out);\n"
            "    } catch (const std::exception& e) {\n"
            "        PyErr_SetString(PyExc_RuntimeError, e.what());\n"
            "        return nullptr;\n"
            "    } catch (...) {\n"
            "        PyErr_SetString(PyExc_RuntimeError, \"unhandled kernel error\");\n"
            "        return nullptr;\n"
            "    }\n"
            "}\n"
        )
        extra_includes = (
            "#include <Python.h>\n"
            "#include <vector>\n"
            "#include \"Tensor.h\"\n"
            "\n"
            "namespace tensorplay { namespace python_c {\n"
            "const Tensor& tpx_py_tensor_cref(PyObject* obj);\n"
            "PyObject* tpx_py_wrap(const Tensor& t);\n"
            "}}  // namespace tensorplay::python_c\n"
            "\n"
        )

    return (
        '#include "cpu/vec/vec.h"\n'
        "using V = tensorplay::vec::Vectorized<float>;\n"
        "\n"
        f"{extra_includes}"
        "typedef void (*tp_parallel_body_c)(void* ctx, long long b, long long e);\n"
        'extern "C" void tp_parallel_for_c('
        "long long begin, long long end, long long grain, "
        "tp_parallel_body_c body, void* ctx);\n"
        "\n"
        "typedef struct TP_Ctx {\n"
        f"{ctx_fields}"
        "    float* out;\n"
        "} TP_Ctx;\n"
        "\n"
        "static void tp_body(void* ctxp, long long b, long long e) {\n"
        "    const TP_Ctx* c = (const TP_Ctx*)ctxp;\n"
        f"{ctx_loads}"
        "    float* __restrict__ out = c->out;\n"
        "    const long W = V::size();\n"
        f"{const_decls}\n"
        f"{splat_hoists}"
        "    long i = b;\n"
        f"{peel_loop}"
        f"{unrolled_loop}"
        "    #pragma GCC ivdep\n"
        "    for (; i + W <= e; i += W) {\n"
        f"{single_body}\n"
        "    }\n"
        "    if (i < e) {\n"
        "        const long count = e - i;\n"
        f"{tail_body}\n"
        "    }\n"
        "}\n"
        "\n"
        f'extern "C" void {entry}(long n, {input_params}, float* __restrict__ out) {{\n'
        f"    TP_Ctx ctx{{{ctx_init}}};\n"
        f"    if (n < {serial_cutoff}LL) {{\n"
        "        tp_body(&ctx, 0, n);\n"
        "    } else {\n"
        f"        tp_parallel_for_c(0, n, {min_chunk}LL, tp_body, &ctx);\n"
        "    }\n"
        "}\n"
        f"{runner_section}"
        f"{direct_section}"
    )


def _pool_threads() -> int:
    """Intra-op pool size at codegen time (baked into the entry check)."""

    try:
        import tensorplay

        threads = int(tensorplay.get_num_threads())
    except Exception:
        threads = 0
    if threads < 1:
        threads = (os.cpu_count() or 2) // 2
    return max(1, threads)


# ---------------------------------------------------------------------------
# Compile + load
# ---------------------------------------------------------------------------

_LIBS_STATE: dict[str, dict[str, Any]] = {"libs": {}}


def _kill_switch() -> bool:
    return os.environ.get("TP_STAX_CPU_NATIVE", "") == "0"


def compile_translation_unit(
    source: str,
    entry: str,
    *,
    isa: VecISA,
    paths: tuple[str, str, str],
    compiler: str,
    version_info: str,
    pinned: bool,
    bind_tag: str,
) -> Any:
    """Build one generated translation unit and return the loaded library.

    The unit is keyed by its own text plus the toolchain fingerprint, written
    once under the shared kernel cache, and loaded through a process-level
    handle table so repeated lowerings of the same program reuse one
    ``dlopen``.  ``pinned`` selects the link set: units that allocate their
    own output tensor and wrap it for the interpreter additionally need the
    Python bridge library.  Returns ``None`` when the toolchain, the build,
    or the load fails -- every caller keeps a working non-generated route.
    """

    include_dir, generated_include_dir, lib_dir = paths
    cache = default_cache("stax-cpu-native")
    key_options = {
        "tier": isa.name,
        "flags": " ".join(isa.build_arch_flags()),
        "ver": version_info[:32],
        "entry": entry,
        "bind": bind_tag,
    }
    key = cache.cache_key(source, entry, key_options)
    source_path = cache.path_for(key, "cpp")
    output_path = cache.path_for(key, "so")

    if not os.path.exists(output_path):
        try:
            import sysconfig

            python_include = sysconfig.get_paths()["include"]
            generated_ops_include = os.path.join(
                os.path.dirname(generated_include_dir), "generated"
            )
            os.makedirs(os.path.dirname(source_path), exist_ok=True)
            with file_lock(output_path + ".lock"):
                if not os.path.exists(output_path):
                    with open(source_path, "w") as fh:
                        fh.write(source)
                    include_dirs = [
                        include_dir,
                        generated_include_dir,
                        python_include,
                    ]
                    if os.path.isdir(generated_ops_include):
                        # Tensor.h pulls the generated op declarations when
                        # the runtime headers are present.
                        include_dirs.append(generated_ops_include)
                    options = CppOptions(
                        compiler=compiler,
                        definitions=isa.definitions(),
                        include_dirs=include_dirs,
                        cflags=[
                            "-std=c++20",
                            "-O3",
                            "-fno-math-errno",
                            "-fPIC",
                            "-shared",
                            *isa.build_arch_flags(),
                        ],
                        library_dirs=[lib_dir],
                        # ``tpx`` must stay on the link line: libp10 carries
                        # undefined references into the tpx ops namespace,
                        # and a kernel module that omits it fails to dlopen
                        # under RTLD_LOCAL (then silently loses the compiled
                        # route to the interpreter fallback).
                        libraries=["p10", "tpx", "tp_python"]
                        if pinned
                        else ["p10", "tpx"],
                        ldflags=[f"-Wl,-rpath,{lib_dir}"],
                    )
                    builder = CppBuilder(
                        name=os.path.basename(output_path),
                        sources=[source_path],
                        options=options,
                        output_dir=os.path.dirname(source_path),
                    )
                    builder.build()
        except Exception:
            if not os.path.exists(output_path):
                return None

    libs = _LIBS_STATE.setdefault("libs", {})
    lib = libs.get(output_path)
    if lib is None:
        try:
            lib = ctypes.CDLL(output_path)
        except Exception:
            return None
        libs[output_path] = lib
    return lib


def _digest_entry_key(
    instructions,
    constants,
    input_count,
    output_ref,
    tier,
    version_info,
    pinned_shape=None,
    layout_key=None,
) -> str:
    # The pinned shape and per-input layouts are baked into the generated
    # unit (output allocation, addressing modes), so they belong in the
    # entry symbol: two specializations sharing one program must never
    # collide on a cached artifact.
    return hashlib.sha256(
        repr(
            (
                tuple(instructions),
                tuple(constants),
                input_count,
                output_ref,
                tier,
                version_info,
                pinned_shape,
                layout_key,
            )
        ).encode()
    ).hexdigest()[:16]


def build_cpu_native_kernel(
    instructions: list[tuple[str, int, int, int]],
    constants: list[float],
    input_count: int,
    output_ref: int | None = None,
    *,
    shape: Any = None,
    device: Any = None,
    input_shapes: tuple[tuple[int, ...], ...] | None = None,
    input_strides: tuple[tuple[int, ...], ...] | None = None,
) -> Optional[Callable[[list[Any]], Any]]:
    """Compile the program once and return ``run(inputs) -> Tensor``.

    Returns ``None`` when the path is disabled, the development headers are
    missing, the program cannot be rendered, or the build/load fails.

    ``shape``/``device`` pin the specialization: the lowering route already
    verified every input against them, so with both present the unit emits
    a METH_FASTCALL runner that keeps the steady-state call entirely in C
    (pointer extraction, output allocation, result wrapping), matching the
    binding pattern of the in-tree kernel loader.

    ``input_shapes``/``input_strides`` describe each input's layout at
    compile time (the pinned route guarantees they hold for every call);
    together with ``shape`` they enable broadcast-splat hoisting and
    row-broadcast ``i % S`` addressing.  The caller remains responsible for
    the route check -- the generated unit trusts these layouts.
    """

    if _kill_switch():
        return None
    if output_ref is None:
        output_ref = input_count + len(instructions) - 1
    try:
        _analyze_instructions(instructions, constants, input_count, output_ref)
    except _ProgramError:
        return None
    paths = package_paths()
    if paths is None:
        return None
    compiler = get_cpp_compiler()
    if not compiler:
        return None
    include_dir, generated_include_dir, lib_dir = paths

    isa: VecISA = pick_vec_isa(paths)
    if not isa:
        return None

    version_info = get_compiler_version_info(compiler)

    pinned_shape: tuple[int, ...] | None = None
    out_device_code: tuple[int, int] | None = None
    if shape is not None:
        try:
            pinned_shape = tuple(int(item) for item in shape)
        except (TypeError, ValueError):
            pinned_shape = None
    if pinned_shape is not None and device is not None:
        device_ordinals = {"cpu": 0, "cuda": 1}
        try:
            out_device_code = (
                device_ordinals[str(device.type)],
                int(device.index) if device.index is not None else -1,
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            out_device_code = None
    if out_device_code is None:
        pinned_shape = None

    layout_key: Any = None
    if (
        pinned_shape is not None
        and input_shapes is not None
        and input_strides is not None
    ):
        layout_key = (
            tuple(tuple(int(d) for d in s) for s in input_shapes),
            tuple(tuple(int(s) for s in st) for st in input_strides),
        )

    entry = f"tp_native_{_digest_entry_key(instructions, constants, input_count, output_ref, isa.name, version_info, pinned_shape, layout_key)}"
    source = render_kernel_source(
        instructions,
        constants,
        input_count,
        output_ref,
        entry,
        out_shape=pinned_shape,
        out_device=out_device_code,
        input_shapes=input_shapes if layout_key is not None else None,
        input_strides=input_strides if layout_key is not None else None,
        lane_count=isa.nelements(),
    )
    pinned = pinned_shape is not None and out_device_code is not None

    lib = compile_translation_unit(
        source,
        entry,
        isa=isa,
        paths=paths,
        compiler=compiler,
        version_info=version_info,
        pinned=pinned,
        bind_tag="plan-v2" if pinned else "py",
    )
    if lib is None:
        return None
    fn = getattr(lib, entry, None)
    if fn is None:
        return None
    fn.restype = None
    fn.argtypes = (
        [ctypes.c_long] + [ctypes.c_void_p] * input_count + [ctypes.c_void_p]
    )

    import tensorplay

    runner: Any = None
    direct_addr = 0
    if pinned:
        # The generated unit carries a METH_FASTCALL runner: it receives the
        # input tensor list, extracts the data pointers in C, allocates the
        # output in C, and wraps the result — the steady-state call never
        # re-enters Python.  The ``tp_direct`` export additionally hands the
        # compiled-call trampoline a pointer-level entry, skipping the list
        # hop entirely.
        make_runner = getattr(lib, "tp_make_runner", None)
        if make_runner is not None:
            make_runner.restype = ctypes.py_object
            make_runner.argtypes = []
            try:
                runner = make_runner()
            except Exception:
                return None
        direct_fn = getattr(lib, "tp_direct", None)
        if direct_fn is not None:
            try:
                direct_addr = ctypes.cast(direct_fn, ctypes.c_void_p).value or 0
            except (ValueError, TypeError, OSError):
                direct_addr = 0
        if runner is not None:
            return (runner, direct_addr)
        return None

    if pinned_shape is None:
        # Unpinned builds keep the runtime shape read; the route check
        # still guarantees all inputs match, so reading input 0 once per
        # call is faithful.
        def run(inputs: list[Any]) -> Any:
            out = tensorplay.empty(
                tuple(int(item) for item in inputs[0].shape),
                dtype=tensorplay.float32,
                device=inputs[0].device,
            )
            fn(
                inputs[0].numel(),
                *[t.data_ptr() for t in inputs],
                out.data_ptr(),
            )
            return out

        return run

    empty = tensorplay.empty
    f32 = tensorplay.float32
    numel = 1
    for item in pinned_shape:
        numel *= item
    pinned_device = device

    def run(inputs: list[Any]) -> Any:
        out = empty(
            pinned_shape,
            dtype=f32,
            device=pinned_device,
        )
        fn(
            numel,
            *[t.data_ptr() for t in inputs],
            out.data_ptr(),
        )
        return out

    return run
