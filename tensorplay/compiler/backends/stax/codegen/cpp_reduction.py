"""Runtime C++ code generation for fused CPU reduction programs.

A reduction region is one pointwise expression program followed by a single
reduction over a set of dimensions.  The generated translation unit evaluates
the expression and the reduction in **one pass**: the elementwise result is
never materialized, so a chain such as ``(x * 2).tanh().sum(dim=1)`` reads its
input once and writes only the reduced output.

Loop structure
--------------

The input shape is first collapsed into *runs* of adjacent dimensions that
share a classification (kept or reduced).  The runs are then scheduled as::

    for row in rows:            # every kept run except a trailing one
        accumulate over the reduced runs
        store

The trailing kept run, when present, becomes the innermost ``post`` extent and
is stride-1 in both the input and the output.  Kept runs are hoisted above the
reduced runs so one accumulator group serves the whole inner nest; the reduced
runs keep their declaration order, so the innermost one is always the stride-1
axis of its level.

Two schedules follow from that layout:

* **vertical** (``post > 1``) -- the vector axis is the kept trailing run, so
  every lane accumulates an independent output element.  No cross-lane step is
  needed; accumulators are stored straight to the output.
* **horizontal** (``post == 1``) -- the vector axis is the innermost reduced
  run.  Lanes are folded once, at the end of the row.

Both schedules run four independent accumulator groups so the machine has four
disjoint dependency chains to interleave, and both share the masked-tail form:
the partial vector is combined through ``V::set`` so lanes past the end keep
the accumulator's own value regardless of what the expression produced for the
zero-filled remainder.

Numerics
--------

Float sums accumulate through a four-level cascade (pairwise summation): each
accumulator group owns a small stack of partial sums promoted on a
power-of-two schedule, which turns the error growth of a length-``n`` sum from
``O(n)`` into ``O(log n)`` while keeping one add per element in the inner
loop.  The promotion period is derived from the trip count, so short
reductions never execute a promotion at all.  ``max``/``min`` propagate NaN in
both the vector and the cross-lane step, and ``mean`` divides the completed
sum by the reduced element count.

Parallelism
-----------

The entry picks its worksharing strategy at code generation time, when every
extent is already known:

* enough independent output rows -> split the row loop, no combining at all;
* few rows but a long reduction -> split the reduction itself into one slot
  per worker and fold the slots in a fixed order, which keeps the result
  reproducible for a given worker count;
* below the worksharing floor -> run the body inline with no dispatch.

The module degrades gracefully: without a system compiler, on an unsupported
program or layout, or on any build/load failure it returns ``None`` and the
caller keeps its existing route.  ``TP_STAX_CPU_NATIVE=0`` disables it.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..cpp_builder import (
    get_compiler_version_info,
    get_cpp_compiler,
    package_paths,
)
from ..cpu_vec_isa import VecISA, pick_vec_isa
from .cpp import (
    _ProgramError,
    _analyze_instructions,
    _const_text,
    _pool_threads,
    analyze_input_modes,
    compile_translation_unit,
    emit_value_program,
)

# Independent accumulator groups per reduction.  Four disjoint chains cover
# the latency of a vector add or max on current cores without exhausting the
# register file once the cascade stack is resident as well.
_ILP = 4

# Cascade depth for float summation.  Level 0 takes every element; each
# promotion folds one level into the next.
_LEVELS = 4

# Widest lane count of any supported tier; sizes the cross-lane staging array.
_MAX_LANES = 16

# Reduction spellings the float program can express.
_REDUCE_OPS = ("sum", "mean", "max", "min", "amax", "amin", "prod")

_VEC_COMBINE = {
    "sum": "({a} + {b})",
    "prod": "({a} * {b})",
    "max": "tensorplay::vec::maximum({a}, {b})",
    "min": "tensorplay::vec::minimum({a}, {b})",
}
_SCALAR_COMBINE = {
    "sum": "({a} + {b})",
    "prod": "({a} * {b})",
    "max": "tp_max_s({a}, {b})",
    "min": "tp_min_s({a}, {b})",
}
_IDENTITY = {
    "sum": "0.0f",
    "prod": "1.0f",
    "max": "-std::numeric_limits<float>::infinity()",
    "min": "std::numeric_limits<float>::infinity()",
}

# Element count below which a region never reaches the worksharing pool.
_MIN_CHUNK = 4096

# Per-worker slot buffer budget for the split strategy; it lives on the stack
# of the calling thread, so the output footprint is bounded against it.
_SPLIT_SLOT_BYTES = 65536


def _family(op: str) -> str:
    """Collapse spelling variants onto one combine family."""

    if op in ("amax", "max"):
        return "max"
    if op in ("amin", "min"):
        return "min"
    if op in ("sum", "mean"):
        return "sum"
    return op


def _combine(op: str, acc: str, value: str) -> str:
    return _VEC_COMBINE[_family(op)].format(a=acc, b=value)


def _scombine(op: str, acc: str, value: str) -> str:
    return _SCALAR_COMBINE[_family(op)].format(a=acc, b=value)


def _identity(op: str) -> str:
    return _IDENTITY[_family(op)]


def _cascades(op: str) -> bool:
    """Whether this reduction accumulates through the cascade stack."""

    return _family(op) == "sum"


def _finalize_scalar(op: str, value: str, red: int) -> str:
    # ``mean`` is a completed sum divided by the reduced element count, in
    # float, which is the same value the non-compiled reduction produces.
    return f"({value} / (float){red}L)" if op == "mean" else value


def _finalize_vector(op: str, value: str, red: int) -> str:
    return f"({value} / V((float){red}L))" if op == "mean" else value


@dataclass(frozen=True)
class ReduceSpec:
    """One reduction: an operation, the reduced axes, and the output rank."""

    op: str
    dims: tuple[int, ...]
    keepdim: bool

    def normalized(self, rank: int) -> "ReduceSpec | None":
        """Resolve negative axes; reject duplicates and out-of-range axes."""

        if self.op not in _REDUCE_OPS:
            return None
        resolved: list[int] = []
        for dim in self.dims:
            value = int(dim)
            if value < 0:
                value += rank
            if value < 0 or value >= rank or value in resolved:
                return None
            resolved.append(value)
        return ReduceSpec(self.op, tuple(sorted(resolved)), self.keepdim)


@dataclass(frozen=True)
class LoopPlan:
    """Collapsed loop nest for one reduction over a contiguous input."""

    # (extent, input stride) per kept run driving the row loop, outermost
    # first; their mixed-radix order is the output's element order.
    row_dims: tuple[tuple[int, int], ...]
    # (extent, input stride) per reduced run, declaration order kept so the
    # last entry is the stride-1 axis whenever ``post`` is 1.
    red_dims: tuple[tuple[int, int], ...]
    # Trailing contiguous kept extent; 1 when the innermost run is reduced.
    post: int
    rows: int
    red: int
    out_shape: tuple[int, ...]

    @property
    def schedule(self) -> str:
        return "vertical" if self.post > 1 else "horizontal"


def _collapse_runs(
    shape: tuple[int, ...], reduced: frozenset[int]
) -> list[tuple[int, bool, int]]:
    """Merge adjacent dimensions sharing a classification.

    Returns ``(extent, is_reduced, stride)`` per run in dimension order, with
    strides from the contiguous layout of ``shape``.
    """

    rank = len(shape)
    strides = [1] * rank
    for dim in range(rank - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]

    runs: list[tuple[int, bool, int]] = []
    for dim in range(rank):
        is_reduced = dim in reduced
        if runs and runs[-1][1] == is_reduced:
            extent, flag, _ = runs[-1]
            runs[-1] = (extent * shape[dim], flag, strides[dim])
        else:
            runs.append((shape[dim], is_reduced, strides[dim]))
    return runs


def plan_reduction(
    in_shape: tuple[int, ...], spec: ReduceSpec
) -> LoopPlan | None:
    """Build the loop nest for ``spec`` over a contiguous ``in_shape``."""

    rank = len(in_shape)
    if rank == 0 or any(int(extent) <= 0 for extent in in_shape):
        return None
    normalized = spec.normalized(rank)
    if normalized is None or not normalized.dims:
        return None
    reduced = frozenset(normalized.dims)

    runs = _collapse_runs(tuple(int(extent) for extent in in_shape), reduced)
    post = 1
    if not runs[-1][1]:
        post = runs[-1][0]
        runs = runs[:-1]

    row_dims = tuple((extent, stride) for extent, flag, stride in runs if not flag)
    red_dims = tuple((extent, stride) for extent, flag, stride in runs if flag)
    if not red_dims:
        return None

    rows = 1
    for extent, _ in row_dims:
        rows *= extent
    red = 1
    for extent, _ in red_dims:
        red *= extent

    out_shape: list[int] = []
    for dim in range(rank):
        if dim in reduced:
            if normalized.keepdim:
                out_shape.append(1)
        else:
            out_shape.append(int(in_shape[dim]))

    return LoopPlan(
        row_dims=row_dims,
        red_dims=red_dims,
        post=post,
        rows=rows,
        red=red,
        out_shape=tuple(out_shape),
    )


# --------------------------------------------------------------------------
# emission helpers
# --------------------------------------------------------------------------


def _row_base_lines(plan: LoopPlan, indent: str, extra: str = "") -> list[str]:
    """Decompose the flat row index into an input offset.

    The row dimensions form a mixed-radix counter in declaration order, which
    is the output's own element order, so the decomposition costs one division
    per row dimension and none at all for the common single-run case.
    ``extra`` is added to every offset (the split strategy passes the base of
    its chunk).
    """

    prefix = f"{extra} + " if extra else ""
    if not plan.row_dims:
        return [f"{indent}const long base = {prefix}0L;"]
    if len(plan.row_dims) == 1:
        stride = plan.row_dims[0][1]
        return [f"{indent}const long base = {prefix}row * {stride}L;"]
    lines = [f"{indent}long rem_ = row;", f"{indent}long acc_base_ = 0;"]
    for extent, stride in reversed(plan.row_dims[1:]):
        lines.append(f"{indent}acc_base_ += (rem_ % {extent}L) * {stride}L;")
        lines.append(f"{indent}rem_ /= {extent}L;")
    lines.append(f"{indent}acc_base_ += rem_ * {plan.row_dims[0][1]}L;")
    lines.append(f"{indent}const long base = {prefix}acc_base_;")
    return lines


def _cascade_setup(op: str, blocks_expr: str, indent: str, groups: int) -> list[str]:
    """Declare the cascade stack and derive its promotion period.

    The stack is only ever reached through the out-of-line promotion helper,
    which keeps it in memory: the hot loop holds nothing but the accumulator
    group itself, so a wide reduction never spills its live values.
    """

    if not _cascades(op):
        return []
    depth = (_LEVELS - 1) * groups
    return [
        f"{indent}long lp_ = tp_ceil_log2({blocks_expr}) / {_LEVELS}L;",
        f"{indent}if (lp_ < 4L) lp_ = 4L;",
        f"{indent}const long level_power = lp_;",
        f"{indent}const long level_mask = (1L << level_power) - 1L;",
        f"{indent}long blk_ = 0;",
        f"{indent}V lvl_[{depth}];",
        f"{indent}for (int li_ = 0; li_ < {depth}; ++li_) lvl_[li_] = V(0.0f);",
    ]


def _cascade_tick(op: str, indent: str, groups: int) -> list[str]:
    """Promote finished chunks up the cascade stack.

    The block counter decides how far the promotion carries; the work itself
    goes through an out-of-line helper so the accumulators stay in registers.
    """

    if not _cascades(op):
        return []
    accs = ", ".join(f"a{group}" for group in range(groups))
    lines = [
        f"{indent}++blk_;",
        f"{indent}if ((blk_ & level_mask) == 0) {{",
        f"{indent}    const V accs_[{groups}] = {{{accs}}};",
        f"{indent}    tp_cascade_promote<{groups}>("
        f"lvl_, accs_, blk_, level_power, level_mask);",
    ]
    for group in range(groups):
        lines.append(f"{indent}    a{group} = V(0.0f);")
    lines.append(f"{indent}}}")
    return lines


def _cascade_flush(op: str, indent: str, groups: int) -> list[str]:
    """Fold the cascade stack back into the named accumulators."""

    if not _cascades(op):
        return []
    return [
        f"{indent}a{group} = a{group} + lvl_[{level * groups + group}];"
        for group in range(groups)
        for level in range(_LEVELS - 1)
    ]


def _open_reduced_nest(
    dims: tuple[tuple[int, int], ...],
    indent: str,
    first_range: tuple[str, str] | None,
) -> tuple[list[str], str, str]:
    """Open loops for ``dims`` and return (lines, offset name, body indent)."""

    lines: list[str] = []
    offset = "base"
    body = indent
    for level, (extent, stride) in enumerate(dims):
        start, end = ("0L", f"{extent}L")
        if level == 0 and first_range is not None:
            start, end = first_range
        nxt = f"ro{level}"
        lines.append(
            f"{body}for (long r{level} = {start}, {nxt} = base_at_{level}; "
            f"r{level} < {end}; ++r{level}, {nxt} += {stride}L) {{"
        )
        # ``base_at_<level>`` names the offset at this level's start index so
        # a split range does not have to re-derive it from the loop variable.
        lines.insert(
            len(lines) - 1,
            f"{body}const long base_at_{level} = {offset} + ({start}) * {stride}L;",
        )
        offset = nxt
        body += "    "
    return lines, offset, body


def _close_reduced_nest(count: int, indent: str) -> list[str]:
    lines = []
    body = indent + "    " * count
    for _ in range(count):
        body = body[:-4]
        lines.append(f"{body}}}")
    return lines


def _emit_program(
    program: dict[str, Any], indent: str, offset: str, count_expr: str, suffix: str
) -> tuple[list[str], str]:
    return emit_value_program(
        program["instructions"],
        program["constants"],
        program["input_count"],
        program["output_ref"],
        program["used_inputs"],
        indent=indent,
        offset=offset,
        count_expr=count_expr,
        suffix=suffix,
        input_modes=program["input_modes"],
    )


def _emit_horizontal(
    plan: LoopPlan,
    op: str,
    program: dict[str, Any],
    indent: str,
    *,
    inner_range: tuple[str, str],
    outer_range: tuple[str, str] | None,
    blocks_expr: str,
) -> list[str]:
    """Accumulate one row along the innermost reduced run; leave it in ``acc_``.

    ``inner_range`` bounds the vectorized axis and ``outer_range`` the
    outermost reduced run, so the split strategy reuses this body verbatim for
    a chunk of the reduction.
    """

    outer_dims = plan.red_dims[:-1]
    lines: list[str] = []
    for group in range(_ILP):
        lines.append(f"{indent}V a{group} = V({_identity(op)});")
    lines.extend(_cascade_setup(op, blocks_expr, indent, _ILP))

    open_lines, offset, body = _open_reduced_nest(
        outer_dims, indent, outer_range if outer_dims else None
    )
    lines.extend(open_lines)

    start, end = inner_range
    lines.append(f"{body}long i = {start};")
    lines.append(f"{body}for (; i + {_ILP}L * W <= {end}; i += {_ILP}L * W) {{")
    lane = body + "    "
    results: list[str] = []
    for group in range(_ILP):
        lane_offset = (
            f"{offset} + i" if group == 0 else f"{offset} + i + {group}L * W"
        )
        group_lines, result = _emit_program(
            program, lane, lane_offset, "W", f"_l{group}"
        )
        lines.extend(group_lines)
        results.append(result)
    for group, result in enumerate(results):
        lines.append(f"{lane}a{group} = {_combine(op, f'a{group}', result)};")
    lines.extend(_cascade_tick(op, lane, _ILP))
    lines.append(f"{body}}}")

    lines.append(f"{body}for (; i + W <= {end}; i += W) {{")
    step_lines, result = _emit_program(program, lane, f"{offset} + i", "W", "_s")
    lines.extend(step_lines)
    lines.append(f"{lane}a0 = {_combine(op, 'a0', result)};")
    lines.append(f"{body}}}")

    lines.append(f"{body}if (i < {end}) {{")
    lines.append(f"{lane}const long count = ({end}) - i;")
    tail_lines, result = _emit_program(
        program, lane, f"{offset} + i", "count", "_p"
    )
    lines.extend(tail_lines)
    # Lanes past ``count`` carry whatever the expression made of the
    # zero-filled remainder, so the combine is applied under a lane mask.
    lines.append(f"{lane}a0 = V::set(a0, {_combine(op, 'a0', result)}, count);")
    lines.append(f"{body}}}")

    lines.extend(_close_reduced_nest(len(outer_dims), indent))
    lines.extend(_cascade_flush(op, indent, _ILP))
    lines.append(f"{indent}a0 = {_combine(op, 'a0', 'a1')};")
    lines.append(f"{indent}a2 = {_combine(op, 'a2', 'a3')};")
    lines.append(f"{indent}a0 = {_combine(op, 'a0', 'a2')};")
    lines.append(f"{indent}float lanes_[{_MAX_LANES}];")
    lines.append(f"{indent}a0.store(lanes_);")
    lines.append(f"{indent}float acc_ = lanes_[0];")
    lines.append(f"{indent}for (long k_ = 1; k_ < W; ++k_)")
    lines.append(f"{indent}    acc_ = {_scombine(op, 'acc_', 'lanes_[k_]')};")
    return lines


def _emit_vertical(
    plan: LoopPlan,
    op: str,
    program: dict[str, Any],
    indent: str,
    *,
    out_ptr: str,
    outer_range: tuple[str, str] | None,
    blocks_expr: str,
    accumulate_into_out: bool,
    red_for_mean: int,
) -> list[str]:
    """Accumulate one row along the kept trailing run and store it.

    ``accumulate_into_out`` folds into the destination instead of overwriting
    it, which is how a split chunk contributes to its worker slot.
    """

    post = plan.post
    lines: list[str] = []

    def group_body(group_indent: str, groups: int, count_expr: str) -> list[str]:
        out: list[str] = []
        for group in range(groups):
            out.append(f"{group_indent}V a{group} = V({_identity(op)});")
        out.extend(_cascade_setup(op, blocks_expr, group_indent, groups))
        open_lines, offset, body = _open_reduced_nest(
            plan.red_dims, group_indent, outer_range
        )
        out.extend(open_lines)
        results: list[str] = []
        for group in range(groups):
            lane_offset = (
                f"{offset} + p" if group == 0 else f"{offset} + p + {group}L * W"
            )
            group_lines, result = _emit_program(
                program, body, lane_offset, count_expr, f"_v{group}"
            )
            out.extend(group_lines)
            results.append(result)
        for group, result in enumerate(results):
            combined = _combine(op, f"a{group}", result)
            if count_expr == "W":
                out.append(f"{body}a{group} = {combined};")
            else:
                out.append(
                    f"{body}a{group} = V::set(a{group}, {combined}, {count_expr});"
                )
        out.extend(_cascade_tick(op, body, groups))
        out.extend(_close_reduced_nest(len(plan.red_dims), group_indent))
        out.extend(_cascade_flush(op, group_indent, groups))
        for group in range(groups):
            target = (
                f"{out_ptr} + p" if group == 0 else f"{out_ptr} + p + {group}L * W"
            )
            if accumulate_into_out:
                out.append(
                    f"{group_indent}V prev{group}_ = V::loadu({target}, {count_expr});"
                )
                out.append(
                    f"{group_indent}V res{group}_ = "
                    f"{_combine(op, f'prev{group}_', f'a{group}')};"
                )
                out.append(
                    f"{group_indent}res{group}_.store({target}, {count_expr});"
                )
            else:
                value = _finalize_vector(op, f"a{group}", red_for_mean)
                out.append(f"{group_indent}{value}.store({target}, {count_expr});")
        return out

    lines.append(f"{indent}long p = 0;")
    lines.append(f"{indent}for (; p + {_ILP}L * W <= {post}L; p += {_ILP}L * W) {{")
    lines.extend(group_body(indent + "    ", _ILP, "W"))
    lines.append(f"{indent}}}")
    lines.append(f"{indent}for (; p + W <= {post}L; p += W) {{")
    lines.extend(group_body(indent + "    ", 1, "W"))
    lines.append(f"{indent}}}")
    lines.append(f"{indent}if (p < {post}L) {{")
    lines.append(f"{indent}    const long count = {post}L - p;")
    lines.extend(group_body(indent + "    ", 1, "count"))
    lines.append(f"{indent}}}")
    return lines


def _select_strategy(plan: LoopPlan, threads: int) -> str:
    """Choose the worksharing strategy from the pinned extents."""

    work = plan.rows * plan.post * plan.red
    if threads <= 1 or work < _MIN_CHUNK * threads:
        return "serial"
    if plan.rows >= threads:
        return "rows"
    # Too few rows to fill the pool: splitting the reduction itself keeps
    # every worker busy, as long as its slot buffer stays within budget and
    # the split axis has enough extent to hand each worker real work.
    out_elements = plan.rows * plan.post
    if (
        threads * out_elements * 4 <= _SPLIT_SLOT_BYTES
        and plan.red_dims[0][0] >= 2 * threads
    ):
        return "split"
    # Otherwise a partial fill still beats a serial run whenever one row is
    # itself above the worksharing floor.
    if plan.rows >= 2 and plan.post * plan.red >= _MIN_CHUNK:
        return "rows"
    return "serial"


def _prologue() -> str:
    return (
        '#include "cpu/vec/vec.h"\n'
        "#include <cmath>\n"
        "#include <limits>\n"
        "#include <Python.h>\n"
        "#include <vector>\n"
        '#include "Tensor.h"\n'
        "using V = tensorplay::vec::Vectorized<float>;\n"
        "\n"
        "namespace tensorplay { namespace python_c {\n"
        "const Tensor& tpx_py_tensor_cref(PyObject* obj);\n"
        "PyObject* tpx_py_wrap(const Tensor& t);\n"
        "}}  // namespace tensorplay::python_c\n"
        "\n"
        "// Order relations propagate NaN, matching the vector forms.\n"
        "static inline float tp_max_s(float a, float b) {\n"
        "    const float c = (a > b) ? a : b;\n"
        "    return std::isnan(a) ? a : c;\n"
        "}\n"
        "static inline float tp_min_s(float a, float b) {\n"
        "    const float c = (a < b) ? a : b;\n"
        "    return std::isnan(a) ? a : c;\n"
        "}\n"
        "// Cascade promotion, kept out of line so the stack it walks stays in\n"
        "// memory and the caller's accumulators stay in registers.\n"
        f"template <int G>\n"
        "__attribute__((noinline)) static void tp_cascade_promote(\n"
        "    V* lvl, const V* accs, long blk, long level_power, long level_mask) {\n"
        "    for (int g = 0; g < G; ++g) lvl[g] = lvl[g] + accs[g];\n"
        f"    for (long j = 1; j < {_LEVELS - 1}L; ++j) {{\n"
        "        if ((blk & (level_mask << (j * level_power))) != 0) return;\n"
        "        for (int g = 0; g < G; ++g) {\n"
        "            lvl[j * G + g] = lvl[j * G + g] + lvl[(j - 1) * G + g];\n"
        "            lvl[(j - 1) * G + g] = V(0.0f);\n"
        "        }\n"
        "    }\n"
        "}\n"
        "static inline long tp_ceil_log2(long value) {\n"
        "    if (value <= 2L) return 1L;\n"
        "    unsigned long remaining = (unsigned long)(value - 1L);\n"
        "    long result = 0;\n"
        "    while (remaining != 0UL) { ++result; remaining >>= 1; }\n"
        "    return result;\n"
        "}\n"
        "\n"
        "typedef void (*tp_parallel_body_c)(void* ctx, long long b, long long e);\n"
        'extern "C" void tp_parallel_for_c('
        "long long begin, long long end, long long grain, "
        "tp_parallel_body_c body, void* ctx);\n"
        "\n"
    )


def render_reduction_source(
    instructions: list[tuple[str, int, int, int]],
    constants: list[float],
    input_count: int,
    output_ref: int,
    spec: ReduceSpec,
    plan: LoopPlan,
    entry: str,
    *,
    out_device: tuple[int, int],
    input_shapes: tuple[tuple[int, ...], ...],
    input_strides: tuple[tuple[int, ...], ...],
    in_shape: tuple[int, ...],
    lane_count: int,
) -> str:
    """Render the full translation unit for one fused CPU reduction."""

    used_inputs = _analyze_instructions(
        instructions, constants, input_count, output_ref, allow_empty=True
    )
    if output_ref < 0 or output_ref >= input_count + len(instructions):
        raise _ProgramError("output reference out of range")

    input_modes = analyze_input_modes(
        input_shapes, input_strides, in_shape, lane_count
    )
    if input_modes is None:
        raise _ProgramError("unsupported input layout")
    inner_extent = plan.post if plan.post > 1 else plan.red_dims[-1][0]
    for mode, width in input_modes:
        if mode == "colmod" and inner_extent % width != 0:
            # A row-broadcast address stays contiguous inside a vector only
            # when the vector never straddles a row.  The nest enters its
            # inner axis at a multiple of that axis's extent, so the row width
            # has to divide it.
            raise _ProgramError("row-broadcast width does not tile the inner axis")

    program = {
        "instructions": instructions,
        "constants": constants,
        "input_count": input_count,
        "output_ref": output_ref,
        "used_inputs": used_inputs,
        "input_modes": input_modes,
    }

    op = spec.op
    threads = _pool_threads()
    strategy = _select_strategy(plan, threads)
    out_elements = plan.rows * plan.post

    const_decls = "\n".join(
        f"    const V c{index} = {_const_text(value)};"
        for index, value in enumerate(constants)
    )
    splat_decls = "\n".join(
        f"    const V h{ref} = V(in{ref}[0]);"
        for ref in sorted(used_inputs)
        if input_modes[ref][0] == "splat"
    )

    ctx_fields = "".join(f"    const float* in{i};\n" for i in range(input_count))
    ctx_fields += "    float* out;\n"
    if strategy == "split":
        ctx_fields += "    float* slots;\n    long grain;\n"
    ctx_loads = "".join(
        f"    const float* __restrict__ in{i} = c->in{i};\n"
        for i in range(input_count)
    )

    # ---- body -----------------------------------------------------------
    body_lines: list[str] = []
    if strategy == "split":
        # One worker slot per chunk of the outermost reduced run.  The bridge
        # advances chunk starts by at least the requested grain, so the slot
        # index derived from the start is unique per chunk.
        body_lines.append("    const long slot = b / c->grain;")
        body_lines.append(
            f"    float* __restrict__ out = c->slots + slot * {out_elements}L;"
        )
        outer_range = ("b", "e")
    else:
        body_lines.append("    float* __restrict__ out = c->out;")
        outer_range = None

    row_start, row_end = ("b", "e") if strategy != "split" else ("0L", f"{plan.rows}L")
    body_lines.append(f"    for (long row = {row_start}; row < {row_end}; ++row) {{")
    body_lines.extend(_row_base_lines(plan, "        "))

    single_red_run = len(plan.red_dims) == 1
    if plan.post > 1:
        blocks = (
            f"((e - b) * {plan.red // plan.red_dims[0][0]}L)"
            if strategy == "split"
            else f"{plan.red}L"
        )
        body_lines.append(
            f"        float* __restrict__ orow = out + row * {plan.post}L;"
        )
        body_lines.extend(
            _emit_vertical(
                plan,
                op,
                program,
                "        ",
                out_ptr="orow",
                outer_range=outer_range,
                blocks_expr=blocks,
                accumulate_into_out=(strategy == "split"),
                red_for_mean=plan.red,
            )
        )
    else:
        inner_extent_expr = f"{plan.red_dims[-1][0]}L"
        inner_range = ("0L", inner_extent_expr)
        horizontal_outer = outer_range
        if strategy == "split" and single_red_run:
            # The split axis *is* the vector axis: the chunk becomes the
            # bounds of the vectorized loop instead of an enclosing loop.
            inner_range = ("b", "e")
            horizontal_outer = None
            blocks = f"((e - b) / ({_ILP}L * W))"
        elif strategy == "split":
            blocks = f"((e - b) * {plan.red // plan.red_dims[0][0]}L / ({_ILP}L * W))"
        else:
            blocks = f"({plan.red}L / ({_ILP}L * W))"
        body_lines.extend(
            _emit_horizontal(
                plan,
                op,
                program,
                "        ",
                inner_range=inner_range,
                outer_range=horizontal_outer,
                blocks_expr=blocks,
            )
        )
        if strategy == "split":
            body_lines.append(
                f"        out[row] = {_scombine(op, 'out[row]', 'acc_')};"
            )
        else:
            body_lines.append(
                f"        out[row] = {_finalize_scalar(op, 'acc_', plan.red)};"
            )
    body_lines.append("    }")
    body = "\n".join(body_lines)

    # ---- entry ----------------------------------------------------------
    input_params = ", ".join(
        f"const float* __restrict__ in{i}" for i in range(input_count)
    )
    ctx_init = ", ".join(f"in{i}" for i in range(input_count)) + ", out"
    entry_lines: list[str] = []
    if strategy == "split":
        outer_extent = plan.red_dims[0][0]
        entry_lines.append(f"    const long slots = {threads}L;")
        entry_lines.append(
            f"    long grain = ({outer_extent}L + slots - 1L) / slots;"
        )
        entry_lines.append("    if (grain < 1L) grain = 1L;")
        entry_lines.append(
            f"    const long nslots = ({outer_extent}L + grain - 1L) / grain;"
        )
        entry_lines.append(f"    float slot_buf[{threads}L * {out_elements}L];")
        entry_lines.append(
            f"    for (long s = 0; s < nslots * {out_elements}L; ++s)"
        )
        entry_lines.append(f"        slot_buf[s] = {_identity(op)};")
        entry_lines.append(f"    TP_Ctx ctx{{{ctx_init}, slot_buf, grain}};")
        entry_lines.append(
            f"    tp_parallel_for_c(0, {outer_extent}L, grain, tp_body, &ctx);"
        )
        entry_lines.append(f"    for (long j = 0; j < {out_elements}L; ++j) {{")
        entry_lines.append("        float acc = slot_buf[j];")
        entry_lines.append("        for (long s = 1; s < nslots; ++s)")
        entry_lines.append(
            "            acc = "
            + _scombine(op, "acc", f"slot_buf[s * {out_elements}L + j]")
            + ";"
        )
        entry_lines.append(
            f"        out[j] = {_finalize_scalar(op, 'acc', plan.red)};"
        )
        entry_lines.append("    }")
    elif strategy == "rows":
        per_row = max(1, plan.post * plan.red)
        row_grain = max(1, (_MIN_CHUNK + per_row - 1) // per_row)
        entry_lines.append(f"    TP_Ctx ctx{{{ctx_init}}};")
        entry_lines.append(
            f"    tp_parallel_for_c(0, {plan.rows}L, {row_grain}L, tp_body, &ctx);"
        )
    else:
        entry_lines.append(f"    TP_Ctx ctx{{{ctx_init}}};")
        entry_lines.append(f"    tp_body(&ctx, 0, {plan.rows}L);")
    entry_body = "\n".join(entry_lines)

    dev_ordinal, dev_index = out_device
    dev_name = {0: "CPU", 1: "CUDA"}[dev_ordinal]
    shape_init = ", ".join(f"{int(d)}LL" for d in plan.out_shape)
    call_args = ", ".join(f"in[{i}]" for i in range(input_count))

    alloc = (
        "        tensorplay::Tensor out = tensorplay::Tensor::empty(\n"
        f"            std::vector<int64_t>{{{shape_init}}},\n"
        "            tensorplay::ScalarType::Float32,\n"
        f"            tensorplay::Device(tensorplay::DeviceType::{dev_name}, "
        f"{int(dev_index)}LL), false);\n"
        f"        {entry}({call_args}, out.data_ptr<float>());\n"
        "        return tensorplay::python_c::tpx_py_wrap(out);\n"
    )

    runner_section = (
        "\n"
        "static PyObject* tp_runner(PyObject*, PyObject* const* args, "
        "Py_ssize_t nargs) {\n"
        "    try {\n"
        "        if (nargs != 1)\n"
        '            throw std::runtime_error("runner expects the input list");\n'
        "        PyObject* inputs = args[0];\n"
        "        if (!PyList_CheckExact(inputs) ||\n"
        f"            PyList_GET_SIZE(inputs) != {input_count})\n"
        '            throw std::runtime_error("runner input list mismatch");\n'
        f"        const float* in[{input_count}];\n"
        f"        for (long i = 0; i < {input_count}; ++i) {{\n"
        "            in[i] = static_cast<const float*>(\n"
        "                tensorplay::python_c::tpx_py_tensor_cref(\n"
        "                    PyList_GET_ITEM(inputs, i)).data_ptr());\n"
        "        }\n"
        f"{alloc}"
        "    } catch (const std::exception& e) {\n"
        "        PyErr_SetString(PyExc_RuntimeError, e.what());\n"
        "        return nullptr;\n"
        "    } catch (...) {\n"
        '        PyErr_SetString(PyExc_RuntimeError, "unhandled kernel runner error");\n'
        "        return nullptr;\n"
        "    }\n"
        "}\n"
        "\n"
        "static PyMethodDef tp_runner_def = {\n"
        '    "runner", (PyCFunction)(void (*)(void))tp_runner,\n'
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
    direct_section = (
        "\n"
        'extern "C" PyObject* tp_direct(const void* const* ins) {\n'
        "    try {\n"
        f"        const float* in[{input_count}];\n"
        f"        for (int i = 0; i < {input_count}; ++i)\n"
        "            in[i] = static_cast<const float*>(ins[i]);\n"
        f"{alloc}"
        "    } catch (const std::exception& e) {\n"
        "        PyErr_SetString(PyExc_RuntimeError, e.what());\n"
        "        return nullptr;\n"
        "    } catch (...) {\n"
        '        PyErr_SetString(PyExc_RuntimeError, "unhandled kernel error");\n'
        "        return nullptr;\n"
        "    }\n"
        "}\n"
    )

    return (
        f"{_prologue()}"
        "typedef struct TP_Ctx {\n"
        f"{ctx_fields}"
        "} TP_Ctx;\n"
        "\n"
        "static void tp_body(void* ctxp, long long b, long long e) {\n"
        "    const TP_Ctx* c = (const TP_Ctx*)ctxp;\n"
        f"{ctx_loads}"
        "    const long W = V::size();\n"
        "    (void)b; (void)e;\n"
        f"{const_decls}\n"
        f"{splat_decls}\n"
        f"{body}\n"
        "}\n"
        "\n"
        f'extern "C" void {entry}({input_params}, float* __restrict__ out) {{\n'
        f"{entry_body}\n"
        "}\n"
        f"{runner_section}"
        f"{direct_section}"
    )


def _digest_key(*parts: Any) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


def _kill_switch() -> bool:
    return os.environ.get("TP_STAX_CPU_NATIVE", "") == "0"


def build_cpu_reduction_kernel(
    instructions: list[tuple[str, int, int, int]],
    constants: list[float],
    input_count: int,
    output_ref: int,
    spec: ReduceSpec,
    *,
    in_shape: tuple[int, ...],
    device: Any,
    input_shapes: tuple[tuple[int, ...], ...],
    input_strides: tuple[tuple[int, ...], ...],
) -> Optional[tuple[Callable[[list[Any]], Any], int, tuple[int, ...]]]:
    """Compile one fused reduction; return ``(runner, direct, out_shape)``.

    ``runner`` takes the input tensor list and returns the reduced tensor;
    ``direct`` is the address of the pointer-level entry used by the C
    steady-state trampoline (0 when absent).  Returns ``None`` whenever the
    path is unavailable, so the caller keeps its existing route.
    """

    if _kill_switch():
        return None
    plan = plan_reduction(in_shape, spec)
    if plan is None:
        return None
    paths = package_paths()
    if paths is None:
        return None
    compiler = get_cpp_compiler()
    if not compiler:
        return None
    isa: VecISA = pick_vec_isa(paths)
    if not isa:
        return None
    version_info = get_compiler_version_info(compiler)

    device_ordinals = {"cpu": 0, "cuda": 1}
    try:
        out_device = (
            device_ordinals[str(device.type)],
            int(device.index) if device.index is not None else -1,
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return None

    layout_key = (
        tuple(tuple(int(d) for d in s) for s in input_shapes),
        tuple(tuple(int(s) for s in st) for st in input_strides),
    )
    entry = "tp_reduce_" + _digest_key(
        tuple(instructions),
        tuple(constants),
        input_count,
        output_ref,
        (spec.op, spec.dims, spec.keepdim),
        tuple(int(dim) for dim in in_shape),
        layout_key,
        out_device,
        isa.name,
        version_info,
        _pool_threads(),
    )

    try:
        source = render_reduction_source(
            instructions,
            constants,
            input_count,
            output_ref,
            spec,
            plan,
            entry,
            out_device=out_device,
            input_shapes=input_shapes,
            input_strides=input_strides,
            in_shape=tuple(int(dim) for dim in in_shape),
            lane_count=isa.nelements(),
        )
    except _ProgramError:
        return None

    lib = compile_translation_unit(
        source,
        entry,
        isa=isa,
        paths=paths,
        compiler=compiler,
        version_info=version_info,
        pinned=True,
        bind_tag="reduce-v1",
    )
    if lib is None:
        return None
    if getattr(lib, entry, None) is None:
        return None

    make_runner = getattr(lib, "tp_make_runner", None)
    if make_runner is None:
        return None
    make_runner.restype = ctypes.py_object
    make_runner.argtypes = []
    try:
        runner = make_runner()
    except Exception:
        return None

    direct_addr = 0
    direct_fn = getattr(lib, "tp_direct", None)
    if direct_fn is not None:
        try:
            direct_addr = ctypes.cast(direct_fn, ctypes.c_void_p).value or 0
        except (ValueError, TypeError, OSError):
            direct_addr = 0
    return runner, direct_addr, plan.out_shape
