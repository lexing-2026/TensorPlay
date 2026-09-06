"""Runtime C++ code generation for row-staged CPU fusion.

Where :mod:`cpp_reduction` compiles *one* reduction at the tail of a
pointwise region, this module compiles a region whose reductions sit in the
**middle**: values reduced along the trailing axis flow back into elementwise
work over that same axis.  That is the shape of the normalizations that
dominate transformer code::

    m = x.amax(dim=-1, keepdim=True)
    e = (x - m).exp()
    y = e / e.sum(dim=-1, keepdim=True)

The generated kernel walks output rows.  Within a row it runs a sequence of
*stages*: each reduction is a vectorized pass over the row that ends in one
scalar, and every later stage may use that scalar as a broadcast.  The final
stage is either another per-row value or a full pass that stores the row.

The payoff is the memory schedule.  Splitting the example above into separate
reduction and normalization kernels streams ``x`` from memory once per kernel;
here every stage of a row runs while the row is still in the nearest cache, so
the region touches main memory once no matter how many stages it has.  The
expression work each stage repeats is recomputation out of cache, which is far
cheaper than the traffic it replaces.

Numerics and worksharing come from the same places as the tail-reduction
kernel: four independent accumulator groups, cascade summation for float sums,
NaN-propagating order relations, masked partial-vector tails, and row-parallel
dispatch through the shared worksharing bridge.  Rows are independent, so no
stage ever needs a cross-thread combine.

The module degrades gracefully: on an unsupported region, a missing
toolchain, or any build/load failure it returns ``None`` and the caller keeps
its existing route.  ``TP_STAX_CPU_NATIVE=0`` disables it.
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
    _pool_threads,
    analyze_input_modes,
    compile_translation_unit,
    emit_value_program,
)
from .cpp_reduction import (
    _ILP,
    _MAX_LANES,
    _MIN_CHUNK,
    _cascade_flush,
    _cascade_setup,
    _cascade_tick,
    _combine,
    _identity,
    _prologue,
    _finalize_scalar,
    _scombine,
)

# Pointwise expressions on per-row values are evaluated on broadcast vectors
# rather than scalars, so a row value passes through exactly the same
# operation the elementwise path would apply to it.
_ROW_BINARY = {
    "add": "({a} + {b})",
    "sub": "({a} - {b})",
    "mul": "({a} * {b})",
    "div": "({a} / {b})",
    "pow": "{a}.pow({b})",
    "minimum": "tensorplay::vec::minimum({a}, {b})",
    "maximum": "tensorplay::vec::maximum({a}, {b})",
    "clamp_min": "tensorplay::vec::maximum({a}, {b})",
    "clamp_max": "tensorplay::vec::minimum({a}, {b})",
}
_ROW_UNARY = {
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
    "rsqrt": "{a}.rsqrt()",
    "exp2": "{a}.exp2()",
    "erf": "{a}.erf()",
}

ROW_OPS = frozenset(_ROW_BINARY) | frozenset(_ROW_UNARY)


@dataclass(frozen=True)
class RowStep:
    """One stage of a row: a reduction, or an operation on row values."""

    kind: str  # "reduce" | "rowop"
    slot: int
    op: str
    # reduce: the elementwise program feeding the accumulator
    instructions: tuple[tuple[str, int, int, int], ...] = ()
    output_ref: int = 0
    # rowop: operand references in the shared ref space
    lhs: int = 0
    rhs: int = -1


@dataclass(frozen=True)
class RowFusion:
    """A region reduced to row stages plus a final per-row or per-element value.

    References follow the shared program encoding, extended so the first
    ``input_count`` slots are tensor inputs and the next ``row_slots`` are the
    per-row values produced by the stages.
    """

    input_count: int
    row_slots: int
    constants: tuple[float, ...]
    steps: tuple[RowStep, ...]
    output_kind: str  # "elem" | "row"
    out_instructions: tuple[tuple[str, int, int, int], ...]
    out_ref: int
    reduce_extent: int
    rows: int
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]


def _row_expr(op: str, lhs: str, rhs: str) -> str:
    if op in _ROW_BINARY:
        return _ROW_BINARY[op].format(a=lhs, b=rhs)
    if op in _ROW_UNARY:
        return _ROW_UNARY[op].format(a=lhs)
    raise _ProgramError(f"unsupported row operation: {op}")


def _ref_name(ref: int, fusion: RowFusion) -> str:
    """Name a reference that a row-level expression can read."""

    if ref < 0:
        value = fusion.constants[-ref - 1]
        text = f"{value:.17g}"
        if not any(ch in text for ch in ".eE"):
            text += ".0"
        return f"V({text}f)"
    if ref < fusion.input_count:
        # A tensor input used at row level must be a broadcast scalar; the
        # planner only admits those.
        return f"h{ref}"
    return f"s{ref - fusion.input_count}"


def _elem_modes(
    fusion: RowFusion,
    input_modes: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...]:
    """Extend the tensor addressing modes with the per-row value slots."""

    return tuple(input_modes) + tuple(
        ("rowval", slot) for slot in range(fusion.row_slots)
    )


def _emit_reduce_stage(
    step: RowStep,
    fusion: RowFusion,
    modes: tuple[tuple[str, int], ...],
    used: set[int],
    indent: str,
) -> list[str]:
    """One vectorized pass over the row, ending in a broadcast row value."""

    extent = fusion.reduce_extent
    op = step.op
    total_inputs = fusion.input_count + fusion.row_slots
    lines: list[str] = []
    lines.append(f"{indent}{{")
    body = indent + "    "
    for group in range(_ILP):
        lines.append(f"{body}V a{group} = V({_identity(op)});")
    lines.extend(
        _cascade_setup(op, f"({extent}L / ({_ILP}L * W))", body, _ILP)
    )

    def program(target_indent: str, offset: str, count: str, suffix: str):
        return emit_value_program(
            list(step.instructions),
            list(fusion.constants),
            total_inputs,
            step.output_ref,
            used,
            indent=target_indent,
            offset=offset,
            count_expr=count,
            suffix=suffix,
            input_modes=modes,
        )

    lines.append(f"{body}long i = 0;")
    lines.append(f"{body}for (; i + {_ILP}L * W <= {extent}L; i += {_ILP}L * W) {{")
    lane = body + "    "
    results = []
    for group in range(_ILP):
        offset = "base + i" if group == 0 else f"base + i + {group}L * W"
        group_lines, result = program(lane, offset, "W", f"_r{step.slot}_{group}")
        lines.extend(group_lines)
        results.append(result)
    for group, result in enumerate(results):
        lines.append(f"{lane}a{group} = {_combine(op, f'a{group}', result)};")
    lines.extend(_cascade_tick(op, lane, _ILP))
    lines.append(f"{body}}}")

    lines.append(f"{body}for (; i + W <= {extent}L; i += W) {{")
    step_lines, result = program(lane, "base + i", "W", f"_r{step.slot}_s")
    lines.extend(step_lines)
    lines.append(f"{lane}a0 = {_combine(op, 'a0', result)};")
    lines.append(f"{body}}}")

    lines.append(f"{body}if (i < {extent}L) {{")
    lines.append(f"{lane}const long count = {extent}L - i;")
    tail_lines, result = program(lane, "base + i", "count", f"_r{step.slot}_p")
    lines.extend(tail_lines)
    lines.append(f"{lane}a0 = V::set(a0, {_combine(op, 'a0', result)}, count);")
    lines.append(f"{body}}}")

    lines.extend(_cascade_flush(op, body, _ILP))
    lines.append(f"{body}a0 = {_combine(op, 'a0', 'a1')};")
    lines.append(f"{body}a2 = {_combine(op, 'a2', 'a3')};")
    lines.append(f"{body}a0 = {_combine(op, 'a0', 'a2')};")
    lines.append(f"{body}float lanes_[{_MAX_LANES}];")
    lines.append(f"{body}a0.store(lanes_);")
    lines.append(f"{body}float fold_ = lanes_[0];")
    lines.append(f"{body}for (long k_ = 1; k_ < W; ++k_)")
    lines.append(f"{body}    fold_ = {_scombine(op, 'fold_', 'lanes_[k_]')};")
    lines.append(f"{body}s{step.slot} = V({_finalize_scalar(op, 'fold_', extent)});")
    lines.append(f"{indent}}}")
    return lines


def _emit_output_stage(
    fusion: RowFusion,
    modes: tuple[tuple[str, int], ...],
    used: set[int],
    indent: str,
) -> list[str]:
    """Store the row: a full elementwise pass, or the single row value."""

    total_inputs = fusion.input_count + fusion.row_slots
    if fusion.output_kind == "row":
        return [
            f"{indent}float out_lanes_[{_MAX_LANES}];",
            f"{indent}{_ref_name(fusion.out_ref, fusion)}.store(out_lanes_);",
            f"{indent}out[row] = out_lanes_[0];",
        ]

    extent = fusion.reduce_extent
    lines: list[str] = []
    lines.append(f"{indent}float* __restrict__ orow = out + base;")
    lines.append(f"{indent}long q = 0;")

    def program(target_indent: str, offset: str, count: str, suffix: str):
        return emit_value_program(
            list(fusion.out_instructions),
            list(fusion.constants),
            total_inputs,
            fusion.out_ref,
            used,
            indent=target_indent,
            offset=offset,
            count_expr=count,
            suffix=suffix,
            input_modes=modes,
        )

    lines.append(f"{indent}for (; q + {_ILP}L * W <= {extent}L; q += {_ILP}L * W) {{")
    lane = indent + "    "
    # Every group is evaluated before any of them stores, so the four
    # dependency chains overlap instead of serializing on the store port.
    stores: list[tuple[str, str]] = []
    for group in range(_ILP):
        offset = "base + q" if group == 0 else f"base + q + {group}L * W"
        group_lines, result = program(lane, offset, "W", f"_o{group}")
        lines.extend(group_lines)
        target = "orow + q" if group == 0 else f"orow + q + {group}L * W"
        stores.append((result, target))
    for result, target in stores:
        lines.append(f"{lane}{result}.store({target}, W);")
    lines.append(f"{indent}}}")
    lines.append(f"{indent}for (; q + W <= {extent}L; q += W) {{")
    step_lines, result = program(lane, "base + q", "W", "_os")
    lines.extend(step_lines)
    lines.append(f"{lane}{result}.store(orow + q, W);")
    lines.append(f"{indent}}}")
    lines.append(f"{indent}if (q < {extent}L) {{")
    lines.append(f"{lane}const long count = {extent}L - q;")
    tail_lines, result = program(lane, "base + q", "count", "_op")
    lines.extend(tail_lines)
    lines.append(f"{lane}{result}.store(orow + q, count);")
    lines.append(f"{indent}}}")
    return lines


def render_row_fusion_source(
    fusion: RowFusion,
    entry: str,
    *,
    out_device: tuple[int, int],
    input_shapes: tuple[tuple[int, ...], ...],
    input_strides: tuple[tuple[int, ...], ...],
) -> str:
    """Render the full translation unit for one row-staged region."""

    # Row-broadcast inputs are addressed by the position *within* the row,
    # which the row loop bounds by the row extent and masks at its tail, so a
    # lane can never straddle the boundary the flat pointwise nest has to
    # guard against.  Classifying at one lane keeps that guard out of the way;
    # the width equality below is what makes the address ``q`` on its own.
    tensor_modes = analyze_input_modes(
        input_shapes, input_strides, fusion.in_shape, 1
    )
    if tensor_modes is None:
        raise _ProgramError("unsupported input layout")
    for mode, width in tensor_modes:
        if mode == "colmod" and width != fusion.reduce_extent:
            raise _ProgramError("row-broadcast width is not the row extent")
    modes = _elem_modes(fusion, tensor_modes)
    total_inputs = fusion.input_count + fusion.row_slots

    # Validate every program against the shared encoding before emitting.
    used_per_step: list[set[int]] = []
    for step in fusion.steps:
        if step.kind == "reduce":
            used_per_step.append(
                _analyze_instructions(
                    list(step.instructions),
                    list(fusion.constants),
                    total_inputs,
                    step.output_ref,
                    allow_empty=True,
                )
            )
        else:
            used_per_step.append(set())
    out_used: set[int] = set()
    if fusion.output_kind == "elem":
        out_used = _analyze_instructions(
            list(fusion.out_instructions),
            list(fusion.constants),
            total_inputs,
            fusion.out_ref,
            allow_empty=True,
        )

    # A tensor read at row level has no lane index to address with, so it has
    # to be a broadcast scalar.
    row_level: set[int] = set()
    for step in fusion.steps:
        if step.kind != "rowop":
            continue
        operands = (step.lhs,) if step.op in _ROW_UNARY else (step.lhs, step.rhs)
        for ref in operands:
            if 0 <= ref < fusion.input_count:
                if tensor_modes[ref][0] != "splat":
                    raise _ProgramError("row expression reads a shaped tensor")
                row_level.add(ref)

    # Broadcast scalars are hoisted once for the whole kernel; a row value is
    # never hoisted because it changes every row.
    hoisted = sorted(
        row_level
        | {
            ref
            for refs in (*used_per_step, out_used)
            for ref in refs
            if ref < fusion.input_count and tensor_modes[ref][0] == "splat"
        }
    )
    splat_decls = "\n".join(
        f"    const V h{ref} = V(in{ref}[0]);" for ref in hoisted
    )

    body_lines: list[str] = [
        "    float* __restrict__ out = c->out;",
        "    for (long row = b; row < e; ++row) {",
        f"        const long base = row * {fusion.reduce_extent}L;",
    ]
    for slot in range(fusion.row_slots):
        body_lines.append(f"        V s{slot} = V(0.0f);")
    for index, step in enumerate(fusion.steps):
        if step.kind == "reduce":
            body_lines.extend(
                _emit_reduce_stage(
                    step, fusion, modes, used_per_step[index], "        "
                )
            )
        else:
            lhs = _ref_name(step.lhs, fusion)
            rhs = "" if step.op in _ROW_UNARY else _ref_name(step.rhs, fusion)
            body_lines.append(
                f"        s{step.slot} = {_row_expr(step.op, lhs, rhs)};"
            )
    body_lines.extend(_emit_output_stage(fusion, modes, out_used, "        "))
    body_lines.append("    }")
    body = "\n".join(body_lines)

    threads = _pool_threads()
    work = fusion.rows * fusion.reduce_extent
    input_params = ", ".join(
        f"const float* __restrict__ in{i}" for i in range(fusion.input_count)
    )
    ctx_fields = "".join(
        f"    const float* in{i};\n" for i in range(fusion.input_count)
    )
    ctx_fields += "    float* out;\n"
    ctx_loads = "".join(
        f"    const float* __restrict__ in{i} = c->in{i};\n"
        for i in range(fusion.input_count)
    )
    ctx_init = (
        ", ".join(f"in{i}" for i in range(fusion.input_count)) + ", out"
    )
    if threads > 1 and work >= _MIN_CHUNK * threads and fusion.rows > 1:
        grain = max(1, (_MIN_CHUNK + fusion.reduce_extent - 1) // fusion.reduce_extent)
        dispatch = (
            f"    tp_parallel_for_c(0, {fusion.rows}L, {grain}L, tp_body, &ctx);"
        )
    else:
        dispatch = f"    tp_body(&ctx, 0, {fusion.rows}L);"

    dev_ordinal, dev_index = out_device
    dev_name = {0: "CPU", 1: "CUDA"}[dev_ordinal]
    shape_init = ", ".join(f"{int(dim)}LL" for dim in fusion.out_shape)
    call_args = ", ".join(f"in[{i}]" for i in range(fusion.input_count))
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
        f"            PyList_GET_SIZE(inputs) != {fusion.input_count})\n"
        '            throw std::runtime_error("runner input list mismatch");\n'
        f"        const float* in[{fusion.input_count}];\n"
        f"        for (long i = 0; i < {fusion.input_count}; ++i) {{\n"
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
        f"        const float* in[{fusion.input_count}];\n"
        f"        for (int i = 0; i < {fusion.input_count}; ++i)\n"
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
        f"{splat_decls}\n"
        f"{body}\n"
        "}\n"
        "\n"
        f'extern "C" void {entry}({input_params}, float* __restrict__ out) {{\n'
        f"    TP_Ctx ctx{{{ctx_init}}};\n"
        f"{dispatch}\n"
        "}\n"
        f"{runner_section}"
        f"{direct_section}"
    )


def _kill_switch() -> bool:
    return os.environ.get("TP_STAX_CPU_NATIVE", "") == "0"


def build_cpu_row_fusion_kernel(
    fusion: RowFusion,
    *,
    device: Any,
    input_shapes: tuple[tuple[int, ...], ...],
    input_strides: tuple[tuple[int, ...], ...],
) -> Optional[tuple[Callable[[list[Any]], Any], int]]:
    """Compile one row-staged region; return ``(runner, direct)``."""

    if _kill_switch():
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

    entry = "tp_rowfuse_" + hashlib.sha256(
        repr(
            (
                fusion,
                tuple(tuple(int(d) for d in s) for s in input_shapes),
                tuple(tuple(int(s) for s in st) for st in input_strides),
                out_device,
                isa.name,
                version_info,
                _pool_threads(),
            )
        ).encode()
    ).hexdigest()[:16]

    try:
        source = render_row_fusion_source(
            fusion,
            entry,
            out_device=out_device,
            input_shapes=input_shapes,
            input_strides=input_strides,
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
        bind_tag="rowfuse-v1",
    )
    if lib is None or getattr(lib, entry, None) is None:
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
    return runner, direct_addr
