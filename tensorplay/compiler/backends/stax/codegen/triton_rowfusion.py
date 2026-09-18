"""Triton code generation for row-staged CUDA fusion.

This is the CUDA half of the row-staged plan :mod:`cpp_rowfusion` describes:
a region whose reductions sit in the *middle*, feeding elementwise work over
the axis they were reduced along.  The plan itself is device independent, so
both generators consume the same :class:`RowFusion`.

The kernel keeps its rows resident for the length of the region.  Every
stage -- each reduction, each value derived from one, and the final store --
reads registers instead of memory, so a region with any number of stages
loads its inputs once and writes its output once.  Splitting the same region
into a kernel per reduction streams the input again for every one of them and
materializes an intermediate between each pair.

One program takes a tile of rows -- as many as fill it, the full width of
each -- so a narrow row shares a program with its neighbours instead of
leaving most of one idle.  How many rows that is, and how many warps split
them, is measured on the shape itself rather than guessed.

The module degrades gracefully: an unsupported region, a missing Triton, or
any build failure returns ``None`` and the caller keeps its existing route.
"""

from __future__ import annotations

import hashlib
import linecache
from typing import Any, Callable, Optional

from .cpp_rowfusion import _ROW_BINARY, _ROW_UNARY, RowFusion

# Values one program holds at once.  The row is resident while the stages
# run, so this is the register budget: a narrow row shares the tile with its
# neighbours, and a row wider than the whole budget keeps its existing route
# rather than compiling to a kernel that would spill.
_TILE_BUDGET = 8192


def _triton_module():
    try:
        import triton
        import triton.language as tl
    except Exception:  # noqa: BLE001 - Triton is optional
        return None
    return triton, tl


def _emitter():
    """A codegen instance used only for its expression table."""

    from .triton import TritonProgramCodegen

    return TritonProgramCodegen([], [], (0,), 1)


def _encode(
    instructions: tuple[tuple[str, int, int, int], ...]
) -> list[int]:
    """Flatten instruction tuples into the shared opcode triples."""

    from .triton import _TRITON_OPCODES

    program: list[int] = []
    for name, lhs, rhs, _result in instructions:
        code = _TRITON_OPCODES.get(name)
        if code is None:
            raise ValueError(f"unsupported row-fusion operation: {name}")
        program.extend((code, lhs, rhs))
    return program


def _literal(value: float) -> str:
    text = f"{float(value):.17g}"
    if not any(ch in text for ch in ".eE"):
        text += ".0"
    return text


# Order reductions fold through an explicit combine so they carry NaN the
# way the reduction they stand for does; the built-in maximum drops it and
# would disagree with the same region run uncompiled.
_REDUCE_FOLD = {
    "sum": "tl.sum({value}, axis=1)",
    "mean": "tl.sum({value}, axis=1)",
    "max": "tl.reduce({value}, 1, _tp_max)",
    "amax": "tl.reduce({value}, 1, _tp_max)",
    "min": "tl.reduce({value}, 1, _tp_min)",
    "amin": "tl.reduce({value}, 1, _tp_min)",
}
_REDUCE_HELPER = {
    "max": "_tp_max",
    "amax": "_tp_max",
    "min": "_tp_min",
    "amin": "_tp_min",
}
_HELPER_SOURCE = {
    "_tp_max": (
        "@triton.jit\n"
        "def _tp_max(a, b):\n"
        "    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)\n"
    ),
    "_tp_min": (
        "@triton.jit\n"
        "def _tp_min(a, b):\n"
        "    return tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)\n"
    ),
}
_REDUCE_NEUTRAL = {
    "sum": "0.0",
    "mean": "0.0",
    "max": "float('-inf')",
    "amax": "float('-inf')",
    "min": "float('inf')",
    "amin": "float('inf')",
}
_REDUCE_COMBINE = {
    "sum": "{acc} + {value}",
    "mean": "{acc} + {value}",
    "max": "tl.maximum({acc}, {value})",
    "amax": "tl.maximum({acc}, {value})",
    "min": "tl.minimum({acc}, {value})",
    "amin": "tl.minimum({acc}, {value})",
}


def supported(fusion: RowFusion) -> bool:
    """Whether this plan has a Triton form."""

    if fusion.reduce_extent <= 0 or fusion.rows <= 0:
        return False
    if fusion.reduce_extent > _TILE_BUDGET:
        return False
    if any(step.op not in _REDUCE_FOLD for step in fusion.steps if step.kind == "reduce"):
        return False
    for step in fusion.steps:
        if step.kind == "rowop" and step.op not in _ROW_BINARY and step.op not in _ROW_UNARY:
            return False
    return True


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result <<= 1
    return result


def _row_expression(op: str, lhs: str, rhs: str) -> str:
    """One operation on values that are already single numbers per row."""

    from .triton import _TRITON_OPCODES

    emitter = _emitter()
    code = _TRITON_OPCODES.get(op)
    if code is None:
        raise ValueError(f"unsupported row operation: {op}")
    names = {"lhs": lhs, "rhs": rhs}
    return emitter._expression(
        code, 0, 1, lambda ref: names["lhs"] if ref == 0 else names["rhs"]
    )


class _Names:
    """Resolve the shared reference space to kernel-side registers."""

    def __init__(self, fusion: RowFusion) -> None:
        self.fusion = fusion
        self.base = fusion.input_count + fusion.row_slots + fusion.stage_slots
        self.staged: dict[int, str] = {}

    def resolver(self, prefix: str) -> Callable[[int], str]:
        fusion = self.fusion
        staged = self.staged

        def resolve(ref: int) -> str:
            if ref < 0:
                return _literal(fusion.constants[-ref - 1])
            if ref < fusion.input_count:
                return f"x{ref}"
            slot = ref - fusion.input_count
            if slot < fusion.row_slots:
                return f"s{slot}"
            slot -= fusion.row_slots
            if slot < fusion.stage_slots:
                name = staged.get(slot)
                if name is None:
                    raise ValueError("staged value read before it is produced")
                return name
            return f"{prefix}{ref - self.base}"

        return resolve


def _load_line(fusion: RowFusion, modes: tuple[tuple[str, int], ...], ref: int) -> str:
    """Bring one tensor value into registers."""

    mode, _width = modes[ref]
    if mode == "splat":
        return f"x{ref} = tl.load(in_ptr{ref})"
    if mode == "colmod":
        # Constant along the row axis: one row's worth is read and broadcast
        # over the tile rather than re-read for every row in it.
        return (
            f"x{ref} = tl.load(in_ptr{ref} + cols1, mask=cmask, "
            "other=0.0)[None, :]"
        )
    return f"x{ref} = tl.load(in_ptr{ref} + base + cols, mask=mask, other=0.0)"


def _program_inputs(
    fusion: RowFusion,
    instructions: tuple[tuple[str, int, int, int], ...],
    output_ref: int,
) -> set[int]:
    """Tensor references one program reads."""

    from .cpp import _analyze_instructions

    total = fusion.input_count + fusion.row_slots + fusion.stage_slots
    used = _analyze_instructions(
        list(instructions), list(fusion.constants), total, output_ref,
        allow_empty=True,
    )
    if not instructions and 0 <= output_ref < fusion.input_count:
        used.add(output_ref)
    return {ref for ref in used if ref < fusion.input_count}


def _program_body(
    program: list[int],
    resolve: Callable[[int], str],
    prefix: str,
    output_ref: int,
) -> tuple[list[str], str]:
    """Emit one program and name the register holding its result.

    A program with no instructions is a reduction reading one of its inputs
    straight through; the reference resolves to that input's register.
    """

    emitter = _emitter()
    lines = emitter._program_lines(program, resolve, prefix)
    return lines, resolve(output_ref)


def _tile_configs(block: int) -> list[tuple[int, int]]:
    """(rows per program, warps) pairs worth trying for this row width.

    A narrow row does not fill a program on its own, so several rows share
    one; a wide row fills it by itself.  The product is held near one tile's
    worth of values so the register file stays inside the budget.
    """

    rows_per_program = [
        count for count in (1, 2, 4, 8, 16) if count * block <= _TILE_BUDGET
    ] or [1]
    configs: list[tuple[int, int]] = []
    for count in rows_per_program[:4]:
        # One warp per 256 values of the tile is the shape that keeps every
        # lane busy without splitting the reduction tree too finely; the
        # wider neighbour covers the cases where it does not.  Six candidates
        # is enough to find the corner and cheap enough to measure.
        centre = min(max((count * block) // 256, 1), 8)
        for warps in dict.fromkeys((centre, min(centre * 2, 8))):
            configs.append((count, warps))
    return configs


def render_row_fusion_kernel(
    fusion: RowFusion,
    kernel_name: str,
    modes: tuple[tuple[str, int], ...],
    *,
    out_dtype: str = "tp.float32",
) -> str:
    """Render the whole translation unit for one row-staged region.

    The kernel walks a tile of rows: ``XBLOCK`` of them at a time, the full
    row width in each.  Every reduction folds along the row axis and comes
    back as one value per row, which the stages after it read as a broadcast
    -- so a stage never returns to memory for a value an earlier stage
    already produced.
    """

    names = _Names(fusion)
    helpers: set[str] = set()
    extent = fusion.reduce_extent
    block = _next_power_of_two(extent)
    body: list[str] = [
        "xoffset = tl.program_id(0) * XBLOCK",
        "rows = xoffset + tl.arange(0, XBLOCK)",
        f"rmask = rows < {fusion.rows}",
        "cols1 = tl.arange(0, BLOCK)",
        f"cmask = cols1 < {extent}",
        f"base = rows[:, None] * {extent}",
        "cols = cols1[None, :]",
        "mask = rmask[:, None] & cmask[None, :]",
    ]

    # A value is loaded where it is first needed rather than up front: the
    # whole tile lives in registers, so a load hoisted above the stage that
    # wants it holds a register set the reductions could otherwise use.
    loaded: set[int] = set()

    def load(refs: set[int]) -> None:
        for ref in sorted(refs - loaded):
            body.append(_load_line(fusion, modes, ref))
            loaded.add(ref)

    for index, step in enumerate(fusion.steps):
        if step.kind == "rowop":
            resolve = names.resolver(f"r{index}_")
            operands = (
                (step.lhs,) if step.op in _ROW_UNARY else (step.lhs, step.rhs)
            )
            load({ref for ref in operands if 0 <= ref < fusion.input_count})
            lhs = resolve(step.lhs)
            rhs = "" if step.op in _ROW_UNARY else resolve(step.rhs)
            body.append(f"s{step.slot} = {_row_expression(step.op, lhs, rhs)}")
            continue
        load(_program_inputs(fusion, step.instructions, step.output_ref))
        resolve = names.resolver(f"p{index}_")
        program = _encode(step.instructions)
        lines, result = _program_body(
            program, resolve, f"p{index}_", step.output_ref
        )
        body.extend(lines)
        neutral = _REDUCE_NEUTRAL[step.op]
        helper = _REDUCE_HELPER.get(step.op)
        if helper is not None:
            helpers.add(helper)
        fold = _REDUCE_FOLD[step.op].format(
            value=f"tl.where(mask, {result}, {neutral})"
        )
        body.append(f"s{step.slot} = ({fold})[:, None]")
        if step.op == "mean":
            body.append(f"s{step.slot} = s{step.slot} / {float(extent)!r}")
        if step.stage >= 0:
            names.staged[step.stage] = result

    if fusion.output_kind == "row":
        resolve = names.resolver("out_")
        body.append(
            f"tl.store(out_ptr + rows[:, None], {resolve(fusion.out_ref)}, "
            "mask=rmask[:, None])"
        )
    else:
        load(_program_inputs(fusion, fusion.out_instructions, fusion.out_ref))
        resolve = names.resolver("out_")
        program = _encode(fusion.out_instructions)
        lines, result = _program_body(program, resolve, "out_", fusion.out_ref)
        body.extend(lines)
        body.append(f"tl.store(out_ptr + base + cols, {result}, mask=mask)")

    signature = [
        *(f"in_ptr{index}" for index in range(fusion.input_count)),
        "out_ptr",
        "XBLOCK: tl.constexpr",
        "BLOCK: tl.constexpr",
    ]
    indented = "\n".join(f"    {line}" for line in body)
    shape = ", ".join(f"{int(size)}" for size in fusion.out_shape)
    if len(fusion.out_shape) == 1:
        shape += ","
    call_args = ", ".join(
        [*(f"inputs[{index}]" for index in range(fusion.input_count)), "out"]
    )
    configs = ", ".join(
        f"triton.Config({{'XBLOCK': {rows}}}, num_warps={warps}, num_stages=1)"
        for rows, warps in _tile_configs(block)
    )
    return (
        "import triton\n"
        "import triton.language as tl\n"
        "import triton.language.extra.cuda.libdevice as libdevice\n"
        "import tensorplay as tp\n"
        "\n"
        + "".join(_HELPER_SOURCE[name] for name in sorted(helpers))
        + "\n"
        "@triton.autotune(\n"
        f"    configs=[{configs}],\n"
        "    key=[],\n"
        ")\n"
        "@triton.jit\n"
        f"def {kernel_name}({', '.join(signature)}):\n"
        f"{indented}\n"
        "\n"
        f"_grid = lambda meta: (triton.cdiv({fusion.rows}, meta['XBLOCK']),)\n"
        "\n"
        "def kernel_launch(inputs):\n"
        f"    out = tp.empty(({shape}), dtype={out_dtype}, "
        "device=inputs[0].device)\n"
        f"    {kernel_name}[_grid]({call_args}, BLOCK={block})\n"
        "    return out\n"
    )


_launch_memo: dict[str, Any] = {}


def build_cuda_row_fusion_kernel(
    fusion: RowFusion,
    *,
    input_shapes: tuple[tuple[int, ...], ...],
    input_strides: tuple[tuple[int, ...], ...],
) -> Optional[Callable[[list[Any]], Any]]:
    """Compile one row-staged region for CUDA; return its launch callable."""

    if not supported(fusion):
        return None
    if _triton_module() is None:
        return None
    from .triton import runtime_available

    # The probe is what brings the runtime up: a kernel cannot launch until
    # the device context exists, and a machine where it cannot launch at all
    # keeps every other route.
    if not runtime_available():
        return None
    from .cpp import _ProgramError, analyze_input_modes

    modes = analyze_input_modes(
        input_shapes, input_strides, fusion.in_shape, 1
    )
    if modes is None:
        return None
    for mode, width in modes:
        if mode == "colmod" and width != fusion.reduce_extent:
            return None

    digest = hashlib.sha256(
        repr(
            (
                fusion,
                tuple(tuple(int(d) for d in s) for s in input_shapes),
                tuple(tuple(int(s) for s in st) for st in input_strides),
            )
        ).encode()
    ).hexdigest()[:16]
    cached = _launch_memo.get(digest)
    if cached is not None:
        return cached

    kernel_name = f"stax_triton_rowfuse_{digest}"
    try:
        source = render_row_fusion_kernel(fusion, kernel_name, modes)
    except (ValueError, KeyError, _ProgramError):
        return None

    try:
        from ..codecache import default_cache

        cache = default_cache("triton")
        key = cache.cache_key(source)
        if cache.load(key, ext="py") is None:
            cache.store(key, source.encode(), ext="py")
    except Exception:  # noqa: BLE001 - cache is best-effort
        pass

    fake_file = f"<tensorplay-stax-triton-rowfuse-{digest}>"
    linecache.cache[fake_file] = (
        len(source),
        None,
        source.splitlines(True),
        fake_file,
    )
    # The unit is exec'd as a real module: a kernel that folds through a
    # helper needs that helper resolvable by name and by source location,
    # which only a module with an identity provides.
    import sys
    import types

    module_name = f"tensorplay_stax_rowfuse_{digest}"
    module = types.ModuleType(module_name)
    module.__file__ = fake_file
    try:
        exec(compile(source, fake_file, "exec"), module.__dict__)
        sys.modules[module_name] = module
        launch = module.__dict__["kernel_launch"]
    except Exception:  # noqa: BLE001 - a build failure keeps the old route
        return None
    _launch_memo[digest] = launch
    return launch
