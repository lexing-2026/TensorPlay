from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from ..pointwise import _TRITON_OPCODES


_MAX_INPUTS = 32
_MAX_OUTPUTS = 32
_MAX_INSTRUCTIONS = 128
_BINARY_OPS = {
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "ne",
    "minimum",
    "maximum",
    "clamp_min",
    "clamp_max",
}
_UNARY_OPS = {
    "neg",
    "pos",
    "abs",
    "sin",
    "cos",
    "exp",
    "log",
    "sigmoid",
    "sqrt",
    "square",
    "tanh",
    "relu",
    "relu_grad",
    "abs_grad",
    "rsqrt",
    "exp2",
    "erf",
}
_OPCODE_TO_NAME = {code: name for name, code in _TRITON_OPCODES.items()}


def _binary_expression(name: str, lhs: str, rhs: str) -> str:
    if name == "add":
        return f"(({lhs}) + ({rhs}))"
    if name == "sub":
        return f"(({lhs}) - ({rhs}))"
    if name == "mul":
        return f"(({lhs}) * ({rhs}))"
    if name == "div":
        return f"(({lhs}) / ({rhs}))"
    if name == "pow":
        return f"::pow({lhs}, {rhs})"
    if name == "lt":
        return f"(({lhs}) < ({rhs}) ? T(1) : T(0))"
    if name == "le":
        return f"(({lhs}) <= ({rhs}) ? T(1) : T(0))"
    if name == "gt":
        return f"(({lhs}) > ({rhs}) ? T(1) : T(0))"
    if name == "ge":
        return f"(({lhs}) >= ({rhs}) ? T(1) : T(0))"
    if name == "eq":
        return f"(({lhs}) == ({rhs}) ? T(1) : T(0))"
    if name == "ne":
        return f"(({lhs}) != ({rhs}) ? T(1) : T(0))"
    if name == "minimum":
        return f"((({lhs}) != ({lhs})) ? ({lhs}) : ((({rhs}) != ({rhs})) ? ({rhs}) : ::fmin({lhs}, {rhs})))"
    if name == "maximum":
        return f"((({lhs}) != ({lhs})) ? ({lhs}) : ((({rhs}) != ({rhs})) ? ({rhs}) : ::fmax({lhs}, {rhs})))"
    if name == "clamp_min":
        return (
            f"((({lhs}) != ({lhs})) ? ({lhs}) : "
            f"((({lhs}) == ({rhs})) ? ({lhs}) : (({lhs}) < ({rhs}) ? ({rhs}) : ({lhs}))))"
        )
    if name == "clamp_max":
        return (
            f"((({lhs}) != ({lhs})) ? ({lhs}) : "
            f"((({lhs}) == ({rhs})) ? ({lhs}) : (({lhs}) > ({rhs}) ? ({rhs}) : ({lhs}))))"
        )
    raise ValueError(f"unsupported binary operation: {name}")


def _unary_expression(name: str, value: str) -> str:
    if name == "neg":
        return f"(-({value}))"
    if name == "pos":
        return f"({value})"
    if name == "abs":
        return f"::fabs({value})"
    if name == "sin":
        return f"::sin({value})"
    if name == "cos":
        return f"::cos({value})"
    if name == "exp":
        return f"::exp({value})"
    if name == "log":
        return f"::log({value})"
    if name == "sigmoid":
        return f"(T(1) / (T(1) + ::exp(-({value}))))"
    if name == "sqrt":
        return f"::sqrt({value})"
    if name == "square":
        return f"(({value}) * ({value}))"
    if name == "tanh":
        return f"::tanh({value})"
    if name == "relu":
        return (
            f"((({value}) != ({value})) ? ({value}) : "
            f"((({value}) == T(0)) ? ({value}) : (({value}) < T(0) ? T(0) : ({value}))))"
        )
    if name == "relu_grad":
        return f"(({value}) > T(0) ? T(1) : T(0))"
    if name == "abs_grad":
        return f"((({value}) > T(0) ? T(1) : T(0)) - (({value}) < T(0) ? T(1) : T(0)))"
    if name == "rsqrt":
        return f"(T(1) / ::sqrt({value}))"
    if name == "exp2":
        return f"::exp2({value})"
    if name == "erf":
        return f"::erf({value})"
    raise ValueError(f"unsupported unary operation: {name}")


def generate_cuda_source(
    program: Sequence[int],
    constants: Sequence[float],
    output_refs: Sequence[int],
    input_count: int,
) -> str:
    if not 1 <= input_count <= _MAX_INPUTS:
        raise ValueError("native CUDA pointwise requires one to thirty-two tensor inputs")
    if not 1 <= len(output_refs) <= _MAX_OUTPUTS:
        raise ValueError("native CUDA pointwise requires one to thirty-two outputs")
    if not program or len(program) % 3:
        raise ValueError("native CUDA pointwise program must contain instruction triples")
    instruction_count = len(program) // 3
    if instruction_count > _MAX_INSTRUCTIONS:
        raise ValueError("native CUDA pointwise program is too large")
    if any(not math.isfinite(float(value)) for value in constants):
        raise ValueError("native CUDA pointwise constants must be finite")

    input_names = [f"x{index}" for index in range(input_count)]
    values: dict[int, str] = {index: name for index, name in enumerate(input_names)}

    def resolve(ref: int, current: int) -> str:
        if ref < 0:
            constant_index = -ref - 1
            if constant_index >= len(constants):
                raise ValueError("native CUDA pointwise constant reference is invalid")
            return f"T({float(constants[constant_index])!r})"
        if ref < input_count:
            return input_names[ref]
        temporary = ref - input_count
        if temporary < 0 or temporary >= current:
            raise ValueError("native CUDA pointwise value reference is invalid")
        return f"t{temporary}"

    lines: list[str] = []
    pending_where: tuple[str, str] | None = None
    for instruction in range(instruction_count):
        offset = instruction * 3
        opcode = int(program[offset])
        name = _OPCODE_TO_NAME.get(opcode)
        if name is None:
            raise ValueError(f"native CUDA pointwise opcode is invalid: {opcode}")
        lhs_ref = int(program[offset + 1])
        rhs_ref = int(program[offset + 2])
        temporary = f"t{instruction}"
        if name == "where":
            if pending_where is not None:
                raise ValueError("native CUDA pointwise where instruction is unpaired")
            pending_where = (
                resolve(lhs_ref, instruction),
                resolve(rhs_ref, instruction),
            )
            expression = "T(0)"
        elif name == "where_rest":
            if pending_where is None:
                raise ValueError("native CUDA pointwise where_rest has no condition")
            condition, then_value = pending_where
            else_value = resolve(rhs_ref, instruction)
            expression = f"(({condition}) != T(0) ? ({then_value}) : ({else_value}))"
            pending_where = None
        elif name in _BINARY_OPS:
            expression = _binary_expression(
                name,
                resolve(lhs_ref, instruction),
                resolve(rhs_ref, instruction),
            )
        elif name in _UNARY_OPS:
            expression = _unary_expression(name, resolve(lhs_ref, instruction))
        else:
            raise ValueError(f"native CUDA pointwise operation is unsupported: {name}")
        values[input_count + instruction] = temporary
        lines.append(f"  T {temporary} = {expression};")

    if pending_where is not None:
        raise ValueError("native CUDA pointwise where instruction is unpaired")
    resolved_outputs: list[str] = []
    for ref in output_refs:
        ref = int(ref)
        if ref < input_count:
            raise ValueError("native CUDA pointwise cannot return an input alias")
        temporary = ref - input_count
        if temporary < 0 or temporary >= instruction_count:
            raise ValueError("native CUDA pointwise output reference is invalid")
        resolved_outputs.append(values[input_count + temporary])

    parameters = ", ".join(f"T {name}" for name in input_names)
    if len(resolved_outputs) == 1:
        signature = f"template <typename T> T stax_program({parameters})"
        tail = f"  return {resolved_outputs[0]};"
    else:
        output_parameters = ", ".join(f"T& out{index}" for index in range(len(resolved_outputs)))
        signature = f"template <typename T> void stax_program({parameters}, {output_parameters})"
        tail = "\n".join(f"  out{index} = {value};" for index, value in enumerate(resolved_outputs))
    body = "\n".join((*lines, tail))
    return f"{signature} {{\n{body}\n}}"


def _direct_runner(
    function: Any,
    source: str,
    output_count: int,
    example_inputs: Sequence[Any] | None,
) -> Callable[[list[Any]], Any] | None:
    if example_inputs is None or not example_inputs:
        return None
    import tensorplay

    first = example_inputs[0]
    reference_shape = tuple(int(item) for item in first.shape)
    if not first.is_contiguous() or any(
        tuple(int(item) for item in value.shape) != reference_shape
        or value.dtype != first.dtype
        or value.device != first.device
        or not value.is_contiguous()
        for value in example_inputs[1:]
    ):
        return None
    native = tensorplay._C._cuda_jiterator_compile_and_launch_kernel
    kernel_name = function.kernel_name
    return_by_ref = function.return_by_ref

    if output_count == 1:

        def run_single_direct(inputs: list[Any]) -> Any:
            result = native(
                source,
                kernel_name,
                return_by_ref,
                output_count,
                tuple(inputs),
                {},
            )
            return result.view(reference_shape)

        return run_single_direct

    def run_multi_direct(inputs: list[Any]) -> tuple[Any, ...]:
        outputs = native(
            source,
            kernel_name,
            return_by_ref,
            output_count,
            tuple(inputs),
            {},
        )
        return tuple(output.view(reference_shape) for output in outputs)

    return run_multi_direct


def compile_program(
    program: Sequence[int],
    constants: Sequence[float],
    output_refs: Sequence[int],
    input_count: int,
    example_inputs: Sequence[Any] | None = None,
) -> Callable[[list[Any]], Any]:
    source = generate_cuda_source(program, constants, output_refs, input_count)
    from tensorplay.cuda import jiterator

    output_count = len(output_refs)
    if output_count == 1:
        function = jiterator._create_jiterator_fn(source)
        direct = _direct_runner(function, source, output_count, example_inputs)
        if direct is not None:
            return direct

        def run_single(inputs: list[Any]) -> Any:
            return function(*inputs)

        return run_single

    function = jiterator._create_multi_output_jiterator_fn(source, output_count)
    direct = _direct_runner(function, source, output_count, example_inputs)
    if direct is not None:
        return direct

    def run_multi(inputs: list[Any]) -> tuple[Any, ...]:
        return function(*inputs)

    return run_multi
