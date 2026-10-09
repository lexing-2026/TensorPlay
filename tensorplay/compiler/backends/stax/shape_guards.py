"""Validate the layout relations and constraints used by generated kernels."""

from __future__ import annotations

import math

import sympy
from sympy.printing.pycode import PythonCodePrinter


class _GuardPrinter(PythonCodePrinter):
    def _print_FloorDiv(self, expr):
        left, right = map(self._print, expr.args)
        return f"(({left}) // ({right}))"

    def _print_ModularIndexing(self, expr):
        value, divisor, modulus = map(self._print, expr.args)
        return f"((({value}) // ({divisor})) % ({modulus}))"


def build_capture_guard(module):
    """Check the decisions and fixed dimensions used during shape capture."""

    dims = module.meta.get("symbolic_dims")
    if dims is None:
        return None
    printer = _GuardPrinter()
    positions = {node.target: i for i, node in enumerate(module.graph.placeholders)}
    replacements = {}
    assignments = []
    checks = []
    for symbol in dims.symbols:
        name, axis, scale, offset = dims.source(symbol)
        replacement = sympy.Symbol(f"capture_{len(replacements)}")
        replacements[symbol] = replacement
        assignments.append(f"    {replacement} = (args[{positions[name]}].size()[{axis}] - {offset}) // {scale}")
    for guard in dims.guards:
        expression = guard.fact.xreplace(replacements)
        if expression.free_symbols - set(replacements.values()):
            raise NotImplementedError("shape decision depends on an unbound dimension")
        checks.append(f"({printer.doprint(expression)})")
    for name, axis, size in module.meta.get("static_shape_dims", ()):
        checks.append(f"args[{positions[name]}].size()[{axis}] == {size}")
    namespace = {"math": math}
    source = "\n".join(["def check(args):", *assignments, "    return " + (" and ".join(checks) or "True")])
    exec(compile(source, "<capture-guards>", "exec"), namespace)
    return namespace["check"]


def build_input_guard(graph):
    """Compile a check over input metadata, without retaining sample tensors."""

    printer = _GuardPrinter()
    bindings = {}
    checks = []
    layouts = []
    scalars = []
    for index, name in enumerate(graph.graph_input_names):
        value = graph.graph_inputs.get(name)
        if isinstance(value, sympy.Symbol):
            bindings.setdefault(value, f"args[{index}]")
        if isinstance(value, sympy.Basic):
            scalars.append((index, value))
        if not hasattr(value, "get_size"):
            continue
        sizes, strides = value.get_size(), value.get_stride()
        layouts.append((index, sizes, strides))
        checks.append(f"len(args[{index}].shape) == {len(sizes)}")
        for kind, expressions in (("size", sizes), ("stride", strides)):
            for dim, expr in enumerate(expressions):
                if isinstance(expr, sympy.Symbol):
                    bindings.setdefault(expr, f"args[{index}].{kind}()[{dim}]")
        offset = graph.graph_input_storage_offsets[name]
        checks.append(f"args[{index}].storage_offset() == {printer.doprint(offset)}")

    def rendered(expr):
        missing = sympy.sympify(expr).free_symbols - bindings.keys()
        if missing:
            raise NotImplementedError(f"shape constraint has unbound symbols: {missing}")
        return printer.doprint(expr)

    for index, sizes, strides in layouts:
        for kind, expressions in (("size", sizes), ("stride", strides)):
            for dim, expr in enumerate(expressions):
                checks.append(f"args[{index}].{kind}()[{dim}] == ({rendered(expr)})")
                if kind == "size" and sympy.sympify(expr).free_symbols:
                    checks.append(f"args[{index}].size()[{dim}] >= 2")
    for index, expression in scalars:
        checks.append(f"args[{index}] == ({rendered(expression)})")
        if expression.free_symbols:
            checks.append(f"args[{index}] >= 2")
    for guard in graph.shape_env.guards:
        checks.append(f"({rendered(guard.expr)})")
    for assertions in graph.shape_env.deferred_runtime_asserts.values():
        for assertion in assertions:
            checks.append(f"({rendered(assertion.expr)})")
    assignments = [f"    {symbol} = {source}" for symbol, source in bindings.items()]
    body = " and ".join(checks) or "True"
    source = "\n".join(["def check(args):", *assignments, f"    return {body}"])
    namespace = {"math": math}
    exec(compile(source, "<shape-guards>", "exec"), namespace)
    check = namespace["check"]
    capture_guard = build_capture_guard(graph.module)

    def validate(args):
        try:
            return bool(check(args)) and (capture_guard is None or bool(capture_guard(args)))
        except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError):
            return False

    validate.source = source
    return validate
