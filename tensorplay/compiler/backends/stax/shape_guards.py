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


def build_input_guard(graph):
    """Compile a check over input metadata, without retaining sample tensors."""

    printer = _GuardPrinter()
    bindings = {}
    checks = []
    layouts = []
    for index, name in enumerate(graph.graph_input_names):
        value = graph.graph_inputs.get(name)
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

    def validate(args):
        try:
            return bool(check(args))
        except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError):
            return False

    validate.source = source
    return validate
