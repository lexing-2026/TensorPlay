from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import MutableMapping
from typing import Any

import sympy

__all__ = ["calculate_value", "infer_symbol_values", "solve_equation", "update_equation"]

_SQUARE = re.compile(r"\[([^]]+)\]")
_PAREN = re.compile(r"\((.*?)\)")
_SYMBOL = re.compile(r"s\d+")


def solve_equation(left_expression: str | Any, right_expression: str | Any) -> tuple[str, int]:
    equation = sympy.sympify(f"{left_expression} - ({right_expression})")
    names = _SYMBOL.findall(str(equation))
    if not names:
        raise ValueError(f"no symbolic variable in {equation}")
    name = names[0]
    symbol = sympy.Symbol(name)
    solutions = sympy.solve(equation, symbol)
    if not solutions:
        raise ValueError(f"equation has no solution: {equation}")
    value = solutions[0]
    if value.free_symbols:
        raise ValueError(f"equation is underdetermined: {equation}")
    return name, int(value)


def calculate_value(
    left_expression: str | Any,
    right_expression: str | Any,
    symints: list[Any],
    symbol_idx_dict: dict[str, int],
) -> None:
    name, value = solve_equation(left_expression, right_expression)
    index = symbol_idx_dict[name]
    symints[index] = sympy.sympify(str(symints[index])).subs(sympy.Symbol(name), value)


def update_equation(
    symints: list[Any],
    init_symints: list[Any],
    padding_constraints: MutableMapping[Any, list[sympy.Expr | int]],
    init_eq: sympy.Expr,
    new_mod_num: int,
    var: Any,
    idx: int,
) -> None:
    values = padding_constraints[var]
    values.append(new_mod_num)
    modulus = 1
    for item in values[1:]:
        modulus = int(sympy.ilcm(modulus, int(item)))
    expression = modulus * sympy.sympify(str(init_symints[idx]))
    constants = [item for item in sympy.sympify(init_eq).args if getattr(item, "is_number", False)]
    if constants:
        expression -= int(constants[0]) % modulus
    symints[idx] = expression


def infer_symbol_values(
    symints: list[Any],
    init_symints: list[Any],
    symbol_idx_dict: dict[str, int],
    padding_constraints: MutableMapping[Any, list[sympy.Expr | int]],
    constraint: str,
) -> None:
    if "non-singleton" in constraint:
        expressions = _PAREN.findall(constraint)
        if len(expressions) >= 2:
            calculate_value(expressions[0], expressions[1], symints, symbol_idx_dict)
        return
    if "first two dimensions of batch2 tensor to be" in constraint:
        matches = _SQUARE.findall(constraint)
        if len(matches) >= 2:
            calculate_value(
                matches[0].split(",")[1].strip(),
                matches[1].split(",")[1].strip(),
                symints,
                symbol_idx_dict,
            )
        return
    if "same reduction dim" in constraint:
        matches = _SQUARE.findall(constraint)
        if len(matches) >= 2:
            calculate_value(
                matches[0].split(",")[-1].strip(),
                matches[1].split(",")[0].strip(),
                symints,
                symbol_idx_dict,
            )
        return
    if "Split sizes add up to" in constraint:
        left = re.search(r"to\s+(.*?)\s+but", constraint)
        right = re.search(r"of\s+(.*?)$", constraint)
        if left and right:
            calculate_value(left.group(1), right.group(1), symints, symbol_idx_dict)
        return
    if "is invalid for input of size" not in constraint:
        return
    matches = _SQUARE.findall(constraint)
    if not matches:
        return
    dimensions = [sympy.sympify(value.strip()) for value in matches[0].split(",")]
    total = sympy.sympify(constraint.split("size", 1)[1].strip())
    known = sympy.Integer(1)
    unknown = []
    for dimension in dimensions:
        if dimension == -1:
            continue
        if dimension.is_number:
            known *= dimension
        else:
            unknown.append(dimension)
    equation = sympy.cancel(total / known)
    variables = list(equation.free_symbols)
    if len(variables) != 1:
        return
    variable = variables[0]
    index = symbol_idx_dict.get(str(variable))
    if index is None:
        return
    values = padding_constraints.setdefault(variable, [])
    if not values:
        values.append(equation)
    update_equation(symints, init_symints, padding_constraints, values[0], int(known), variable, index)
