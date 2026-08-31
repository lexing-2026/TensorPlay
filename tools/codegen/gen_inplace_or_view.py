"""Validate schema metadata used by mutation and alias handling."""

from __future__ import annotations

from .main import CodegenContext, register_generator


def _records(functions):
    return tuple(getattr(functions, "reference_functions", functions))


def validate_inplace_or_view(functions) -> list[str]:
    """Return invalid mutation or alias records without dropping any record."""
    problems: list[str] = []
    for function in _records(functions):
        name = str(function.func.name)
        if function.is_view_op:
            try:
                function.view_schema_kind
            except AssertionError as error:
                problems.append(f"{name}: {error}")
            if not any(argument.type.is_tensor_like() for argument in
                       function.func.arguments.flat_all):
                problems.append(f"{name}: a view must have a tensor argument")

        if function.func.kind().name == "inplace":
            if function.func.name.name.inplace is not True:
                problems.append(f"{name}: inplace schema has no inplace name")
            if function.func.arguments.self_arg is None:
                problems.append(f"{name}: inplace schema has no self argument")

    return problems


@register_generator("InplaceOrView")
def _gen_inplace_or_view(ctx: CodegenContext) -> None:
    problems = validate_inplace_or_view(ctx.funcs)
    if problems:
        raise ValueError(
            "mutation and alias schema validation failed:\n  "
            + "\n  ".join(problems)
        )
