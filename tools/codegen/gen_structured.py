"""Validate structured schema groups before native code generation."""

from __future__ import annotations

from .main import CodegenContext, register_generator


def _records(functions):
    return tuple(getattr(functions, "reference_functions", functions))


def _name(function) -> str:
    return str(function.func.name)


def _kind(function) -> str:
    return function.func.kind().name


def validate_structured(functions) -> list[str]:
    """Return all invalid structured relationships in the schema."""
    records = _records(functions)
    by_name = {_name(function): function for function in records}
    problems: list[str] = []

    for function in records:
        delegate = function.structured_delegate
        if delegate is not None:
            delegate_name = str(delegate)
            target = by_name.get(delegate_name)
            if target is None:
                problems.append(
                    f"{_name(function)}: structured_delegate "
                    f"'{delegate_name}' is not present"
                )
                continue
            if not target.structured:
                problems.append(
                    f"{_name(function)}: delegate '{delegate_name}' "
                    "is not a structured out function"
                )
            if _kind(target) != "out":
                problems.append(
                    f"{_name(function)}: delegate '{delegate_name}' "
                    f"has schema kind {_kind(target)}, expected out"
                )
            if _kind(function) == "out":
                problems.append(
                    f"{_name(function)}: out functions cannot delegate"
                )
            if function.func.signature() != target.func.signature():
                problems.append(
                    f"{_name(function)}: delegate signature differs from "
                    f"'{delegate_name}'"
                )

        if function.structured and _kind(function) != "out":
            problems.append(
                f"{_name(function)}: structured is valid only on out functions"
            )
        if function.structured and function.structured_delegate is not None:
            problems.append(
                f"{_name(function)}: structured and structured_delegate "
                "cannot both be set"
            )

    return problems


def _structured_groups(ctx: CodegenContext):
    grouped = getattr(ctx.funcs, "grouped_native_functions", None)
    if grouped is None:
        return ()
    return tuple(group for group in grouped() if getattr(group, "structured", False))


@register_generator("Structured")
def _gen_structured(ctx: CodegenContext) -> None:
    problems = validate_structured(ctx.funcs)
    if problems:
        raise ValueError(
            "structured schema validation failed:\n  "
            + "\n  ".join(problems)
        )

    groups = _structured_groups(ctx)
    if not groups:
        return

    names = ", ".join(str(group.out.func.name) for group in groups)
    raise RuntimeError(
        "structured schemas require a native structured-kernel emitter; "
        f"no emitter is registered for: {names}"
    )
