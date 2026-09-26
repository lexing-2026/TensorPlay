"""Small helpers for asking a backend what it will accept.

A launch carries two different kinds of name: the names the kernel declares as
parameters, and the names the backend reads as options.  Which names are which
is not knowable by reading the launch -- a name is a parameter or an option
depending on the backend that will run it -- so it is asked.

The backend is reached by asking it to parse options, which is the only thing
it exposes for the purpose.  A name it does not recognize is an error rather
than something to pass along, because a launch whose extra names nobody reads
is a launch that quietly does not do what it says.
"""

from __future__ import annotations

import sympy

import tensorplay as tp

from .triton_compat import triton


def get_backend_options_for_target(target, options=None):
    """Every option name the backend for ``target`` recognizes."""

    options = {} if options is None else dict(options)
    backend = triton.compiler.compiler.make_backend(target)
    return backend.parse_options(options).__dict__


def _is_concrete_backend_option_value(value) -> bool:
    """Whether a value is fixed now rather than worked out during the launch.

    An option the backend reads while it is launching has to be a value, not
    an expression: there is nothing to evaluate an expression against at that
    point.  So a symbolic value -- anything standing for a number, whether this
    project's own or a symbolic library's -- is not one.
    """

    if isinstance(
        value,
        (
            tp.Tensor,
            tp.SymInt,
            tp.SymFloat,
            tp.SymBool,
            sympy.Expr,
        ),
    ):
        return False
    if isinstance(value, (tuple, list)):
        return all(_is_concrete_backend_option_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            _is_concrete_backend_option_value(key)
            and _is_concrete_backend_option_value(item)
            for key, item in value.items()
        )
    return True


def try_filter_backend_options_for_target(target, options, kernel_arg_names=()):
    """Split launch names into the ones the backend reads as options.

    A name the backend does not read and the kernel does not declare belongs to
    neither, and is an error: passing it along would let a launch name something
    that nothing reads, and the kernel would run with an option the caller
    believed it had set.  The check is skipped entirely when there is nothing
    to split, which is the common case -- a launch that sets no options.
    """

    parsed_options = get_backend_options_for_target(target)
    kernel_arg_names = tuple(kernel_arg_names)
    filtered_options = {
        name: value for name, value in options.items() if name in parsed_options
    }
    invalid_options = [
        name
        for name in options
        if name not in parsed_options and name not in kernel_arg_names
    ]
    if invalid_options:
        raise RuntimeError(
            "launch names must be kernel parameters or backend options: "
            f"{sorted(invalid_options)!r}"
        )
    dynamic_options = [
        name
        for name, value in filtered_options.items()
        if not _is_concrete_backend_option_value(value)
    ]
    if dynamic_options:
        raise RuntimeError(
            f"backend options must be values, not expressions: {sorted(dynamic_options)!r}"
        )
    return filtered_options
