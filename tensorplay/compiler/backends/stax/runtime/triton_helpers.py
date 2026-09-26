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

from .triton_compat import libdevice, math, triton

from .triton_compat import JITFunction


def set_driver_to_cpu():
    """Make the host backend the one launches are made through.

    A host backend may not be installed at all, which is not fatal here: a
    kernel that cannot be launched is still worth writing, and refusing to
    write it would turn a missing optional piece into a failure to compile.
    """

    import warnings

    import triton.backends
    import triton.runtime.driver

    driver = triton.runtime.driver
    backend = triton.backends.backends.get("cpu", None)
    if backend is None:
        warnings.warn(
            "could not find an active host backend; generated kernels will not "
            "be executable"
        )
        return
    if isinstance(driver.active, backend.driver):
        return
    driver.set_active(backend.driver())


def _is_backend_active(name, backend):
    """Whether this backend is the one the machine can run.

    The backend knows whether it has a device, but it can be wrong about it
    when the check runs in a subprocess where the device is not visible to
    the library that would find it.  So where the answer would be surprising,
    the device is asked directly instead.
    """

    if backend.driver.is_active():
        return True
    if name == "nvidia":
        return tp.cuda.is_available() and tp.version.hip is None
    if name == "amd":
        return tp.cuda.is_available() and tp.version.hip is not None
    return False


def set_driver_to_gpu():
    """Make the device backend the one launches are made through.

    Compiling and launching name a target, and the runtime has to be holding
    the matching driver before it will.  Which backend that is depends on the
    machine, so it is asked for rather than assumed -- and a backend already
    active is left alone, because setting one up again is not free.
    """

    import triton
    import triton.backends
    import triton.runtime.driver

    driver = triton.runtime.driver
    for name, backend in triton.backends.backends.items():
        if name == "cpu" or not _is_backend_active(name, backend):
            continue
        # The active driver may be a lazy proxy, in which case the object it
        # stands for is what tells whether it is already this backend's.
        active = driver.active
        if isinstance(active, backend.driver) or (
            hasattr(active, "_obj")
            and isinstance(active._obj, backend.driver)
        ):
            return
        driver.set_active(backend.driver())
        return
    raise RuntimeError("could not find an active device backend")


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

def get_constexprs(kernel: JITFunction) -> list[int]:
    """Which of a kernel's parameters were fixed when it was written.

    Returned as positions rather than names, because that is what a launch needs
    to decide: a parameter at one of these positions is not a value the caller
    passes, whatever the launch is called and whatever order the names are
    written in.
    """

    return [p.num for p in kernel.params if p.is_constexpr]
