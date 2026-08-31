"""Small shared pieces of the function-transform implementation."""
from typing import Any, Callable, Union

__all__ = ["argnums_t", "exposed_in"]

#: An ``argnums`` argument: one positional index, or several.
argnums_t = Union[int, tuple[int, ...]]


def exposed_in(module: str) -> Callable[[Any], Any]:
    """Re-points a function's reported module to its public namespace.

    The transforms live here but are documented and raised from
    ``tensorplay.func``; this keeps tracebacks and ``repr`` pointing at the
    name a caller actually typed.
    """

    def wrapper(fn: Any) -> Any:
        fn.__module__ = module
        return fn

    return wrapper
