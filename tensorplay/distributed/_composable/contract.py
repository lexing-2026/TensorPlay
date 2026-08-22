# Ported from torch/distributed/_composable/contract.py.
from collections.abc import Callable
from typing import Protocol, TypeVar

from tensorplay.nn.modules.module import Module

__all__: list[str] = []

_T = TypeVar("_T")
_TState = TypeVar("_TState")


def generate_state_key(string="__composable_api_state_key"):
    """Generate a state key that can be used for ``_get_registry``."""
    return string


class RegistryItem:
    pass


class _ContractFn(Protocol):
    def __call__(self, module: Module, *args, **kwargs): ...


def contract(
    state_key,
) -> Callable:
    """
    Decorate a composable API function so it enforces the composable contract.

    The contract ensures the API installs its ``RegistryItem`` into the
    module's registry exactly once and records the returned state.
    """

    def wrapped(func: _ContractFn) -> _ContractFn:
        def wrapper(module: Module, *args, **kwargs):
            registry = _get_registry(module)
            if registry is None:
                raise ValueError(
                    "Every composable API expects a `nn.Module` instance as "
                    "its first argument."
                )
            if state_key in registry:
                raise RuntimeError(
                    f"{func.__name__} has already been applied to this module."
                )
            result = func(module, *args, **kwargs)
            if isinstance(result, RegistryItem):
                registry[state_key] = result
            return result

        return wrapper

    return wrapped


def _get_registry(module: Module) -> dict[str, RegistryItem] | None:
    """Get the ``Module``'s registry, creating it if necessary (torch parity)."""
    if not hasattr(module, "__dict__"):
        raise TypeError(
            "tensorplay.distributed composable APIs do not support modules "
            "that do not have a __dict__, e.g. those that define __slots__."
        )

    # A missing registry indicates this is the first time any composable API
    # is being applied to this module.
    if not hasattr(module, "_registry"):
        module.__dict__["_registry"] = {}
    return module.__dict__["_registry"]
