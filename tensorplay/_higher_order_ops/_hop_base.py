"""Higher-order operators.

A higher-order operator is a callable-dispatched operator: some of its
arguments are functions (graphs) rather than tensors.  The operators in this
package expose a registration surface keyed by dispatch role, with the
composite eager implementation as the base registration; graph capture turns
a call into one opaque node carrying the traced subgraphs.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from typing import Any

import tensorplay
from tensorplay import Tensor


class _AutoDispatchBelowAutograd:
    """Suppress the Autograd routing while an autograd formula recomputes
    through the same operator.  Without it, a ``Function.forward`` that calls
    the operator again would re-enter the Autograd registration forever."""

    def __init__(self) -> None:
        self._previous = False

    def __enter__(self) -> "_AutoDispatchBelowAutograd":
        self._previous = getattr(_below_autograd, "active", False)
        _below_autograd.active = True
        return self

    def __exit__(self, *exc: Any) -> None:
        _below_autograd.active = self._previous


_below_autograd = threading.local()


class HigherOrderOperator:
    """Base registry for an operator that accepts graph arguments.

    Registrations are keyed by role string (``"CompositeExplicitAutograd"``,
    ``"Autograd"``, ...).  Calling the instance resolves the most specific
    registered implementation for the runtime state and invokes it with the
    original arguments.
    """

    def __init__(self, name: str, *, cacheable: bool = False) -> None:
        self._name = name
        self.__name__ = name
        self.__module__ = "tensor.ops.higher_order"
        self._cacheable = cacheable
        self._impls: dict[str, Callable[..., Any]] = {}
        self._fake_impl: Callable[..., Any] | None = None

    @property
    def name(self) -> str:
        return self._name

    def py_impl(self, role: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register one implementation for a dispatch role."""

        def wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._impls[role] = fn
            return fn

        return wrapper

    def py_functionalize_impl(
        self, fn: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Register the functionalized implementation (same surface here)."""
        self._impls["Functionalize"] = fn
        return fn

    def register_fake(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register the meta/abstract implementation."""
        self._fake_impl = fn
        return fn

    def has_impl(self, role: str) -> bool:
        return role in self._impls

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # The Autograd layer sits above the composite one: route to it while
        # gradients are being tracked and no formula is re-entering.
        if (
            not getattr(_below_autograd, "active", False)
            and "Autograd" in self._impls
            and tensorplay.is_grad_enabled()
        ):
            return self._impls["Autograd"](*args, **kwargs)
        if "CompositeExplicitAutograd" in self._impls:
            return self._impls["CompositeExplicitAutograd"](*args, **kwargs)
        raise RuntimeError(f"no implementation registered for {self._name}")


def register_fake(hop: HigherOrderOperator):
    """Decorator form of :meth:`HigherOrderOperator.register_fake`."""

    def wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
        return hop.register_fake(fn)

    return wrapper


@contextlib.contextmanager
def suspend_functionalization():
    yield


@contextlib.contextmanager
def disable_functional_mode():
    yield


@contextlib.contextmanager
def disable_proxy_modes_tracing():
    yield


class FakeTensorMode:
    """Placeholder mode: tensors pass through unchanged in this build."""

    def __init__(self, allow_non_fake_inputs: bool = True) -> None:
        self.allow_non_fake_inputs = allow_non_fake_inputs

    def __enter__(self) -> "FakeTensorMode":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def is_fake_tensor(t: Any) -> bool:
    del t
    return False


def detect_fake_mode(values: Any = None) -> FakeTensorMode | None:
    del values
    return None
