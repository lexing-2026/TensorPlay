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
_autocast_excluded = threading.local()


class _ExcludeAutocastGuard:
    """Suppress the Autocast routing while a cast-down implementation
    redispatches with the already-cast operands."""

    def __enter__(self) -> "_ExcludeAutocastGuard":
        self._previous = getattr(_autocast_excluded, "active", False)
        _autocast_excluded.active = True
        return self

    def __exit__(self, *exc: Any) -> None:
        _autocast_excluded.active = self._previous


def _first_tensor_device_type(args: tuple[Any, ...]) -> str | None:
    stack = list(args)
    while stack:
        arg = stack.pop(0)
        if isinstance(arg, (tuple, list)):
            stack.extend(arg)
            continue
        if isinstance(arg, Tensor):
            return arg.device.type
    return None


class HigherOrderOperator:
    """Base registry for an operator that accepts graph arguments.

    Registrations are keyed by dispatch role.  A role is either a string
    (``"CompositeExplicitAutograd"``, ``"Autograd"``, ``"AutocastCUDA"``,
    ``"AutocastCPU"``, ``"ProxyTorchDispatchMode"``, ``"Functionalize"``,
    ``"PyAutograd"``, ...) or a mode class.  Calling the instance resolves the
    most specific registered implementation for the runtime state and invokes
    it with the original arguments.

    The routing order follows the dispatch-stack priorities: an active proxy
    capture first, then the autograd layer (unless a formula is re-entering
    below it), then the autocast layer (unless excluded), then the composite
    eager base registration.
    """

    def __init__(self, name: str, *, cacheable: bool = False) -> None:
        self._name = name
        self.__name__ = name
        self.__module__ = "tensor.ops.higher_order"
        self._cacheable = cacheable
        self._impls: dict[Any, Callable[..., Any]] = {}
        self._fake_impl: Callable[..., Any] | None = None

    @property
    def name(self) -> str:
        return self._name

    def py_impl(self, role: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register one implementation for a dispatch role."""

        def wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._impls[role] = fn
            return fn

        return wrapper

    def py_autograd_impl(
        self, fn: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Register the implementation that runs at the autograd layer.

        The registered callable receives every call before the composite one
        and is responsible for its own grad-state handling.
        """
        self._impls["PyAutograd"] = fn
        return fn

    def py_functionalize_impl(
        self, fn: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Register the functionalized implementation."""
        self._impls["Functionalize"] = fn
        return fn

    def functionalize_call(self, ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """Invoke the registered functionalize implementation with a caller
        supplied functionalization context."""
        impl = self._impls.get("Functionalize")
        if impl is None:
            raise RuntimeError(
                f"no functionalize implementation registered for {self._name}"
            )
        return impl(ctx, *args, **kwargs)

    def register_fake(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register the meta/abstract implementation."""
        self._fake_impl = fn
        return fn

    def has_impl(self, role: str) -> bool:
        return role in self._impls

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # A running proxy capture sees every call first so it can record the
        # operator node instead of executing the eager implementation.
        if "ProxyTorchDispatchMode" in self._impls:
            from tensorplay.graph.experimental.proxy_tensor import get_proxy_mode

            mode = get_proxy_mode()
            if mode is not None:
                return self._impls["ProxyTorchDispatchMode"](mode, *args, **kwargs)

        # The Autograd layer sits above autocast and the composite one: route
        # to it while gradients are being tracked and no formula is
        # re-entering.  A PyAutograd registration handles its own grad state.
        if not getattr(_below_autograd, "active", False):
            if "PyAutograd" in self._impls:
                return self._impls["PyAutograd"](*args, **kwargs)
            if "Autograd" in self._impls and tensorplay.is_grad_enabled():
                return self._impls["Autograd"](*args, **kwargs)

        # Between autograd and the composite layer sit the autocast keys: cast
        # the operands to the active autocast dtype when one is enabled.
        if not getattr(_autocast_excluded, "active", False):
            device_type = _first_tensor_device_type(args)
            autocast_role = None
            if device_type == "cuda":
                autocast_role = "AutocastCUDA"
            elif device_type == "cpu":
                autocast_role = "AutocastCPU"
            if autocast_role is not None and autocast_role in self._impls:
                if tensorplay.is_autocast_enabled(device_type):
                    return self._impls[autocast_role](*args, **kwargs)

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
