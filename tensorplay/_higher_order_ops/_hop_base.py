"""Higher-order operators.

A higher-order operator is a callable-dispatched operator: some of its
arguments are functions (graphs) rather than tensors.  The operators in this
package expose a registration surface keyed by dispatch role, with the
composite eager implementation as the base registration; graph capture turns
a call into one opaque node carrying the traced subgraphs.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
import weakref
from collections.abc import Callable
from typing import Any, Optional

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
_recorded_whole = threading.local()


def recording_whole_call() -> bool:
    """Whether a trace is recording the operator call now running as one node.

    The call runs on real values with recording switched off, so nothing on
    the dispatch stack says a trace is watching -- yet the gradient of what it
    returns is asked for later, through the trace, and has to come back as an
    operator the trace can keep whole, not as the steps this one run took.
    """

    return getattr(_recorded_whole, "depth", 0) > 0


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
    ``"AutocastCPU"``, ``"ProxyDispatchMode"``, ``"Functionalize"``,
    ``"PyAutograd"``, ...) or a mode class.  Calling the instance resolves the
    most specific registered implementation for the runtime state and invokes
    it with the original arguments.

    The routing order follows the dispatch-stack priorities: an active proxy
    capture first, then the autograd layer (unless a formula is re-entering
    below it), then the autocast layer (unless excluded), then the composite
    eager base registration.
    """

    #: Whether a trace that follows real values records this operator's steps
    #: rather than the operator.  A loop over a known number of steps is what
    #: such a trace sees anyway -- every extent is the one it was run with --
    #: and its steps lower and fuse as ordinary operations, where a loop kept
    #: whole would have to run one step at a time.  A capture still records it
    #: as one node.
    runs_inline_under_value_trace = False

    def __init__(self, name: str, *, cacheable: bool = False) -> None:
        self._name = name
        self.__name__ = name
        self.__module__ = "tensorplay.ops.higher_order"
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

    def _record_real_call(self, tracer: Any, args: tuple, kwargs: dict) -> Any:
        """Run the call on the values it was given and record it as one node.

        A trace that records values rather than stand-ins has no stand-in to
        hand back: the program it is following goes on computing with whatever
        this returns, so what is returned is the real result.  The call runs
        with recording off, so the operations inside it stay inside it, and it
        is recorded whole, with the subgraphs it was handed as its arguments --
        the same node a capture of the program holds for it.  A gradient the
        trace asks for later reaches the backward operator the same way.
        """

        from tensorplay.graph.experimental.proxy_tensor import (
            disable_proxy_modes_tracing,
        )

        _recorded_whole.depth = getattr(_recorded_whole, "depth", 0) + 1
        try:
            with disable_proxy_modes_tracing():
                out = self(*args, **kwargs)
        finally:
            _recorded_whole.depth -= 1
        with disable_proxy_modes_tracing():
            args, kwargs = self.traced_arguments(args, kwargs)
        tracer.record(self, args, kwargs, out)
        return out

    def traced_arguments(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """The call's arguments as a recorded node holds them.

        A node can hold a function only as a traced graph, and an operator
        called from inside a gradient formula is handed plain functions -- the
        formula builds them as it goes.  An operator that can be handed one
        traces it here on the values it was given; the default has nothing to
        trace.
        """

        return args, kwargs

    def __call__(self, /, *args: Any, **kwargs: Any) -> Any:
        # Positional-only receiver: operator arguments may be named ``self``.
        # A running proxy capture sees every call first so it can record the
        # operator node instead of executing the eager implementation.
        if "ProxyDispatchMode" in self._impls:
            from tensorplay.graph.experimental.proxy_tensor import get_proxy_mode

            mode = get_proxy_mode()
            if mode is not None:
                if getattr(mode.tracer, "records_real_values", False):
                    if self.runs_inline_under_value_trace:
                        return self._impls["CompositeExplicitAutograd"](*args, **kwargs)
                    return self._record_real_call(mode.tracer, args, kwargs)
                return self._impls["ProxyDispatchMode"](mode, *args, **kwargs)

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


def disable_proxy_modes_tracing():
    """Run a block with no capture recording it.

    An operator whose body is a whole region of the program is asked, before it
    runs, whether a capture is recording.  A call made *by* that operator, to
    find out what it returns, is not part of the region and must not be
    recorded -- and it must not be recorded by the same rule that recorded the
    operator, or the operator evaluates itself looking for the answer it was
    supposed to be asked for, forever.

    Imported where it is used rather than defined here, because the state it
    suspends is the one the graph tracer owns, and a second copy of the answer
    to "is a capture running" is a second thing to keep in step with the first.
    """

    from tensorplay.graph.experimental.proxy_tensor import (
        disable_proxy_modes_tracing as _suspend,
    )

    return _suspend()


class FakeTensorMode:
    """A region in which a tensor stands for a value rather than holding one.

    Tracing a function means running it, and running it on values that were
    chosen for the purpose says nothing about what it will do on the values it
    will be given. So inside this region a tensor carries the value it was made
    from and is only a stand-in for it: what the function computes is recorded,
    and the value it computed it from is still there to be read back.

    Whether a tensor that is not a stand-in may be handed in is a question about
    the call rather than about the region, so it is asked of each region: a
    region that forbids them is refusing a call whose answer would not mean
    anything, and a region that allows them is accepting one whose meaning
    depends on values chosen outside.
    """

    def __init__(self, allow_non_fake_inputs: bool = True) -> None:
        self.allow_non_fake_inputs = allow_non_fake_inputs
        self._saved: dict[int, Any] = {}
        #: What the extents in here are worked out against.  A value inside the
        #: region describes a shape, and a shape that is not yet a number has
        #: to be something: the environment is what holds the difference
        #: between the part that is settled and the part that is not, and it is
        #: what a guard about the settled part is written in terms of.
        from tensorplay.graph.experimental.symbolic_shapes import ShapeEnv

        self.shape_env = ShapeEnv()

    def __enter__(self) -> "FakeTensorMode":
        # Pushed as well as remembered: a tensor is only a stand-in while this
        # is on the stack of modes the dispatcher consults, and an operator
        # called with the language's key held open reaches whatever is on that
        # stack and pops it as it goes.  A region that was entered but not
        # pushed would answer nothing to the operation it exists to trace.
        from tensorplay import _C

        self._dispatch_token = _C._push_dispatch_mode(self)
        self._token = _ACTIVE_FAKE_MODE.set(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        from tensorplay import _C

        _C._pop_dispatch_mode()
        _ACTIVE_FAKE_MODE.reset(self._token)

    def mark(self, tensor: Any) -> Any:
        """Record ``tensor`` as standing for the value it was made from."""

        if isinstance(tensor, Tensor):
            _FAKE_CONSTANTS[id(tensor)] = tensor
        elif isinstance(tensor, (tuple, list)):
            for value in tensor:
                self.mark(value)
        return tensor

    def __tensorplay_dispatch__(self, func, types, args=(), kwargs=None):
        """Run the operation here, and record what it made.

        A tensor inside the region is a stand-in rather than a value, so the
        operation is run on stand-ins and whatever it returns is one too. What
        is being recorded is the value it was computed from, so a stand-in
        that outlives the region still says what it stands for.

        The extents it comes back with are the ones the operation would produce
        on the values it was given, which is why it is asked rather than taken
        from what was handed in: an operation whose result depends on its input
        values produces a shape that is not in the inputs.
        """

        kwargs = {} if kwargs is None else kwargs
        result = func(*args, **kwargs)
        if isinstance(result, (Tensor, tuple, list)) and all(
            isinstance(r, Tensor) for r in (result if isinstance(result, (tuple, list)) else (result,))
        ):
            return self.mark(result)
        return result


#: The region currently being traced through, if any.  Held here rather than
#: passed down because a tensor becomes a stand-in by being made inside the
#: region, and what makes it one is not something the code making it is told.
_ACTIVE_FAKE_MODE: contextvars.ContextVar[Optional["FakeTensorMode"]] = (
    contextvars.ContextVar("active_fake_mode", default=None)
)

#: What each stand-in stands for, held beside the values rather than on them.
#: A tensor carries no room of its own for a note saying what it is standing
#: in for.  Entries last as long as their tensors do; recording a value does
#: not extend the lifetime of its storage.
_FAKE_CONSTANTS: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()


def is_fake_tensor(t: Any) -> bool:
    """Whether this value is standing in for another rather than being one.

    Asked of the value rather than of the region it was made in, because a
    stand-in outlives its region: a graph captured under one is read long after
    that region has closed, and asking the region then would say no about
    something that is.
    """

    return isinstance(t, Tensor) and id(t) in _FAKE_CONSTANTS


def maybe_get_fake_constant(t: Any) -> Any | None:
    """The value a stand-in stands for, or nothing if it is not a stand-in.

    Nothing rather than the value itself when asked about something that is not
    a stand-in, so that a caller asking about every value it handles can tell
    the two apart instead of finding a value where there was none.
    """

    if is_fake_tensor(t):
        return _FAKE_CONSTANTS.get(id(t))
    return None


def detect_fake_mode(values: Any = None) -> FakeTensorMode | None:
    """The region being traced through, if the values were made inside one.

    A value made inside a region names that region, and a value made outside
    one names none -- so the question is answered by the values rather than by
    asking whether some region happens to be open, which would say yes about
    values that were not made in it.
    """

    del values
    return _ACTIVE_FAKE_MODE.get()
