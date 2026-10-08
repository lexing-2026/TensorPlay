"""What every value of a captured program is on the example inputs.

Export records the program symbolically, so a value the program computes has
no number behind it -- yet the program reads the shapes of those values, and
branches on them, the same way it reads the shapes of its inputs.  So as each
call is recorded it is also run, once, on what its arguments were on the
example: the program's own inputs (copies of them, so nothing the program
writes reaches the caller), its parameters, its buffers.  That gives every
computed tensor an example, and with it a dtype, a device and a shape.

A shape that does not depend on a dimension declared dynamic is the example's.
One that does is derived by the shape rule of the operation, over the symbols
those dimensions stand for, and checked against the example; an operation with
no rule, or whose result is sized by what a tensor holds rather than by its
shape, leaves extents nothing is known about -- reading them is fine, deciding
anything on them is not.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import sympy

from ..graph.node import Node
from ._shape_rules import (
    DATA_DEPENDENT,
    Data,
    DataDependent,
    Shaped,
    find,
    operation,
)

if TYPE_CHECKING:
    from ._symbolic_dims import SymbolicDims

log = logging.getLogger(__name__)

MISSING = object()


class _Unresolved(Exception):
    pass


#: Attributes of a tensor that are themselves tensors.
_TENSOR_ATTRIBUTES = frozenset({"T", "mT", "H", "mH", "real", "imag", "data", "grad"})
#: Attributes that are not facts about the example alone.
_UNSETTLED_ATTRIBUTES = frozenset({"requires_grad", "grad_fn", "is_leaf", "_version"})
#: Methods that answer a question about a tensor rather than compute one.
_METADATA_METHODS = frozenset(
    {
        "dim",
        "ndimension",
        "is_floating_point",
        "is_complex",
        "is_contiguous",
        "is_signed",
        "element_size",
        "get_device",
    }
)
_COUNTING_METHODS = frozenset({"numel", "nelement"})


def _is_tensor(value: Any) -> bool:
    import tensorplay as tp

    return isinstance(value, tp.Tensor)


class ExampleRun:
    """The example of every value a capture recorded, and its extents."""

    def __init__(
        self,
        dims: SymbolicDims,
        samples: dict[str, Any],
        state: Callable[[Node], Any],
    ) -> None:
        self.dims = dims
        #: The tracer's samples: example inputs, and sizes as symbols.
        self._samples = samples
        #: Reads a lifted parameter, buffer or constant off the model.
        self._state = state
        self._values: dict[str, Any] = {}
        #: Parameters the run reads in place, copied before it writes one.
        self._shared: set[int] = set()

    def release(self) -> None:
        self._values.clear()
        self._shared.clear()

    # -- reading ---------------------------------------------------------------

    def value(self, node: Node) -> Any:
        """What ``node`` was on the example, as a rule sees it, or ``MISSING``."""

        try:
            return self._resolve(node)
        except _Unresolved:
            return MISSING

    def _resolve(self, node: Node) -> Any:
        known = self._values.get(node.name, MISSING)
        if known is not MISSING:
            return known
        from ..graph.experimental.sym_node import SymNode

        sample = self._samples.get(node.name)
        if isinstance(sample, SymNode):
            expr = sympy.sympify(sample.expr)
            return int(expr) if expr.is_Integer else expr
        if node.op == "placeholder":
            value = self._placeholder(node, sample)
        elif sample is not None:
            value = self._wrap_fixed(sample, copy=False)
        else:
            raise _Unresolved(node.name)
        self._values[node.name] = value
        return value

    def _placeholder(self, node: Node, sample: Any) -> Any:
        if sample is None:
            target = node.meta.get("state_target")
            if target is None:
                raise _Unresolved(node.name)
            value = self._state(node)
            if not _is_tensor(value):
                return value
            if node.meta.get("state_kind") == "parameter":
                # Parameters are only read; one written in place is copied first.
                self._shared.add(id(value))
                return Shaped(tuple(int(size) for size in value.shape), value)
            return Shaped(tuple(int(size) for size in value.shape), value.detach().clone())
        origins = frozenset({str(node.target)})
        if _is_tensor(sample):
            sites = self.dims.sites.get(str(node.target), {})
            extents = tuple(
                sites[axis][0] if axis in sites else int(size)
                for axis, size in enumerate(sample.shape)
            )
            return Shaped(extents, sample.detach().clone(), origins)
        if isinstance(sample, (tuple, list, dict)):
            return self._wrap_fixed(sample, copy=True, origins=origins)
        # A Python value the caller passes may differ on every call.
        return Data(sample)

    def _wrap_fixed(
        self, value: Any, *, copy: bool, origins: frozenset[str] = frozenset()
    ) -> Any:
        if _is_tensor(value):
            example = value.detach().clone() if copy else value
            return Shaped(tuple(int(size) for size in value.shape), example, origins)
        if isinstance(value, (tuple, list)):
            return type(value)(
                self._wrap_fixed(item, copy=copy, origins=origins) for item in value
            )
        if isinstance(value, dict):
            return {
                key: self._wrap_fixed(item, copy=copy, origins=origins)
                for key, item in value.items()
            }
        return value

    def _resolve_all(self, value: Any) -> Any:
        if isinstance(value, Node):
            return self._resolve(value)
        if isinstance(value, tuple):
            items = [self._resolve_all(item) for item in value]
            return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
        if isinstance(value, list):
            return [self._resolve_all(item) for item in value]
        if isinstance(value, dict):
            return {key: self._resolve_all(item) for key, item in value.items()}
        if isinstance(value, slice):
            return slice(
                self._resolve_all(value.start),
                self._resolve_all(value.stop),
                self._resolve_all(value.step),
            )
        return value

    def _concrete(self, value: Any) -> Any:
        """``value`` as the call is run with it: examples and example sizes."""

        if isinstance(value, Shaped):
            return value.example
        if isinstance(value, Data):
            return value.example
        if isinstance(value, sympy.Basic):
            return self.dims.example(value)
        if isinstance(value, tuple):
            items = [self._concrete(item) for item in value]
            return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
        if isinstance(value, list):
            return [self._concrete(item) for item in value]
        if isinstance(value, dict):
            return {key: self._concrete(item) for key, item in value.items()}
        if isinstance(value, slice):
            return slice(
                self._concrete(value.start), self._concrete(value.stop), self._concrete(value.step)
            )
        return value

    # -- recording -------------------------------------------------------------

    def record(self, node: Node) -> Any:
        """Run the call ``node`` records on the example and derive its extents.

        Returns what was recorded for it, or ``MISSING`` when the call could
        not be run -- an argument without an example, or a call that fails on
        it -- and the value stays without one, as it was before this run.
        """

        from ..graph.tracer import _is_higher_order

        if node.op not in ("call_function", "call_method") or _is_higher_order(node.target):
            return MISSING
        try:
            args = self._resolve_all(node.args)
            kwargs = self._resolve_all(node.kwargs)
        except _Unresolved:
            return MISSING
        name = operation(node.op, node.target)
        value = self._read(name, args)
        if value is MISSING:
            value = self._run(node, name, args, kwargs)
        if value is not MISSING:
            self._values[node.name] = value
        return value

    def _read(self, name: str | None, args: tuple[Any, ...]) -> Any:
        """A read of a tensor's metadata, or of an element of a sequence."""

        if name == "getattr" and len(args) >= 2 and isinstance(args[1], str):
            owner, attribute = args[0], args[1]
            if isinstance(owner, Shaped):
                if attribute == "shape":
                    return owner.extents
                if attribute == "ndim":
                    return owner.rank
                if attribute in _TENSOR_ATTRIBUTES or attribute in _UNSETTLED_ATTRIBUTES:
                    return MISSING
                try:
                    return getattr(owner.example, attribute)
                except AttributeError:
                    return MISSING
            if isinstance(owner, tuple) and hasattr(owner, "_fields"):
                return getattr(owner, attribute, MISSING)
            return MISSING
        if name == "getitem" and len(args) == 2 and isinstance(args[0], (tuple, list, Data)):
            container, key = args
            if isinstance(container, Data):
                try:
                    return Data(container.example[self._concrete(key)])
                except Exception:
                    return MISSING
            if isinstance(key, sympy.Basic) or (
                isinstance(key, slice) and any(isinstance(part, sympy.Basic) for part in (key.start, key.stop, key.step))
            ):
                return MISSING
            try:
                return container[key]
            except (IndexError, TypeError, KeyError):
                return MISSING
        return MISSING

    def _run(self, node: Node, name: str | None, args: Any, kwargs: Any) -> Any:
        self._unshare(name, args, kwargs)
        try:
            output = self._execute(node, self._concrete(args), self._concrete(kwargs))
        except Exception as error:  # an example run that fails leaves the value unknown
            log.debug("example run of %s failed: %s", node.name, error)
            return MISSING
        if name in _METADATA_METHODS and not _is_tensor(output):
            return output
        if name in _COUNTING_METHODS and args and isinstance(args[0], Shaped):
            count = sympy.sympify(1)
            for extent in args[0].extents:
                count = count * extent
            return int(count) if count.is_Integer else count
        if not self._holds_tensor(output):
            if self._mentions_tensor((args, kwargs)):
                return Data(output)
            return output
        extents, reason = self._derive(name, args, kwargs, output)
        reason = reason or f"computed by {name or node.target!r}"
        return self._wrap(output, extents, reason, self._origins((args, kwargs)))

    def _execute(self, node: Node, args: Any, kwargs: Any) -> Any:
        from ..graph import _utils

        try:
            from ..autograd.grad_mode import no_grad
        except ImportError:  # pragma: no cover - the frontend always has it
            no_grad = contextlib.nullcontext
        token = _utils._executing_sample.set(True)
        try:
            with no_grad():
                if node.op == "call_function":
                    return node.target(*args, **kwargs)
                return getattr(args[0], node.target)(*args[1:], **kwargs)
        finally:
            _utils._executing_sample.reset(token)

    def _unshare(self, name: str | None, args: Any, kwargs: Any) -> None:
        """Copy a parameter before a call that writes to it runs on it."""

        if not self._shared or not name:
            return
        writes = (
            (name.endswith("_") and not name.endswith("__"))
            or name == "setitem"
            or kwargs.get("inplace") is True
            or "out" in kwargs
        )
        if not writes:
            return
        for value in (*args, *kwargs.values()):
            if isinstance(value, Shaped) and id(value.example) in self._shared:
                self._shared.discard(id(value.example))
                value.example = value.example.detach().clone()

    def _derive(self, name: str | None, args: Any, kwargs: Any, output: Any) -> tuple[Any, str | None]:
        """The result's extents, or ``None`` with the reason they are unknown."""

        if name in DATA_DEPENDENT:
            return None, f"sized by the values of a tensor ({name})"
        rule = find(name) if name else None
        if rule is not None:
            mark = len(self.dims.guards)
            try:
                extents = rule(self.dims, *args, **kwargs)
            except DataDependent:
                del self.dims.guards[mark:]
                return None, f"sized by the values of a tensor ({name})"
            except Exception as error:
                log.debug("shape rule for %s gave up: %s", name, error)
                extents = None
            if extents is not None and self._agrees(extents, output):
                return extents, None
            del self.dims.guards[mark:]
        if self._symbolic((args, kwargs)) or self._mentions_data((args, kwargs)):
            what = name or "the call"
            return None, f"{what} has no shape rule for sizes that vary"
        return self._fixed(output), None

    def _agrees(self, extents: Any, output: Any) -> bool:
        if _is_tensor(output):
            if not isinstance(extents, tuple) or len(extents) != output.dim():
                return False
            try:
                return all(
                    self.dims.example(extent) == int(size)
                    for extent, size in zip(extents, output.shape)
                )
            except (ValueError, TypeError):
                return False
        if isinstance(output, (tuple, list)):
            if not isinstance(extents, list) or len(extents) != len(output):
                return False
            return all(
                entry is None if not _is_tensor(item) else self._agrees(entry, item)
                for entry, item in zip(extents, output)
            )
        return False

    def _fixed(self, output: Any) -> Any:
        if _is_tensor(output):
            return tuple(int(size) for size in output.shape)
        if isinstance(output, (tuple, list)):
            return [self._fixed(item) if self._holds_tensor(item) else None for item in output]
        return None

    def _wrap(self, output: Any, extents: Any, reason: str, origins: frozenset[str]) -> Any:
        if _is_tensor(output):
            if extents is None:
                extents = tuple(self.dims.fresh(int(size), reason) for size in output.shape)
            return Shaped(extents, output, origins)
        if isinstance(output, (tuple, list)):
            entries = extents if isinstance(extents, list) else [None] * len(output)
            items = [
                self._wrap(item, entry, reason, origins) if self._holds_tensor(item) else Data(item)
                for item, entry in zip(output, entries)
            ]
            if hasattr(output, "_fields"):
                return type(output)(*items)
            try:
                return type(output)(items)
            except TypeError:
                return tuple(items)
        return Data(output)

    # -- inspecting arguments --------------------------------------------------

    def _holds_tensor(self, value: Any) -> bool:
        if _is_tensor(value):
            return True
        if isinstance(value, (tuple, list)):
            return any(self._holds_tensor(item) for item in value)
        return False

    def _origins(self, value: Any) -> frozenset[str]:
        if isinstance(value, Shaped):
            return value.origins
        if isinstance(value, (tuple, list)):
            items = value
        elif isinstance(value, dict):
            items = list(value.values())
        else:
            return frozenset()
        return frozenset().union(*(self._origins(item) for item in items))

    def _mentions_tensor(self, value: Any) -> bool:
        return self._any(value, lambda item: isinstance(item, (Shaped, Data)))

    def _mentions_data(self, value: Any) -> bool:
        return self._any(value, lambda item: isinstance(item, Data))

    def _symbolic(self, value: Any) -> bool:
        def varies(item: Any) -> bool:
            if isinstance(item, Shaped):
                return any(not isinstance(extent, int) for extent in item.extents)
            return isinstance(item, sympy.Basic) and bool(item.free_symbols)

        return self._any(value, varies)

    def _any(self, value: Any, test: Callable[[Any], bool]) -> bool:
        if test(value):
            return True
        if isinstance(value, (tuple, list)):
            return any(self._any(item, test) for item in value)
        if isinstance(value, dict):
            return any(self._any(item, test) for item in value.values())
        if isinstance(value, slice):
            return any(self._any(item, test) for item in (value.start, value.stop, value.step))
        return False
