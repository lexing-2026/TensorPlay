"""Keep native symbolic metadata as scalar values in a dispatcher graph."""

from __future__ import annotations

import ast
import operator

import tensorplay as tp

from ._dispatch_trace import ProxyTensorDispatchMode
from ..interpreter import Interpreter


class SymbolicGraphInterpreter(Interpreter):
    """Execute shape methods through operators that preserve scalar expressions."""

    def call_function(self, target, args, kwargs):
        if target is operator.getitem and isinstance(args[0], tp.Tensor):
            result = self.basic_index(*args)
            if result is not NotImplemented:
                return result
        scalar_ops = {operator.add: "add", operator.sub: "sub", operator.mul: "mul", operator.truediv: "div"}
        scalar_op = next((name for func, name in scalar_ops.items() if target is func), None)
        if scalar_op is not None and len(args) == 2:
            left, right = args
            if isinstance(left, tp.Tensor) and isinstance(right, tp.SymInt):
                return getattr(tp.ops.tp, scalar_op).Scalar(left, right, **kwargs)
            if isinstance(right, tp.Tensor) and isinstance(left, tp.SymInt):
                if target in (operator.add, operator.mul):
                    return getattr(tp.ops.tp, scalar_op).Scalar(right, left, **kwargs)
                if target is operator.sub:
                    return tp.ops.tp.rsub.Scalar(right, left, **kwargs)
        if target is tp.reshape:
            if not args:
                kwargs = dict(kwargs)
                args = (kwargs.pop("input"),)
            return self.call_method("reshape", args, kwargs)
        if target is tp.flatten:
            return self.call_method("flatten", args, kwargs)
        return super().call_function(target, args, kwargs)

    def basic_index(self, value, index):
        items = index if isinstance(index, tuple) else (index,)
        integer_types = (int, tp.SymInt)
        if any(
            isinstance(item, bool) or not isinstance(item, (*integer_types, slice, type(None), type(Ellipsis)))
            for item in items
        ):
            return NotImplemented
        if any(
            bound is not None and not isinstance(bound, integer_types)
            for item in items if isinstance(item, slice)
            for bound in (item.start, item.stop, item.step)
        ):
            return NotImplemented
        consumed = sum(item is not None and item is not Ellipsis for item in items)
        if consumed > value.ndim or sum(item is Ellipsis for item in items) > 1:
            return NotImplemented
        ellipsis_axes = value.ndim - consumed
        axis = 0
        for item in items:
            if item is Ellipsis:
                axis += ellipsis_axes
            elif item is None:
                value = tp.ops.tp.unsqueeze.default(value, axis)
                axis += 1
            elif isinstance(item, slice):
                value = tp.ops.tp.slice.Tensor(value, axis, item.start, item.stop, 1 if item.step is None else item.step)
                axis += 1
            else:
                value = tp.ops.tp.select.int(value, axis, item)
        return value

    def call_method(self, target, args, kwargs):
        value, *rest = args
        if isinstance(value, tp.Tensor):
            if target == "repeat":
                repeats = rest[0] if len(rest) == 1 and isinstance(rest[0], (list, tuple)) else rest
                return tp.ops.tp.repeat.default(value, repeats, **kwargs)
            if target in ("reshape_as", "view_as", "expand_as"):
                other = rest[0]
                sizes = [tp.ops.tp.sym_size.int(other, axis) for axis in range(other.ndim)]
                return getattr(tp.ops.tp, target.removesuffix("_as")).default(value, sizes, **kwargs)
            if target in ("reshape", "view", "expand"):
                kwargs = dict(kwargs)
                if not rest:
                    sizes = kwargs.pop("shape" if target == "reshape" else "size")
                else:
                    sizes = rest[0] if len(rest) == 1 and isinstance(rest[0], (list, tuple)) else rest
                return getattr(tp.ops.tp, target).default(value, sizes, **kwargs)
            if target == "flatten":
                start = kwargs.get("start_dim", rest[0] if rest else 0)
                end = kwargs.get("end_dim", rest[1] if len(rest) > 1 else -1)
                rank = value.ndim
                if rank == 0:
                    return tp.ops.tp.reshape.default(value, [1])
                if not (-rank <= start < rank and -rank <= end < rank):
                    return super().call_method(target, args, kwargs)
                start, end = start % rank, end % rank
                if start > end:
                    return super().call_method(target, args, kwargs)
                if start == end:
                    return value
                shape = [tp.ops.tp.sym_size.int(value, axis) for axis in range(rank)]
                extent = 1
                for size in shape[start : end + 1]:
                    extent *= size
                return tp.ops.tp.reshape.default(value, [*shape[:start], extent, *shape[end + 1 :]])
        return super().call_method(target, args, kwargs)


def map_native_symbol(tracer, value):
    """Materialize a scalar expression using the metadata nodes it reads."""
    text = str(value)
    known = tracer.native_shape_nodes.get(text)
    if known is not None:
        return known
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
    }

    def emit(expr):
        if isinstance(expr, ast.Name):
            return tracer.native_shape_nodes[expr.id]
        if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
            return expr.value
        if isinstance(expr, ast.BinOp) and type(expr.op) in operators:
            node = tracer.graph.call_function(operators[type(expr.op)], (emit(expr.left), emit(expr.right)))
        elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            node = tracer.graph.call_function(operator.neg, (emit(expr.operand),))
        else:
            raise NotImplementedError("unsupported native shape arithmetic")
        node.meta["symbolic_scalar"] = True
        if tracer.backward:
            node.meta["is_backward"] = True
        return node

    result = emit(ast.parse(text, mode="eval").body)
    tracer.native_shape_nodes[text] = result
    return result


class SymbolicDispatchMode(ProxyTensorDispatchMode):
    def __enter__(self):
        from tensorplay._ops import _symbolic_call_mode

        super().__enter__()
        if not hasattr(self, "_call_tokens"):
            self._call_tokens = []
            self._pending_calls = []
        self._call_tokens.append(_symbolic_call_mode.set(self))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        from tensorplay._ops import _symbolic_call_mode

        _symbolic_call_mode.reset(self._call_tokens.pop())
        super().__exit__(exc_type, exc_value, traceback)

    def call_with_symbolic_args(self, func, args, kwargs):
        from ..node import map_aggregate

        symbolic = []
        map_aggregate(
            (args, kwargs), lambda x: symbolic.append(x) if isinstance(x, tp.SymInt) and x.is_symbolic() else None
        )
        if not symbolic:
            return tp._C._call_overload(func._key, args, kwargs)
        if not self.tracer.backward:
            tensors = []
            map_aggregate((args, kwargs), lambda x: tensors.append(x) if isinstance(x, tp.Tensor) else None)
            if any(value.requires_grad for value in tensors):
                for index, argument in enumerate(func._schema.arguments):
                    value = args[index] if index < len(args) else kwargs.get(argument.name)
                    if str(argument.type) == "Scalar" or f"autograd_saved_arg:{argument.name}" in func._tags:
                        saved_symbols = []
                        map_aggregate(
                            value,
                            lambda x: saved_symbols.append(x) if isinstance(x, tp.SymInt) and x.is_symbolic() else None,
                        )
                        if saved_symbols:
                            self.tracer.static_autograd_metadata = True
        pending = [func, args, kwargs, False]
        self._pending_calls.append(pending)
        try:

            def concrete(value):
                return value.guard_int("", 0) if isinstance(value, tp.SymInt) else value

            return tp._C._call_overload(func._key, map_aggregate(args, concrete), map_aggregate(kwargs, concrete))
        finally:
            self._pending_calls.pop()

    def __tensorplay_dispatch__(self, func, types, args=(), kwargs=None):
        from ..node import map_aggregate

        kwargs = kwargs or {}
        if self._pending_calls:
            pending = self._pending_calls[-1]
            if pending[0] is func and not pending[3]:
                args, kwargs = pending[1:3]
                pending[3] = True
        if not self.tracer.backward and "static_autograd_metadata" in func._tags:
            tensors = []
            map_aggregate((args, kwargs), lambda x: tensors.append(x) if isinstance(x, tp.Tensor) else None)
            if any(value.requires_grad for value in tensors):
                self.tracer.static_autograd_metadata = True
        if (
            not self.tracer.backward
            and "autograd_variance_count" in func._tags
            and any(isinstance(value, tp.Tensor) and value.requires_grad for value in args)
        ):
            for index, argument in enumerate(func._schema.arguments):
                if argument.name == "correction":
                    correction = args[index] if index < len(args) else kwargs.get(argument.name, 1)
                    if correction is not None and (not isinstance(correction, int) or correction > 1):
                        self.tracer.static_autograd_metadata = True
        if func is tp.ops.tp.sym_size.int:
            source = self.tracer.node_for(args[0])
            key = (source, args[1])
            cached = self.tracer.native_shape_values.get(key)
            if cached is not None:
                return cached
            hint = int(args[0].size(args[1]))
            if hint in (0, 1):
                return hint
            node = self.tracer.record(func, args, kwargs, hint)
            node.meta["symbolic_scalar"] = True
            value = tp.SymInt.symbolic(node.name, hint)
            self.tracer.native_shape_nodes[node.name] = node
            self.tracer.native_shape_values[key] = value
            return value

        def concrete(value):
            return value.guard_int("", 0) if isinstance(value, tp.SymInt) else value

        decomposition = self.decomposition_table.get(func)
        if decomposition is not None:
            with self:
                result = decomposition(*args, **kwargs)
            if result is not NotImplemented:
                return result
        out = func(*map_aggregate(args, concrete), **map_aggregate(kwargs, concrete))
        self.tracer.record(func, args, kwargs, out)
        return out
