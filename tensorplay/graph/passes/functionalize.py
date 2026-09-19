"""Functionalization of operator-level graphs.

Rewrites a graph whose ``call_function`` targets are operator overloads so
that no node mutates a value it did not create, while computing the same
results:

* An in-place or ``out=`` overload is replaced by its functional overload,
  found from the schema (``add_.Tensor`` -> ``add.Tensor``; ``add.out`` ->
  the overload with the same arguments minus the destination).  An overload
  without a functional form becomes one ``auto_functionalized`` node, which
  runs it on private copies and returns the outputs plus the new values.
* Every tensor value is tracked as a *base* plus the chain of view
  operators that produced it.  A mutation of a view writes the new value
  back into its base through the inverse of each view step
  (``slice_scatter``, ``select_scatter``, reshaping back, the inverse
  permutation, ...); views read afterwards are replayed from the updated
  base, so every alias observes the write.
* A graph input whose base was mutated ends the graph with one ``copy_``
  of its final value, so callers still see the update.

Alias relations come from the schema annotations (``Tensor(a)`` returns),
so the pass needs no per-operator knowledge besides the view inverses.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

from ..graph import Graph
from ..graph_module import GraphModule
from ..node import Node

__all__ = ["FunctionalizationError", "functionalize"]


class FunctionalizationError(RuntimeError):
    """A mutation whose effect cannot be expressed functionally."""


def _ops() -> Any:
    import tensorplay
    from tensorplay._ops import NATIVE_NAMESPACE

    return getattr(tensorplay.ops, NATIVE_NAMESPACE)


def _schema(target: Any) -> Any:
    return getattr(target, "_schema", None)


_SELF = object()  # marks the aliased input inside a recorded view step


@dataclass(frozen=True)
class _Step:
    """One view application: ``target(*args, **kwargs)`` with ``_SELF`` in
    the aliased input's position, or ``getitem`` of a multi-output view."""

    target: Any
    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...]
    index: int | None = None
    # Traced values this step produced and was applied to (inverses read
    # shapes and strides from them).
    val: Any = None
    source_val: Any = None


# ---------------------------------------------------------------------------
# Functional variants
# ---------------------------------------------------------------------------


def _signature_key(arguments) -> tuple:
    return tuple(
        (a.name, str(a.type).replace("(", "").split(")")[-1], a.kwarg_only)
        for a in arguments
    )


def _functional_variant(overload: Any) -> Any:
    """The overload computing ``overload``'s result without mutating."""

    schema = overload._schema
    packets = _ops()
    out_args = [a for a in schema.arguments if a.is_out]
    if out_args:
        packet = getattr(packets, schema.name, None)
        wanted = _signature_key([a for a in schema.arguments if not a.is_out])
        for name in packet.overloads() if packet is not None else ():
            candidate = getattr(packet, name)
            cs = candidate._schema
            if cs.is_mutable or len(cs.returns) != len(out_args):
                continue
            if _signature_key(cs.arguments) == wanted:
                return candidate
        return None
    if schema.name.endswith("_"):
        packet = getattr(packets, schema.name[:-1], None)
        if packet is None:
            return None
        wanted = _signature_key(schema.arguments)
        for name in packet.overloads():
            candidate = getattr(packet, name)
            cs = candidate._schema
            if not cs.is_mutable and _signature_key(cs.arguments) == wanted:
                return candidate
    return None


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


class _Functionalizer:
    def __init__(self, gm: GraphModule) -> None:
        self.gm = gm
        self.graph = Graph()
        self.env: dict[Node, Any] = {}
        # old node -> (base old node, view chain)
        self.base_of: dict[Node, tuple[Node, tuple[_Step, ...]]] = {}
        self.base_value: dict[Node, Node] = {}
        self.version: dict[Node, int] = {}
        self.cache: dict[tuple[Node, int], Any] = {}
        self.inputs: list[Node] = []
        self.list_views: set[Node] = set()
        self.processing: Node | None = None

    # -- value access -------------------------------------------------------

    def new_base(self, old: Node, value: Any) -> None:
        self.base_of[old] = (old, ())
        self.base_value[old] = value
        self.version[old] = 0

    def current(self, old: Any) -> Any:
        if not isinstance(old, Node):
            return old
        entry = self.base_of.get(old)
        if entry is None:
            return self.env[old]
        base, chain = entry
        if not chain:
            return self.base_value[base]
        key = (old, self.version[base])
        cached = self.cache.get(key)
        if cached is None:
            cached = self.replay(self.base_value[base], chain)
            self.cache[key] = cached
        return cached

    def map(self, value: Any) -> Any:
        if isinstance(value, Node):
            return self.current(value)
        if isinstance(value, tuple):
            return tuple(self.map(v) for v in value)
        if isinstance(value, list):
            return [self.map(v) for v in value]
        if isinstance(value, dict):
            return {k: self.map(v) for k, v in value.items()}
        return value

    def emit(self, target: Any, args: tuple, kwargs: dict | None = None, name: str | None = None) -> Node:
        node = self.graph.create_node(
            "call_function", target, tuple(args), dict(kwargs or {}),
            name=name or getattr(target, "_opname", None),
        )
        # Nodes emitted while rewriting a node belong to the same phase
        # (joint graphs tag their backward half).
        if self.processing is not None and self.processing.meta.get("is_backward"):
            node.meta["is_backward"] = True
        return node

    def apply_step(self, step: _Step, value: Any) -> Any:
        if step.index is not None:
            return self.emit(operator.getitem, (value, step.index), name="getitem")
        args = tuple(value if a is _SELF else self.map(a) for a in step.args)
        kwargs = {k: (value if v is _SELF else self.map(v)) for k, v in step.kwargs}
        return self.emit(step.target, args, kwargs)

    def replay(self, value: Any, chain: tuple[_Step, ...]) -> Any:
        for step in chain:
            value = self.apply_step(step, value)
        return value

    # -- writes -------------------------------------------------------------

    def write(self, old: Node, value: Any) -> None:
        base, chain = self.base_of[old]
        # parents[i] is the value step i was applied to.
        parents = [self.base_value[base]]
        for step in chain[:-1]:
            parents.append(self.apply_step(step, parents[-1]))
        child = value
        i = len(chain) - 1
        while i >= 0:
            step = chain[i]
            if step.index is not None:
                # Element of a multi-output view: write straight into the
                # value the list-producing view was applied to.
                child = self.inverse_getitem(step, parents[i - 1], child)
                i -= 2
                continue
            child = self.inverse(step, parents[i], child)
            i -= 1
        self.base_value[base] = child
        self.version[base] += 1

    def inverse(self, step: _Step, parent: Any, child: Any) -> Any:
        ops = _ops()
        if step.index is not None:
            return self.inverse_getitem(step, parent, child)
        schema = step.target._schema
        name = schema.name
        args = {a.name: v for a, v in zip(schema.arguments, step.args)}
        args.update(dict(step.kwargs))

        def arg(key: str, default: Any = None) -> Any:
            value = args.get(key, default)
            return self.map(value) if value is not _SELF else parent

        if name == "slice":
            return self.emit(ops.slice_scatter.default, (parent, child, arg("dim", 0), arg("start"), arg("end"), arg("step", 1)))
        if name == "select":
            return self.emit(ops.select_scatter.default, (parent, child, arg("dim"), arg("index")))
        if name == "diagonal":
            return self.emit(
                ops.diagonal_scatter.default,
                (parent, child, arg("offset", 0), arg("dim1", 0), arg("dim2", 1)),
            )
        if name == "as_strided":
            return self.emit(
                ops.as_strided_scatter.default,
                (parent, child, arg("size"), arg("stride"), arg("storage_offset")),
            )
        if name in {
            "view", "reshape", "_reshape_alias", "flatten", "unflatten", "squeeze",
            "unsqueeze", "view_as", "reshape_as",
        }:
            shape = list(step.source_val.shape)
            return self.emit(ops.reshape.default, (child, shape))
        if name in {"alias", "detach", "contiguous", "lift_fresh", "resolve_conj",
                    "resolve_neg", "positive", "_autocast_to_full_precision",
                    "_autocast_to_reduced_precision", "to", "pin_memory"}:
            return child
        if name in {"transpose", "swapaxes", "swapdims"}:
            dims = [v for a, v in zip(schema.arguments[1:], step.args[1:])]
            return self.emit(ops.transpose.int, (child, self.map(dims[0]), self.map(dims[1])))
        if name in {"t", "numpy_T", "mT", "adjoint", "mH"}:
            if name in {"mH", "adjoint"}:
                child = self.emit(ops.conj.default, (child,))
            if name == "t" or name == "numpy_T":
                ndim = step.source_val.dim()
                if ndim < 2:
                    return child
                perm = list(reversed(range(ndim)))
                return self.emit(ops.permute.default, (child, perm))
            return self.emit(ops.transpose.int, (child, -2, -1))
        if name == "permute":
            dims = [int(d) for d in self.map(arg("dims"))]
            ndim = len(dims)
            inverse = [0] * ndim
            for position, dim in enumerate(dims):
                inverse[dim % ndim] = position
            return self.emit(ops.permute.default, (child, inverse))
        if name in {"movedim", "moveaxis"}:
            return self.emit(step.target, (child, self.map(arg("destination")), self.map(arg("source"))))
        if name == "narrow":
            start = self.map(arg("start"))
            length = self.map(arg("length"))
            dim = self.map(arg("dim"))
            return self.emit(ops.slice_scatter.default, (parent, child, dim, start, start + length, 1))
        if name in {"_conj", "conj"}:
            return self.emit(ops._conj.default, (child,))
        if name == "_neg_view":
            return self.emit(ops.neg.default, (child,))
        if name == "real":
            imag = self.emit(ops.imag.default, (parent,))
            return self.emit(ops.complex.default, (child, imag))
        if name == "imag":
            real = self.emit(ops.real.default, (parent,))
            return self.emit(ops.complex.default, (real, child))
        if name == "view_as_real":
            return self.emit(ops.view_as_complex.default, (child,))
        if name == "view_as_complex":
            return self.emit(ops.view_as_real.default, (child,))
        # Any other strided view: write through its concrete geometry.
        view = step.val
        base = step.source_val
        if view is None or base is None:
            raise FunctionalizationError(f"cannot write through view operator {step.target}")
        offset = view.storage_offset() - base.storage_offset()
        return self.emit(
            ops.as_strided_scatter.default,
            (parent, child, list(view.shape), list(view.stride()), offset),
        )

    def inverse_getitem(self, step: _Step, parent: Any, child: Any) -> Any:
        ops = _ops()
        producer = step.target  # the multi-output view step
        schema = producer.target._schema
        pieces = producer.val
        base = producer.source_val
        if schema.name == "unbind":
            dim = _arg(schema, producer, "dim", 0)
            return self.emit(ops.select_scatter.default, (parent, child, dim, step.index))
        # split / chunk / tensor_split family: slices along one dimension.
        dim = _arg(schema, producer, "dim", 0)
        dim = dim % base.dim()
        start = sum(int(piece.shape[dim]) for piece in pieces[: step.index])
        length = int(pieces[step.index].shape[dim])
        return self.emit(ops.slice_scatter.default, (parent, child, dim, start, start + length, 1))

    # -- nodes ---------------------------------------------------------------

    def run(self) -> GraphModule:
        for node in self.gm.graph.nodes:
            self.processing = node if node.op != "output" else None
            handler = getattr(self, f"do_{node.op}")
            handler(node)
        return GraphModule(self.gm.root, self.graph)

    def do_placeholder(self, node: Node) -> None:
        new = self.graph.placeholder(node.target if isinstance(node.target, str) else node.name)
        new.meta.update(node.meta)
        self.env[node] = new
        self.inputs.append(node)
        self.new_base(node, new)

    def do_get_attr(self, node: Node) -> None:
        new = self.graph.get_attr(node.target)
        new.meta.update(node.meta)
        self.env[node] = new
        self.inputs.append(node)
        self.new_base(node, new)

    def do_call_module(self, node: Node) -> None:
        raise FunctionalizationError("call_module nodes must be inlined before functionalization")

    def do_call_method(self, node: Node) -> None:
        raise FunctionalizationError(
            "functionalization runs on operator-level graphs; found a method call"
        )

    def do_call_function(self, node: Node) -> None:
        target = node.target
        if target is operator.getitem:
            return self.do_getitem(node)
        schema = _schema(target)
        if schema is None:
            new = self.emit(target, self.map(node.args), self.map(node.kwargs), name=node.name)
            new.meta.update(node.meta)
            self.env[node] = new
            self.new_base(node, new)
            return
        mutated = schema.mutated_arguments()
        if mutated:
            return self.do_mutation(node, schema, mutated)
        new = self.emit(target, self.map(node.args), self.map(node.kwargs))
        new.meta.update(node.meta)
        self.env[node] = new
        source = _view_source(schema, node) if schema._is_view_op() else None
        if source is not None and not _shares_storage(node, source):
            # Maybe-aliasing operators (reshape, contiguous, to) returned a
            # copy in this trace: the result is a value of its own.
            source = None
        if source is None:
            self.new_base(node, new)
            return
        base, chain = self.base_of[source]
        step = _Step(
            target,
            tuple(_SELF if a is source else a for a in node.args),
            tuple((k, _SELF if v is source else v) for k, v in node.kwargs.items()),
            val=node.meta.get("val"),
            source_val=source.meta.get("val"),
        )
        self.base_of[node] = (base, chain + (step,))
        self.cache[(node, self.version[base])] = new
        if len(schema.returns) == 1 and "[]" in str(schema.returns[0].type):
            self.list_views.add(node)

    def do_getitem(self, node: Node) -> None:
        source, index = node.args
        alias = getattr(self, "_aliased_results", {}).get((source, index))
        remap = getattr(self, "_hop_index", {}).get(source)
        if remap is not None:
            if alias is not None:
                # The return aliases a mutated argument: read its new value.
                self.env[node] = self.current(alias)
                self.base_of[node] = self.base_of[alias]
                return
            index = remap[index]
        new = self.emit(operator.getitem, (self.current(source), index), name=node.name)
        new.meta.update(node.meta)
        self.env[node] = new
        if source in self.list_views:
            base, chain = self.base_of[source]
            step = _Step(
                chain[-1], (), (), index=index, val=node.meta.get("val"),
                source_val=chain[-1].source_val,
            )
            self.base_of[node] = (base, chain + (step,))
            self.cache[(node, self.version[base])] = new
            return
        alias = getattr(self, "_aliased_results", {}).get((source, index))
        if alias is not None:
            self.base_of[node] = self.base_of[alias]
            return
        self.new_base(node, new)

    def do_mutation(self, node: Node, schema: Any, mutated) -> None:
        by_name = {a.name: a for a in schema.arguments}
        values: dict[str, Any] = {}
        for argument, value in zip(schema.arguments, node.args):
            values[argument.name] = value
        values.update(node.kwargs)
        targets = [values[a.name] for a in mutated]
        functional = _functional_variant(node.target)
        if functional is not None:
            out_names = {a.name for a in schema.arguments if a.is_out}
            args = [self.map(values[a.name]) for a in schema.arguments
                    if a.name not in out_names and not a.kwarg_only and a.name in values]
            kwargs = {a.name: self.map(values[a.name]) for a in schema.arguments
                      if a.name not in out_names and a.kwarg_only and a.name in values}
            result = self.emit(functional, tuple(args), kwargs)
            result.meta.update(node.meta)
            if len(targets) == 1:
                new_values = [result]
            else:
                new_values = [
                    self.emit(operator.getitem, (result, i), name="getitem")
                    for i in range(len(targets))
                ]
        else:
            # No functional overload: one auto_functionalized node runs the
            # mutation on private copies and returns (*outputs, *new values).
            from tensorplay._higher_order_ops.auto_functionalize import (
                auto_functionalized,
                returns_without_aliases,
            )

            call_kwargs = {
                a.name: self.map(values[a.name]) if a.name in values else a.default_value
                for a in schema.arguments
            }
            result = self.emit(auto_functionalized, (node.target,), call_kwargs,
                               name="auto_functionalized")
            result.meta.update(node.meta)
            kept = returns_without_aliases(node.target)
            if not hasattr(self, "_hop_index"):
                self._hop_index = {}
            self._hop_index[node] = {orig: pos for pos, orig in enumerate(kept)}
            new_values = [
                self.emit(operator.getitem, (result, len(kept) + i), name="getitem")
                for i in range(len(mutated))
            ]
            if len(schema.returns) == 1:
                # The single return is either a fresh output or an alias of
                # a mutated argument, whose new value then stands for it.
                result = (self.emit(operator.getitem, (result, 0), name="getitem")
                          if kept else result)
        for old, value in zip(targets, new_values):
            if isinstance(old, Node):
                self.write(old, value)
            elif isinstance(old, (list, tuple)):
                for index, element in enumerate(old):
                    if isinstance(element, Node):
                        self.write(element, self.emit(operator.getitem, (value, index), name="getitem"))
        self.env[node] = result
        # Returns annotated as writes alias the mutated argument.
        aliases = {}
        for index, ret in enumerate(schema.returns):
            if ret.alias_info is None or not ret.alias_info.is_write:
                continue
            for argument in mutated:
                if argument.alias_info.before_set & ret.alias_info.before_set:
                    aliases[index] = values[argument.name]
        if len(schema.returns) == 1 and 0 in aliases:
            self.base_of[node] = self.base_of[aliases[0]]
        else:
            if not hasattr(self, "_aliased_results"):
                self._aliased_results = {}
            for index, old in aliases.items():
                self._aliased_results[(node, index)] = old
            if node not in self.base_of:
                self.new_base(node, result)

    def do_output(self, node: Node) -> None:
        outputs = self.map(node.args[0])
        ops = _ops()
        for old in self.inputs:
            if self.version.get(old, 0) and _is_tensor(old.meta.get("val")):
                self.emit(ops.copy_.default, (self.env[old], self.base_value[old]))
        self.graph.output(outputs)


def _is_tensor(value: Any) -> bool:
    import tensorplay

    return isinstance(value, tensorplay.Tensor)


def _storage_key(value: Any) -> Any:
    tensors = value if isinstance(value, (list, tuple)) else [value]
    keys = set()
    for tensor in tensors:
        if _is_tensor(tensor):
            try:
                keys.add(tensor.untyped_storage().data_ptr())
            except Exception:  # noqa: BLE001 - storage-less tensors alias nothing
                pass
    return keys


def _shares_storage(node: Node, source: Node) -> bool:
    produced = _storage_key(node.meta.get("val"))
    origin = _storage_key(source.meta.get("val"))
    if not produced or not origin:
        # Without traced values the annotation is the only evidence.
        return True
    return bool(produced & origin)


def _val(value: Any) -> Any:
    return value.meta.get("val") if isinstance(value, Node) else value


def _arg(schema: Any, step: _Step, name: str, default: Any) -> Any:
    for argument, value in zip(schema.arguments, step.args):
        if argument.name == name:
            return value
    for key, value in step.kwargs:
        if key == name:
            return value
    return default


def _view_source(schema: Any, node: Node) -> Node | None:
    """The input the node's return aliases, per the alias annotations."""

    ret_sets = set()
    for ret in schema.returns:
        if ret.alias_info is not None:
            ret_sets |= ret.alias_info.before_set
    for argument, value in zip(schema.arguments, node.args):
        if argument.alias_info is not None and argument.alias_info.before_set & ret_sets:
            return value if isinstance(value, Node) else None
    for argument in schema.arguments:
        if argument.name in node.kwargs and argument.alias_info is not None:
            if argument.alias_info.before_set & ret_sets:
                value = node.kwargs[argument.name]
                return value if isinstance(value, Node) else None
    return None


def functionalize(gm: GraphModule) -> GraphModule:
    """Return an equivalent graph module free of mutations (see module docs)."""

    return _Functionalizer(gm).run()
