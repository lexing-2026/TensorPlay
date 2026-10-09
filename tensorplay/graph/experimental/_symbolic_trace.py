"""Capture shape reads and Python decisions as expressions with backed hints."""

from __future__ import annotations

import sympy

from ..tracer import Tracer
from .sym_node import SymInt, SymNode


class SymbolicShapeTracer(Tracer):
    def __init__(self, samples, *, specialize_zero_one=True, decomposition_table=None):
        super().__init__(execute=False)
        self.decomposition_table = dict(decomposition_table or {})
        from tensorplay.export._symbolic_dims import SymbolicDims
        from tensorplay.export.dynamic_shapes import Dim

        spec = {
            name: {
                axis: Dim.AUTO for axis, size in enumerate(value.shape) if not specialize_zero_one or size not in (0, 1)
            }
            for name, value in samples.items()
            if hasattr(value, "shape")
        }
        self.dims = SymbolicDims.from_spec(samples, spec)
        self.static_dims = tuple(
            (name, axis, int(size))
            for name, value in samples.items()
            if hasattr(value, "shape")
            for axis, size in enumerate(value.shape)
            if axis not in spec[name]
        )
        from tensorplay.utils._pytree import tree_leaves

        self.static_params = tuple(
            name
            for name, value in samples.items()
            if not hasattr(value, "shape") and any(hasattr(leaf, "shape") for leaf in tree_leaves(value))
        )
        self.example_run = None
        self._shapes = {}

    def trace(self, root, sample_inputs=None):
        from tensorplay.export._example_run import ExampleRun

        self.example_run = ExampleRun(self.dims, self._node_samples, lambda node: self._graph_attrs[node.target])
        try:
            graph = super().trace(root, sample_inputs)
            graph.meta["symbolic_shapes"] = True
            graph.meta["symbolic_dims"] = self.dims
            graph.meta["static_shape_dims"] = self.static_dims
            graph.meta["static_shape_params"] = self.static_params
            return graph
        finally:
            self.example_run.release()
            self.example_run = None

    def symbolic_shape(self, proxy):
        from tensorplay.export._shape_rules import Shaped

        value = self.example_run.value(proxy.node)
        if not isinstance(value, Shaped):
            return None
        if proxy.node.name not in self._shapes:
            shape = []
            for axis, extent in enumerate(value.extents):
                if isinstance(extent, int):
                    shape.append(extent)
                    continue
                import tensorplay as tp

                item = self.create_proxy("call_function", tp.ops.tp.sym_size.int, (proxy, axis), {})
                self._node_samples[item.node.name] = SymInt(extent, None, int, self.dims.example(extent))
                shape.append(item)
            self._shapes[proxy.node.name] = tuple(shape)
        return self._shapes[proxy.node.name]

    def known_metadata(self, proxy, name):
        from tensorplay.export._shape_rules import Shaped

        value = self.example_run.value(proxy.node)
        if isinstance(value, Shaped) and name in ("dtype", "device", "ndim"):
            return value.rank if name == "ndim" else getattr(value.example, name)
        return None

    def symbolic_gate(self, proxy, kind):
        sample = self._node_samples.get(proxy.node.name)
        if isinstance(sample, SymNode) and sample.hint is not None:
            return self.dims.decide(proxy.node, sample, kind)
        return None

    def create_proxy(self, kind, target, args, kwargs):
        decomposition = self.decomposition_table.get(target) if kind == "call_function" else None
        if decomposition is not None:
            result = decomposition(*args, **kwargs)
            if result is not NotImplemented:
                return result
        proxy = super().create_proxy(kind, target, args, kwargs)
        if kind in ("call_function", "call_method") and self.example_run is not None:
            from tensorplay.export._shape_rules import Shaped
            from ..node import map_aggregate

            leaves = []
            map_aggregate(self.resolve_sample((proxy.node.args, proxy.node.kwargs)), lambda value: leaves.append(value))
            if any(isinstance(value, SymNode) for value in leaves) and not any(
                hasattr(value, "shape") for value in leaves
            ):
                self._execute_node(proxy.node)
            if isinstance(self._node_samples.get(proxy.node.name), SymNode):
                return proxy

            value = self.example_run.record(proxy.node)
            if isinstance(value, sympy.Basic) and value.free_symbols:
                pytype = bool if isinstance(value, sympy.logic.boolalg.Boolean) else int
                from .sym_node import SymBool

                node_type = SymBool if pytype is bool else SymInt
                hint = value.xreplace(self.dims.hints)
                self._node_samples[proxy.node.name] = node_type(
                    value, None, pytype, bool(hint) if pytype is bool else int(hint)
                )
            elif isinstance(value, Shaped):
                proxy.node.meta["val"] = value.example
        return proxy
