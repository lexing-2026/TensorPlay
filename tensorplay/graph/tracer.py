from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Tuple

from . import _utils
from ._utils import (
    GraphCaptureError,
    _capture_disabled,
    _active_tracer,
    _iter_nodes,
    compiler_context,
    gate_outcome,
)
from .graph import Graph
from .graph_module import GraphModule
from .node import Node
from .proxy import Proxy, _apply_preserved_node_meta


def _is_module(value: Any) -> bool:
    return callable(getattr(value, "forward", None)) and callable(
        getattr(value, "named_children", None)
    )

class Tracer:
    """Capture a callable into the canonical graph.

    This is intentionally a frontend primitive.  It is not part of the Stax
    backend and may later be replaced by a frame-evaluation frontend without
    changing the backend contract.

    Args:
        concrete_args: Optional mapping of argument name to a concrete value.
            Listed arguments are specialized away during capture: they do not
            become placeholders of the resulting graph.
        execute: Hybrid execution mode (the D1 trace-resume foundation).
            When true, every recorded node is also executed eagerly on its
            argument sample values, so each proxy carries a concrete sample.
            Python control flow over tensor data (``if x.sum() > 0``) then
            specializes on the evaluated value instead of raising; api.py
            promotes the feeding placeholders into data guards so cached
            specializations are never reused across differing data.  This is
            the execution-tracer path for recording a concrete control-flow
            outcome while retaining the symbolic graph.  Capture cost is one eager pass;
            RNG-consuming regions retain the traced-state behavior of the
            execution tracer.
    """

    def __init__(
        self,
        concrete_args: Optional[Dict[str, Any]] = None,
        *,
        execute: bool = False,
    ) -> None:
        self.graph = Graph()
        self.root: Any = None
        self.signature: Optional[inspect.Signature] = None
        self.execute = bool(execute)
        # node.name -> eagerly evaluated sample value (execute mode).  The
        # placeholder subset duplicates ``_samples`` so one lookup serves
        # every consumer.
        self._node_samples: Dict[str, Any] = {}
        # (node.name, gate) pairs for every sample consumption by Python
        # control flow ("bool"/"int"/"float"/"index"/"iter").  Stamped onto
        # GraphModule.meta as ``data_specializations``; api.py derives the
        # data-guarded placeholder set from these.
        self.data_specializations: list[Tuple[str, str]] = []
        # Node names routed through graph.gate() (UPV-native): their
        # values stay symbolic inside the graph and are excluded from the
        # cache-key tail; plain int()/float() consumption stays baked+keyed.
        self.symbolic_gate_nodes: set[str] = set()
        self.concrete_args: Dict[str, Any] = dict(concrete_args or {})
        # Example values bound to placeholders during capture.  Metadata reads
        # (shape/dtype/device/...) resolve against them so Python control flow
        # can specialize statically; tensor DATA stays symbolic.
        self.sample_inputs: Dict[str, Any] = {}
        self._samples: Dict[str, Any] = {}
        self._graph_attrs: Dict[str, Any] = {}
        self._tensorplay_nested_regions: Dict[Any, Any] = {}
        # Placeholder-name -> attribute reads performed during capture
        # ({"shape", "len", "dtype", ...}).  Feeds dynamic-mode shape guards.
        self.metadata_touches: set[Tuple[str, str]] = set()
        # Qualified module path recorded per ``call_module`` node.  Modules
        # executed twice produce distinct ``path_0``/``path_1`` style entries,
        self.node_to_qualname: Dict[Node, str] = {}
        self._recorded_qualnames: set[str] = set()

    def is_leaf_module(self, module: Any, qualified_name: str) -> bool:
        """Return whether ``module`` should be traced as a single unit.

        The compiler frontend defaults to fully inlined behavior: every child
        module is inlined so backends receive primitive operations.  Frontends
        that need submodule boundaries in the graph (feature extraction)
        override this hook; returning ``True`` makes the tracer emit a
        ``call_module`` node targeting the module's qualified name instead of
        descending into ``forward``.
        """

        del module, qualified_name
        return False

    def path_of_module(self, module: Any) -> str:
        named_modules = getattr(self.root, "named_modules", None)
        if callable(named_modules):
            for name, candidate in named_modules():
                if candidate is module:
                    return name
        raise NameError("module is not installed as a submodule")

    def call_module(
        self,
        module: Any,
        forward: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        qualified_name = self.path_of_module(module)
        if not self.is_leaf_module(module, qualified_name):
            return forward(*args, **kwargs)
        return self.create_proxy("call_module", qualified_name, args, kwargs)

    def _record_call_module(self, node: Node, qualified_name: str) -> None:
        if qualified_name not in self._recorded_qualnames:
            self._recorded_qualnames.add(qualified_name)
            self.node_to_qualname[node] = qualified_name
            return
        suffix = 0
        candidate = f"{qualified_name}_{suffix}"
        while candidate in self._recorded_qualnames:
            suffix += 1
            candidate = f"{qualified_name}_{suffix}"
        self._recorded_qualnames.add(candidate)
        self.node_to_qualname[node] = candidate

    # -- hybrid execution (sample propagation) -------------------------------

    def resolve_sample(self, value: Any) -> Any:
        """Resolve the concrete sample behind a captured value.

        Proxies look up ``_node_samples``; containers recurse (returning
        ``None`` when any element is unresolved); everything else is itself.
        """

        if isinstance(value, Proxy):
            return self._node_samples.get(value.node.name)
        if isinstance(value, Node):
            return self._node_samples.get(value.name)
        if isinstance(value, tuple):
            resolved = [self.resolve_sample(item) for item in value]
            return None if any(item is None for item in resolved) else tuple(resolved)
        if isinstance(value, list):
            resolved = [self.resolve_sample(item) for item in value]
            return None if any(item is None for item in resolved) else resolved
        if isinstance(value, dict):
            resolved = {
                key: self.resolve_sample(item) for key, item in value.items()
            }
            return None if any(item is None for item in resolved.values()) else resolved
        if isinstance(value, slice):
            start = self.resolve_sample(value.start)
            stop = self.resolve_sample(value.stop)
            step = self.resolve_sample(value.step)
            if start is None or stop is None or step is None:
                return None
            return slice(start, stop, step)
        return value

    def _execute_node(self, node: Node) -> None:
        """Eagerly evaluate one recorded node to obtain its sample value.

        Execution is advisory: any failure leaves the node without a sample
        and downstream gates raise GraphCaptureError exactly as they did in
        purely symbolic mode.  Autograd is disabled so capture does not grow
        a backward graph for the sample pass.
        """

        args = node.args
        kwargs = node.kwargs
        if node.op == "get_attr":
            try:
                value = self.root
                for part in node.target.split("."):
                    value = getattr(value, part)
            except AttributeError:
                return
            self._node_samples[node.name] = value
            return
        sample_args = self.resolve_sample(args)
        sample_kwargs = self.resolve_sample(kwargs)
        if sample_args is None or sample_kwargs is None:
            return

        def _run() -> Any:
            if node.op == "call_function":
                if isinstance(node.target, (Proxy, Node)):
                    raise TypeError("dynamically produced callables stay symbolic")
                return node.target(*sample_args, **sample_kwargs)
            if node.op == "call_method":
                receiver = getattr(sample_args[0], node.target)
                return receiver(*sample_args[1:], **sample_kwargs)
            if node.op == "call_module":
                module = self.root
                for part in str(node.target).split("."):
                    module = getattr(module, part)
                return module(*sample_args, **sample_kwargs)
            raise TypeError(f"sample execution unsupported for node kind {node.op!r}")

        try:
            try:
                from tensorplay.autograd.grad_mode import no_grad
            except ImportError:
                value = _run()
            else:
                with no_grad():
                    value = _run()
        except Exception:
            # Advisory: an op that fails eagerly simply stays symbolic.
            return
        self._node_samples[node.name] = value

    def create_proxy(
        self,
        kind: str,
        target: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Proxy:
        if _capture_disabled.get():
            raise GraphCaptureError("graph capture is disabled for this operation")
        proxy = Proxy(self.graph.create_node(kind, target, args, kwargs), self)
        _apply_preserved_node_meta(proxy.node)
        if self.execute and kind != "placeholder":
            self._execute_node(proxy.node)
        return proxy

    def proxy(self, node: Node) -> Proxy:
        """Create the proxy object associated with an existing node."""

        return Proxy(node, self)

    def trace(
        self, root: Any, sample_inputs: Optional[Dict[str, Any]] = None
    ) -> "GraphModule":
        self.root = root
        self.sample_inputs = dict(sample_inputs or {})
        tracer_token = _active_tracer.set(self)
        _utils._TRACE_DEPTH += 1
        try:
            return self._trace_impl(root)
        finally:
            _utils._TRACE_DEPTH -= 1
            _active_tracer.reset(tracer_token)

    def _trace_impl(self, root: Any) -> "GraphModule":
        if _is_module(root):
            function = root.forward
        elif callable(root):
            function = root
        else:
            raise TypeError(f"compile() expected a callable, got {type(root)!r}")

        self.signature = inspect.signature(function)
        parameters = list(self.signature.parameters.values())
        if any(
            parameter.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for parameter in parameters
        ):
            raise GraphCaptureError(
                "varargs and varkw arguments are not supported by this compiler frontend"
            )

        unknown_concrete = set(self.concrete_args) - {
            parameter.name for parameter in parameters
        }
        if unknown_concrete:
            raise GraphCaptureError(
                f"concrete arguments {sorted(unknown_concrete)} are not "
                "parameters of the traced callable"
            )

        values: Dict[str, Any] = {}
        for parameter in parameters:
            if parameter.name in self.concrete_args:
                # Specialized arguments never reach the graph.
                values[parameter.name] = self.concrete_args[parameter.name]
            else:
                placeholder_node = self.graph.placeholder(
                    parameter.name,
                    parameter.annotation
                    if parameter.annotation is not inspect.Parameter.empty
                    else None,
                    default_value=parameter.default,
                )
                sample = self.sample_inputs.get(parameter.name)
                if sample is not None:
                    self._samples[parameter.name] = sample
                    self._node_samples[placeholder_node.name] = sample
                values[parameter.name] = self.proxy(placeholder_node)

        with compiler_context():
            if _is_module(root):
                output = self._trace_module(root, function, parameters, values)
            else:
                output = self._invoke(function, parameters, values)

        self.graph.output(output)
        self.graph.lint()
        # Specialized (concrete) parameters disappear from the graph contract
        # together with their placeholders; every other parameter keeps its
        # kind and default so ``bind_partial().apply_defaults()`` keeps working.
        symbolic_parameters = [
            parameter for parameter in parameters if parameter.name not in self.concrete_args
        ]
        specialized_signature = inspect.Signature(symbolic_parameters)
        graph_module = GraphModule(root, self.graph, specialized_signature)
        graph_module._graph_attrs.update(self._graph_attrs)
        if self._samples:
            graph_module.meta["sample_inputs"] = dict(self._samples)
        if self.metadata_touches:
            graph_module.meta["metadata_touches"] = sorted(self.metadata_touches)
        if self.data_specializations:
            graph_module.meta["data_specializations"] = tuple(
                self.data_specializations
            )
            # Stamped while the graph is still complete: later passes may
            # constant-fold or DCE away the consumed condition subtree, which
            # would make any post-hoc producer walk lose the dependency.
            graph_module.meta["data_guard_params"] = tuple(
                sorted(self._data_guard_params())
            )
            self.validate_capture()
            replay = self._extract_guard_replay()
            if replay is not None:
                graph_module.meta["guard_replay"] = replay
        return graph_module


    def validate_capture(self) -> None:
        """Post-trace invariants (fail fast instead of late unwrap errors).

        Every data-specialization consumer must still exist with a usable
        sample, and every symbolic gate must be resolvable — otherwise the
        error surfaces far from its cause (e.g. at backend lowering or first
        cache lookup) and the "why no sample" hunt spans the whole trace.
        """

        nodes = {node.name: node for node in self.graph.nodes}
        for name, kind in self.data_specializations:
            if name not in nodes:
                raise GraphCaptureError(
                    f"specialization consumer {name!r} ({kind}) vanished "
                    "before validation; capture bookkeeping is inconsistent"
                )
            if kind not in ("int", "float", "index", "iter") and (
                self.resolve_sample(nodes[name]) is None
            ):
                raise GraphCaptureError(
                    f"control-flow gate on {name!r} ({kind}) has no sample; "
                    "the producing subgraph failed eager execution during "
                    "capture, so this branch cannot be specialized"
                )

    def _extract_guard_replay(self) -> Optional[Dict[str, Any]]:
        """Copy the condition subgraph feeding every gate (pre-passes).

        api.py re-evaluates this mini-graph at guard-check time and keys the
        specialization cache on gate outcomes, so two inputs share a compiled
        artifact whenever they take the same branches regardless of raw bytes.
        Extraction must happen before optimization passes: DeadCodeElimination
        removes the (output-unreachable) condition subtree.
        """

        if not self.data_specializations:
            return None
        producers: Dict[str, Node] = {}
        placeholder_nodes: Dict[str, Node] = {}
        for node in self.graph.nodes:
            if node.op == "placeholder":
                placeholder_nodes[node.name] = node
            else:
                producers[node.name] = node

        needed: set[str] = set()
        pending = [name for name, _kind in self.data_specializations]
        while pending:
            name = pending.pop()
            if name in needed:
                continue
            needed.add(name)
            node = producers.get(name)
            if node is not None:
                pending.extend(item.name for item in _iter_nodes(node.args))
                pending.extend(item.name for item in _iter_nodes(node.kwargs))

        mini = Graph()
        mapping: Dict[str, Node] = {}
        for node in self.graph.nodes:
            if node.op != "placeholder" or node.name not in needed:
                continue
            mapping[node.name] = mini.placeholder(node.name)
        for node in self.graph.nodes:
            if node.op == "placeholder" or node.name not in needed:
                continue
            mapping[node.name] = mini.create_node(
                node.op,
                node.target,
                self._remap_guard_args(node.args, mapping),
                self._remap_guard_args(node.kwargs, mapping),
                name=node.name,
            )
        gates = tuple(
            (mapping[name].name, kind) for name, kind in self.data_specializations
        )
        values = tuple(
            gate_outcome(kind, self._node_samples[name])
            for name, kind in self.data_specializations
        )
        outputs = [mapping[name] for name, _kind in self.data_specializations]
        mini.output(outputs[0] if len(outputs) == 1 else tuple(outputs))
        mini.lint()
        return {
            "graph": mini,
            "placeholders": tuple(
                node.name for node in mini.placeholders
            ),
            "gates": gates,
            "values": values,
            "symbolic": tuple(sorted(self.symbolic_gate_nodes)),
        }

    @staticmethod
    def _remap_guard_args(value: Any, mapping: Dict[str, Node]) -> Any:
        if isinstance(value, Node):
            return mapping[value.name]
        if isinstance(value, tuple):
            return tuple(Tracer._remap_guard_args(item, mapping) for item in value)
        if isinstance(value, list):
            return [Tracer._remap_guard_args(item, mapping) for item in value]
        if isinstance(value, dict):
            return {
                key: Tracer._remap_guard_args(item, mapping)
                for key, item in value.items()
            }
        if isinstance(value, slice):
            return slice(
                Tracer._remap_guard_args(value.start, mapping),
                Tracer._remap_guard_args(value.stop, mapping),
                Tracer._remap_guard_args(value.step, mapping),
            )
        return value

    def _data_guard_params(self) -> set[str]:
        """Placeholders whose contents fed a data specialization."""

        producers: Dict[str, set[str]] = {}
        placeholders: set[str] = set()
        for node in self.graph.nodes:
            if node.op == "placeholder":
                placeholders.add(node.name)
                continue
            feeds = {item.name for item in _iter_nodes(node.args)}
            feeds |= {item.name for item in _iter_nodes(node.kwargs)}
            producers[node.name] = feeds
        pending = [name for name, _kind in self.data_specializations]
        guarded: set[str] = set()
        seen: set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            if name in placeholders:
                guarded.add(name)
                continue
            pending.extend(producers.get(name, ()))
        return guarded

    @staticmethod
    def _invoke(
        function: Callable[..., Any],
        parameters: list[inspect.Parameter],
        values: Dict[str, Any],
    ) -> Any:
        """Call a traced function while preserving Python parameter kinds.

        Passing every symbolic parameter positionally breaks keyword-only
        arguments and changes the call contract before the backend ever sees
        the graph.  The graph preserves the signature at this boundary; the
        small explicit dispatcher gives the same behavior for the canonical
        TensorPlay graph.  ``values`` may contain plain Python objects for
        arguments specialized through ``concrete_args`` alongside proxies.
        """

        positional: list[Any] = []
        keyword: dict[str, Any] = {}
        for parameter in parameters:
            value = values[parameter.name]
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional.append(value)
            elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                keyword[parameter.name] = value
            else:
                raise GraphCaptureError(
                    "varargs and varkw arguments are not supported by this compiler frontend"
                )
        return function(*positional, **keyword)

    def _trace_module(
        self,
        root: Any,
        function: Callable[..., Any],
        parameters: list[inspect.Parameter],
        values: Dict[str, Any],
    ) -> Any:
        missing = object()
        patches: list[tuple[Any, str, Any]] = []

        def patch_attribute(module: Any, name: str, value: Any) -> None:
            previous = module.__dict__.get(name, missing)
            module.__dict__[name] = value
            patches.append((module, name, previous))

        def qualified(module_name: str, attribute: str) -> str:
            return f"{module_name}.{attribute}" if module_name else attribute

        try:
            # Child modules are inlined so the backend receives the operations
            # inside them.  ``is_leaf_module`` may
            # opt a child out of inlining; such children become a single
            # ``call_module`` node targeting their qualified path, which is
            # what feature extraction needs to locate submodule outputs.
            leaf_qualnames: set[str] = set()
            for module_name, module in root.named_modules(remove_duplicate=True):
                for child_name, child in module.named_children():
                    qualname = qualified(module_name, child_name)

                    if self.is_leaf_module(child, qualname):
                        leaf_qualnames.add(qualname)

                        def call_module_child(
                            *args: Any,
                            _qualname: str = qualname,
                            **kwargs: Any,
                        ) -> Any:
                            proxy = self.create_proxy(
                                "call_module", _qualname, args, kwargs
                            )
                            self._record_call_module(proxy.node, _qualname)
                            return proxy

                        patch_attribute(module, child_name, call_module_child)
                    else:
                        def inline_child(
                            *args: Any, _child: Any = child, **kwargs: Any
                        ) -> Any:
                            return _child.forward(*args, **kwargs)

                        patch_attribute(module, child_name, inline_child)

            def under_leaf(module_qualname: str) -> bool:
                parts = module_qualname.split(".")
                return any(
                    ".".join(parts[: index + 1]) in leaf_qualnames
                    for index in range(len(parts))
                )

            # Parameters and buffers are graph attributes, not frozen Python
            # constants.  This keeps the compiled graph tied to the live
            # module state and preserves parameter autograd edges.  Parameters
            # below a leaf module stay untouched: the leaf is invoked as a
            # whole and its state must not appear as dangling graph inputs.
            for module_name, module in root.named_modules(remove_duplicate=True):
                if under_leaf(module_name):
                    continue
                for attribute_name, value in (
                    *getattr(module, "_parameters", {}).items(),
                    *getattr(module, "_buffers", {}).items(),
                ):
                    if value is None:
                        continue
                    if not hasattr(value, "shape") or not hasattr(value, "requires_grad"):
                        continue
                    patch_attribute(
                        module,
                        attribute_name,
                        self.create_proxy(
                            "get_attr",
                            qualified(module_name, attribute_name),
                            (),
                            {},
                        ),
                    )

            return self._invoke(function, parameters, values)
        finally:
            for module, name, previous in reversed(patches):
                if previous is missing:
                    module.__dict__.pop(name, None)
                else:
                    module.__dict__[name] = previous

class NodePathTracer(Tracer):
    def is_leaf_module(self, module: Any, qualified_name: str) -> bool:
        del qualified_name
        return next(module.named_children(), None) is None


__all__ = ["NodePathTracer", "Tracer"]
