from __future__ import annotations

import copy
import base64
import contextlib
import functools
import hashlib
import inspect
import types
import itertools
import linecache
import pickle
import sys
import warnings
import weakref
from pathlib import Path
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Callable, Optional

from ._utils import GraphCaptureError, _iter_nodes
from .graph import Graph
from .node import Node


_MISSING = object()


class _HookHandle:
    _next_id = itertools.count()

    def __init__(self, *tables: dict[int, Any]) -> None:
        self.id = next(self._next_id)
        self._table_refs = tuple(weakref.ref(table) for table in tables)

    def remove(self) -> None:
        for table_ref in self._table_refs:
            table = table_ref()
            if table is not None:
                table.pop(self.id, None)

    def __enter__(self) -> "_HookHandle":
        return self

    def __exit__(self, *_: Any) -> None:
        self.remove()


class _EvalCacheLoader:
    """Keep generated source available to inspection and traceback tools."""

    def __init__(self) -> None:
        self.sources: dict[str, str] = {}
        self._counter = itertools.count()

    def cache(
        self,
        source: str,
        globals_dict: dict[str, Any] | None = None,
        code_fields: dict[str, Any] | None = None,
    ) -> str:
        del globals_dict
        filename = str(code_fields.get("co_filename")) if code_fields and code_fields.get("co_filename") else f"<tensorplay-generated-{next(self._counter)}>"
        self.sources[filename] = source
        linecache.cache[filename] = (
            len(source), None, source.splitlines(True), filename
        )
        return filename

    def get_source(self, module_name: str) -> str | None:
        return self.sources.get(module_name)


_loader = _EvalCacheLoader()


def _exec_with_source(
    source: str,
    globals_dict: dict[str, Any],
    code_fields: dict[str, Any] | None = None,
) -> None:
    filename = _loader.cache(source, globals_dict, code_fields)
    exec(compile(source, filename, "exec", dont_inherit=True), globals_dict)


def _method_from_src(
    method_name: str,
    source: str,
    globals_dict: dict[str, Any],
    code_fields: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    namespace = dict(globals_dict)
    _exec_with_source(source, namespace, code_fields)
    try:
        return namespace[method_name]
    except KeyError as exc:
        raise GraphCaptureError(
            f"generated source did not define {method_name!r}"
        ) from exc


def _forward_from_src(
    source: str,
    globals_dict: dict[str, Any],
    code_fields: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    return _method_from_src("forward", source, globals_dict, code_fields)


def _get_attr_via_attr_list(value: Any, parts: list[str]) -> Any:
    current = value
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _lookup_path(value: Any, target: str) -> Any:
    if isinstance(value, dict) and target in value:
        return value[target]
    if not target:
        return value
    return _get_attr_via_attr_list(value, target.split("."))


class _ModuleContainer:
    """Small registration container used only while the package is booting."""

    def __init__(self) -> None:
        self.training = True
        self._modules: dict[str, Any] = {}
        self._parameters: dict[str, Any] = {}
        self._buffers: dict[str, Any] = {}
        self._non_persistent_buffers_set: set[str] = set()

    def add_module(self, name: str, module: Any) -> None:
        self._modules[name] = module

    def __getattr__(self, name: str) -> Any:
        for table_name in ("_parameters", "_buffers", "_modules"):
            table = self.__dict__.get(table_name, {})
            if name in table:
                return table[name]
        raise AttributeError(name)


def _module_type() -> Any:
    try:
        from ..nn.modules.module import Module
    except (ImportError, AttributeError):
        return ()
    return Module


def _is_module(value: Any) -> bool:
    graph_module_type = globals().get("GraphModule", ())
    module_type = _module_type()
    candidates = tuple(
        item for item in (module_type, graph_module_type) if isinstance(item, type)
    )
    return bool(candidates) and isinstance(value, candidates)


def _new_module() -> Any:
    module_type = _module_type()
    if module_type:
        return module_type()
    return _ModuleContainer()


def _init_module_state(instance: Any) -> None:
    object.__setattr__(instance, "training", True)
    object.__setattr__(instance, "_modules", OrderedDict())
    object.__setattr__(instance, "_parameters", OrderedDict())
    object.__setattr__(instance, "_buffers", OrderedDict())
    object.__setattr__(instance, "_non_persistent_buffers_set", set())
    object.__setattr__(instance, "_backward_pre_hooks", OrderedDict())
    object.__setattr__(instance, "_backward_hooks", OrderedDict())
    object.__setattr__(instance, "_is_full_backward_hook", None)
    object.__setattr__(instance, "_forward_pre_hooks", OrderedDict())
    object.__setattr__(instance, "_forward_hooks", OrderedDict())
    object.__setattr__(instance, "_forward_pre_hooks_with_kwargs", {})
    object.__setattr__(instance, "_forward_hooks_with_kwargs", {})
    object.__setattr__(instance, "_forward_hooks_always_called", {})
    object.__setattr__(instance, "_state_dict_hooks", OrderedDict())
    object.__setattr__(instance, "_state_dict_pre_hooks", OrderedDict())
    object.__setattr__(instance, "_load_state_dict_pre_hooks", OrderedDict())
    object.__setattr__(instance, "_load_state_dict_post_hooks", OrderedDict())
    object.__setattr__(instance, "_compiled_call_impl", None)


def _module_repr(instance: Any) -> str:
    name = instance._get_name()
    children = []
    for child_name, child in instance.named_children():
        children.append(f"  ({child_name}): {child!r}")
    if not children:
        return f"{name}()"
    return f"{name}(\n" + "\n".join(children) + "\n)"


def _is_tensor(value: Any) -> bool:
    try:
        import tensorplay

        tensor_type = getattr(tensorplay, "Tensor", ())
        return bool(tensor_type) and isinstance(value, tensor_type)
    except (ImportError, TypeError):
        return False


def _is_parameter(value: Any) -> bool:
    try:
        from ..nn.parameter import Parameter

        return isinstance(value, Parameter)
    except ImportError:
        return False


def _is_buffer(value: Any) -> bool:
    try:
        from ..nn.parameter import Buffer

        return isinstance(value, Buffer)
    except ImportError:
        return False


def _module_hook_tables() -> tuple[dict[int, Any], ...]:
    try:
        from ..nn.modules import module as module_impl

        return (
            module_impl._global_forward_pre_hooks,
            module_impl._global_forward_hooks,
            module_impl._global_forward_hooks_with_kwargs,
            module_impl._global_forward_hooks_always_called,
            module_impl._global_buffer_registration_hooks,
            module_impl._global_module_registration_hooks,
            module_impl._global_parameter_registration_hooks,
        )
    except (ImportError, AttributeError):
        return ()


def _assign_value(destination: Any, field: str, value: Any, persistent: bool = True) -> None:
    if _is_tensor(value) and not _is_parameter(value):
        register_buffer = getattr(destination, "register_buffer", None)
        if callable(register_buffer):
            register_buffer(field, value, persistent=persistent)
            return
    setattr(destination, field, value)


def _copy_attr(from_obj: Any, to_module: Any, target: str) -> None:
    if not isinstance(target, str) or not target:
        raise ValueError("attribute target must be a non-empty string")
    parts = target.split(".")
    source = from_obj
    destination = to_module
    for part in parts[:-1]:
        source = _lookup_path(source, part)
        current = getattr(destination, part, _MISSING)
        if current is source:
            return
        if current is _MISSING or current is None:
            current = _new_module()
            destination.add_module(part, current)
        if not _is_module(current):
            raise RuntimeError(
                f"attribute {part!r} in target {target!r} is not a module"
            )
        destination = current
    field = parts[-1]
    value = _lookup_path(source, field)
    non_persistent = getattr(source, "_non_persistent_buffers_set", ())
    _assign_value(destination, field, value, field not in non_persistent)


def _assign_attr(value: Any, destination: Any, target: str) -> None:
    if not isinstance(target, str) or not target:
        raise ValueError("attribute target must be a non-empty string")
    parts = target.split(".")
    holder = destination
    for part in parts[:-1]:
        current = getattr(holder, part, _MISSING)
        if current is _MISSING or current is None:
            current = _new_module()
            if hasattr(holder, "add_module"):
                holder.add_module(part, current)
            else:
                setattr(holder, part, current)
        if not _is_module(current):
            raise RuntimeError(
                f"attribute {part!r} in target {target!r} is not a module"
            )
        holder = current
    _assign_value(holder, parts[-1], value)


def _get_attr(model: Any, attr_name: str) -> Any:
    return _lookup_path(model, attr_name)


def _del_attr(model: Any, attr_name: str) -> None:
    if not attr_name:
        raise AttributeError("cannot delete an empty attribute name")
    parts = attr_name.split(".")
    holder = _get_attr_via_attr_list(model, parts[:-1]) if len(parts) > 1 else model
    delattr(holder, parts[-1])


def _has_attr(model: Any, attr_name: str) -> bool:
    try:
        _lookup_path(model, attr_name)
    except (AttributeError, KeyError, IndexError, TypeError):
        return False
    return True


def _format_import_statement(name: str, value: Any, importer: Any = None) -> str:
    if isinstance(value, str):
        return f"{name} = {value!r}"
    if importer is not None:
        try:
            module_name, attr_name = importer.get_name(value)
            return f"from {module_name} import {attr_name} as {name}"
        except Exception:
            pass
    module_name = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", getattr(value, "__name__", None))
    if not module_name or not qualname or "<locals>" in str(qualname):
        raise TypeError(f"cannot import generated global {name!r}")
    return f"from {module_name} import {str(qualname).split('.', 1)[0]} as {name}"


def _format_import_block(globals_dict: dict[str, Any], importer: Any = None) -> str:
    statements = {
        _format_import_statement(name, value, importer)
        for name, value in globals_dict.items()
    }
    return "\n".join(sorted(statements)) + ("\n" if statements else "")


def _metadata_hash(code: str, metadata: Mapping[int, Mapping[str, Any]]) -> str:
    payload = repr((code, sorted((int(key), dict(value)) for key, value in metadata.items())))
    return base64.b32encode(hashlib.sha256(payload.encode()).digest())[:51].decode().lower()


def _rebuild_graph_module(
    root: Any,
    graph: Graph,
    signature: inspect.Signature | None,
    state: dict[str, Any],
) -> "GraphModule":
    result = GraphModule(root, graph, signature)
    result.__dict__.update(state)
    object.__setattr__(result, "_graph", graph)
    graph.owning_module = result
    object.__setattr__(result, "_compiled_forward", None)
    object.__setattr__(result, "_compiled_impl", None)
    result.__dict__.pop("forward", None)
    result.recompile()
    return result


def reduce_graph_module(body: dict[str, Any], import_block: str = "") -> "GraphModule":
    if import_block:
        _exec_with_source(import_block, {})
    root = body.pop("_root", None)
    graph = body.pop("_graph")
    signature = body.pop("signature", None)
    return _rebuild_graph_module(root, graph, signature, body)


def reduce_package_graph_module(
    importer: Any, body: dict[str, Any], generated_module_name: str
) -> "GraphModule":
    module = importer.import_module(generated_module_name)
    if not callable(getattr(module, "forward", None)):
        raise RuntimeError(
            f"generated package module {generated_module_name!r} has no forward"
        )
    root = body.pop("_root", None)
    graph = body.pop("_graph")
    signature = body.pop("signature", None)
    return _rebuild_graph_module(root, graph, signature, body)


class GraphModule:
    """Executable graph wrapper passed to compiler backends."""

    def __init__(
        self, root: Any, graph: Graph, signature: inspect.Signature | None = None
    ) -> None:
        self.root = root
        self.graph = graph
        self.graph.owning_module = self
        self.signature = signature
        self.code = graph.python_code()
        self.meta: dict[str, Any] = {}
        self._graph_attrs: dict[str, Any] = {}
        self._compiled_forward: Optional[Callable[..., Any]] = None
        self._compiled_targets: dict[str, Any] = {}
        self._compiled_constants: list[Any] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._compiled_forward is not None:
            return self._compiled_forward(*args, **kwargs)
        return self._interpret(*args, **kwargs)

    def _invalidate_compiled_executor(self) -> None:
        """Drop the generated executor (graph surgery invalidated it).

        ``recompile`` binds the compiled program as the instance
        ``forward``; both bindings must go so calls fall back to the live
        interpreter until the module explicitly recompiles.
        """

        object.__setattr__(self, "_compiled_forward", None)
        self.__dict__.pop("forward", None)

    def recompile(self) -> Callable[..., Any]:
        """Generate an explicit Python executor for custom backend use.

        This is useful for frontend tests and deliberately opt-in fallback
        backends.  A performance backend must not use this executor: the
        ResNet benchmark requests ``strict_native`` and rejects it outright.
        """

        self._compiled_targets = {}
        self._compiled_constants = []
        lines = ["def _compiled(self, *args, **kwargs):"]
        if self.signature is None:
            lines.append("    _bound = None")
        else:
            lines.append("    _bound = self.signature.bind_partial(*args, **kwargs)")
            lines.append("    _bound.apply_defaults()")

        for node in self.graph.placeholders:
            if self.signature is None:
                index = self.graph.placeholders.index(node)
                default = node.args[0] if node.args else inspect.Parameter.empty
                if default is inspect.Parameter.empty:
                    lines.append(
                        f"    {node.name} = args[{index}] if len(args) > {index} "
                        f"else kwargs[{node.name!r}]"
                    )
                else:
                    lines.append(
                        f"    {node.name} = args[{index}] if len(args) > {index} "
                        f"else kwargs.get({node.name!r}, {self._expr(default)})"
                    )
            else:
                parameter_name = node.target if isinstance(node.target, str) else node.name
                lines.append(
                    f"    {node.name} = _bound.arguments[{parameter_name!r}]"
                )

        for node in self.graph.nodes:
            if node.op in {"placeholder", "output"}:
                continue
            if node.op == "call_function":
                target_name = f"_target_{len(self._compiled_targets)}"
                self._compiled_targets[target_name] = self._resolve_target(node.target)
                args_expr = ", ".join(self._expr(arg) for arg in node.args)
                kwargs_expr = self._kwargs_expr(node.kwargs)
                call = f"{target_name}({args_expr}"
                if kwargs_expr:
                    call += f", {kwargs_expr}"
                call += ")"
                lines.append(f"    {node.name} = {call}")
            elif node.op == "call_method":
                resolved = list(node.args)
                if not resolved:
                    raise GraphCaptureError("call_method node has no receiver")
                receiver = self._expr(resolved[0])
                method_args = ", ".join(self._expr(arg) for arg in resolved[1:])
                kwargs_expr = self._kwargs_expr(node.kwargs)
                call = f"{receiver}.{node.target}({method_args}"
                if kwargs_expr:
                    if method_args:
                        call += ", "
                    call += kwargs_expr
                call += ")"
                lines.append(f"    {node.name} = {call}")
            elif node.op == "call_module":
                args_expr = ", ".join(self._expr(arg) for arg in node.args)
                kwargs_expr = self._kwargs_expr(node.kwargs)
                call = f"self._get_attr({node.target!r})({args_expr}"
                if kwargs_expr:
                    if args_expr:
                        call += ", "
                    call += kwargs_expr
                call += ")"
                lines.append(f"    {node.name} = {call}")
            elif node.op == "get_attr":
                lines.append(f"    {node.name} = self._get_attr({node.target!r})")
            else:
                raise GraphCaptureError(f"unsupported graph node kind: {node.op}")

        output_nodes = self.graph.outputs
        if not output_nodes:
            raise GraphCaptureError("graph has no output node")
        lines.append(f"    return {self._expr(output_nodes[-1].args[0])}")
        lines = self.graph._apply_code_transformers(lines)
        source = "\n".join(lines) + "\n"

        namespace: dict[str, Any] = {}
        exec(compile(source, "<tensorplay-compiled-graph>", "exec"), namespace)
        for name, target in self._compiled_targets.items():
            namespace[name] = target
        function = namespace["_compiled"]
        self._compiled_forward = types.MethodType(function, self)
        self.code = source
        return self.forward

    def _expr(self, value: Any) -> str:
        if isinstance(value, Node):
            return value.name
        if isinstance(value, tuple):
            items = ", ".join(self._expr(item) for item in value)
            if len(value) == 1:
                items += ","
            return f"({items})"
        if isinstance(value, list):
            return "[" + ", ".join(self._expr(item) for item in value) + "]"
        if isinstance(value, dict):
            items = ", ".join(
                f"{key!r}: {self._expr(item)}" for key, item in value.items()
            )
            return "{" + items + "}"
        if isinstance(value, slice):
            return (
                f"slice({self._expr(value.start)}, {self._expr(value.stop)}, "
                f"{self._expr(value.step)})"
            )
        if isinstance(value, range):
            return f"range({self._expr(value.start)}, {self._expr(value.stop)}, {self._expr(value.step)})"
        if isinstance(value, set):
            return "{" + ", ".join(self._expr(item) for item in value) + "}"
        if isinstance(value, frozenset):
            return "frozenset({" + ", ".join(self._expr(item) for item in value) + "})"
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return repr(value)
        index = len(self._compiled_constants)
        self._compiled_constants.append(value)
        return f"self._compiled_constants[{index}]"

    def _kwargs_expr(self, kwargs: dict[str, Any]) -> str:
        if not kwargs:
            return ""
        return "**{" + ", ".join(
            f"{key!r}: {self._expr(value)}" for key, value in kwargs.items()
        ) + "}"

    def _interpret(self, *args: Any, _record_meta: bool = False, **kwargs: Any) -> Any:
        try:
            if self.signature is None:
                bound_arguments = {}
                for index, node in enumerate(self.graph.placeholders):
                    parameter_name = node.target if isinstance(node.target, str) else node.name
                    if index < len(args):
                        bound_arguments[parameter_name] = args[index]
                    elif parameter_name in kwargs:
                        bound_arguments[parameter_name] = kwargs[parameter_name]
                    elif node.args:
                        bound_arguments[parameter_name] = node.args[0]
                    else:
                        raise TypeError(
                            f"missing required graph input: {parameter_name}"
                        )
                unknown = set(kwargs) - {
                    node.target if isinstance(node.target, str) else node.name
                    for node in self.graph.placeholders
                }
                if unknown:
                    raise TypeError(
                        f"unexpected graph inputs: {', '.join(sorted(unknown))}"
                    )
            else:
                bound = self.signature.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                bound_arguments = bound.arguments
        except TypeError:
            raise

        def keep(node: Node, value: Any) -> Any:
            env[node] = value
            if _record_meta:
                node.meta["val"] = value
                shape = getattr(value, "shape", None)
                if shape is not None:
                    try:
                        node.meta["tensor_shape"] = tuple(int(d) for d in shape())
                    except (TypeError, ValueError):
                        try:
                            node.meta["tensor_shape"] = tuple(int(d) for d in shape)
                        except (TypeError, ValueError):
                            pass
            return value

        env: dict[Node, Any] = {}
        for node in self.graph.placeholders:
            parameter_name = node.target if isinstance(node.target, str) else node.name
            if parameter_name not in bound_arguments:
                raise TypeError(f"missing required compiler input: {parameter_name}")
            keep(node, bound_arguments[parameter_name])

        for node in self.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "call_function":
                target = self._resolve_target(node.target)
                keep(node, target(
                    *self._resolve(node.args, env),
                    **self._resolve(node.kwargs, env),
                ))
            elif node.op == "call_method":
                resolved_args = self._resolve(node.args, env)
                receiver, *method_args = resolved_args
                keep(node, getattr(receiver, node.target)(*method_args, **self._resolve(node.kwargs, env)))
            elif node.op == "call_module":
                module = self._get_attr(node.target)
                keep(node, module(
                    *self._resolve(node.args, env),
                    **self._resolve(node.kwargs, env),
                ))
            elif node.op == "get_attr":
                keep(node, self._get_attr(node.target))
            elif node.op == "output":
                return self._resolve(node.args[0], env)
            else:
                raise GraphCaptureError(f"unsupported graph node kind: {node.op}")

        raise GraphCaptureError("graph has no output node")

    @staticmethod
    def _resolve(value: Any, env: dict[Node, Any]) -> Any:
        if isinstance(value, Node):
            return env[value]
        if isinstance(value, tuple):
            return tuple(GraphModule._resolve(item, env) for item in value)
        if isinstance(value, list):
            return [GraphModule._resolve(item, env) for item in value]
        if isinstance(value, dict):
            return {key: GraphModule._resolve(item, env) for key, item in value.items()}
        if isinstance(value, slice):
            return slice(
                GraphModule._resolve(value.start, env),
                GraphModule._resolve(value.stop, env),
                GraphModule._resolve(value.step, env),
            )
        if isinstance(value, range):
            return (
                f"range({self._expr(value.start)}, {self._expr(value.stop)}, "
                f"{self._expr(value.step)})"
            )
        return value

    def _get_attr(self, target: str) -> Any:
        if target == "":
            return self.root
        if target in self._graph_attrs:
            return self._graph_attrs[target]
        value = self.root
        for part in target.split("."):
            value = getattr(value, part)
        return value

    def register_graph_attr(self, target: str, value: Any) -> None:
        """Register a graph-owned runtime object under a qualified name."""

        if not isinstance(target, str) or not target or "." in target:
            raise ValueError("graph-owned attribute names must be simple identifiers")
        if target in self._graph_attrs:
            raise RuntimeError(f"graph-owned attribute {target!r} is already registered")
        self._graph_attrs[target] = value

    def __getattr__(self, name: str) -> Any:
        root = self.__dict__.get("root")
        if root is not None:
            try:
                return getattr(root, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"{type(self).__name__!s} has no attribute {name!r}"
        )

    def named_children(self) -> Iterator[tuple[str, Any]]:
        root = self.root
        if root is None:
            return
        method = getattr(root, "named_children", None)
        if callable(method):
            yield from method()
            return
        modules = getattr(root, "_modules", None)
        if isinstance(modules, dict):
            for name, module in modules.items():
                if module is not None:
                    yield name, module
        else:
            for name, value in vars(root).items():
                if callable(value) and not name.startswith("_"):
                    yield name, value

    def children(self) -> Iterator[Any]:
        for _, module in self.named_children():
            yield module

    def named_modules(
        self,
        memo: set[int] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Any]]:
        if memo is None:
            memo = set()
        if id(self) in memo:
            return
        if remove_duplicate:
            memo.add(id(self))
        yield prefix, self
        for name, module in self.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            if hasattr(module, "named_modules"):
                yield from module.named_modules(memo, child_prefix, remove_duplicate)
            else:
                if not remove_duplicate or id(module) not in memo:
                    if remove_duplicate:
                        memo.add(id(module))
                    yield child_prefix, module

    def modules(self) -> Iterator[Any]:
        for _, module in self.named_modules():
            yield module

    def get_submodule(self, target: str) -> Any:
        if target == "":
            return self
        current: Any = self
        for atom in target.split("."):
            if not _is_module(current):
                raise AttributeError(
                    f"{type(current).__name__!s} has no submodule {atom!r}"
                )
            try:
                current = getattr(current, atom)
            except AttributeError as exc:
                raise AttributeError(
                    f"{type(current).__name__!s} has no attribute {atom!r}"
                ) from exc
            if not _is_module(current):
                raise AttributeError(f"attribute {atom!r} is not a module")
        return current

    def get_parameter(self, target: str) -> Any:
        parent, _, name = target.rpartition(".")
        module = self.get_submodule(parent)
        parameters = getattr(module, "_parameters", {})
        if name not in parameters or parameters[name] is None:
            raise AttributeError(f"parameter {target!r} is not registered")
        return parameters[name]

    def get_buffer(self, target: str) -> Any:
        parent, _, name = target.rpartition(".")
        module = self.get_submodule(parent)
        buffers = getattr(module, "_buffers", {})
        if name not in buffers or buffers[name] is None:
            raise AttributeError(f"buffer {target!r} is not registered")
        return buffers[name]

    def set_submodule(self, target: str, module: Any, strict: bool = False) -> None:
        if not target:
            raise ValueError("cannot set a submodule without a target")
        if not _is_module(module):
            raise ValueError(f"replacement {module!r} is not a module")
        parent_name, _, name = target.rpartition(".")
        parent = self.get_submodule(parent_name)
        existing = getattr(parent, name, _MISSING)
        if strict and existing is _MISSING:
            raise AttributeError(f"submodule {target!r} does not exist")
        if existing is not _MISSING and not _is_module(existing):
            raise AttributeError(f"attribute {target!r} is not a module")
        parent.add_module(name, module)

    def get_submodule(self, target: str) -> Any:
        if not target:
            return self
        value: Any = self
        for atom in target.split("."):
            if value is self and atom == "root":
                value = self.root
            else:
                value = getattr(value, atom)
        return value

    def add_submodule(self, target: str, module: Any) -> bool:
        if not target or not isinstance(target, str):
            raise ValueError("submodule target must be a non-empty string")
        parent_name, _, name = target.rpartition(".")
        parent = self.root if not parent_name else self._get_attr(parent_name)
        add_module = getattr(parent, "add_module", None)
        if callable(add_module):
            add_module(name, module)
        else:
            setattr(parent, name, module)
        return True

    def delete_submodule(self, target: str) -> bool:
        if not target or not isinstance(target, str):
            raise ValueError("submodule target must be a non-empty string")
        parent_name, _, name = target.rpartition(".")
        parent = self.root if not parent_name else self._get_attr(parent_name)
        modules = getattr(parent, "_modules", None)
        if isinstance(modules, dict) and name in modules:
            del modules[name]
        if not hasattr(parent, name):
            return False
        try:
            delattr(parent, name)
        except AttributeError:
            return False
        return True

    def delete_all_unused_submodules(self) -> None:
        used = {
            node.target
            for node in self.graph.nodes
            if node.op == "call_module" and isinstance(node.target, str)
        }
        for name, _module in list(self.named_children()):
            if name not in used:
                self.delete_submodule(name)

    def print_readable(self, print_output: bool = False) -> str:
        text = self.code if isinstance(self.code, str) else str(self.code)
        if print_output:
            print(text)
        return text

    def __str__(self) -> str:
        return self.print_readable()

    def __repr__(self) -> str:
        return f"GraphModule({self.graph!s})"

    def __deepcopy__(self, memo: dict[int, Any]) -> "GraphModule":
        if id(self) in memo:
            return memo[id(self)]
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key in {"_compiled_forward", "_compiled_targets", "_compiled_constants"}:
                continue
            setattr(result, key, copy.deepcopy(value, memo))
        result._compiled_forward = None
        result._compiled_targets = {}
        result._compiled_constants = []
        graph = result.__dict__.get("graph")
        if graph is not None and getattr(graph, "owning_module", None) is not result:
            graph.owning_module = result
        if result.__dict__.get("code") is None:
            result.code = graph.python_code() if graph is not None else ""
        return result

    @staticmethod
    def _resolve_target(target: Any) -> Any:
        if isinstance(target, Node):
            raise GraphCaptureError("calling a dynamically produced function is unsupported")
        return target

    # The definitions below are deliberately kept on the class so old instances
    # created before a source refresh acquire the same lifecycle as new ones.
    def __init__(
        self,
        root: Any,
        graph: Graph,
        signature: inspect.Signature | None = None,
        *,
        class_name: str = "GraphModule",
    ) -> None:
        _init_module_state(self)
        if not isinstance(graph, Graph):
            raise AssertionError(f"Expected a Graph instance, but got {type(graph)}")
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_graph", None)
        object.__setattr__(self, "_code", "")
        object.__setattr__(self, "_python_code", None)
        object.__setattr__(self, "_compiled_impl", None)
        object.__setattr__(self, "_compiled_forward", None)
        object.__setattr__(self, "_compiled_constants", [])
        object.__setattr__(self, "_is_replica", False)
        self.signature = signature
        self._class_name = class_name
        self.meta: dict[str, Any] = {}
        self._graph_attrs: dict[str, Any] = {}
        self.shape_env: Any = None
        self._replace_hooks: list[Callable[[Node, str, Node], object]] = []
        self._create_node_hooks: list[Callable[[Node], object]] = []
        self._erase_node_hooks: list[Callable[[Node], object]] = []
        self._deepcopy_hooks: list[Callable[["GraphModule"], object]] = []
        self._tracer_cls = getattr(graph, "_tracer_cls", None)
        self._tracer_extras = dict(getattr(graph, "_tracer_extras", {}) or {})
        self._copy_graph_references(root, graph)
        self.graph = graph

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_with_hooks(*args, **kwargs)

    @property
    def root(self) -> Any:
        return self.__dict__.get("_root")

    @root.setter
    def root(self, value: Any) -> None:
        object.__setattr__(self, "_root", value)

    @property
    def graph(self) -> Graph:
        graph = self.__dict__.get("_graph")
        if graph is None:
            raise AttributeError("graph has not been initialized")
        return graph

    @graph.setter
    def graph(self, value: Graph) -> None:
        if not isinstance(value, Graph):
            raise AssertionError(f"Expected a Graph instance, but got {type(value)}")
        object.__setattr__(self, "_graph", value)
        value.owning_module = self
        self._tracer_cls = getattr(value, "_tracer_cls", None)
        self._tracer_extras = dict(getattr(value, "_tracer_extras", {}) or {})
        self.recompile()

    @property
    def code(self) -> str:
        code = self.__dict__.get("_code")
        if not code:
            raise RuntimeError("generated code is not available")
        return code

    @property
    def _boxed_call(self) -> bool:
        try:
            from .graph import _BoxedCodeGen

            return isinstance(self.graph._codegen, _BoxedCodeGen)
        except ImportError:
            return False

    def _copy_graph_references(self, root: Any, graph: Graph) -> None:
        if root is None:
            return
        targets = sorted(
            {
                str(node.target)
                for node in graph.nodes
                if node.op in {"get_attr", "call_module"}
            },
            key=lambda item: (item.count("."), item),
        )
        if isinstance(root, dict):
            for target in targets:
                if target not in root:
                    raise RuntimeError(
                        f"graph node references target {target!r}, but no value was supplied"
                    )
                _assign_attr(root[target], self, target)
            return
        if _is_module(root):
            if hasattr(root, "training"):
                self.training = bool(root.training)
            for target in targets:
                try:
                    _copy_attr(root, self, target)
                except (AttributeError, KeyError) as exc:
                    raise RuntimeError(
                        f"graph node references missing attribute {target!r}"
                    ) from exc

    def __getattr__(self, name: str) -> Any:
        for table_name in ("_parameters", "_buffers", "_modules"):
            table = self.__dict__.get(table_name, {})
            if name in table:
                return table[name]
        root = self.__dict__.get("_root", _MISSING)
        if root is not _MISSING and root is not None:
            try:
                return getattr(root, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"{type(self).__name__!s} has no attribute {name!r}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or "_modules" not in self.__dict__:
            object.__setattr__(self, name, value)
            return

        def remove_from(*containers: Any) -> None:
            for container in containers:
                if isinstance(container, dict):
                    container.pop(name, None)
                elif isinstance(container, set):
                    container.discard(name)

        parameters = self.__dict__.get("_parameters", {})
        buffers = self.__dict__.get("_buffers", {})
        modules = self.__dict__.get("_modules", {})
        if _is_parameter(value):
            remove_from(self.__dict__, buffers, modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
            return
        if _is_module(value):
            remove_from(self.__dict__, parameters, buffers, self._non_persistent_buffers_set)
            self.add_module(name, value)
            return
        if _is_buffer(value):
            remove_from(self.__dict__, parameters, modules)
            self.register_buffer(name, value, persistent=bool(getattr(value, "persistent", True)))
            return
        if name in parameters:
            if value is not None:
                raise TypeError(f"cannot assign {type(value)!r} as parameter {name!r}")
            self.register_parameter(name, None)
            return
        if name in modules:
            if value is not None and not _is_module(value):
                raise TypeError(f"cannot assign {type(value)!r} as module {name!r}")
            modules[name] = value
            return
        if name in buffers:
            if value is not None and not _is_tensor(value):
                raise TypeError(f"cannot assign {type(value)!r} as buffer {name!r}")
            buffers[name] = value
            return
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._parameters:
            del self._parameters[name]
            return
        if name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
            return
        if name in self._modules:
            del self._modules[name]
            return
        object.__delattr__(self, name)

    def _has_local_attribute(self, name: str) -> bool:
        # Registration collision checks must not traverse the root fallback
        # in __getattr__: while _copy_graph_references populates this module
        # from the root, the root exposes every same-named attribute, so a
        # plain hasattr() would report "already exists" for the first
        # parameter copy.  Only real local slots count as collisions.
        return (
            name in self.__dict__
            or name in self._parameters
            or name in self._buffers
            or name in self._modules
        )

    def add_module(self, name: str, module: Any) -> None:
        if not isinstance(name, str):
            raise TypeError(f"module name should be a string, got {type(name)!r}")
        if not name or "." in name:
            raise KeyError(f"invalid module name {name!r}")
        if module is not None and not _is_module(module):
            raise TypeError(f"{type(module)!r} is not a module")
        if self._has_local_attribute(name) and name not in self._modules:
            raise KeyError(f"attribute {name!r} already exists")
        for hook in _module_hook_tables()[5:6]:
            for registration_hook in hook.values():
                replacement = registration_hook(self, name, module)
                if replacement is not None:
                    module = replacement
        self._modules[name] = module

    register_module = add_module

    def register_parameter(self, name: str, parameter: Any) -> None:
        if not isinstance(name, str):
            raise TypeError(f"parameter name should be a string, got {type(name)!r}")
        if not name or "." in name:
            raise KeyError(f"invalid parameter name {name!r}")
        if parameter is not None and not _is_parameter(parameter):
            raise TypeError(f"{type(parameter)!r} is not a parameter")
        if self._has_local_attribute(name) and name not in self._parameters:
            raise KeyError(f"attribute {name!r} already exists")
        for hook in _module_hook_tables()[6:7]:
            for registration_hook in hook.values():
                replacement = registration_hook(self, name, parameter)
                if replacement is not None:
                    parameter = replacement
        self._buffers.pop(name, None)
        self._modules.pop(name, None)
        self._non_persistent_buffers_set.discard(name)
        self._parameters[name] = parameter

    def register_buffer(self, name: str, value: Any, persistent: bool = True) -> None:
        if not isinstance(name, str):
            raise TypeError(f"buffer name should be a string, got {type(name)!r}")
        if not name or "." in name:
            raise KeyError(f"invalid buffer name {name!r}")
        if value is not None and not _is_tensor(value):
            raise TypeError(f"{type(value)!r} is not a tensor or None")
        if self._has_local_attribute(name) and name not in self._buffers:
            raise KeyError(f"attribute {name!r} already exists")
        for hook in _module_hook_tables()[4:5]:
            for registration_hook in hook.values():
                replacement = registration_hook(self, name, value)
                if replacement is not None:
                    value = replacement
        self._parameters.pop(name, None)
        self._modules.pop(name, None)
        self._buffers[name] = value
        if persistent:
            self._non_persistent_buffers_set.discard(name)
        else:
            self._non_persistent_buffers_set.add(name)

    def named_children(self) -> Iterator[tuple[str, Any]]:
        seen: set[int] = set()
        for name, module in self._modules.items():
            if module is not None and id(module) not in seen:
                seen.add(id(module))
                yield name, module

    def children(self) -> Iterator[Any]:
        for _, module in self.named_children():
            yield module

    def named_modules(
        self,
        memo: set[int] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Any]]:
        if memo is None:
            memo = set()
        marker = id(self)
        if remove_duplicate and marker in memo:
            return
        if remove_duplicate:
            memo.add(marker)
        yield prefix, self
        for name, module in self.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            method = getattr(module, "named_modules", None)
            if callable(method):
                yield from method(memo, child_prefix, remove_duplicate)
            else:
                yield child_prefix, module

    def modules(self) -> Iterator[Any]:
        for _, module in self.named_modules():
            yield module

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Any]]:
        seen: set[int] = set()
        for name, parameter in self._parameters.items():
            if parameter is not None and (not remove_duplicate or id(parameter) not in seen):
                if remove_duplicate:
                    seen.add(id(parameter))
                yield (f"{prefix}.{name}" if prefix else name), parameter
        if recurse:
            for name, module in self.named_children():
                method = getattr(module, "named_parameters", None)
                if callable(method):
                    yield from method(
                        f"{prefix}.{name}" if prefix else name,
                        recurse=True,
                        remove_duplicate=remove_duplicate,
                    )

    def parameters(self, recurse: bool = True) -> Iterator[Any]:
        for _, parameter in self.named_parameters(recurse=recurse):
            yield parameter

    def buffers(self, recurse: bool = True) -> Iterator[Any]:
        for _, value in self.named_buffers(recurse=recurse):
            yield value

    def named_buffers(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Any]]:
        seen: set[int] = set()
        for name, value in self._buffers.items():
            if value is not None and (not remove_duplicate or id(value) not in seen):
                if remove_duplicate:
                    seen.add(id(value))
                yield (f"{prefix}.{name}" if prefix else name), value
        if recurse:
            for name, module in self.named_children():
                method = getattr(module, "named_buffers", None)
                if callable(method):
                    yield from method(
                        f"{prefix}.{name}" if prefix else name,
                        recurse=True,
                        remove_duplicate=remove_duplicate,
                    )

    def train(self, mode: bool = True) -> "GraphModule":
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            train = getattr(module, "train", None)
            if callable(train):
                train(mode)
        return self

    def eval(self) -> "GraphModule":
        return self.train(False)

    def _get_name(self) -> str:
        return str(self.__dict__.get("_class_name", type(self).__name__))

    def _save_to_state_dict(
        self, destination: dict[str, Any], prefix: str, keep_vars: bool
    ) -> None:
        for name, value in self._parameters.items():
            if value is not None:
                destination[prefix + name] = (
                    value if keep_vars else getattr(value, "detach", lambda: value)()
                )
        for name, value in self._buffers.items():
            if value is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = (
                    value if keep_vars else getattr(value, "detach", lambda: value)()
                )

    def state_dict(
        self,
        *args: Any,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        if args:
            warnings.warn(
                "positional state_dict arguments are deprecated; use keyword arguments",
                FutureWarning,
                stacklevel=2,
            )
            if destination is None:
                destination = args[0]
            if len(args) > 1:
                prefix = args[1]
            if len(args) > 2:
                keep_vars = args[2]
            if len(args) > 3:
                raise TypeError(f"state_dict() takes at most 3 positional arguments ({len(args)} given)")
        if destination is None:
            destination = OrderedDict()
        for hook in self._state_dict_pre_hooks.values():
            hook(self, prefix, keep_vars)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self.named_children():
            state_dict = getattr(module, "state_dict", None)
            if callable(state_dict):
                state_dict(
                    destination=destination,
                    prefix=prefix + name + ".",
                    keep_vars=keep_vars,
                )
        for hook in self._state_dict_hooks.values():
            result = hook(self, destination, prefix, {})
            if result is not None:
                destination = result
        return destination

    def _register_state_dict_hook(self, hook: Callable[..., Any]) -> _HookHandle:
        if not callable(hook):
            raise TypeError("state_dict hook must be callable")
        handle = _HookHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.id] = hook
        return handle

    def register_state_dict_post_hook(self, hook: Callable[..., Any]) -> _HookHandle:
        return self._register_state_dict_hook(hook)

    def register_state_dict_pre_hook(self, hook: Callable[..., Any]) -> _HookHandle:
        if not callable(hook):
            raise TypeError("state_dict pre-hook must be callable")
        handle = _HookHandle(self._state_dict_pre_hooks)
        self._state_dict_pre_hooks[handle.id] = hook
        return handle

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"state_dict must be a mapping, got {type(state_dict)!r}")
        for hook in self._load_state_dict_pre_hooks.values():
            hook(self, state_dict, "", {}, strict, [], [], [])
        expected = dict(self.named_parameters())
        expected.update(dict(self.named_buffers()))
        missing: list[str] = []
        unexpected = [key for key in state_dict if key not in expected]
        for key, target in expected.items():
            if key not in state_dict:
                missing.append(key)
                continue
            value = state_dict[key]
            if assign:
                _assign_attr(value, self, key)
                continue
            copier = getattr(target, "copy_", None)
            if callable(copier):
                copier(value)
            else:
                _assign_attr(value, self, key)
        result_type = type(
            "IncompatibleKeys", (tuple,),
            {"__new__": staticmethod(lambda cls, missing_keys, unexpected_keys: tuple.__new__(cls, (missing_keys, unexpected_keys))),
             "missing_keys": property(lambda self: self[0]),
             "unexpected_keys": property(lambda self: self[1]),
             "__repr__": lambda self: "<All keys matched successfully>" if not self[0] and not self[1] else tuple.__repr__(self)},
        )
        result = result_type(missing, unexpected)
        for hook in self._load_state_dict_post_hooks.values():
            hook(self, result)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "Error(s) in loading state_dict: "
                f"missing keys={missing!r}, unexpected keys={unexpected!r}"
            )
        return result

    def _invoke_forward(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if self._boxed_call:
            if kwargs:
                raise TypeError("boxed graph modules do not accept keyword arguments")
            boxed_args = list(args)
            result = self.forward(boxed_args)
        elif kwargs:
            result = self.forward(*args, **kwargs)
        else:
            processed = self.graph.process_inputs(*args)
            if isinstance(processed, tuple):
                result = self.forward(*processed)
            elif isinstance(processed, list):
                result = self.forward(*processed)
            else:
                result = self.forward(processed)
        return self.graph.process_outputs(result)

    def _call_with_hooks(self, *args: Any, **kwargs: Any) -> Any:
        global_tables = _module_hook_tables()
        global_pre = global_tables[0] if len(global_tables) > 0 else {}
        global_post = global_tables[1] if len(global_tables) > 1 else {}
        global_pre_kwargs = global_tables[2] if len(global_tables) > 2 else {}
        global_always = global_tables[3] if len(global_tables) > 3 else {}
        for hook_id, hook in (*global_pre.items(), *self._forward_pre_hooks.items()):
            if hook_id in global_pre_kwargs or hook_id in self._forward_pre_hooks_with_kwargs:
                result = hook(self, args, kwargs)
                if result is not None:
                    if not isinstance(result, tuple) or len(result) != 2:
                        raise RuntimeError(
                            "forward pre-hook must return None or a tuple of (new_args, new_kwargs)"
                        )
                    args, kwargs = result
            else:
                result = hook(self, args)
                if result is not None:
                    args = result if isinstance(result, tuple) else (result,)

        called_always: set[int] = set()
        hooks = (*global_post.items(), *self._forward_hooks.items())
        try:
            result = self._invoke_forward(args, kwargs)
            for hook_id, hook in hooks:
                if hook_id in global_always or hook_id in self._forward_hooks_always_called:
                    called_always.add(hook_id)
                if hook_id in global_pre_kwargs or hook_id in self._forward_hooks_with_kwargs:
                    hook_result = hook(self, args, kwargs, result)
                else:
                    hook_result = hook(self, args, result)
                if hook_result is not None:
                    result = hook_result
            return result
        except Exception:
            for hook_id, hook in hooks:
                if hook_id not in global_always and hook_id not in self._forward_hooks_always_called:
                    continue
                if hook_id in called_always:
                    continue
                try:
                    if hook_id in global_pre_kwargs or hook_id in self._forward_hooks_with_kwargs:
                        hook_result = hook(self, args, kwargs, None)
                    else:
                        hook_result = hook(self, args, None)
                    if hook_result is not None:
                        result = hook_result
                except Exception as hook_error:
                    warnings.warn(
                        "a forward hook marked always_call raised while another error was active: "
                        f"{hook_error}",
                        stacklevel=2,
                    )
            raise

    def register_forward_pre_hook(
        self, hook: Callable[..., Any], *, prepend: bool = False, with_kwargs: bool = False
    ) -> Any:
        handle = _HookHandle(self._forward_pre_hooks, self._forward_pre_hooks_with_kwargs)
        self._forward_pre_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_pre_hooks_with_kwargs[handle.id] = True
        if prepend:
            self._forward_pre_hooks.move_to_end(handle.id, last=False)
        return handle

    def register_forward_hook(
        self,
        hook: Callable[..., Any],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> Any:
        handle = _HookHandle(
            self._forward_hooks,
            self._forward_hooks_with_kwargs,
            self._forward_hooks_always_called,
        )
        self._forward_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_hooks_with_kwargs[handle.id] = True
        if always_call:
            self._forward_hooks_always_called[handle.id] = True
        if prepend:
            self._forward_hooks.move_to_end(handle.id, last=False)
        return handle

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        compiled = self.__dict__.get("_compiled_forward")
        if compiled is not None:
            return compiled(*args, **kwargs)
        return self._interpret(*args, **kwargs)

    def recompile(self) -> Callable[..., Any]:
        python_code = self.graph.python_code(root_module="self")
        if not hasattr(python_code, "src"):
            raise GraphCaptureError("graph code generation did not return PythonCode")
        source = python_code.src
        code_fields = dict(getattr(self.graph, "_co_fields", {}) or {})
        implementation = _forward_from_src(source, dict(python_code.globals), code_fields)
        implementation.__module__ = type(self).__module__
        implementation.__qualname__ = f"{type(self).__qualname__}.forward"
        bound_implementation = types.MethodType(implementation, self)

        if self.signature is not None and not self._boxed_call:
            @functools.wraps(implementation)
            def checked_forward(*args: Any, **kwargs: Any) -> Any:
                bound = self.signature.bind(*args, **kwargs)
                bound.apply_defaults()
                return bound_implementation(*bound.args, **bound.kwargs)

            checked_forward.__signature__ = self.signature  # type: ignore[attr-defined]
            forward = checked_forward
        else:
            forward = bound_implementation

        object.__setattr__(self, "_code", source)
        object.__setattr__(self, "_python_code", python_code)
        object.__setattr__(self, "_compiled_impl", bound_implementation)
        object.__setattr__(self, "_compiled_forward", forward)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "_lineno_map", python_code._lineno_map)
        object.__setattr__(self, "_prologue_start", python_code._prologue_start)
        self._recompile_submodules()
        return forward

    def _interpret(self, *args: Any, _record_meta: bool = False, **kwargs: Any) -> Any:
        if self.signature is not None:
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            bound_arguments = dict(bound.arguments)
        else:
            placeholders = self.graph.placeholders
            if len(args) > len(placeholders):
                raise TypeError(
                    f"expected at most {len(placeholders)} positional inputs, got {len(args)}"
                )
            bound_arguments: dict[str, Any] = {}
            for index, node in enumerate(placeholders):
                if index < len(args):
                    bound_arguments[node.name] = args[index]
                elif node.name in kwargs:
                    bound_arguments[node.name] = kwargs[node.name]
                elif node.args:
                    bound_arguments[node.name] = node.args[0]
                else:
                    raise TypeError(f"missing required graph input: {node.name}")
            unknown = set(kwargs) - {node.name for node in placeholders}
            if unknown:
                raise TypeError(
                    "unexpected graph inputs: " + ", ".join(sorted(unknown))
                )

        environment: dict[Node, Any] = {}

        def keep(node: Node, value: Any) -> Any:
            environment[node] = value
            if _record_meta:
                node.meta["val"] = value
                shape = getattr(value, "shape", None)
                if callable(shape):
                    shape = shape()
                if shape is not None:
                    try:
                        node.meta["tensor_shape"] = tuple(int(dim) for dim in shape)
                    except (TypeError, ValueError):
                        pass
            return value

        for node in self.graph.placeholders:
            parameter_name = node.target if isinstance(node.target, str) else node.name
            if parameter_name not in bound_arguments:
                if node.name not in bound_arguments:
                    raise TypeError(f"missing required graph input: {parameter_name}")
                keep(node, bound_arguments[node.name])
            else:
                keep(node, bound_arguments[parameter_name])

        for node in self.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "call_function":
                target = self._resolve_target(node.target)
                keep(
                    node,
                    target(
                        *self._resolve(node.args, environment),
                        **self._resolve(node.kwargs, environment),
                    ),
                )
            elif node.op == "call_method":
                resolved = self._resolve(node.args, environment)
                if not resolved:
                    raise GraphCaptureError("call_method node has no receiver")
                receiver, *method_args = resolved
                keep(
                    node,
                    getattr(receiver, node.target)(
                        *method_args,
                        **self._resolve(node.kwargs, environment),
                    ),
                )
            elif node.op == "call_module":
                module = self._get_attr(str(node.target))
                keep(
                    node,
                    module(
                        *self._resolve(node.args, environment),
                        **self._resolve(node.kwargs, environment),
                    ),
                )
            elif node.op == "get_attr":
                keep(node, self._get_attr(str(node.target)))
            elif node.op == "output":
                if len(node.args) != 1:
                    raise GraphCaptureError("output node must contain one value")
                return self._resolve(node.args[0], environment)
            else:
                raise GraphCaptureError(f"unsupported graph node kind: {node.op!r}")
        raise GraphCaptureError("graph has no output node")

    @staticmethod
    def _resolve(value: Any, environment: dict[Node, Any]) -> Any:
        from .immutable_collections import immutable_dict, immutable_list

        if isinstance(value, Node):
            try:
                return environment[value]
            except KeyError as exc:
                raise GraphCaptureError(
                    f"value for node {value.name!r} is not available"
                ) from exc
        if isinstance(value, immutable_list):
            return immutable_list(GraphModule._resolve(item, environment) for item in value)
        if isinstance(value, list):
            return [GraphModule._resolve(item, environment) for item in value]
        if isinstance(value, tuple):
            items = [GraphModule._resolve(item, environment) for item in value]
            return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
        if isinstance(value, immutable_dict):
            return immutable_dict(
                (GraphModule._resolve(key, environment), GraphModule._resolve(item, environment))
                for key, item in value.items()
            )
        if isinstance(value, dict):
            return {
                GraphModule._resolve(key, environment): GraphModule._resolve(item, environment)
                for key, item in value.items()
            }
        if isinstance(value, slice):
            return slice(
                GraphModule._resolve(value.start, environment),
                GraphModule._resolve(value.stop, environment),
                GraphModule._resolve(value.step, environment),
            )
        if isinstance(value, range):
            return range(
                GraphModule._resolve(value.start, environment),
                GraphModule._resolve(value.stop, environment),
                GraphModule._resolve(value.step, environment),
            )
        if isinstance(value, set):
            return {GraphModule._resolve(item, environment) for item in value}
        if isinstance(value, frozenset):
            return frozenset(GraphModule._resolve(item, environment) for item in value)
        return value

    def _get_attr(self, target: str) -> Any:
        if target in self.__dict__.get("_graph_attrs", {}):
            return self.__dict__["_graph_attrs"][target]
        try:
            return _lookup_path(self, target)
        except (AttributeError, KeyError, IndexError, TypeError):
            root = self.__dict__.get("_root", _MISSING)
            if root is not _MISSING and root is not None:
                return _lookup_path(root, target)
            raise

    @staticmethod
    def _resolve_target(target: Any) -> Any:
        if isinstance(target, Node):
            raise GraphCaptureError("dynamically produced callables are unsupported")
        if not callable(target):
            raise TypeError(f"call_function target is not callable: {target!r}")
        return target

    def add_submodule(self, target: str, module: Any) -> bool:
        if not isinstance(target, str) or not target:
            raise ValueError("submodule target must be a non-empty string")
        if not _is_module(module):
            raise TypeError(f"{type(module)!r} is not a TensorPlay module")
        parts = target.split(".")
        holder: Any = self
        for part in parts[:-1]:
            current = getattr(holder, part, None)
            if current is None:
                current = _new_module()
                holder.add_module(part, current)
            if not _is_module(current):
                return False
            holder = current
        holder.add_module(parts[-1], module)
        return True

    def delete_submodule(self, target: str) -> bool:
        if not isinstance(target, str) or not target:
            return False
        parts = target.split(".")
        holder: Any = self
        for part in parts[:-1]:
            try:
                holder = getattr(holder, part)
            except AttributeError:
                return False
            if not _is_module(holder):
                return False
        name = parts[-1]
        current = getattr(holder, name, _MISSING)
        if current is _MISSING or not _is_module(current):
            return False
        delattr(holder, name)
        return True

    def delete_all_unused_submodules(self) -> None:
        used: set[str] = set()
        for node in self.graph.nodes:
            if node.op not in {"call_module", "get_attr"} or not isinstance(node.target, str):
                continue
            parts = node.target.split(".")
            used.update(".".join(parts[:index]) for index in range(1, len(parts) + 1))
            if node.op == "call_module":
                try:
                    module = self.get_submodule(node.target)
                except AttributeError:
                    continue
                for child_name, _ in module.named_modules():
                    if child_name:
                        used.add(f"{node.target}.{child_name}")
        for name, _ in list(self.named_modules()):
            if name and name not in used:
                self.delete_submodule(name)

    def _recompile_submodules(self) -> list[tuple[str, Any]]:
        results: list[tuple[str, Any]] = []
        for name, module in self.named_children():
            if isinstance(module, GraphModule):
                results.append((name, module.recompile()))
        return results

    def to_folder(self, folder: str | Path, module_name: str = "GraphModule") -> None:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "state.pkl").open("wb") as stream:
            pickle.dump(self, stream)
        source = (
            "import pickle\n"
            "from pathlib import Path\n\n"
            "def load():\n"
            "    with (Path(__file__).parent / 'state.pkl').open('rb') as f:\n"
            "        return pickle.load(f)\n\n"
            f"{module_name} = load()\n"
        )
        (path / "module.py").write_text(source)
        (path / "__init__.py").write_text(f"from .module import {module_name}\n")

    def print_readable(
        self,
        print_output: bool = True,
        include_stride: bool = False,
        include_device: bool = False,
        colored: bool = False,
        *,
        fast_sympy_print: bool = False,
        expanded_def: bool = False,
        additional_meta: list[str] | None = None,
    ) -> str:
        del fast_sympy_print
        code = self.graph.python_code(
            root_module="self",
            verbose=True,
            include_stride=include_stride,
            include_device=include_device,
            colored=colored,
            expanded_def=expanded_def,
            additional_meta=additional_meta,
        )
        text = f"class {self._class_name}(Module):\n"
        text += "\n".join(
            "    " + line if line else "" for line in code.src.splitlines()
        )
        text += "\n"
        if print_output:
            print(text, end="")
        return text

    def __str__(self) -> str:
        return f"{_module_repr(self)}\n{self.code}\n# use print_readable() for graph details"

    def __repr__(self) -> str:
        return _module_repr(self)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_compiled_forward", None)
        state.pop("_compiled_impl", None)
        state.pop("_python_code", None)
        forward = state.get("forward")
        if isinstance(forward, types.MethodType) and forward.__self__ is self:
            state.pop("forward", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("_replace_hooks", [])
        self.__dict__.setdefault("_create_node_hooks", [])
        self.__dict__.setdefault("_erase_node_hooks", [])
        self.__dict__.setdefault("_deepcopy_hooks", [])
        self.__dict__.setdefault("meta", {})
        self.__dict__["_compiled_forward"] = None
        self.__dict__["_compiled_impl"] = None
        graph = self.__dict__.get("_graph")
        if graph is not None:
            graph.owning_module = self

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        state = self.__getstate__()
        graph = copy.deepcopy(self.graph)
        root = self.root
        state.pop("_graph", None)
        state.pop("_root", None)
        state.pop("forward", None)
        return _rebuild_graph_module, (root, graph, self.signature, state)

    def _deepcopy_init(self) -> Callable[..., None]:
        return GraphModule.__init__

    def __deepcopy__(self, memo: dict[int, Any]) -> "GraphModule":
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        result = object.__new__(type(self))
        memo[id(self)] = result
        root = copy.deepcopy(self.root, memo)
        graph = copy.deepcopy(self.graph, memo)
        GraphModule.__init__(
            result,
            root,
            graph,
            copy.deepcopy(self.signature, memo),
            class_name=self._class_name,
        )
        state = self.__getstate__()
        state.pop("_graph", None)
        state.pop("_root", None)
        state.pop("forward", None)
        result.__dict__.update(copy.deepcopy(state, memo))
        object.__setattr__(result, "_graph", graph)
        graph.owning_module = result
        object.__setattr__(result, "_compiled_forward", None)
        object.__setattr__(result, "_compiled_impl", None)
        result.recompile()
        for hook in getattr(result, "_deepcopy_hooks", ()):
            hook(result)
        return result

    def __copy__(self) -> "GraphModule":
        result = GraphModule(self, self.graph, self.signature, class_name=self._class_name)
        result.meta = self.meta.copy()
        return result

    def _replicate_for_data_parallel(self) -> "GraphModule":
        result = copy.copy(self)
        object.__setattr__(result, "_is_replica", True)
        return result

    @contextlib.contextmanager
    def _set_replace_hook(
        self, hook: Callable[[Node, str, Node], object]
    ) -> Iterator[None]:
        self._register_replace_node_hook(hook)
        try:
            yield
        finally:
            self._unregister_replace_node_hook(hook)

    def _register_replace_node_hook(self, hook: Callable[[Node, str, Node], object]) -> None:
        if not callable(hook):
            raise AssertionError("replace hook must be callable")
        self._replace_hooks.append(hook)

    def _unregister_replace_node_hook(self, hook: Callable[[Node, str, Node], object]) -> None:
        if not callable(hook):
            raise AssertionError("replace hook must be callable")
        self._replace_hooks.remove(hook)

    def _register_create_node_hook(self, hook: Callable[[Node], object]) -> None:
        if not callable(hook):
            raise AssertionError("create hook must be callable")
        self._create_node_hooks.append(hook)

    def _unregister_create_node_hook(self, hook: Callable[[Node], object]) -> None:
        if not callable(hook):
            raise AssertionError("create hook must be callable")
        self._create_node_hooks.remove(hook)

    def _register_erase_node_hook(self, hook: Callable[[Node], object]) -> None:
        if not callable(hook):
            raise AssertionError("erase hook must be callable")
        self._erase_node_hooks.append(hook)

    def _unregister_erase_node_hook(self, hook: Callable[[Node], object]) -> None:
        if not callable(hook):
            raise AssertionError("erase hook must be callable")
        self._erase_node_hooks.remove(hook)

    def _register_deepcopy_hook(self, hook: Callable[["GraphModule"], object]) -> None:
        if not callable(hook):
            raise AssertionError("deepcopy hook must be callable")
        self._deepcopy_hooks.append(hook)

    def _unregister_deepcopy_hook(self, hook: Callable[["GraphModule"], object]) -> None:
        if not callable(hook):
            raise AssertionError("deepcopy hook must be callable")
        self._deepcopy_hooks.remove(hook)

__all__ = ["GraphModule"]
