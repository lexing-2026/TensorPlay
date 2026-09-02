from __future__ import annotations

import builtins
import copy
import enum
import inspect
import keyword
import math
import operator
import re
import typing
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NamedTuple

from ._compatibility import compatibility
from ._utils import (
    GraphCaptureError,
    _format_target,
    _iter_nodes,
    _sanitize_name,
)
from .immutable_collections import immutable_dict, immutable_list
from .node import Node, map_arg
from .proxy import Proxy

__all__ = [
    "PythonCode",
    "CodeGen",
    "Graph",
    "dead_code_elimination",
    "magic_methods",
    "reflectable_magic_methods",
]


_legal_ops = frozenset(
    {
        "call_function",
        "call_method",
        "get_attr",
        "call_module",
        "placeholder",
        "output",
    }
)


def _graph_size(value: Any, dim: int) -> Any:
    return value.size(dim)


def _graph_stride(value: Any, dim: int) -> Any:
    return value.stride(dim)


def _graph_storage_offset(value: Any) -> Any:
    return value.storage_offset()


def _graph_getattr(value: Any, name: str) -> Any:
    return getattr(value, name)


def _graph_bool_to_int(value: Any) -> int:
    return int(bool(value))


def _graph_sym_min(left: Any, right: Any) -> Any:
    from .experimental.sym_node import SymNode

    if isinstance(left, SymNode):
        return left.sym_min(right)
    if isinstance(right, SymNode):
        return right.sym_min(left)
    return min(left, right)


def _graph_sym_max(left: Any, right: Any) -> Any:
    from .experimental.sym_node import SymNode

    if isinstance(left, SymNode):
        return left.sym_max(right)
    if isinstance(right, SymNode):
        return right.sym_max(left)
    return max(left, right)


def _graph_floor(value: Any) -> Any:
    from .experimental.sym_node import SymNode

    if isinstance(value, SymNode):
        return value.floor()
    return math.floor(value)


def _graph_ceil(value: Any) -> Any:
    from .experimental.sym_node import SymNode

    if isinstance(value, SymNode):
        return value.ceil()
    return math.ceil(value)


def _graph_abs(value: Any) -> Any:
    from .experimental.sym_node import SymNode

    if isinstance(value, SymNode):
        return value.abs()
    return abs(value)


def _graph_sym_and(left: Any, right: Any) -> Any:
    from .experimental.sym_node import SymNode

    if isinstance(left, SymNode):
        return left.and_(right)
    if isinstance(right, SymNode):
        return right.and_(left)
    return bool(left) and bool(right)


def _graph_sym_or(left: Any, right: Any) -> Any:
    from .experimental.sym_node import SymNode

    if isinstance(left, SymNode):
        return left.or_(right)
    if isinstance(right, SymNode):
        return right.or_(left)
    return bool(left) or bool(right)


def _graph_sym_xor(left: Any, right: Any) -> Any:
    from .experimental.sym_node import SymNode

    if isinstance(left, SymNode):
        return left.xor(right)
    if isinstance(right, SymNode):
        return right.xor(left)
    return bool(left) != bool(right)


def _graph_sym_not(value: Any) -> Any:
    from .experimental.sym_node import SymNode

    if isinstance(value, SymNode):
        return value.sym_not()
    return not bool(value)


def _graph_sym_ite(condition: Any, true_value: Any, false_value: Any) -> Any:
    from .experimental.sym_node import SymNode, sym_ite

    if isinstance(condition, SymNode):
        return sym_ite(condition, true_value, false_value)
    return true_value if bool(condition) else false_value


def _graph_sym_is_integer(value: Any) -> Any:
    from .experimental.sym_node import SymNode

    if isinstance(value, SymNode):
        return value.is_integer()
    return float(value).is_integer()


def _graph_sym_trunc(value: Any) -> Any:
    from .experimental.sym_node import SymNode

    if isinstance(value, SymNode):
        return value.trunc()
    return math.trunc(value)


def _graph_sym_float(value: Any) -> Any:
    from .experimental.sym_node import SymNode

    if isinstance(value, SymNode):
        return value.sym_float()
    return float(value)


def _snake_case(value: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value).lower()


def _qualified_name(value: Any) -> str:
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if name is None:
        name = type(value).__qualname__
    module = getattr(value, "__module__", None)
    return f"{module}.{name}" if module and module != "builtins" else str(name)


def _type_repr(value: Any) -> str:
    if value is None:
        return "None"
    if value is Ellipsis:
        return "..."
    if value is typing.Any:
        return "typing.Any"
    if isinstance(value, type):
        if value.__module__ == "builtins":
            return value.__qualname__
        return _qualified_name(value)
    origin = typing.get_origin(value)
    if origin is not None:
        args = typing.get_args(value)
        return f"{_type_repr(origin)}[{', '.join(_type_repr(arg) for arg in args)}]"
    return _qualified_name(value) if hasattr(value, "__module__") else repr(value)


_name_pattern = re.compile(r"^([a-zA-Z_][0-9a-zA-Z_]*?)(?:_(\d+))?$")
_illegal_names: dict[str, object] = {
    name: object() for name in (*keyword.kwlist, *builtins.__dict__)
}


@compatibility(is_backward_compatible=True)
class _Namespace:
    """Assign valid, unique names to local and external graph objects."""

    def __init__(self) -> None:
        self._obj_to_name: dict[Any, str] = {}
        self._id_to_name: dict[int, str] = {}
        self._used_names: set[str] = set()
        self._base_count: dict[str, int] = {}

    def _lookup(self, obj: object) -> str | None:
        try:
            return self._obj_to_name.get(obj)
        except TypeError:
            return self._id_to_name.get(id(obj))

    def release_name(self, name: str) -> None:
        """Return an erased node's name to the pool for reuse."""

        self._used_names.discard(name)
        for obj, held in list(self._obj_to_name.items()):
            if held == name:
                del self._obj_to_name[obj]
        for key, held in list(self._id_to_name.items()):
            if held == name:
                del self._id_to_name[key]

    def _remember(self, obj: object, name: str) -> None:
        try:
            self._obj_to_name[obj] = name
        except TypeError:
            self._id_to_name[id(obj)] = name

    def create_name(self, candidate: str, obj: object | None = None) -> str:
        if obj is not None:
            existing = self._lookup(obj)
            if existing is not None:
                return existing
        candidate = _sanitize_name(str(candidate)) or "_unnamed"
        match = _name_pattern.match(candidate)
        if match is None:
            candidate = f"_{candidate}"
            match = _name_pattern.match(candidate)
        if match is None:
            candidate = "_unnamed"
            match = _name_pattern.match(candidate)
        assert match is not None
        base, suffix = match.groups()
        # Collisions number from zero (add, add_0, add_1, ...), whether the
        # candidate arrives with or without a numeric suffix.  Only keywords
        # are guarded: assigning to them is a syntax error in generated code.
        # Builtin spellings (sum, abs, ...) are safe as node names because
        # generated code invokes targets through resolved identifiers, never
        # by node name.
        if candidate in keyword.kwlist:
            number = 0 if suffix is None else int(suffix) + 1
            candidate = f"{base}_{number}"
        number = int(suffix) if suffix is not None else None
        while candidate in self._used_names:
            number = 0 if number is None else number + 1
            candidate = f"{base}_{number}"
        self._used_names.add(candidate)
        if obj is not None:
            self._remember(obj, candidate)
        return candidate

    def associate_name_with_obj(self, name: str, obj: object) -> None:
        existing = self._lookup(obj)
        if existing is not None and existing != name:
            raise AssertionError("object already has a different name")
        self._used_names.add(name)
        self._remember(obj, name)

    def _rename_object(self, obj: object, name: str) -> None:
        self._remember(obj, name)
        self._used_names.add(name)


@compatibility(is_backward_compatible=True)
@dataclass
class PythonCode:
    """Source text and the runtime namespace required to execute it."""

    src: str
    globals: dict[str, Any]
    _lineno_map: dict[int, int | None] | None = None
    _prologue_start: int = 0

    def __str__(self) -> str:
        return self.src


def _format_target_path(root: str, target: str) -> str:
    result = root
    for part in target.split("."):
        if part.isidentifier() and not keyword.iskeyword(part):
            result = f"{result}.{part}"
        else:
            result = f"getattr({result}, {part!r})"
    return result


class _InsertPoint:
    __slots__ = ("graph", "orig_insert")

    def __init__(self, graph: "Graph", new_insert: Callable[[Node], None]) -> None:
        self.graph = graph
        self.orig_insert, graph._insert = graph._insert, new_insert

    def __enter__(self) -> "_InsertPoint":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.graph._insert = self.orig_insert


class _NodeList(list[Node]):
    """List view with graph bookkeeping for direct list mutations."""

    __slots__ = ("graph",)

    def __init__(self, graph: "Graph", values: Iterable[Node] = ()) -> None:
        self.graph = graph
        super().__init__(values)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        # The owner slot must be installed before list items are restored.  A
        # normal list-subclass reduction restores items first, which makes any
        # bookkeeping callback see an uninitialized owner.
        return type(self), (self.graph, tuple(self)), None

    def _mutated(self, old: Iterable[Node]) -> None:
        self.graph._sync_nodes(self, old)

    def append(self, node: Node) -> None:
        old = tuple(self)
        super().append(node)
        self._mutated(old)

    def extend(self, values: Iterable[Node]) -> None:
        old = tuple(self)
        super().extend(values)
        self._mutated(old)

    def insert(self, index: int, node: Node) -> None:
        old = tuple(self)
        super().insert(index, node)
        self._mutated(old)

    def remove(self, node: Node) -> None:
        old = tuple(self)
        super().remove(node)
        self._mutated(old)

    def pop(self, index: int = -1) -> Node:
        old = tuple(self)
        result = super().pop(index)
        self._mutated(old)
        return result

    def clear(self) -> None:
        old = tuple(self)
        super().clear()
        self._mutated(old)

    def reverse(self) -> None:
        old = tuple(self)
        super().reverse()
        self._mutated(old)

    def sort(self, *args: Any, **kwargs: Any) -> None:
        old = tuple(self)
        super().sort(*args, **kwargs)
        self._mutated(old)

    def __setitem__(self, index: int | slice, value: Any) -> None:
        old = tuple(self)
        super().__setitem__(index, value)
        self._mutated(old)

    def __delitem__(self, index: int | slice) -> None:
        old = tuple(self)
        super().__delitem__(index)
        self._mutated(old)

    def __iadd__(self, values: Iterable[Node]) -> "_NodeList":
        old = tuple(self)
        super().__iadd__(values)
        self._mutated(old)
        return self

    def __imul__(self, count: int) -> "_NodeList":
        old = tuple(self)
        super().__imul__(count)
        self._mutated(old)
        return self


# The lower-case spelling is part of the graph internals used by code
# generators and graph transformations.  It names the concrete list view.
_node_list = _NodeList


class _FindNodesLookupTable:
    """Index nodes by opcode and callable identity for fast queries."""

    def __init__(self) -> None:
        self.table: dict[tuple[str, Any], dict[Node, None]] = defaultdict(dict)
        self.by_op: dict[str, dict[Node, None]] = defaultdict(dict)

    @staticmethod
    def _target_key(target: Any) -> Any:
        try:
            hash(target)
        except TypeError:
            return ("id", id(target))
        return ("value", target)

    def _key(self, node: Node) -> tuple[str, Any]:
        return (
            node.op,
            self._target_key(node.target) if node.op == "call_function" else None,
        )

    def insert(self, node: Node) -> None:
        self.table[self._key(node)][node] = None
        self.by_op[node.op][node] = None

    def remove(self, node: Node) -> None:
        self.table[self._key(node)].pop(node, None)
        self.by_op[node.op].pop(node, None)

    def rebuild(self, nodes: Iterable[Node]) -> None:
        self.table.clear()
        self.by_op.clear()
        for node in nodes:
            self.insert(node)

    def find_nodes(self, op: str, target: Any | None = None) -> list[Node]:
        if target is None:
            return list(self.by_op.get(op, {}))
        if op == "call_function":
            result = list(self.table.get((op, self._target_key(target)), {}))
            return [node for node in result if _same_target(node.target, target)]
        return [node for node in self.by_op.get(op, {}) if _same_target(node.target, target)]

    def __contains__(self, node: Node) -> bool:
        if not hasattr(node, "op") or not hasattr(node, "target"):
            return False
        return node in self.table.get(self._key(node), {})


class _PyTreeInfo(NamedTuple):
    orig_args: list[str]
    in_spec: Any
    out_spec: Any | None


@dataclass(frozen=True)
class _ParsedStackTrace:
    file: str
    lineno: str
    name: str
    code: str

    def get_summary_str(self) -> str:
        return f"File: {self.file}:{self.lineno} in {self.name}, code: {self.code}"


def _parse_stack_trace(
    stack_trace: str | None,
    filter_fn: Callable[[str, str, str], bool] | None = None,
) -> _ParsedStackTrace | None:
    if stack_trace is None:
        return None
    pattern = re.compile(r'^File "(.+)", line (\d+), in (.+)$')
    lines = stack_trace.strip().splitlines()
    for index in range(len(lines) - 2, -1, -1):
        match = pattern.match(lines[index].strip())
        if match is None:
            continue
        file_name, line_number, name = match.groups()
        code = lines[index + 1].strip()
        if filter_fn is None or filter_fn(file_name, name, code):
            return _ParsedStackTrace(file_name, line_number, name, code)
    return None


reflectable_magic_methods = {
    "add": "{} + {}",
    "sub": "{} - {}",
    "mul": "{} * {}",
    "floordiv": "{} // {}",
    "truediv": "{} / {}",
    "div": "{} / {}",
    "mod": "{} % {}",
    "pow": "{} ** {}",
    "lshift": "{} << {}",
    "rshift": "{} >> {}",
    "and_": "{} & {}",
    "or_": "{} | {}",
    "xor": "{} ^ {}",
    "getitem": "{}[{}]",
    "matmul": "{} @ {}",
}

magic_methods = {
    "eq": "{} == {}",
    "ne": "{} != {}",
    "lt": "{} < {}",
    "gt": "{} > {}",
    "le": "{} <= {}",
    "ge": "{} >= {}",
    "pos": "+{}",
    "neg": "-{}",
    "invert": "~{}",
    **reflectable_magic_methods,
}


_inplace_methods = {
    "iadd": "{} += {}",
    "iand": "{} &= {}",
    "ifloordiv": "{} //= {}",
    "ilshift": "{} <<= {}",
    "imod": "{} %= {}",
    "imul": "{} *= {}",
    "imatmul": "{} @= {}",
    "ior": "{} |= {}",
    "ipow": "{} **= {}",
    "irshift": "{} >>= {}",
    "isub": "{} -= {}",
    "itruediv": "{} /= {}",
    "ixor": "{} ^= {}",
    "setitem": "{}[{}] = {}",
}


def _callable_name(value: Any) -> str:
    return str(getattr(value, "__name__", None) or type(value).__name__)


def _same_target(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        result = left == right
    except Exception:
        return False
    return isinstance(result, bool) and result


@compatibility(is_backward_compatible=False)
class CodeGen:
    """Generate executable Python source from graph nodes."""

    _sym_repr: Callable[[Any], str] = repr

    def __init__(self) -> None:
        self._body_transformer: Callable[[list[str]], list[str]] | None = None
        self._func_name = "forward"

    def _format_multiline_args(self, args: Sequence[str]) -> str:
        return "".join(self._format_single_arg(arg) for arg in args)

    @staticmethod
    def _format_single_arg(arg: str) -> str:
        if "#" in arg:
            argument, comment = arg.split("#", 1)
            return f"    {argument.rstrip()},  # {comment.lstrip()}\n"
        return f"    {arg},\n"

    @staticmethod
    def _get_delimiters(container: Sequence[Any]) -> tuple[str, str]:
        return ("(", ")") if isinstance(container, tuple) else ("[", "]")

    def _get_desc_trailers(
        self, items: Sequence[Any], descs: Sequence[str] | None
    ) -> list[str]:
        if descs is None:
            return [""] * len(items)
        if len(descs) != len(items):
            raise ValueError("description count must match container length")
        return [f"  # {desc}" for desc in descs]

    def _format_multiline_container(
        self,
        items: Sequence[Any],
        descs: Sequence[str] | None = None,
        prefix: str = "",
        repr_fn: Callable[[Any], str] | None = None,
    ) -> str:
        render = repr if repr_fn is None else repr_fn
        left, right = self._get_delimiters(items)
        trailers = self._get_desc_trailers(items, descs)
        lines = [f"{prefix}{left}\n"]
        lines.extend(
            f"    {render(item)},{trailer}\n"
            for item, trailer in zip(items, trailers)
        )
        lines.append(right)
        return "".join(lines)

    @staticmethod
    def _call_method_with_signature_check(
        method: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        signature = inspect.signature(method)
        filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return method(*args, **filtered)

    def gen_fn_def(
        self,
        free_vars: list[str],
        maybe_return_annotation: str = "",
        *,
        expanded_def: bool = False,
        root_module: str | None = "self",
    ) -> str:
        variables = list(free_vars)
        if root_module is not None and (not variables or variables[0] != root_module):
            variables.insert(0, root_module)
        if expanded_def:
            return (
                f"def {self._func_name}(\n"
                f"{self._format_multiline_args(variables)})"
                f"{maybe_return_annotation}:"
            )
        return f"def {self._func_name}({', '.join(variables)}){maybe_return_annotation}:"

    def generate_output(
        self,
        output_args: Any,
        *,
        descs: Sequence[str] | None = None,
        repr_fn: Callable[[Any], str] | None = None,
    ) -> str:
        render = repr if repr_fn is None else repr_fn
        if descs is not None and isinstance(output_args, (list, tuple)):
            return self._format_multiline_container(
                output_args, descs, "return ", repr_fn=render
            )
        return f"return {render(output_args)}"

    def process_inputs(self, *args: Any) -> Any:
        return args

    def process_outputs(self, outputs: Any) -> Any:
        return outputs

    def additional_globals(self) -> list[tuple[str, Any]]:
        return []

    def _gen_python_code(
        self,
        nodes: Iterable[Node],
        root_module: str | None,
        namespace: _Namespace,
        *,
        verbose: bool = False,
        include_stride: bool = False,
        include_device: bool = False,
        colored: bool = False,
        expanded_def: bool = False,
        record_func: bool = False,
        additional_meta: list[str] | None = None,
    ) -> PythonCode:
        del include_stride, include_device, colored, record_func, additional_meta
        ordered_nodes = list(nodes)
        root = root_module if root_module is not None else "self"
        use_root = root_module is not None or any(
            node.op in {"get_attr", "call_module"} for node in ordered_nodes
        )
        if not use_root:
            root = None

        globals_: dict[str, Any] = {}
        node_names: dict[Node, str] = {}
        free_vars: list[str] = []
        body: list[str] = []
        counter_lines: list[tuple[int, int]] = []

        def add_global(name_hint: str, value: Any) -> str:
            existing = namespace._lookup(value)
            if existing is not None:
                globals_.setdefault(existing, value)
                return existing
            name = namespace.create_name(_snake_case(name_hint), value)
            globals_[name] = value
            return name

        def node_ref(node: Node) -> str:
            existing = node_names.get(node)
            if existing is None:
                existing = namespace.create_name(node.name, node)
                node_names[node] = existing
            return existing

        def render(value: Any) -> str:
            if isinstance(value, Node):
                return node_ref(value)
            if value is None or value is True or value is False or value is Ellipsis:
                return repr(value)
            if isinstance(value, (str, bytes, int, float, complex)):
                if isinstance(value, complex) and (
                    not math.isfinite(value.real) or not math.isfinite(value.imag)
                ):
                    return f"complex({value.real!r}, {value.imag!r})"
                return repr(value)
            if isinstance(value, enum.Enum):
                cls = add_global(value.__class__.__name__, value.__class__)
                return f"{cls}.{value.name}"
            if isinstance(value, tuple) and hasattr(value, "_fields"):
                cls = add_global(_qualified_name(type(value)), type(value))
                return f"{cls}({', '.join(render(item) for item in value)})"
            if isinstance(value, tuple):
                items = ", ".join(render(item) for item in value)
                return f"({items}{',' if len(value) == 1 else ''})"
            if isinstance(value, immutable_list):
                ctor = add_global("immutable_list", immutable_list)
                return f"{ctor}([{', '.join(render(item) for item in value)}])"
            if isinstance(value, list):
                return "[" + ", ".join(render(item) for item in value) + "]"
            if isinstance(value, immutable_dict):
                items = ", ".join(
                    f"{render(key)}: {render(item)}" for key, item in value.items()
                )
                ctor = add_global("immutable_dict", immutable_dict)
                return f"{ctor}({{{items}}})"
            if isinstance(value, dict):
                return "{" + ", ".join(
                    f"{render(key)}: {render(item)}" for key, item in value.items()
                ) + "}"
            if isinstance(value, slice):
                return f"slice({render(value.start)}, {render(value.stop)}, {render(value.step)})"
            if isinstance(value, range):
                return f"range({render(value.start)}, {render(value.stop)}, {render(value.step)})"
            if isinstance(value, set):
                return "{" + ", ".join(render(item) for item in value) + "}"
            if isinstance(value, frozenset):
                return "frozenset({" + ", ".join(render(item) for item in value) + "})"
            if isinstance(value, type):
                return add_global(_qualified_name(value), value)
            if callable(value):
                return add_global(_qualified_name(value), value)
            return add_global(type(value).__name__, value)

        def argument_list(args: Sequence[Any], kwargs: Mapping[str, Any]) -> str:
            rendered = [render(item) for item in args]
            simple_kwargs: list[str] = []
            mapping_kwargs: list[str] = []
            for key, value in kwargs.items():
                value_text = render(value)
                if isinstance(key, str) and key.isidentifier() and not keyword.iskeyword(key):
                    simple_kwargs.append(f"{key}={value_text}")
                else:
                    mapping_kwargs.append(f"{render(key)}: {value_text}")
            rendered.extend(simple_kwargs)
            if mapping_kwargs:
                rendered.append("**{" + ", ".join(mapping_kwargs) + "}")
            return ", ".join(rendered)

        def emit(node: Node, node_index: int) -> list[str]:
            result = [f"# COUNTER: {node_index}"]
            counter_lines.append((node_index, len(body) + len(result)))
            if node.op == "placeholder":
                if not isinstance(node.target, str):
                    raise GraphCaptureError("placeholder target must be a string")
                default = f"={render(node.args[0])}" if node.args else ""
                free_vars.append(f"{node.target}{default}")
                if node.target.lstrip("*") != node_ref(node):
                    result.append(f"{node_ref(node)} = {node.target.lstrip('*')}")
                return result
            if node.op == "output":
                result.append(self.generate_output(node.args[0], repr_fn=render))
                return result
            lhs = node_ref(node)
            if node.op == "call_method":
                if not node.args:
                    raise GraphCaptureError("call_method node has no receiver")
                rhs = f"{_format_target_path(render(node.args[0]), str(node.target))}({argument_list(node.args[1:], node.kwargs)})"
            elif node.op == "call_module":
                if root is None:
                    raise GraphCaptureError("call_module requires a root module")
                rhs = f"{_format_target_path(root, str(node.target))}({argument_list(node.args, node.kwargs)})"
            elif node.op == "get_attr":
                if root is None:
                    raise GraphCaptureError("get_attr requires a root module")
                rhs = _format_target_path(root, str(node.target))
            elif node.op == "call_function":
                target = node.target
                target_name = getattr(target, "__name__", "")
                target_module = getattr(target, "__module__", "")
                if target_module in {"_operator", "operator"} and target_name in magic_methods and not node.kwargs:
                    rhs = magic_methods[target_name].format(*(render(item) for item in node.args))
                elif target_module in {"_operator", "operator"} and target_name in _inplace_methods and not node.kwargs:
                    rhs = _inplace_methods[target_name].format(*(render(item) for item in node.args))
                else:
                    rhs = f"{render(target)}({argument_list(node.args, node.kwargs)})"
            else:
                raise GraphCaptureError(f"unsupported graph node kind: {node.op}")
            if node.op == "call_function" and target_module in {"_operator", "operator"} and target_name in _inplace_methods and not node.kwargs:
                return [*result, rhs, f"{lhs} = {render(node.args[0])}"]
            return [*result, f"{lhs} = {rhs}"]

        for name, value in self.additional_globals():
            add_global(name, value)
        for index, node in enumerate(ordered_nodes):
            body.extend(emit(node, index))
        if not body:
            body = ["pass"]
        if self._body_transformer is not None:
            body = list(self._body_transformer(body))
        output_nodes = [node for node in ordered_nodes if node.op == "output"]
        if not output_nodes:
            raise GraphCaptureError("graph has no output node")
        output_type = output_nodes[-1].type
        return_annotation = "" if output_type is None else f" -> {_type_repr(output_type)}"
        # A qualified-name annotation (`tensorplay._C.TensorBase`) only
        # resolves when the referenced module is in the exec globals; the
        # namespace sweep below only covers values rendered from the graph.
        if return_annotation:
            annotation_module = (
                return_annotation.split(" -> ", 1)[1].lstrip().split(".", 1)[0]
            )
            if (
                annotation_module
                and annotation_module not in {"None", "typing"}
                and annotation_module not in globals_
            ):
                try:
                    import importlib as _importlib

                    globals_[annotation_module] = _importlib.import_module(
                        annotation_module
                    )
                except ImportError:
                    pass
        prologue = self.gen_fn_def(
            free_vars,
            return_annotation,
            expanded_def=expanded_def,
            root_module=root,
        )
        source_lines = [prologue, *(f"    {line}" for line in body)]
        source = "\n".join(source_lines) + "\n"
        line_map: dict[int, int | None] = {}
        for line_number, line in enumerate(source.splitlines(), 1):
            line_map[line_number] = None
            if line.lstrip().startswith("# COUNTER:"):
                try:
                    line_map[line_number] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        return PythonCode(source, globals_, line_map, prologue.count("\n") + 1)


@compatibility(is_backward_compatible=False)
class _BoxedCodeGen(CodeGen):
    """Generate a function accepting and clearing one mutable argument list."""

    def gen_fn_def(
        self,
        free_vars: list[str],
        maybe_return_annotation: str = "",
        *,
        expanded_def: bool = False,
        root_module: str | None = "self",
    ) -> str:
        del expanded_def, root_module
        names = [item.split(":", 1)[0].split("=", 1)[0].strip() for item in free_vars]
        lines = [f"def {self._func_name}(self, args_list){maybe_return_annotation}:"]
        if names:
            lines.append(f"    {', '.join(names)}, = args_list")
            lines.append("    args_list.clear()")
        return "\n".join(lines)


@compatibility(is_backward_compatible=False)
class _PyTreeCodeGen(CodeGen):
    """Code generator that flattens structured inputs and restores outputs."""

    def __init__(self, pytree_info: _PyTreeInfo) -> None:
        super().__init__()
        self.pytree_info = pytree_info

    def process_inputs(self, *inputs: Any) -> Any:
        from ._pytree import tree_flatten

        leaves, _ = tree_flatten(inputs)
        return leaves

    def process_outputs(self, out: Any) -> Any:
        if self.pytree_info.out_spec is None:
            return out
        from ._pytree import tree_unflatten

        values = out if isinstance(out, (list, tuple)) else [out]
        return tree_unflatten(values, self.pytree_info.out_spec)


@compatibility(is_backward_compatible=False)
class _ExportCodeGen(_PyTreeCodeGen):
    """Code generator for explicit input and output shuffle graphs."""

    def __init__(
        self,
        pytree_info: _PyTreeInfo,
        in_shuffle_graph: Any,
        out_shuffle_graph: Any,
        tree_leaf_names: list[str],
        root: Any = None,
    ) -> None:
        super().__init__(pytree_info)
        self.in_shuffle_graph = in_shuffle_graph
        self.out_shuffle_graph = out_shuffle_graph
        self.tree_leaf_names = tree_leaf_names
        self.root = root
        self.flat_args: Any = None

    def process_inputs(self, *inputs: Any) -> Any:
        flat = super().process_inputs(*inputs)
        self.flat_args = (self.root, *flat) if self.root is not None else flat
        return self.in_shuffle_graph(*self.flat_args)

    def process_outputs(self, out: Any) -> Any:
        shuffled = self.out_shuffle_graph(*self.flat_args, *out)
        return super().process_outputs(shuffled)


class Graph:
    """Mutable, topologically ordered intermediate representation."""

    def __init__(
        self,
        owning_module: Any = None,
        tracer_cls: type[Any] | None = None,
        tracer_extras: dict[str, Any] | None = None,
    ) -> None:
        self._nodes = _NodeList(self)
        self._index: dict[Node, int] = {}
        self._live_names: set[str] = set()
        self._next_sort_key = 0
        self._insert: Callable[[Node], None] = self._append_node
        self._owning_module = owning_module
        self._tracer_cls = tracer_cls
        self._tracer_extras = tracer_extras
        self._graph_namespace = _Namespace()
        self._find_nodes_lookup_table = _FindNodesLookupTable()
        self._codegen: CodeGen = CodeGen()
        self._codegen_hooks: list[Callable[[list[str]], list[str]]] = []
        self._co_fields: dict[str, Any] = {}

    def __getstate__(self) -> dict[str, Any]:
        # The owning module is a runtime back-reference.  Keeping it in a
        # graph serialization creates a graph/module cycle and duplicates the
        # executable wrapper while the graph itself is being restored.
        return {
            "nodes": tuple(self.nodes),
            "tracer_cls": self._tracer_cls,
            "tracer_extras": self._tracer_extras,
            "codegen": self._codegen,
            "codegen_hooks": self._codegen_hooks,
            "co_fields": self._co_fields,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._nodes = _NodeList(self)
        self._index = {}
        self._live_names = set()
        self._next_sort_key = 0
        self._insert = self._append_node
        self._owning_module = None
        self._tracer_cls = state.get("tracer_cls")
        self._tracer_extras = state.get("tracer_extras")
        self._graph_namespace = _Namespace()
        self._find_nodes_lookup_table = _FindNodesLookupTable()
        self._codegen = state.get("codegen", CodeGen())
        self._codegen_hooks = list(state.get("codegen_hooks", ()))
        self._co_fields = dict(state.get("co_fields", {}))
        nodes = tuple(state.get("nodes", ()))
        list.extend(self._nodes, nodes)
        self._sync_nodes(self._nodes)

    @property
    def nodes(self) -> _NodeList:
        return self._nodes

    @nodes.setter
    def nodes(self, value: Iterable[Node]) -> None:
        old = tuple(getattr(self, "_nodes", ()))
        self._nodes = _NodeList(self, value)
        self._sync_nodes(self._nodes, old)

    def _sync_nodes(self, values: Iterable[Node], old: Iterable[Node] = ()) -> None:
        current = list(values)
        current_set = set(current)
        for node in old:
            if node not in current_set and node.graph is self:
                node.graph = None
                node._erased = True
                self._live_names.discard(node.name)
        for index, node in enumerate(current):
            node.graph = self
            node._erased = False
            node._sort_key = index
        self._index = {node: index for index, node in enumerate(current)}
        self._live_names = {node.name for node in current}
        self._next_sort_key = len(current)
        if hasattr(self, "_find_nodes_lookup_table"):
            self._find_nodes_lookup_table.rebuild(current)

    def _append_node(self, node: Node) -> None:
        self.nodes.append(node)

    def _remove_node(self, node: Node) -> None:
        if node.graph is not self:
            raise GraphCaptureError(f"attempting to remove {node} from the wrong graph")
        if node.users:
            raise GraphCaptureError(
                f"cannot erase {node.name} because it still has {len(node.users)} user(s)"
            )
        for input_node in node.all_input_nodes:
            input_node.users.discard(node)
        old = tuple(self.nodes)
        list.remove(self.nodes, node)
        self._sync_nodes(self.nodes, old)
        # Free the name so a replacement node (an output re-registration,
        # a rewrite inserting over an erased site) can reclaim it.
        self._graph_namespace.release_name(node.name)
        node.graph = None
        node._erased = True
        node._args = ()
        node._kwargs = {}
        node._input_nodes.clear()
        node.users.clear()
        self._notify_owner_mutated()

    def _notify_owner_mutated(self) -> None:
        """Drop the owning module's generated executor after graph surgery.

        The generated code is a snapshot of the graph at recompile time; a
        rewrite (substitution, erasure) that kept executing it would run the
        pre-rewrite program.  Calls fall back to the live interpreter until
        the module explicitly recompiles.
        """

        owner = self.owning_module
        invalidate = getattr(owner, "_invalidate_compiled_executor", None)
        if callable(invalidate):
            invalidate()

    @property
    def owning_module(self) -> Any:
        return self._owning_module

    @owning_module.setter
    def owning_module(self, value: Any) -> None:
        self._owning_module = value

    @property
    def placeholders(self) -> list[Node]:
        return [node for node in self.nodes if node.op == "placeholder"]

    @property
    def outputs(self) -> list[Node]:
        return [node for node in self.nodes if node.op == "output"]

    @property
    def output_node(self) -> Node:
        outputs = self.outputs
        if not outputs:
            raise GraphCaptureError("graph has no output node")
        return outputs[-1]

    def _target_to_str(self, target: Any) -> str:
        value = target if isinstance(target, str) else _callable_name(target)
        if value.startswith("__") and value.endswith("__"):
            value = value[2:-2]
        # Module and attribute targets are named after the referenced object,
        # not its qualified path: "backbone.conv1" derives "conv1" and
        # "backbone.conv1.weight" derives "weight".
        value = value.rsplit(".", 1)[-1]
        return _snake_case(value)

    def _create_unique_name(self, candidate: str) -> str:
        return self._graph_namespace.create_name(candidate, None)

    def create_node(
        self,
        op: str,
        target: Any,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        name: str | None = None,
        type_expr: Any | None = None,
        *,
        return_type: Any | None = None,
    ) -> Node:
        if op not in _legal_ops:
            raise GraphCaptureError(f"unsupported graph operation kind: {op!r}")
        if args is None:
            args = ()
        if not isinstance(args, tuple):
            raise AssertionError(f"args must be a tuple, got {type(args)}")
        if kwargs is None:
            kwargs = immutable_dict()
        if not isinstance(kwargs, dict):
            raise AssertionError(f"kwargs must be a dict, got {type(kwargs)}")

        def normalize(value: Any) -> Any:
            return value.node if isinstance(value, Proxy) else value

        args = map_arg(args, normalize)
        kwargs = map_arg(kwargs, normalize)
        if not isinstance(args, tuple) or not isinstance(kwargs, dict):
            raise AssertionError("node arguments lost their container types")
        candidate = name if name is not None else self._target_to_str(target)
        node_name = self._graph_namespace.create_name(candidate, None)
        node = Node(
            self,
            node_name,
            op,
            target,
            args,
            kwargs,
            return_type if return_type is not None else type_expr,
        )
        node._sort_key = self._next_sort_key
        self._next_sort_key += 1
        if self.owning_module is not None:
            for hook in getattr(self.owning_module, "_create_node_hooks", ()):
                hook(node)
        self._insert(node)
        self._graph_namespace.associate_name_with_obj(node.name, node)
        return node

    def placeholder(
        self,
        name: str,
        type_expr: Any = None,
        default_value: Any = inspect.Signature.empty,
        *,
        default: Any = inspect.Signature.empty,
    ) -> Node:
        if default is not inspect.Signature.empty:
            if default_value is not inspect.Signature.empty:
                raise TypeError("placeholder received two default values")
            default_value = default
        args = () if default_value is inspect.Signature.empty else (default_value,)
        node = self.create_node("placeholder", name, args, {}, name=name, type_expr=type_expr)
        if args:
            node.meta["default"] = args[0]
        return node

    def get_attr(self, qualified_name: str, type_expr: Any = None) -> Node:
        return self.create_node("get_attr", qualified_name, type_expr=type_expr)

    def call_module(
        self,
        module_name: str,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        type_expr: Any = None,
        *,
        name: str | None = None,
    ) -> Node:
        return self.create_node("call_module", module_name, args, kwargs, name, type_expr)

    def call_method(
        self,
        method_name: str,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        type_expr: Any = None,
        *,
        name: str | None = None,
    ) -> Node:
        return self.create_node("call_method", method_name, args, kwargs, name, type_expr)

    def call_function(
        self,
        the_function: Callable[..., Any],
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        type_expr: Any = None,
        name: str | None = None,
    ) -> Node:
        return self.create_node("call_function", the_function, args, kwargs, name, type_expr)

    def output(self, result: Any, type_expr: Any = None) -> Node:
        for old_output in list(self.outputs):
            self._remove_node(old_output)
        return self.create_node("output", "output", (result,), {}, "output", type_expr)

    def find_nodes(
        self,
        *,
        op: str | None = None,
        target: Any | None = None,
        sort: bool = True,
    ) -> list[Node]:
        if op is None:
            result = [
                node
                for node in self.nodes
                if target is None or node.target is target or node.target == target
            ]
        else:
            result = self._find_nodes_lookup_table.find_nodes(op, target)
        if sort:
            result.sort(key=lambda node: self._index.get(node, -1))
        return result

    def graph_copy(
        self,
        graph: "Graph",
        val_map: dict[Node, Any],
        return_output_node: bool = False,
    ) -> Any:
        for node in graph.nodes:
            if node in val_map:
                continue
            if node.op == "output":
                result = map_arg(node.args[0], lambda value: val_map[value])
                return (result, node) if return_output_node else result
            val_map[node] = self.node_copy(node, lambda value: val_map[value])
        return None

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> "Graph":
        if memo is None:
            memo = {}
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        graph = type(self)(
            tracer_cls=self._tracer_cls,
            tracer_extras=copy.deepcopy(self._tracer_extras, memo),
        )
        memo[id(self)] = graph
        val_map: dict[Node, Node] = {}
        for node in self.nodes:
            if node.op == "output":
                continue
            copied = graph.node_copy(node, lambda value: val_map[value])
            val_map[node] = copied
            memo[id(node)] = copied
            copied.meta = copy.deepcopy(node.meta, memo)
        for node in self.nodes:
            if node.op == "output":
                output = graph.output(map_arg(node.args[0], lambda value: val_map[value]), node.type)
                output.meta = copy.deepcopy(node.meta, memo)
                memo[id(node)] = output
        graph._codegen = copy.deepcopy(self._codegen, memo)
        graph._co_fields = copy.deepcopy(self._co_fields, memo)
        return graph

    def node_copy(
        self,
        node: Node,
        arg_transform: Callable[[Node], Any] = lambda value: value,
    ) -> Node:
        args = map_arg(node.args, arg_transform)
        kwargs = map_arg(node.kwargs, arg_transform)
        if not isinstance(args, tuple) or not isinstance(kwargs, dict):
            raise AssertionError("node copy must preserve args and kwargs containers")
        result = self.create_node(node.op, node.target, args, kwargs, node.name, node.type)
        result.meta = copy.copy(node.meta)
        result.tag = node.tag
        result.size_bytes = node.size_bytes
        return result

    def erase_node(self, to_erase: Node) -> None:
        if to_erase.graph is not self:
            raise GraphCaptureError("attempting to remove a node from the wrong graph")
        if to_erase._erased:
            return
        for hook in getattr(self.owning_module, "_erase_node_hooks", ()):
            hook(to_erase)
        self._remove_node(to_erase)

    def inserting_before(self, node: Node | None = None) -> _InsertPoint:
        if node is None:
            return _InsertPoint(self, lambda value: self.nodes.insert(0, value))
        if node.graph is not self:
            raise GraphCaptureError(f"{node.name} is not part of this graph")
        cursor = [self._index[node]]

        def insert(value: Node) -> None:
            position = cursor[0]
            if position >= len(self.nodes) or self.nodes[position] is not node:
                try:
                    position = self.nodes.index(node)
                except ValueError as exc:
                    raise GraphCaptureError(f"insertion anchor {node.name} was erased") from exc
            self.nodes.insert(position, value)
            cursor[0] = position + 1

        return _InsertPoint(self, insert)

    def inserting_after(self, node: Node | None = None) -> _InsertPoint:
        if node is None:
            return _InsertPoint(self, self._append_node)
        if node.graph is not self:
            raise GraphCaptureError(f"{node.name} is not part of this graph")
        cursor = [self._index[node] + 1]

        def insert(value: Node) -> None:
            position = cursor[0]
            if position > len(self.nodes) or (
                position == len(self.nodes) and (not self.nodes or self.nodes[-1] is not node)
            ):
                try:
                    position = self.nodes.index(node) + 1
                except ValueError as exc:
                    raise GraphCaptureError(f"insertion anchor {node.name} was erased") from exc
            self.nodes.insert(position, value)
            cursor[0] = position

        return _InsertPoint(self, insert)

    def _get_tensor_meta_val(self, tensor_node: Node) -> tuple[Any, str]:
        if "val" in tensor_node.meta:
            return tensor_node.meta["val"], "val"
        if "example_value" in tensor_node.meta:
            return tensor_node.meta["example_value"], "example_value"
        return None, "val"

    @staticmethod
    def _set_tensor_meta_val(node: Node, value: Any, key: str) -> None:
        node.meta[key] = value
        if key == "val" and "example_value" in node.meta:
            node.meta["example_value"] = value

    def create_size_node(self, tensor_node: Node, dim: int) -> Node:
        value, key = self._get_tensor_meta_val(tensor_node)
        node = self.call_function(_graph_size, (tensor_node, dim), name="sym_size")
        if value is not None:
            self._set_tensor_meta_val(node, value.size(dim), key)
        return node

    def create_stride_node(self, tensor_node: Node, dim: int) -> Node:
        value, key = self._get_tensor_meta_val(tensor_node)
        node = self.call_function(_graph_stride, (tensor_node, dim), name="sym_stride")
        if value is not None:
            self._set_tensor_meta_val(node, value.stride(dim), key)
        return node

    def create_storage_offset_node(self, tensor_node: Node) -> Node:
        value, key = self._get_tensor_meta_val(tensor_node)
        node = self.call_function(_graph_storage_offset, (tensor_node,), name="sym_storage_offset")
        if value is not None:
            self._set_tensor_meta_val(node, value.storage_offset(), key)
        return node

    def _resolve_unbacked_binding(
        self,
        producer: Node,
        keypath: tuple[Any, ...],
        lower_symint: Callable[[Any], Node | int],
    ) -> Node:
        from .experimental.symbolic_shapes import (
            CallMethodKey,
            ConvertIntKey,
            DivideByKey,
            InnerTensorKey,
            SequenceKey,
        )

        node = producer
        index = 0
        while index < len(keypath):
            key = keypath[index]
            next_key = keypath[index + 1] if index + 1 < len(keypath) else None
            if isinstance(key, CallMethodKey) and isinstance(next_key, SequenceKey):
                if key.name == "size":
                    node = self.create_size_node(node, next_key.index)
                elif key.name == "stride":
                    node = self.create_stride_node(node, next_key.index)
                else:
                    node = self.call_method(key.name, (node, next_key.index))
                index += 2
            elif isinstance(key, CallMethodKey):
                if key.name == "storage_offset":
                    node = self.create_storage_offset_node(node)
                else:
                    node = self.call_method(key.name, (node,))
                index += 1
            elif isinstance(key, SequenceKey):
                node = self.call_function(operator.getitem, (node, key.index))
                index += 1
            elif isinstance(key, ConvertIntKey):
                node = self.call_function(_graph_bool_to_int, (node,))
                index += 1
            elif isinstance(key, DivideByKey):
                divisor = key.divisor
                if hasattr(divisor, "expr"):
                    divisor = lower_symint(divisor)
                node = self.call_function(operator.floordiv, (node, divisor))
                index += 1
            elif isinstance(key, InnerTensorKey):
                node = self.call_function(_graph_getattr, (node, key.inner_name))
                index += 1
            else:
                raise GraphCaptureError(f"unrecognized symbolic keypath component {key!r}")
        return node

    @staticmethod
    def _symbolic_expr(value: Any) -> Any:
        import sympy

        return value.expr if hasattr(value, "expr") else sympy.sympify(value)

    def materialize_symints(self, values: Sequence[Any]) -> list[Any]:
        import math
        import sympy
        from sympy.logic.boolalg import BooleanAtom

        from .experimental.sym_node import SymNode

        expression_to_value: dict[sympy.Basic, Any] = {}
        symbol_to_node: dict[sympy.Symbol, Node] = {}
        size_sources: dict[sympy.Symbol, tuple[Node, int, int]] = {}
        binding_sources: dict[sympy.Symbol, tuple[Node, tuple[Any, ...]]] = {}

        def node_value(node: Node) -> Any:
            value = node.meta.get("val")
            return value if value is not None else node.meta.get("example_value")

        def symbolic_expr(value: Any) -> sympy.Basic:
            expression = self._symbolic_expr(value)
            if not isinstance(expression, sympy.Basic):
                expression = sympy.sympify(expression)
            return expression

        def record_shape_source(node: Node, value: Any) -> None:
            shape = getattr(value, "shape", None)
            if callable(shape):
                shape = shape()
            if shape is None:
                return
            for dim, item in enumerate(shape):
                if not isinstance(item, SymNode):
                    continue
                expression = symbolic_expr(item)
                if isinstance(expression, sympy.Symbol):
                    size_sources.setdefault(expression, (node, dim, 1))
                    continue
                if not isinstance(expression, sympy.Mul) or len(expression.args) != 2:
                    continue
                first, second = expression.args
                if isinstance(first, sympy.Integer) and isinstance(second, sympy.Symbol):
                    first, second = second, first
                if (
                    isinstance(first, sympy.Symbol)
                    and isinstance(second, sympy.Integer)
                    and int(second) > 0
                ):
                    size_sources.setdefault(first, (node, dim, int(second)))

        for node in self.nodes:
            value = node_value(node)
            if node.op == "placeholder" and isinstance(value, SymNode):
                expression = symbolic_expr(value)
                if isinstance(expression, sympy.Symbol):
                    symbol_to_node.setdefault(expression, node)
            if node.op == "placeholder":
                record_shape_source(node, value)
            symbol = node.meta.get("symbol")
            if isinstance(symbol, sympy.Symbol):
                symbol_to_node.setdefault(symbol, node)
            bindings = node.meta.get("unbacked_bindings")
            if bindings:
                for symbol, keypath in bindings.items():
                    if isinstance(symbol, sympy.Symbol):
                        binding_sources.setdefault(symbol, (node, tuple(keypath)))

        def set_node_value(node: Node, value: Any) -> None:
            node.meta["val"] = value
            if any(
                isinstance(input_node, Node)
                and "example_value" in input_node.meta
                for input_node in node.all_input_nodes
            ):
                node.meta["example_value"] = value

        def ensure_symbol(symbol: sympy.Symbol) -> Node:
            existing = symbol_to_node.get(symbol)
            if existing is not None:
                return existing
            source = size_sources.get(symbol)
            if source is not None:
                producer, dim, divisor = source
                result = self.create_size_node(producer, dim)
                if divisor != 1:
                    result = self.call_function(operator.floordiv, (result, divisor))
                symbol_to_node[symbol] = result
                return result
            binding = binding_sources.get(symbol)
            if binding is not None:
                producer, keypath = binding
                for item in keypath:
                    divisor = getattr(item, "divisor", None)
                    if hasattr(divisor, "expr"):
                        for free_symbol in symbolic_expr(divisor).free_symbols:
                            ensure_symbol(free_symbol)
                result = self._resolve_unbacked_binding(
                    producer,
                    keypath,
                    lower_symint=materialize,
                )
                symbol_to_node[symbol] = result
                return result
            raise GraphCaptureError(f"symbol {symbol} has no graph producer")

        def annotate_result(value: Any) -> Any:
            if not isinstance(value, Node):
                return value
            target = value.target
            if not callable(target):
                return value
            try:
                args = map_arg(value.args, lambda node: node_value(node))
                kwargs = map_arg(value.kwargs, lambda node: node_value(node))
                set_node_value(value, target(*args, **kwargs))
            except (AttributeError, TypeError, ValueError):
                pass
            return value

        def fold(function: Callable[..., Any], arguments: list[Any]) -> Any:
            if not arguments:
                raise GraphCaptureError(
                    f"cannot materialize empty {getattr(function, '__name__', function)}"
                )
            result = arguments[0]
            for argument in arguments[1:]:
                result = annotate_result(self.call_function(function, (result, argument)))
            return result

        comparison_functions: dict[Any, Callable[..., Any]] = {
            sympy.Eq: operator.eq,
            sympy.Ne: operator.ne,
            sympy.Gt: operator.gt,
            sympy.Lt: operator.lt,
            sympy.Ge: operator.ge,
            sympy.Le: operator.le,
        }
        boolean_functions: dict[Any, Callable[..., Any]] = {
            sympy.And: _graph_sym_and,
            sympy.Or: _graph_sym_or,
            sympy.Xor: _graph_sym_xor,
        }
        function_targets: dict[str, Callable[..., Any]] = {
            "lshift": operator.lshift,
            "rshift": operator.rshift,
            "trunc": _graph_sym_trunc,
            "to_float": _graph_sym_float,
            "is_integer": _graph_sym_is_integer,
            "bitwise_and": operator.and_,
            "bitwise_or": operator.or_,
            "bitwise_xor": operator.xor,
        }

        def materialize(value: Any) -> Any:
            if isinstance(value, Node):
                return value
            if isinstance(value, bool):
                return bool(value)
            if isinstance(value, int) and not isinstance(value, SymNode):
                return int(value)
            if isinstance(value, float) and not isinstance(value, SymNode):
                return float(value)

            expression = symbolic_expr(value)
            if expression in expression_to_value:
                return expression_to_value[expression]
            if isinstance(expression, BooleanAtom):
                result: Any = bool(expression)
            elif isinstance(expression, sympy.Integer):
                result = int(expression)
            elif isinstance(expression, sympy.Float):
                result = float(expression)
            elif isinstance(expression, sympy.Rational):
                result = int(expression) if expression.q == 1 else float(expression)
            elif isinstance(expression, sympy.Symbol):
                result = ensure_symbol(expression)
            elif isinstance(expression, sympy.Add):
                result = fold(operator.add, [materialize(item) for item in expression.args])
            elif isinstance(expression, sympy.Mul):
                result = fold(operator.mul, [materialize(item) for item in expression.args])
            elif isinstance(expression, sympy.Pow):
                result = annotate_result(
                    self.call_function(
                        operator.pow,
                        tuple(materialize(item) for item in expression.args),
                    )
                )
            elif isinstance(expression, sympy.Mod):
                result = annotate_result(
                    self.call_function(
                        operator.mod,
                        tuple(materialize(item) for item in expression.args),
                    )
                )
            elif expression.func is sympy.floor:
                result = annotate_result(
                    self.call_function(_graph_floor, (materialize(expression.args[0]),))
                )
            elif expression.func is sympy.ceiling:
                result = annotate_result(
                    self.call_function(_graph_ceil, (materialize(expression.args[0]),))
                )
            elif expression.func is sympy.Abs:
                result = annotate_result(
                    self.call_function(_graph_abs, (materialize(expression.args[0]),))
                )
            elif expression.func is sympy.Min:
                result = fold(_graph_sym_min, [materialize(item) for item in expression.args])
            elif expression.func is sympy.Max:
                result = fold(_graph_sym_max, [materialize(item) for item in expression.args])
            elif expression.func in comparison_functions:
                result = annotate_result(
                    self.call_function(
                        comparison_functions[expression.func],
                        tuple(materialize(item) for item in expression.args),
                    )
                )
            elif expression.func in boolean_functions:
                result = fold(
                    boolean_functions[expression.func],
                    [materialize(item) for item in expression.args],
                )
            elif expression.func is sympy.Not:
                result = annotate_result(
                    self.call_function(
                        _graph_sym_not,
                        (materialize(expression.args[0]),),
                    )
                )
            elif expression.func is sympy.Piecewise:
                pairs = list(expression.args)
                result = materialize(pairs[-1].args[0])
                for pair in reversed(pairs[:-1]):
                    result = annotate_result(
                        self.call_function(
                            _graph_sym_ite,
                            (
                                materialize(pair.args[1]),
                                materialize(pair.args[0]),
                                result,
                            ),
                        )
                    )
            else:
                function = function_targets.get(getattr(expression.func, "__name__", ""))
                if function is None:
                    raise GraphCaptureError(
                        f"cannot materialize symbolic expression {expression!r}"
                    )
                result = annotate_result(
                    self.call_function(
                        function,
                        tuple(materialize(item) for item in expression.args),
                    )
                )

            if isinstance(result, Node):
                expression_to_value[expression] = annotate_result(result)
            else:
                expression_to_value[expression] = result
            return result

        result_values: list[Any] = []
        for value in values:
            result = materialize(value)
            if isinstance(result, Node) and isinstance(value, SymNode):
                set_node_value(result, value)
            result_values.append(result)
        return result_values

    def materialize_symint(self, value: Any) -> Any:
        return self.materialize_symints([value])[0]

    def process_inputs(self, *args: Any) -> Any:
        return self._codegen.process_inputs(*args)

    def process_outputs(self, value: Any) -> Any:
        return self._codegen.process_outputs(value)

    def set_codegen(self, codegen: CodeGen) -> None:
        self._codegen = codegen

    @contextmanager
    def on_generate_code(
        self,
        make_transformer: Callable[
            [Callable[[list[str]], list[str]] | None],
            Callable[[list[str]], list[str]],
        ],
    ) -> Iterator[None]:
        previous_hooks = list(self._codegen_hooks)
        current = self._codegen_hooks[-1] if self._codegen_hooks else None
        self._codegen_hooks.append(make_transformer(current))
        previous_transformer = self._codegen._body_transformer
        self._codegen._body_transformer = make_transformer(previous_transformer)
        try:
            yield
        finally:
            self._codegen_hooks = previous_hooks
            self._codegen._body_transformer = previous_transformer

    def _apply_code_transformers(self, lines: list[str]) -> list[str]:
        if not lines:
            return []
        body = [line[4:] if line.startswith("    ") else line for line in lines[1:]]
        for transformer in self._codegen_hooks:
            transformed = transformer(body)
            if transformed is None:
                raise GraphCaptureError("code generation hook returned None")
            body = [str(line).rstrip("\n") for line in transformed]
        return [lines[0], *[(f"    {line}" if line else "") for line in body]]

    def _python_code(self, root_module: str | None = "self", **kwargs: Any) -> PythonCode:
        return self._codegen._gen_python_code(self.nodes, root_module, _Namespace(), **kwargs)

    def python_code(self, root_module: str | None = None, **kwargs: Any) -> PythonCode | str:
        code = self._python_code("self" if root_module is None else root_module, **kwargs)
        return code.src if root_module is None else code

    def eliminate_dead_code(self) -> int:
        return dead_code_elimination(self)

    def lint(self) -> None:
        positions = self._index
        for node in self.nodes:
            if node.graph is not self:
                raise GraphCaptureError(f"node {node.name} is attached to another graph")
            for input_node in (*_iter_nodes(node.args), *_iter_nodes(node.kwargs)):
                if input_node.graph is not self:
                    raise GraphCaptureError(
                        f"node {node.name} references {input_node.name} from another graph"
                    )
                if positions.get(input_node, len(self.nodes)) >= positions[node]:
                    raise GraphCaptureError(
                        f"graph is not topologically ordered: {input_node} -> {node}"
                    )
        if len(self.outputs) > 1:
            raise GraphCaptureError("graph contains more than one output node")

    def _clear_nodes(self) -> None:
        old = tuple(self.nodes)
        list.clear(self.nodes)
        self._sync_nodes(self.nodes, old)

    def _format_value(self, value: Any) -> str:
        if isinstance(value, Node):
            return value.name
        if isinstance(value, tuple):
            items = ", ".join(self._format_value(item) for item in value)
            return f"({items}{',' if len(value) == 1 else ''})"
        if isinstance(value, list):
            return "[" + ", ".join(self._format_value(item) for item in value) + "]"
        if isinstance(value, dict):
            return "{" + ", ".join(
                f"{key!r}: {self._format_value(item)}" for key, item in value.items()
            ) + "}"
        if isinstance(value, slice):
            return f"slice({self._format_value(value.start)}, {self._format_value(value.stop)}, {self._format_value(value.step)})"
        return _format_target(value) if callable(value) else repr(value)

    def to_dot(
        self,
        *,
        graph_name: str = "TensorPlayGraph",
        rankdir: str = "TB",
        show_shapes: bool = True,
    ) -> str:
        styles = {
            "placeholder": ("ellipse", "#aec7e8"),
            "get_attr": ("diamond", "#ffbb78"),
            "call_function": ("box", "#ffffb3"),
            "call_method": ("box", "#ffffb3"),
            "call_module": ("component", "#98df8a"),
            "output": ("ellipse", "#c7e9c0"),
        }

        def escape(value: Any) -> str:
            return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        lines = [
            f"digraph {escape(graph_name)} {{",
            f"    rankdir={rankdir};",
            '    graph [fontname="monospace"];',
            '    node [fontname="monospace" fontsize=10 style=filled];',
            '    edge [fontsize=9];',
        ]
        for node in self.nodes:
            shape, color = styles.get(node.op, ("box", "#d9d9d9"))
            label = f"{node.name}\\n{node.op}[{_format_target(node.target)}]"
            attrs = [f'label="{escape(label)}"']
            if show_shapes and node.meta.get("tensor_shape") is not None:
                attrs.append(f'tooltip="{escape(node.meta["tensor_shape"])}"')
            lines.append(
                f'    "{escape(node.name)}" [shape={shape}, fillcolor="{color}", {", ".join(attrs)}];'
            )
        for node in self.nodes:
            for input_node in _iter_nodes(node.args):
                lines.append(f'    "{escape(input_node.name)}" -> "{escape(node.name)}";')
            for key, value in node.kwargs.items():
                for input_node in _iter_nodes(value):
                    lines.append(
                        f'    "{escape(input_node.name)}" -> "{escape(node.name)}" [label="{escape(key)}"];'
                    )
        lines.append("}")
        return "\n".join(lines)

    def draw(self, filename: str, format: str | None = None, *, rankdir: str = "TB") -> str:
        import shutil
        import subprocess
        from pathlib import Path

        path = Path(filename)
        stem = path.with_suffix("")
        image_format = format or path.suffix.lstrip(".") or "png"
        dot_path = Path(str(stem) + ".gv")
        dot_path.write_text(self.to_dot(rankdir=rankdir))
        dot = shutil.which("dot")
        if dot is None:
            raise RuntimeError(f"Graphviz executable 'dot' not found; wrote {dot_path}")
        output = f"{stem}.{image_format}"
        import subprocess as _subprocess

        _subprocess.run([dot, f"-T{image_format}", str(dot_path), "-o", output], check=True)
        return output

    def __str__(self) -> str:
        lines = ["graph("]
        for node in self.nodes:
            text = node.format_node(include_tensor_metadata=True)
            if text is not None:
                lines.append(f"    {text}")
        lines.append(")")
        return "\n".join(lines)

    def print_tabular(self) -> str:
        rows = [("opcode", "name", "target", "args", "kwargs")]
        rows.extend(
            (
                node.op,
                node.name,
                node._pretty_print_target(node.target),
                repr(node.args),
                repr(node.kwargs),
            )
            for node in self.nodes
        )
        widths = [max(len(str(row[index])) for row in rows) for index in range(len(rows[0]))]
        result = "\n".join(
            "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))
            for row in rows
        )
        print(result)
        return result


def dead_code_elimination(graph: Graph) -> int:
    live: set[Node] = set()
    worklist = list(graph.outputs)
    worklist.extend(node for node in graph.nodes if node.is_impure())
    while worklist:
        node = worklist.pop()
        if node in live:
            continue
        live.add(node)
        worklist.extend(_iter_nodes(node.args))
        worklist.extend(_iter_nodes(node.kwargs))
    old_nodes = list(graph.nodes)
    kept = [node for node in old_nodes if node in live or node.op == "placeholder"]
    removed = [node for node in old_nodes if node not in kept]
    if not removed:
        return 0
    list.__setitem__(graph.nodes, slice(None), kept)
    graph._sync_nodes(graph.nodes, old_nodes)
    for node in removed:
        node.users.clear()
        node._args = ()
        node._kwargs = {}
        node._input_nodes.clear()
        node.graph = None
        node._erased = True
    for node in graph.nodes:
        node.users.clear()
    for node in graph.nodes:
        for input_node in node.all_input_nodes:
            input_node.users.add(node)
    return len(removed)
