"""Self-contained native schema engine.

Parses the operator schema dialect used by config/native_functions.yaml:

    name.overload(Tensor self, int dim=-1, *, bool keepdim=False,
                  Tensor(a!) out) -> (Tensor values, Tensor(b!) indices)

and produces the record model the generators consume: operator names with
inplace/functional suffix rules, argument buckets (positional, self,
keyword-only, out), tensor-options clusters, kinds (functional / inplace /
out), view metadata, and structured groups.  No external schema package is
imported; the parser, type algebra and alias tables live in this file, while
the tag registry is versioned under config/tags.yaml for self-contained builds.
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import yaml


# ---------------------------------------------------------------------------
# Operator names
# ---------------------------------------------------------------------------

class BaseOperatorName:
    """name / name. / name_ / name.functional / __dunder__ spellings."""

    __slots__ = ("base", "inplace", "dunder_method", "functional_overload")

    def __init__(self, base: str, inplace: bool = False,
                 dunder_method: bool = False,
                 functional_overload: bool = False):
        self.base = base
        self.inplace = inplace
        self.dunder_method = dunder_method
        self.functional_overload = functional_overload

    def __str__(self) -> str:
        if self.dunder_method:
            return f"__{'i' if self.inplace else ''}{self.base}__"
        out = self.base
        if self.functional_overload:
            out += ".functional"
        if self.inplace:
            out += "_"
        return out

    def __eq__(self, other) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))


class OperatorName:
    __slots__ = ("name", "overload_name")

    def __init__(self, name: BaseOperatorName, overload_name: str):
        self.name = name
        self.overload_name = overload_name

    def __str__(self) -> str:
        if self.overload_name:
            return f"{self.name}.{self.overload_name}"
        return str(self.name)

    def __repr__(self) -> str:
        return f"OperatorName({self!s})"

    def __eq__(self, other) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))


_DUNDER_RE = re.compile(r"^__i?(?P<base>[a-z][a-z0-9]*(_[a-z0-9]+)*)__$")


def _parse_operator_name(text: str) -> OperatorName:
    overload_name = ""
    if "." in text:
        text, _, overload_name = text.partition(".")
        if "." in overload_name:
            raise ValueError(f"operator name has two overload parts: {text}")
    m = _DUNDER_RE.match(text)
    if m is not None:
        base = m.group("base")
        inplace = text.startswith("__i")
        return OperatorName(
            BaseOperatorName(base, inplace=inplace, dunder_method=True),
            overload_name)
    inplace = text.endswith("_")
    if inplace:
        text = text[:-1]
    return OperatorName(BaseOperatorName(text, inplace=inplace),
                        overload_name)


# ---------------------------------------------------------------------------
# Type expressions
# ---------------------------------------------------------------------------

class TypeExpr:
    """Schema type spellings.  `Tensor?[]` is a list of optional tensors
    (elem_opt); `Tensor[]?` is an optional list (is_opt); both flags can
    combine.  `mutability` marks a write annotation (trailing '!'), `alias`
    a read-only alias annotation."""

    __slots__ = ("name", "is_opt", "is_list", "size", "mutability", "alias",
                 "elem_opt")

    def __init__(self, name: str, is_opt: bool = False, is_list: bool = False,
                 size: int | None = None, mutability: str | None = None,
                 alias: str | None = None, elem_opt: bool = False):
        self.name = name
        self.is_opt = is_opt
        self.is_list = is_list
        self.size = size
        self.mutability = mutability
        self.alias = alias
        self.elem_opt = elem_opt

    def __str__(self) -> str:
        s = self.name
        if self.mutability is not None:
            s = f"{s}({self.mutability}!)"
        elif self.alias is not None:
            s = f"{s}({self.alias})"
        if self.elem_opt:
            s += "?"
        if self.is_list:
            s += f"[{'' if self.size is None else self.size}]"
        if self.is_opt:
            s += "?"
        return s


    def __eq__(self, other) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    def is_tensor_like(self) -> bool:
        return self.name == "Tensor"


_TYPE_RE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
    r"\((?P<ann>[a-zA-Z](?:\s->\s\*)?(?P<bang>!)?|a\s->\s\*)\)"
    r"(?P<elem1>\?(?=\[))?"
    r"(?:\[(?P<size>\d*)\])?"
    r"(?P<opt>\?)?$"
)
_TYPE_NOANN_RE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
    r"(?P<elem1>\?(?=\[))?"
    r"(?:\[(?P<size>\d*)\])?"
    r"(?P<opt>\?)?$"
)


def _parse_type(text: str) -> TypeExpr:
    stripped = text.strip()
    m = _TYPE_RE.match(stripped)
    if m is None:
        m = _TYPE_NOANN_RE.match(stripped)
    if m is None:
        raise ValueError(f"cannot parse schema type: {text!r}")
    ann = m.groupdict().get("ann")
    name = m.group("name")
    bang = bool(m.groupdict().get("bang"))
    is_write = ann is not None and ann.endswith("!")
    plain_ann = ann[:-1] if is_write else ann
    return TypeExpr(
        name=name,
        is_opt=bool(m.group("opt")),
        is_list=m.group("size") is not None,
        size=int(m.group("size")) if m.group("size") else None,
        mutability=plain_ann if is_write else None,
        alias=plain_ann if (ann is not None and not is_write) else None,
        elem_opt=bool(m.group("elem1")),
    )


class OptionalType(TypeExpr):
    def __init__(self, elem: TypeExpr):
        super().__init__(elem.name, is_opt=True, is_list=elem.is_list,
                         size=elem.size, mutability=elem.mutability,
                         alias=elem.alias)
        self.elem = elem


class ListType(TypeExpr):
    def __init__(self, elem: TypeExpr, size: int | None = None):
        super().__init__(elem.name, is_opt=elem.is_opt, is_list=True,
                         size=size, mutability=elem.mutability,
                         alias=elem.alias)
        self.elem = elem


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

class Argument:
    __slots__ = ("name", "type", "default", "annotation", "kwonly")

    def __init__(self, name: str, type_: TypeExpr, default=None,
                 annotation=None, kwonly: bool = False):
        self.name = name
        self.type = type_
        self.default = default
        self.annotation = annotation
        self.kwonly = kwonly

    def __str__(self) -> str:
        s = f"{self.type} {self.name}"
        if self.default is not None:
            s += f"={self.default}"
        return s


class SelfArgument(Argument):
    def __init__(self, argument: Argument):
        super().__init__(argument.name, argument.type, argument.default,
                         argument.annotation, argument.kwonly)
        self.argument = argument


class Annotation:
    __slots__ = ("kind", "write")

    def __init__(self, kind: str, write: bool):
        if not re.fullmatch(r"[a-h](\s->\s\*)?", kind):
            raise ValueError(f"unknown alias annotation: {kind!r}")
        self.kind = kind
        self.write = write

    @property
    def is_write(self) -> bool:
        return self.write

    @property
    def is_alias(self) -> bool:
        return not self.write

    def __str__(self) -> str:
        return f"({self.kind}{'!' if self.write else ''})"


class TensorOptionsArguments:
    """dtype/layout/device[/pin_memory] cluster flattened into one wrapper."""

    def __init__(self, args: list[Argument]):
        self.args = args

    def all(self) -> list[Argument]:
        return list(self.args)

    def __len__(self) -> int:
        return len(self.args)


class Arguments:
    """Bucketed argument list keeping the source order of every slot.

    `ordered` carries the schema sequence with the self wrapper and the
    factory cluster intact; the flat_* views derive from it.  The factory
    cluster (dtype/layout/device[/pin_memory]) is stored once, inside its
    wrapper, at its source position.  `star_index` is the emitted-argument
    ordinal where the '*' separator sits."""

    def __init__(self, ordered, star_index):
        self.ordered = tuple(ordered)
        self.star_index = star_index

    # -- bucket views ------------------------------------------------------
    @property
    def pre_self_positional(self) -> tuple:
        return tuple(a for a in self._leaves()
                     if not a.kwonly and self._before_self(a))

    @property
    def self_arg(self):
        for w in self.ordered:
            if isinstance(w, SelfArgument):
                return w
        return None

    @property
    def post_self_positional(self) -> tuple:
        seen_self = False
        out = []
        for a in self._leaves():
            if a.kwonly:
                continue
            if not seen_self and self._is_self_arg(a):
                seen_self = True
                continue
            if seen_self and not a.kwonly:
                out.append(a)
        return tuple(out)

    @property
    def tensor_options(self):
        for w in self.ordered:
            if isinstance(w, TensorOptionsArguments):
                return w
        return None

    @property
    def pre_tensor_options_kwarg_only(self) -> tuple:
        return ()

    @property
    def post_tensor_options_kwarg_only(self) -> tuple:
        return tuple(a for a in self._kwonly_leaves())

    @property
    def out(self) -> tuple:
        return tuple(a for a in self._leaves()
                     if a.kwonly and a.annotation is not None
                     and a.annotation.is_write and a.type.is_tensor_like())

    # -- flat views --------------------------------------------------------
    @property
    def flat_positional(self) -> tuple:
        return tuple(a for a in self._leaves() if not a.kwonly)

    @property
    def flat_kwarg_only(self) -> tuple:
        kw = self._kwonly_leaves()
        return tuple(a for a in kw
                     if not (a.annotation is not None and a.annotation.is_write
                             and a.type.is_tensor_like()))

    @property
    def flat_non_out(self) -> tuple:
        return self.flat_positional + self.flat_kwarg_only

    @property
    def flat_all(self) -> tuple:
        return self.flat_non_out + self.out

    @property
    def all(self) -> tuple:
        return self.ordered

    def __iter__(self):
        return iter(self._leaves())

    def _leaves(self) -> list:
        out = []
        for w in self.ordered:
            if isinstance(w, TensorOptionsArguments):
                out.extend(w.all())
            elif isinstance(w, SelfArgument):
                out.append(w.argument)
            else:
                out.append(w)
        return out

    def _kwonly_leaves(self) -> list:
        out = []
        for a in self._leaves():
            if a.kwonly and not (a.annotation is not None
                                 and a.annotation.is_write
                                 and a.type.is_tensor_like()):
                out.append(a)
        return out

    def _before_self(self, a) -> bool:
        leaves = self._leaves()
        idx = leaves.index(a)
        for w in self.ordered:
            if isinstance(w, SelfArgument):
                return idx < leaves.index(w.argument)
        return False

    def _is_self_arg(self, a) -> bool:
        w = self.self_arg
        return w is not None and w.argument is a

    # -- rendering ----------------------------------------------------------
    def all_flat_str(self) -> list[str]:
        """Rendered schema argument spellings in source order, '*' at its
        recorded position; the factory cluster renders in place."""
        out: list[str] = []
        star = self.star_index
        emitted = 0
        for w in self.ordered:
            if isinstance(w, TensorOptionsArguments):
                entries = [str(a) for a in w.all()]
            elif isinstance(w, SelfArgument):
                entries = [str(w.argument)]
            else:
                entries = [str(w)]
            for e in entries:
                if star is not None and emitted == star:
                    out.append("*")
                out.append(e)
                emitted += 1
        if star is not None and emitted == star:
            out.append("*")
        return out

    def has_tensor_arg(self) -> bool:
        for a in self.flat_non_out:
            if a.type.is_tensor_like():
                return True
        return self.tensor_options is not None

    def has_generator_arg(self) -> bool:
        return any(a.type.name == "Generator"
                   for a in self.flat_non_out)


# ---------------------------------------------------------------------------
# FunctionSchema
# ---------------------------------------------------------------------------

class SchemaKind(Enum):
    functional = "functional"
    inplace = "inplace"
    out = "out"
    scratch = "scratch"


class ViewSchemaKind(Enum):
    non_aliasing = "non_aliasing"
    aliasing = "aliasing"
    aliasing_inplace = "aliasing_inplace"


@dataclass
class Return:
    type: TypeExpr
    name: str | None = None
    annotation: Annotation | None = None

    def __str__(self) -> str:
        s = str(self.type)
        if self.name:
            s += f" {self.name}"
        return s


_ARG_SPLIT_RE = re.compile(r",(?![^(]*\))")
_DEFAULT_RE = re.compile(r"^(?P<type>.+?)\s+(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
                         r"(?:\s*=\s*(?P<default>.*))?$")


class FunctionSchema:
    def __init__(self, name: OperatorName, arguments: Arguments,
                 returns: list[Return]):
        self.name = name
        self.arguments = arguments
        self.returns = returns

    # -- construction ----------------------------------------------------
    @staticmethod
    def parse(text: str) -> "FunctionSchema":
        arrow = text.find(") -> ")
        if arrow < 0:
            raise ValueError(f"schema has no return arrow: {text!r}")
        head = text[:arrow + 1]
        ret = text[arrow + len(") -> "):]
        name_part = head[:head.index("(")].strip()
        inner = head[head.index("(") + 1:-1]
        op = _parse_operator_name(name_part)

        # returns
        returns: list[Return] = []
        ret = ret.strip()
        if ret:
            if ret.startswith("("):
                body = ret[1:]
                if body.endswith(")"):
                    body = body[:-1]
                if body.endswith(") ->"):
                    raise ValueError(f"nested arrow in returns: {ret!r}")
                ret_parts = _split_top(body)
            else:
                ret_parts = [ret]
            for piece in ret_parts:
                piece = piece.strip()
                if not piece:
                    continue
                r = _parse_return(piece)
                returns.append(r)

        args = _parse_arguments(inner)
        return FunctionSchema(op, args, returns)

    # -- queries ----------------------------------------------------------
    def kind(self) -> SchemaKind:
        base = self.name.name
        if base.inplace:
            return SchemaKind.inplace
        if any(a.name == "out" for a in self.arguments.out):
            return SchemaKind.out
        return SchemaKind.functional

    def signature(self, include_overload_name: bool = False) -> str:
        args = self.arguments
        pieces = []
        for a in args.flat_positional:
            pieces.append(str(a.type))
        if args.flat_kwarg_only or args.out:
            pieces.append("*")
        for a in args.flat_kwarg_only:
            pieces.append(str(a.type))
        # Grouping-key normalization (reference signature() semantics): the
        # out argument and mutability annotations are excluded so
        # functional/inplace/out variants of one op share the same key.
        name = str(self.name) if include_overload_name else str(
            self.name.name)
        ret = ", ".join(_unannotated(r.type) for r in self.returns)
        ret_s = f"({ret})" if len(self.returns) != 1 else ret
        if not self.returns:
            ret_s = "()"
        return f"{name}({', '.join(pieces)}) -> {ret_s}"

    def view_signature(self, include_overload_name: bool = False) -> str:
        return self.signature(include_overload_name)

    def __str__(self) -> str:
        pieces = [str(a) for a in self.arguments.all_flat_str()]
        ret = ", ".join(str(r) for r in self.returns)
        if len(self.returns) != 1:
            ret = f"({ret})"
        if not self.returns:
            ret = "()"
        return f"{self.name}({', '.join(pieces)}) -> {ret}"


def _split_top(text: str) -> list[str]:
    depth = 0
    cur = ""
    parts: list[str] = []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


_TENSOR_OPTIONS_NAMES = ("dtype", "layout", "device", "pin_memory")


def _parse_arguments(inner: str) -> Arguments:
    parts = [p for p in _split_top(inner) if p.strip()]
    positional: list[Argument] = []
    kwonly: list[Argument] = []
    out: list[Argument] = []
    ordered: list = []
    seen_star = False
    star_emitted = None
    emitted = 0
    for p in parts:
        p = p.strip()
        if p == "*":
            seen_star = True
            star_emitted = emitted
            continue
        arg = _parse_argument(p, kwonly=seen_star)
        is_write_out = (
            seen_star
            and arg.annotation is not None
            and arg.annotation.is_write
            and arg.type.is_tensor_like()
        )
        if is_write_out:
            out.append(arg)
            ordered.append(arg)
            emitted += 1
            continue
        if seen_star:
            kwonly.append(arg)
        else:
            positional.append(arg)
        ordered.append(arg)
        emitted += 1

    # self wrapper: a positional argument literally named "self"
    if positional and positional[0].name == "self":
        ordered = ([SelfArgument(positional[0])]
                   + ordered[1:])
    elif positional and positional[0].name != "self" and any(
            a.name == "self" for a in positional):
        idx = next(i for i, a in enumerate(positional) if a.name == "self")
        at = ordered.index(positional[idx])
        ordered = (ordered[:at] + [SelfArgument(positional[idx])]
                   + ordered[at + 1:])

    # factory cluster: dtype/layout/device(/pin_memory) keyword-only slots
    kw_names = [a.name for a in kwonly]
    if all(n in kw_names for n in ("dtype", "layout", "device")):
        cluster_names = [n for n in _TENSOR_OPTIONS_NAMES if n in kw_names]
        cluster_args = [a for a in kwonly if a.name in cluster_names]
        wrapper = TensorOptionsArguments(cluster_args)
        # replace the first cluster member's slot with the wrapper
        first = cluster_args[0]
        at = ordered.index(first)
        ordered = (ordered[:at] + [wrapper]
                   + [w for w in ordered[at + 1:]
                      if not (isinstance(w, Argument)
                              and w.name in cluster_names)])

    return Arguments(ordered, star_emitted)


def _parse_argument(piece: str, kwonly: bool) -> Argument:
    m = _DEFAULT_RE.match(piece.strip())
    if m is None:
        raise ValueError(f"cannot parse schema argument: {piece!r}")
    t = _parse_type(m.group("type").strip())
    annotation = None
    if t.mutability is not None:
        annotation = Annotation(t.mutability, write=True)
    elif t.alias is not None:
        annotation = Annotation(t.alias, write=False)
    default = m.group("default")
    if default is not None:
        default = default.strip()
        if default == "":
            default = None
    return Argument(m.group("name"), t, default, annotation, kwonly)


_RETURN_RE = re.compile(
    r"^(?P<type>[A-Za-z_][A-Za-z_0-9]*(?:\([a-zA-Z](?:\s->\s\*)?!?\))?"
    r"(?:\?)?(?:\[\d*\](?:\?)?)?\??)"
    r"(?:\s+(?P<name>[A-Za-z_][A-Za-z_0-9]*))?$"
)


def _parse_return(piece: str) -> Return:
    # "<type>" or "<type> <name>"; the type spelling carries its own
    # annotation: Tensor(a!) write, Tensor(a) read alias, Tensor(a -> *)
    # directed alias.  Return names may be capitalized (L, U, Q, R).
    m = _RETURN_RE.match(piece.strip())
    if m is None:
        raise ValueError(f"cannot parse schema return: {piece!r}")
    t = _parse_type(m.group("type"))
    annotation = None
    if t.mutability is not None:
        annotation = Annotation(t.mutability.rstrip("!"), write=True)
    elif t.alias is not None:
        annotation = Annotation(t.alias, write=False)
    return Return(t, m.group("name"), annotation)


# ---------------------------------------------------------------------------
# Alias/view metadata (built-in)
# ---------------------------------------------------------------------------

VIEW_FUNCTIONS_WITH_METADATA_CHANGE = [
    "view_as_complex",
    "view_as_real",
    "_conj",
    "_neg_view",
    "_nested_get_values",
    "_nested_view_from_buffer",
    "_nested_view_from_jagged",
]

VIEW_FUNCTIONS = {
    "numpy_T": "self",
    "alias": "self",
    "as_strided": "self",
    "diagonal": "self",
    "expand": "self",
    "permute": "self",
    "select": "self",
    "slice": "self",
    "slice_inverse": "self",
    "split": "self",
    "split_with_sizes": "self",
    "squeeze": "self",
    "t": "self",
    "transpose": "self",
    "unfold": "self",
    "unsqueeze": "self",
    "flatten": "self",
    "view": "self",
    "unbind": "self",
    "_indices": "self",
    "_values": "self",
    "indices": "self",
    "values": "self",
    "crow_indices": "self",
    "col_indices": "self",
    "ccol_indices": "self",
    "row_indices": "self",
    "sparse_coo_tensor_with_dims_and_tensors": "values",
    "_reshape_alias": "self",
    "_test_autograd_multiple_dispatch_view": "self",
}
for _key in VIEW_FUNCTIONS_WITH_METADATA_CHANGE:
    VIEW_FUNCTIONS[_key] = "self"

RETURNS_VIEWS_OF_INPUT = frozenset(VIEW_FUNCTIONS.keys()).union({
    "chunk",
    "detach",
    "contiguous",
    "reshape",
    "reshape_as",
    "expand_as",
    "view_as",
    "real",
    "imag",
    "narrow",
    "movedim",
    "tensor_split",
    "swapdims",
    "swapaxes",
    "mT",
    "mH",
    "adjoint",
    "matrix_H",
})


# ---------------------------------------------------------------------------
# Tag registry
# ---------------------------------------------------------------------------

_TAGS_PATH = Path(__file__).resolve().parents[2] / "config" / "tags.yaml"


def parse_tags_yaml(path: str | Path | None = None) -> dict[str, str | None]:
    """Load the versioned tag registry used by schema parsing."""
    tag_path = Path(path) if path is not None else _TAGS_PATH
    with tag_path.open("r", encoding="utf-8") as tag_file:
        entries = yaml.safe_load(tag_file)
    if not isinstance(entries, list) or not entries:
        raise TypeError(f"tag file must contain a non-empty list: {tag_path}")

    tags: dict[str, str | None] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("tag"), str):
            raise TypeError(f"tag entry {index} must define a string tag: {tag_path}")
        name = entry["tag"]
        if name in tags:
            raise ValueError(f"duplicate tag: {name}")
        description = entry.get("desc")
        if description is not None and not isinstance(description, str):
            raise TypeError(f"tag description must be a string: {name}")
        tags[name] = description
    return tags


VALID_TAGS = frozenset(parse_tags_yaml())


# ---------------------------------------------------------------------------
# Dispatch keys
# ---------------------------------------------------------------------------

class DispatchKey(Enum):
    CompositeImplicitAutograd = "CompositeImplicitAutograd"
    CompositeExplicitAutograd = "CompositeExplicitAutograd"
    Autograd = "Autograd"
    CPU = "CPU"
    CUDA = "CUDA"
    Vulkan = "Vulkan"

    def __str__(self) -> str:
        return self.value


dispatch_keys: list[DispatchKey] = list(DispatchKey)


def _normalize_dispatch_key(name: str):
    try:
        return DispatchKey(name)
    except ValueError:
        return name  # unknown keys pass through verbatim


# ---------------------------------------------------------------------------
# Reference-shaped native function record
# ---------------------------------------------------------------------------

class Location:
    def __init__(self, file: str, line: int):
        self.file = str(file)
        self.line = int(line)

    def __str__(self) -> str:
        return f"{self.file}:{self.line}"


class LineLoader(yaml.SafeLoader):
    """YAML loader that attaches __line__ to every mapping."""


def _line_loader_construct_mapping(loader, node, deep=False):
    mapping = yaml.SafeLoader.construct_mapping(loader, node, deep=deep)
    mapping["__line__"] = node.start_mark.line + 1
    return mapping


LineLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _line_loader_construct_mapping)


class NativeFunctionRecord:
    """Reference-shaped record: schema plus yaml metadata fields."""

    _PASSTHROUGH_FIELDS = (
        "manual_cpp_binding",
        "manual_kernel_registration",
        "python_module",
        "category_override",
        "device_guard",
        "device_check",
        "pkg",
        "autogen",
        "overwrite_package",
        "only_register_dispatcher",
        "skip_dynamo",
    )

    def __init__(self, func: FunctionSchema, item: dict,
                 valid_tags=None, location: Location | None = None):
        self.func = func
        self.loc = location or Location("schema", item.get("__line__", 0))
        self.variants = _split_variants(item.get("variants", "function"))
        self.dispatch = {
            _normalize_dispatch_key(k): v
            for k, v in dict(item.get("dispatch") or {}).items()
        }
        self.structured = bool(item.get("structured", False))
        self.structured_delegate = item.get("structured_delegate")
        self.structured_inherits_delegate = item.get(
            "structured_inherits_delegate")
        self.autogen = _split_autogen(item.get("autogen"))
        self.tags = tuple(_validate_tags(item.get("tags"), valid_tags))
        self.uva = item.get("uva", False)
        self.supports_tensor_options = self.func.arguments.tensor_options \
            is not None
        self.is_view_op = str(self.func.name.name) in VIEW_FUNCTIONS \
            or self.func.name.name.base in VIEW_FUNCTIONS
        for field_name in self._PASSTHROUGH_FIELDS:
            setattr(self, field_name, item.get(field_name))
        # Device guard defaults to on; an explicit false disables it.
        self.device_guard = item.get("device_guard", True)

    @property
    def is_view_op_field(self):
        return self.is_view_op

    @property
    def view_schema_kind(self) -> ViewSchemaKind:
        if self.is_view_op:
            if self.func.kind() == SchemaKind.inplace:
                return ViewSchemaKind.aliasing_inplace
            return ViewSchemaKind.aliasing
        return ViewSchemaKind.non_aliasing

    def kind(self):
        return self.func.kind()

    def signature(self):
        return self.func.signature()

    def __str__(self) -> str:
        return str(self.func)


def _unannotated(t: TypeExpr) -> str:
    """Type spelling with the mutability annotation stripped."""
    if t.mutability is None and t.alias is None:
        return str(t)
    plain = TypeExpr(t.name, t.is_opt, t.is_list, t.size, None, None,
                     t.elem_opt)
    return str(plain)


def _split_variants(raw) -> list[str]:
    if raw is None:
        return ["function"]
    if isinstance(raw, str):
        return [v.strip() for v in raw.split(",") if v.strip()]
    return [str(v).strip() for v in raw]


def _split_autogen(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [v.strip() for v in raw.split(",") if v.strip()]
    return list(raw)


def _validate_tags(raw, valid_tags) -> list[str]:
    if raw is None:
        return []
    tags = [raw] if isinstance(raw, str) else list(raw)
    for tag in tags:
        if valid_tags is not None and tag not in valid_tags:
            raise ValueError(f"unknown schema tag: {tag}")
    return tags


class NativeFunctionsGroup:
    """functional + inplace + out trio sharing a base signature."""

    def __init__(self, functional=None, inplace=None, out=None):
        self.functional = functional
        self.inplace = inplace
        self.out = out
        self.structured = bool(
            getattr(out, "structured", False)
            or getattr(functional, "structured", False)
        )

    @staticmethod
    def from_dict(functions: dict) -> "NativeFunctionsGroup | None":
        functional = functions.get(SchemaKind.functional)
        inplace = functions.get(SchemaKind.inplace)
        out = functions.get(SchemaKind.out)
        if functional is None or out is None:
            return None
        return NativeFunctionsGroup(functional, inplace, out)


class NativeFunctionsViewGroup:
    def __init__(self, view=None, view_copy=None, view_inplace=None):
        self.view = view
        self.view_copy = view_copy
        self.view_inplace = view_inplace


FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT: list[str] = []


class ParsedNativeYaml:
    def __init__(self, native_functions, backend_indices):
        self.native_functions = tuple(native_functions)
        self.backend_indices = backend_indices


def parse_native_yaml_struct(prepared: list[dict], valid_tags=None,
                             path: str = "schema.yaml"
                             ) -> ParsedNativeYaml:
    """Parse prepared schema entries into reference-shaped records."""
    records = []
    for item in prepared:
        loc = Location(path, item.get("__line__", 0))
        func = FunctionSchema.parse(item["func"])
        record = NativeFunctionRecord(func, item, valid_tags, loc)
        records.append(record)
    return ParsedNativeYaml(records, {})


def cpp_name(schema: FunctionSchema) -> str:
    """Reference-compatible C++ symbol for a schema (dispatch-name shape)."""
    return str(schema.name)


def pre_group_native_functions(native_functions):
    grouped = defaultdict(dict)
    for function in native_functions:
        key = function.func.signature()
        kind = function.func.kind()
        while kind in grouped[key]:
            key = (key, function.func.name)
        grouped[key][kind] = function
    return grouped
