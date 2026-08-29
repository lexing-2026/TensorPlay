"""

Type/Argument/NativeFunction records consumed by this repo's generators, so
every generator shares exactly upstream's grammar, validation, and error
messages.  No legacy dialect parser exists anymore: native_functions.yaml is
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .yaml_utils import YamlLoader

import yaml


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

import sys as _sys

_TORCHGEN_READY = False


def _ensure_torchgen():
    """
    grammar predates this checkout)."""
    global _TORCHGEN_READY
    global _TORCHGEN_MODULE
    if _TORCHGEN_READY:
        return
    root = Path(__file__).resolve()
    for cand in root.parents:
        standalone = cand / "third_party" / "torchgen"
        if (standalone / "torchgen" / "model.py").exists():
            pt = str(standalone)
            break
        if (cand / "third_party" / "pytorch" / "torchgen" / "model.py").exists():
            pt = str(cand / "third_party" / "pytorch")
            break
    else:
        raise RuntimeError(
            "or the legacy third_party/pytorch/ layout)")

    for m in [k for k in list(_sys.modules)
              if k == "torchgen" or k.startswith("torchgen.")]:
        del _sys.modules[m]
    saved = list(_sys.path)
    _sys.path.insert(0, pt)
    try:
        import torchgen.model as tgm
    finally:
        _sys.path[:] = saved
    _TORCHGEN_MODULE = tgm
    _TORCHGEN_READY = True


_TORCHGEN_MODULE = None


def _ensure_torchgen_imported():
    _ensure_torchgen()
    assert _TORCHGEN_MODULE is not None
    return _TORCHGEN_MODULE


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

_KIND_ALIASES = {
    "int": "int64_t",
    "SymInt": "int64_t",
    "float": "double",
    "ScalarType": "DType",
}


@dataclass(frozen=True)
class Type:
    kind: str
    is_list: bool = False
    is_opt: bool = False
    mutability: str | None = None
    symint: bool = False

    @property
    def is_tensor_like(self) -> bool:
        return self.kind == "Tensor"

    @property
    def is_mutable_tensor_list(self) -> bool:
        return self.is_list and self.is_tensor_like and self.mutability is not None

    @property
    def is_mutable_ref(self) -> bool:
        return self.is_tensor_like and not self.is_list and self.mutability is not None

    def __str__(self) -> str:
        s = "SymInt" if self.symint else self.kind
        if self.mutability is not None:
            s = f"{s}({self.mutability}!)"
        if self.is_list:
            s += "[]"
        if self.is_opt:
            s += "?"
        return s


_TYPE_CACHE: dict[tuple, Type] = {}


def make_type(kind: str, is_list: bool = False, is_opt: bool = False,
              mutability: str | None = None, symint: bool = False) -> Type:
    key = (kind, is_list, is_opt, mutability, symint)
    t = _TYPE_CACHE.get(key)
    if t is None:
        t = Type(*key)
        _TYPE_CACHE[key] = t
    return t


def _type_from_tg(t) -> Type:
    tgm = _ensure_torchgen_imported()
    # Unwrap Optional<T> first; T may be a BaseType or a ListType
    # (``int[]?`` -> optional list of int64_t).
    is_opt = isinstance(t, tgm.OptionalType)
    inner = t.elem if is_opt else t
    is_list = isinstance(inner, tgm.ListType)
    elem = inner.elem if is_list else inner
    name = elem.name.name if hasattr(elem.name, "name") else str(elem.name)
    symint = str(name) == "SymInt"
    kind = _KIND_ALIASES.get(str(name), str(name))
    return make_type(kind, is_list, is_opt, None, symint)


# ---------------------------------------------------------------------------
# Argument / Return / NativeFunction
# ---------------------------------------------------------------------------

@dataclass
class Argument:
    name: str
    type: Type
    default: str | None = None
    kwonly: bool = False

    @property
    def python_name(self) -> str:
        return "from_" if self.name == "from" else self.name


@dataclass
class ReturnDecl:
    type: Type
    name: str | None = None


@dataclass
class NativeFunction:
    schema: str
    base_name: str
    cpp_name: str
    overload_name: str
    args: list[Argument]
    returns: list[ReturnDecl]
    variants: list[str] = field(default_factory=list)

    dispatch: dict[str, str] = field(default_factory=dict)
    device_check: str | None = None
    skip_implementation: bool = False
    structured_delegate: str | None = None
    structured_outputs: str | None = None
    autograd_meta: dict | None = None
    pydefaults: dict[str, str] = field(default_factory=dict)

    @property
    def func_name(self) -> str:
        if self.overload_name:
            return f"{self.base_name}.{self.overload_name}"
        return self.base_name

    @property
    def positional_args(self) -> list[Argument]:
        return [a for a in self.args if not a.kwonly]

    @property
    def kwarg_args(self) -> list[Argument]:
        return [a for a in self.args if a.kwonly]

    def arg(self, name: str) -> Argument | None:
        for a in self.args:
            if a.name == name:
                return a
        return None

    @property
    def tensor_args(self) -> list[Argument]:
        return [a for a in self.args if a.type.is_tensor_like and not a.type.is_opt]

    @property
    def mutable_args(self) -> list[Argument]:
        return [a for a in self.args
                if a.type.is_mutable_ref or a.type.is_mutable_tensor_list]

    @property
    def returns_tuple(self) -> bool:
        return len(self.returns) > 1

    @property
    def cpp_return_kind(self) -> str:
        r = self.returns
        if len(r) == 0 or (len(r) == 1 and r[0].type.kind == "void"):
            return "void"
        if len(r) > 1:
            return "tuple"
        t = r[0]
        if t.type.is_list:
            return "list"
        return {True: "mut_ref", False: "value"}[t.type.is_mutable_ref]

    def self_arg(self) -> Argument | None:
        return self.arg("self")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _norm_default(d):
    if d is None:
        return None
    if d in ("True", "False"):
        return d.lower()
    return d


def parse_schema(schema: str) -> NativeFunction:
    schema = schema.replace("int64_t", "int")
    tgm = _ensure_torchgen_imported()
    BaseType = tgm.BaseType

    ts = tgm.FunctionSchema.parse(schema)
    # into positional / kwarg-only / out buckets at parse time and `.all`
    # flattens them.  Recover the distinction via the kwarg-only bucket.
    kwarg_names = {a.name for a in ts.arguments.flat_kwarg_only}

    def unwrap(t):
        while not isinstance(t, BaseType):
            t = t.elem
        return t.name.name  # BaseTy enum member -> canonical spelling

    def conv_arg(w):
        a = getattr(w, "argument", w)
        mut = "a" if (a.annotation is not None and a.annotation.is_write) else None
        # ``int[]?`` stacks Optional over List over Base; peel each layer so
        # the element type AND both flags are recovered.
        outer = a.type
        is_opt = isinstance(outer, tgm.OptionalType)
        inner = outer.elem if is_opt else outer
        is_list = isinstance(inner, tgm.ListType)
        base = unwrap(inner)
        symint = base == "SymInt"
        kind = _KIND_ALIASES.get(base, base)
        t = make_type(kind, is_list, is_opt, None, symint)
        if mut and t.is_tensor_like:
            t = make_type(t.kind, t.is_list, t.is_opt, mut, t.symint)
        kwonly = a.name in kwarg_names
        return Argument(name=a.name, type=t,
                        default=_norm_default(a.default), kwonly=kwonly)

    args = [conv_arg(w) for w in ts.arguments.all]

    returns: list[ReturnDecl] = []
    for rw in ts.returns:
        r = getattr(rw, "argument", rw)
        mut = "a" if (r.annotation is not None and r.annotation.is_write) else None
        base = unwrap(r.type)
        symint = base == "SymInt"
        kind = _KIND_ALIASES.get(base, base)
        is_list = "ListType" in type(r.type).__name__
        t = make_type(kind, is_list, False, None, symint)
        if mut and t.is_tensor_like:
            t = make_type(t.kind, t.is_list, False, mut, t.symint)
        returns.append(ReturnDecl(t, getattr(r, "name", None)))

    name = ts.name
    bon = name.name                      # BaseOperatorName
    base_op = bon.base + ("_" if bon.inplace else "")
    # TensorPlay symbols keep it (`add_.Tensor` -> add_).
    if getattr(name.name.base, "inplace", False) and not base_op.endswith("_"):
        base_op += "_"
    overload = str(name.overload_name)

    return NativeFunction(
        schema=schema,
        base_name=base_op,
        cpp_name=base_op,
        overload_name="" if overload == "" else overload,
        args=args,
        returns=returns,
    )


def _native_function_from_yaml(item: dict) -> NativeFunction:
    f = parse_schema(item["func"])
    seen_v: list[str] = []
    for v in item.get("variants", "function").split(","):
        v = v.strip()
        if v and v not in seen_v:
            seen_v.append(v)
    f.variants = seen_v
    disp = item.get("dispatch") or {}
    f.dispatch = dict(disp) if isinstance(disp, dict) else {}
    f.device_check = item.get("device_check")
    f.skip_implementation = bool(item.get("skip_implementation", False))
    f.structured_delegate = item.get("structured_delegate")
    f.structured_outputs = item.get("structured_outputs")
    f.autograd_meta = item.get("autograd")
    pd = item.get("python_defaults") or {}
    f.pydefaults = dict(pd)
    return f


def parse_native_yaml(path: str) -> list[NativeFunction]:
    with open(path, "r") as fh:
        data = yaml.load(fh, Loader=YamlLoader)
    funcs = []
    seen_schemas: set[str] = set()
    for item in data or []:
        # Exact-duplicate `- func:` entries are merge artifacts; keep first.
        if item["func"] in seen_schemas:
            continue
        seen_schemas.add(item["func"])
        funcs.append(_native_function_from_yaml(item))
    return funcs


def parse_derivatives_yaml(path: str) -> list[dict]:
    with open(path, "r") as fh:
        data = yaml.load(fh, Loader=YamlLoader)
    return data or []
