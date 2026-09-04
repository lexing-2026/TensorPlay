"""Schema records and validation helpers consumed by the code generators.

Type, Argument, and NativeFunction records provide one schema grammar,
validation path, and error vocabulary for all generated targets.  The parser
accepts the current schema format used by this repository.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .yaml_utils import YamlLoader

import yaml


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

import sys as _sys

_TORCHGEN_READY = False
_TORCHGEN_PATH: Path | None = None
_VIEW_METADATA_READY = False
_VIEW_FUNCTIONS: dict[str, str] = {}
_RETURNS_VIEWS_OF_INPUT: frozenset[str] = frozenset()
_VIEW_FUNCTIONS_WITH_METADATA_CHANGE: frozenset[str] = frozenset()


def _ensure_torchgen():
    """Load the schema parser from a supported vendored location."""
    global _TORCHGEN_READY
    global _TORCHGEN_MODULE
    global _TORCHGEN_PATH
    if _TORCHGEN_READY:
        return
    root = Path(__file__).resolve()
    candidates = []
    for cand in root.parents:
        candidates.append(cand / "third_party" / "pytorch")
    package_root = next(
        (p for p in candidates if (p / "torchgen" / "model.py").exists()),
        None,
    )
    if package_root is None:
        raise RuntimeError("cannot locate the schema parser package")

    for m in [k for k in list(_sys.modules)
              if k == "torchgen" or k.startswith("torchgen.")]:
        del _sys.modules[m]
    saved = list(_sys.path)
    _sys.path.insert(0, str(package_root))
    try:
        import torchgen.model as tgm
    finally:
        _sys.path[:] = saved
    _TORCHGEN_MODULE = tgm
    _TORCHGEN_PATH = package_root
    _TORCHGEN_READY = True


_TORCHGEN_MODULE = None


def _ensure_torchgen_imported():
    _ensure_torchgen()
    assert _TORCHGEN_MODULE is not None
    return _TORCHGEN_MODULE


def _ensure_torchgen_generator():
    _ensure_torchgen()
    from torchgen import gen
    return gen


def _ensure_view_metadata() -> None:
    """Load the complete alias metadata used by the mutation generator."""
    global _VIEW_METADATA_READY
    global _VIEW_FUNCTIONS
    global _RETURNS_VIEWS_OF_INPUT
    global _VIEW_FUNCTIONS_WITH_METADATA_CHANGE
    if _VIEW_METADATA_READY:
        return
    _ensure_torchgen()
    if _TORCHGEN_PATH is None:
        raise RuntimeError("cannot locate alias metadata")

    import tools as tools_package

    source_path = _TORCHGEN_PATH / "tools"
    if not (source_path / "autograd" / "gen_inplace_or_view_type.py").exists():
        raise RuntimeError("cannot locate alias metadata")
    for module_name in list(_sys.modules):
        if module_name == "tools.autograd" or module_name.startswith(
                "tools.autograd."):
            del _sys.modules[module_name]
    saved_path = list(tools_package.__path__)
    tools_package.__path__.insert(0, str(source_path))
    try:
        from tools.autograd import gen_inplace_or_view_type
    finally:
        tools_package.__path__[:] = saved_path

    _VIEW_FUNCTIONS = dict(gen_inplace_or_view_type.VIEW_FUNCTIONS)
    _RETURNS_VIEWS_OF_INPUT = frozenset(
        gen_inplace_or_view_type.RETURNS_VIEWS_OF_INPUT
    )
    _VIEW_FUNCTIONS_WITH_METADATA_CHANGE = frozenset(
        gen_inplace_or_view_type.VIEW_FUNCTIONS_WITH_METADATA_CHANGE
    )
    _VIEW_METADATA_READY = True


def _tags_path() -> Path:
    if _TORCHGEN_PATH is not None:
        candidate = _TORCHGEN_PATH / "aten" / "src" / "ATen" / "native" / "tags.yaml"
        if candidate.exists():
            return candidate
    root = Path(__file__).resolve()
    for cand in root.parents:
        candidate = cand / "third_party" / "pytorch" / "aten" / "src" / "ATen" / "native" / "tags.yaml"
        if candidate.exists():
            return candidate
    raise RuntimeError("cannot locate the schema tag registry")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

_KIND_ALIASES = {
    "int": "int64_t",
    "SymInt": "int64_t",
    "SymBool": "bool",
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
    symbool: bool = False
    symfloat: bool = False
    list_elem_opt: bool = False
    list_size: int | None = None

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
        if self.symint:
            s = "SymInt"
        elif self.symbool:
            s = "SymBool"
        elif self.symfloat:
            s = "SymFloat"
        else:
            s = self.kind
        if self.mutability is not None:
            s = f"{s}({self.mutability}!)"
        if self.list_elem_opt:
            s += "?"
        if self.is_list:
            s += f"[{'' if self.list_size is None else self.list_size}]"
        if self.is_opt:
            s += "?"
        return s


_TYPE_CACHE: dict[tuple, Type] = {}


def make_type(kind: str, is_list: bool = False, is_opt: bool = False,
              mutability: str | None = None, symint: bool = False,
              symbool: bool = False, symfloat: bool = False,
              list_elem_opt: bool = False,
              list_size: int | None = None) -> Type:
    key = (kind, is_list, is_opt, mutability, symint, symbool, symfloat,
           list_elem_opt, list_size)
    t = _TYPE_CACHE.get(key)
    if t is None:
        t = Type(*key)
        _TYPE_CACHE[key] = t
    return t


def _type_from_tg(t) -> Type:
    tgm = _ensure_torchgen_imported()
    is_opt = isinstance(t, tgm.OptionalType)
    inner = t.elem if is_opt else t
    is_list = isinstance(inner, tgm.ListType)
    elem = inner.elem if is_list else inner
    list_elem_opt = isinstance(elem, tgm.OptionalType)
    if list_elem_opt:
        elem = elem.elem
    list_size = inner.size if is_list else None
    name = elem.name.name if hasattr(elem.name, "name") else str(elem.name)
    symint = str(name) == "SymInt"
    symbool = str(name) == "SymBool"
    symfloat = str(name) == "SymFloat"
    kind = _KIND_ALIASES.get(str(name), str(name))
    return make_type(kind, is_list, is_opt, None, symint, symbool, symfloat,
                     list_elem_opt, list_size)


# ---------------------------------------------------------------------------
# Argument / Return / NativeFunction
# ---------------------------------------------------------------------------

@dataclass
class Argument:
    name: str
    type: Type
    default: str | None = None
    kwonly: bool = False
    source_argument: object | None = field(default=None, repr=False,
                                           compare=False)

    @property
    def python_name(self) -> str:
        return "from_" if self.name == "from" else self.name


@dataclass
class ReturnDecl:
    type: Type
    name: str | None = None
    source_return: object | None = field(default=None, repr=False,
                                         compare=False)


@dataclass
class NativeFunction:
    schema: str
    base_name: str
    cpp_name: str
    overload_name: str
    args: list[Argument]
    returns: list[ReturnDecl]
    out_args: tuple[str, ...] = ()
    variants: list[str] = field(default_factory=list)
    # Names of arguments whose C++ default value is suppressed when an
    # overload group needs an explicit call shape.
    cpp_no_default_args: set[str] = field(default_factory=set)

    dispatch: dict[str, str] = field(default_factory=dict)
    device_check: str | None = None
    structured_delegate: str | None = None
    structured_outputs: str | None = None
    autograd_meta: dict | None = None
    pydefaults: dict[str, str] = field(default_factory=dict)
    dispatcher_name: str | None = None
    unambiguous_operator_name: str | None = None
    schema_kind: str | None = None
    tags: frozenset[str] = frozenset()
    source_native_function: object | None = field(default=None, repr=False,
                                                   compare=False)
    backend_metadata: dict[str, object] = field(default_factory=dict,
                                                repr=False, compare=False)

    @property
    def func_name(self) -> str:
        if self.dispatcher_name is not None:
            return self.dispatcher_name
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

    @property
    def is_out(self) -> bool:
        return bool(self.out_args)

    @property
    def source_function(self):
        return getattr(self.source_native_function, "func", None)

    @property
    def reference(self):
        return self.source_native_function

    @property
    def namespace(self) -> str:
        namespace = getattr(self.source_native_function, "namespace", None)
        return "tensorplay" if namespace in (None, "aten") else namespace

    @property
    def structured(self) -> bool:
        return bool(getattr(self.source_native_function, "structured", False))

    @property
    def use_const_ref_for_mutable_tensors(self) -> bool:
        return bool(getattr(
            self.source_native_function,
            "use_const_ref_for_mutable_tensors",
            False,
        ))

    @property
    def device_guard(self) -> bool:
        return bool(getattr(self.source_native_function, "device_guard", True))

    @property
    def manual_kernel_registration(self) -> bool:
        return bool(getattr(
            self.source_native_function,
            "manual_kernel_registration",
            False,
        ))

    @property
    def manual_cpp_binding(self) -> bool:
        return bool(getattr(
            self.source_native_function,
            "manual_cpp_binding",
            False,
        ))

    @property
    def python_module(self) -> str | None:
        return getattr(self.source_native_function, "python_module", None)

    @property
    def category_override(self) -> str | None:
        return getattr(self.source_native_function, "category_override", None)

    @property
    def autogen(self) -> tuple[str, ...]:
        return tuple(str(value) for value in getattr(
            self.source_native_function, "autogen", ()
        ))

    @property
    def ufunc_inner_loop(self):
        return getattr(self.source_native_function, "ufunc_inner_loop", {})

    @property
    def structured_inherits(self) -> str | None:
        value = getattr(self.source_native_function, "structured_inherits", None)
        return None if value is None else str(value)

    @property
    def precomputed(self):
        return getattr(self.source_native_function, "precomputed", None)

    @property
    def is_abstract(self) -> bool:
        return bool(getattr(self.source_native_function, "is_abstract", False))

    @property
    def location(self):
        return getattr(self.source_native_function, "loc", None)

    @property
    def is_functional(self) -> bool:
        return self.schema_kind == "functional"

    @property
    def is_inplace(self) -> bool:
        return self.schema_kind == "inplace"

    @property
    def is_mutable(self) -> bool:
        return self.schema_kind == "mutable"

    @property
    def has_composite_implicit_autograd_kernel(self) -> bool:
        return bool(getattr(
            self.source_native_function,
            "has_composite_implicit_autograd_kernel",
            False,
        ))

    @property
    def has_composite_implicit_autograd_nested_tensor_kernel(self) -> bool:
        return bool(getattr(
            self.source_native_function,
            "has_composite_implicit_autograd_nested_tensor_kernel",
            False,
        ))

    @property
    def has_composite_explicit_autograd_kernel(self) -> bool:
        return bool(getattr(
            self.source_native_function,
            "has_composite_explicit_autograd_kernel",
            False,
        ))

    @property
    def has_composite_explicit_autograd_non_functional_kernel(self) -> bool:
        return bool(getattr(
            self.source_native_function,
            "has_composite_explicit_autograd_non_functional_kernel",
            False,
        ))

    @property
    def has_composite_kernel(self) -> bool:
        return (
            self.has_composite_implicit_autograd_kernel
            or self.has_composite_implicit_autograd_nested_tensor_kernel
            or self.has_composite_explicit_autograd_kernel
            or self.has_composite_explicit_autograd_non_functional_kernel
        )

    @property
    def is_view_op(self) -> bool:
        source = self.source_native_function
        if source is not None and source.is_view_op:
            return True
        _ensure_view_metadata()
        return self.root_name in _VIEW_FUNCTIONS

    @property
    def returns_view_of_input(self) -> bool:
        source = self.source_native_function
        if source is not None and source.is_view_op:
            return True
        _ensure_view_metadata()
        return self.root_name in _RETURNS_VIEWS_OF_INPUT

    @property
    def view_input_name(self) -> str | None:
        _ensure_view_metadata()
        return _VIEW_FUNCTIONS.get(self.root_name) or (
            "self" if self.root_name in _RETURNS_VIEWS_OF_INPUT else None
        )

    @property
    def view_metadata_changes(self) -> bool:
        _ensure_view_metadata()
        return self.root_name in _VIEW_FUNCTIONS_WITH_METADATA_CHANGE

    @property
    def view_schema_kind(self):
        source = self.source_native_function
        if source is not None and source.is_view_op:
            return source.view_schema_kind
        _ensure_view_metadata()
        if self.root_name in _VIEW_FUNCTIONS:
            return _ensure_torchgen_imported().ViewSchemaKind.aliasing
        return _ensure_torchgen_imported().ViewSchemaKind.non_aliasing

    @property
    def part_of_structured_group(self) -> bool:
        return self.structured or self.structured_delegate is not None

    @property
    def root_name(self) -> str:
        source = self.source_function
        if source is not None:
            return source.name.name.base
        return self.base_name.rstrip("_")

    def backend(self, dispatch_key: str) -> object | None:
        key_name = getattr(dispatch_key, "name", str(dispatch_key))
        return self.backend_metadata.get(key_name)


def operator_unambiguous_name(f: NativeFunction) -> str:
    """Return the identifier form of the complete operator name."""
    name = f.unambiguous_operator_name
    if name is None:
        name = f.func_name.replace(".", "_")
    return name.replace("::", "_")


def redispatch_name(f: NativeFunction, variant: str) -> str:
    """Return the deterministic redispatch helper name for one operator."""
    return f"redispatch_{operator_unambiguous_name(f)}_{variant}"


def redispatch_key(f: NativeFunction, variant: str) -> tuple:
    """Return the generated identity for a redispatch helper."""
    return (f.func_name, variant)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _norm_default(d):
    if d is None:
        return None
    if d in ("True", "False"):
        return d.lower()
    return d


def parse_schema(schema: str, parsed_schema=None) -> NativeFunction:
    source_schema = schema
    schema = schema.replace("int64_t", "int")
    tgm = _ensure_torchgen_imported()

    ts = parsed_schema if parsed_schema is not None else tgm.FunctionSchema.parse(schema)
    if str(ts) != schema:
        raise ValueError(f"schema projection mismatch: {schema} != {ts}")
    # into positional / kwarg-only / out buckets at parse time and `.all`
    # flattens them.  Recover the distinction via the kwarg-only bucket.
    kwarg_names = {a.name for a in ts.arguments.flat_kwarg_only}
    # Write-annotated schema slots are bucketed as out arguments and are
    # keyword-only in the signature, even though the kwarg-only bucket
    # excludes them; keep the kwonly flag faithful to the source schema.
    out_names = {getattr(getattr(w, "argument", w), "name")
                 for w in ts.arguments.out}

    def conv_arg(w):
        a = getattr(w, "argument", w)
        mut = "a" if (a.annotation is not None and a.annotation.is_write) else None
        t = _type_from_tg(a.type)
        if mut and t.is_tensor_like:
            t = make_type(t.kind, t.is_list, t.is_opt, mut, t.symint,
                          t.symbool, t.symfloat, t.list_elem_opt,
                          t.list_size)
        kwonly = a.name in kwarg_names or a.name in out_names
        return Argument(name=a.name, type=t,
                        default=_norm_default(a.default), kwonly=kwonly,
                        source_argument=a)

    # A factory-style kwarg cluster (dtype/layout/device/pin_memory) comes
    # back bundled as a TensorOptionsArguments wrapper; expand it into its
    # four constituent arguments, in schema order.
    def expand(w):
        if isinstance(w, tgm.TensorOptionsArguments):
            out = []
            for part in w.all():
                out.extend(expand(part))
            return out
        return [conv_arg(w)]

    args = [a for w in ts.arguments.all for a in expand(w)]
    # Python-facing parameter names must be unique within one signature.
    # The conventional `self` -> `input` rename can collide with a schema
    # argument literally named `input` (conv_tbc_backward), so later
    # collisions take a numeric suffix.  Positional bindings are unaffected;
    # keyword surfaces are generated from these names everywhere.
    seen_py_names: set[str] = set()
    for a in args:
        py = "input" if a.name == "self" else a.python_name
        if py in seen_py_names:
            suffix = 1
            while f"{py}{suffix}" in seen_py_names:
                suffix += 1
            a.name = f"{py}{suffix}"
        seen_py_names.add("input" if a.name == "self" else a.python_name)
    out_args = tuple(
        getattr(getattr(w, "argument", w), "name")
        for w in ts.arguments.out
    )

    returns: list[ReturnDecl] = []
    for rw in ts.returns:
        r = getattr(rw, "argument", rw)
        mut = "a" if (r.annotation is not None and r.annotation.is_write) else None
        t = _type_from_tg(r.type)
        if mut and t.is_tensor_like:
            t = make_type(t.kind, t.is_list, t.is_opt, mut, t.symint,
                          t.symbool, t.symfloat, t.list_elem_opt,
                          t.list_size)
        returns.append(ReturnDecl(t, getattr(r, "name", None),
                                  source_return=r))

    name = ts.name
    bon = name.name                      # BaseOperatorName
    # Dunder operators (__and__, __iand__, ...) keep their double underscores
    # in the C++ name: stripping them would yield reserved keywords
    # (`and`/`or`/`xor`), which cannot be declared as function names.
    if getattr(bon, "dunder_method", False):
        base_op = f"__{'i' if bon.inplace else ''}{bon.base}__"
    else:
        base_op = bon.base + (
            "_" if bon.inplace else
            "_functional" if getattr(bon, "functional_overload", False) else ""
        )
    overload = str(name.overload_name)

    return NativeFunction(
        schema=source_schema,
        base_name=base_op,
        cpp_name=base_op,
        overload_name="" if overload == "" else overload,
        args=args,
        returns=returns,
        out_args=out_args,
        dispatcher_name=str(ts.name),
        unambiguous_operator_name=ts.name.unambiguous_name(),
    )


def _native_function_from_yaml(
    item: dict,
    reference_function=None,
    backend_indices: dict[object, object] | None = None,
) -> NativeFunction:
    f = parse_schema(
        item["func"],
        getattr(reference_function, "func", None),
    )
    seen_v: list[str] = []
    for v in item.get("variants", "function").split(","):
        v = v.strip()
        if v and v not in seen_v:
            seen_v.append(v)
    f.variants = seen_v
    f.cpp_no_default_args = set(item.get("cpp_no_default_args") or [])
    disp = item.get("dispatch") or {}
    f.dispatch = dict(disp) if isinstance(disp, dict) else {}
    f.device_check = item.get("device_check")
    f.structured_delegate = item.get("structured_delegate")
    f.structured_outputs = item.get("structured_outputs")
    f.autograd_meta = item.get("autograd")
    pd = item.get("python_defaults") or {}
    f.pydefaults = dict(pd)
    if reference_function is not None:
        f.source_native_function = reference_function
        f.dispatcher_name = str(reference_function.func.name)
        f.unambiguous_operator_name = reference_function.func.name.unambiguous_name()
        f.schema_kind = reference_function.func.kind().name
        f.tags = frozenset(reference_function.tags)
        f.device_check = reference_function.device_check.name
        f.structured_delegate = (
            str(reference_function.structured_delegate)
            if reference_function.structured_delegate is not None
            else f.structured_delegate
        )
        if backend_indices:
            for dispatch_key, backend_index in backend_indices.items():
                index = getattr(backend_index, "index", {})
                metadata = index.get(reference_function.func.name)
                if metadata is not None:
                    f.backend_metadata[str(dispatch_key)] = metadata
        _validate_dispatch_projection(f)
        _validate_native_projection(f, reference_function)
    return f


def _validate_dispatch_projection(f: NativeFunction) -> None:
    if not f.dispatch:
        return

    expected: dict[str, str] = {}
    for dispatch_key, metadata in f.backend_metadata.items():
        expected[dispatch_key] = metadata.kernel

    actual: dict[str, str] = {}
    for dispatch_key, kernel in f.dispatch.items():
        if dispatch_key == "__line__":
            continue
        normalized_key = {
            "Composite": "CompositeExplicitAutograd",
        }.get(dispatch_key, dispatch_key)
        actual[normalized_key] = str(kernel).rsplit("::", 1)[-1]

    if actual != expected:
        raise ValueError(
            f"dispatch projection mismatch for {f.func_name}: "
            f"{actual} != {expected}"
        )


def _validate_native_projection(f: NativeFunction, reference_function) -> None:
    expected_schema = str(reference_function.func)
    actual_schema = f.schema.replace("int64_t", "int")
    if actual_schema != expected_schema:
        raise ValueError(
            f"schema projection mismatch for {f.schema}: "
            f"{actual_schema} != {expected_schema}"
        )

    expected_kind = reference_function.func.kind().name
    if f.schema_kind != expected_kind:
        raise ValueError(
            f"schema kind mismatch for {f.func_name}: "
            f"{f.schema_kind} != {expected_kind}"
        )

    expected_variants = {variant.name for variant in reference_function.variants}
    if set(f.variants) != expected_variants:
        raise ValueError(
            f"variant mismatch for {f.func_name}: "
            f"{set(f.variants)} != {expected_variants}"
        )

    expected_out = tuple(
        argument.name for argument in reference_function.func.arguments.out
    )
    if f.out_args != expected_out:
        raise ValueError(
            f"out argument mismatch for {f.func_name}: "
            f"{f.out_args} != {expected_out}"
        )

    expected_no_defaults = set(reference_function.cpp_no_default_args)
    if f.cpp_no_default_args != expected_no_defaults:
        raise ValueError(
            f"cpp default policy mismatch for {f.func_name}: "
            f"{f.cpp_no_default_args} != {expected_no_defaults}"
        )


def _requires_seeded_tag(schema) -> bool:
    name = str(schema.name)
    return (
        "rand" in name
        or (
            ("dropout" in name
             or any("dropout" in a.name for a in schema.arguments.flat_all))
            and "backward" not in name
            and name != "_cudnn_init_dropout_state"
        )
        or schema.arguments.has_generator_arg()
    )


def _prepare_reference_entry(item: dict) -> dict:
    """Normalize project-only fields before the strict schema pass."""
    prepared = copy.deepcopy(item)
    prepared.pop("python_defaults", None)
    source_schema = prepared["func"]
    prepared["func"] = source_schema.replace("int64_t", "int")

    tgm = _ensure_torchgen_imported()
    schema = tgm.FunctionSchema.parse(prepared["func"])
    op_name = str(schema.name.name)

    if op_name.startswith("_foreach"):
        prepared.setdefault("device_check", "NoCheck")

    dispatch = prepared.get("dispatch")
    if isinstance(dispatch, dict):
        translated = dict(dispatch)
        if "Composite" in translated:
            translated["CompositeExplicitAutograd"] = translated.pop("Composite")
        prepared["dispatch"] = translated
    elif (
        op_name.startswith("new_")
        or op_name.endswith("_like")
        or (
            schema.arguments.tensor_options is not None
            and not schema.arguments.has_tensor_arg()
        )
    ):
        from torchgen.api import cpp
        prepared["dispatch"] = {
            "CompositeExplicitAutograd": cpp.name(schema),
        }

    if _requires_seeded_tag(schema):
        tags = prepared.setdefault("tags", [])
        if isinstance(tags, str):
            tags = [tags]
            prepared["tags"] = tags
        else:
            tags = list(tags)
            prepared["tags"] = tags
        if "nondeterministic_seeded" not in tags:
            tags.append("nondeterministic_seeded")
    return prepared


@dataclass(frozen=True)
class _ReferenceParseResult:
    functions: dict[object, object]
    native_functions: tuple[object, ...]
    backend_indices: dict[object, object]


def _reference_pre_group_native_functions(native_functions):
    grouped = defaultdict(dict)
    for function in native_functions:
        key = function.func.signature()
        kind = function.func.kind()
        while kind in grouped[key]:
            key = (key, function.func.name)
        grouped[key][kind] = function
    return grouped


class NativeFunctionCollection(list[NativeFunction]):
    """List-compatible schema collection carrying backend indexes."""

    def __init__(
        self,
        functions: list[NativeFunction],
        reference_functions: tuple[object, ...],
        backend_indices,
    ):
        super().__init__(functions)
        self.reference_functions = reference_functions
        self.backend_indices = backend_indices

    @property
    def reference_by_name(self) -> dict[object, object]:
        return {
            function.func.name: function
            for function in self.reference_functions
        }

    def grouped_native_functions(self):
        """Return the native function groups used by downstream generators."""
        generator = _ensure_torchgen_generator()
        grouped = _reference_pre_group_native_functions(self.reference_functions)
        result = []
        for functions in grouped.values():
            if (generator.SchemaKind.functional not in functions
                    or generator.SchemaKind.out not in functions):
                result.extend(functions.values())
                continue
            group = generator.NativeFunctionsGroup.from_dict(functions)
            if group is None:
                if any("generated" in f.tags for f in functions.values()):
                    raise AssertionError(
                        "generated native functions must form a complete group"
                    )
                result.extend(functions.values())
            else:
                result.append(group)
        return result

    def grouped_view_functions(self):
        model = _ensure_torchgen_imported()
        grouped = defaultdict(dict)
        for function in self.reference_functions:
            schema = function.func.view_signature()
            view_kind = function.view_schema_kind
            kind = (function.func.kind()
                    if view_kind == model.ViewSchemaKind.non_aliasing
                    else view_kind)
            is_view_copy = (
                kind == model.SchemaKind.functional
                and function.func.name.name.base.endswith(("_copy", "_scatter"))
                and "view_copy" in function.tags
            )
            if kind == model.SchemaKind.functional and not is_view_copy:
                schema = (schema, function.func.name)
            while kind in grouped[schema]:
                schema = (schema, function.func.name)
            grouped[schema][kind] = function

        result = []
        for functions in grouped.values():
            view = functions.pop(model.ViewSchemaKind.aliasing, None)
            if view is not None:
                result.append(model.NativeFunctionsViewGroup(
                    view=view,
                    view_copy=functions.pop(model.SchemaKind.functional, None),
                    view_inplace=functions.pop(
                        model.ViewSchemaKind.aliasing_inplace, None),
                ))
            result.extend(functions.values())
        return result


def _parse_reference_native_yaml(path: str) -> _ReferenceParseResult:
    """Run the complete native schema validation and return parsed records."""
    generator = _ensure_torchgen_generator()
    with open(path, "r") as fh:
        data = yaml.load(fh, Loader=generator.LineLoader)
    if not isinstance(data, list):
        raise TypeError(f"schema file must contain a list: {path}")

    prepared = [_prepare_reference_entry(item) for item in data]
    valid_tags = generator.parse_tags_yaml(str(_tags_path()))

    # TP declares a Vulkan backend in its schema file; the vendored reference
    # parser validates dispatch keys against its own closed set, so admit the
    # key for the duration of the parse.
    from torchgen import model as _tg_model
    reference_dispatch_keys = _tg_model.dispatch_keys
    if not any(getattr(k, "name", str(k)) == "Vulkan"
               for k in reference_dispatch_keys):
        class _VulkanKey:
            name = "Vulkan"

            def __str__(self):
                return "Vulkan"

            def __eq__(self, other):
                return self is other or getattr(other, "name", None) == "Vulkan"

            def __hash__(self):
                return hash("Vulkan")
        reference_dispatch_keys.append(_VulkanKey())

    from torchgen import native_function_generation

    original_pre_group = native_function_generation.pre_group_native_functions
    original_no_out = tuple(
        native_function_generation.FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT
    )

    for item in prepared:
        schema = generator.NativeFunction.from_yaml(
            item,
            generator.Location(path, item["__line__"]),
            valid_tags,
        )[0].func
        if (
            schema.kind().name == "functional"
            and not any(return_value.type.is_tensor_like()
                        for return_value in schema.returns)
            and not any("out" in str(value)
                        for value in str(item.get("autogen", "")).split(", "))
        ):
            name = str(schema.name)
            if name not in (
                native_function_generation
                .FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT
            ):
                native_function_generation.FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT.append(
                    name
                )

    native_function_generation.pre_group_native_functions = (
        _reference_pre_group_native_functions
    )
    try:
        parsed = generator.parse_native_yaml_struct(
            prepared,
            valid_tags,
            path=path,
        )
    finally:
        native_function_generation.pre_group_native_functions = original_pre_group
        native_function_generation.FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT[:] = (
            original_no_out
        )

    if len(parsed.native_functions) < len(prepared):
        raise ValueError(
            f"schema parser dropped records for {path}: "
            f"{len(prepared)} > {len(parsed.native_functions)}"
        )
    functions: dict[object, object] = {}
    for function in parsed.native_functions:
        name = function.func.name
        if name in functions:
            raise ValueError(f"duplicate parsed operator schema: {name}")
        functions[name] = function
    return _ReferenceParseResult(
        functions,
        tuple(parsed.native_functions),
        parsed.backend_indices,
    )


def parse_native_yaml(path: str) -> NativeFunctionCollection:
    with open(path, "r") as fh:
        data = yaml.load(fh, Loader=YamlLoader)
    if not isinstance(data, list):
        raise TypeError(f"schema file must contain a list: {path}")
    reference_parse = _parse_reference_native_yaml(path)
    funcs = []
    seen_schemas: set[str] = set()
    seen_operators: dict[str, str] = {}
    for item in data:
        # Exact-duplicate `- func:` entries are merge artifacts; keep first.
        if item["func"] in seen_schemas:
            continue
        seen_schemas.add(item["func"])
        parsed_schema = item["func"].replace("int64_t", "int")
        reference_schema = _ensure_torchgen_imported().FunctionSchema.parse(
            parsed_schema
        )
        reference_function = reference_parse.functions.get(reference_schema.name)
        if reference_function is None:
            raise ValueError(f"schema record was not returned by the parser: {item['func']}")
        function = _native_function_from_yaml(
            item,
            reference_function,
            reference_parse.backend_indices,
        )
        previous = seen_operators.get(function.func_name)
        if previous is not None:
            raise ValueError(
                f"duplicate operator schema {function.func_name}: "
                f"{previous} and {function.schema}"
            )
        seen_operators[function.func_name] = function.schema
        funcs.append(function)
    configured_names = set(seen_operators)
    reference_names = {
        str(function.func.name) for function in reference_parse.native_functions
    }
    if configured_names != reference_names:
        missing = sorted(reference_names - configured_names)
        extra = sorted(configured_names - reference_names)
        raise ValueError(
            "schema parser record set mismatch: "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    return NativeFunctionCollection(
        funcs,
        reference_parse.native_functions,
        reference_parse.backend_indices,
    )


def parse_derivatives_yaml(path: str) -> list[dict]:
    with open(path, "r") as fh:
        data = yaml.load(fh, Loader=YamlLoader)
    return data or []
