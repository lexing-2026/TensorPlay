"""C++ / stub / pybind / pyi type mapping for the schema model.

This mirrors torchgen's api layer (api/cpp.py, api/types.py): the schema
`Type` is translated exactly once into each target spelling, and generators
never re-derive C++ signatures from raw strings.
"""

from __future__ import annotations

import re

from .model import Argument, NativeFunction, Type, make_type

# ---------------------------------------------------------------------------
# p10 backend registration on torchgen's CType algebra
#
# Mirrors how out-of-tree backends adopt torchgen: provide the target's atomics
# and reference/value conventions, then every signature composes through the
# SAME CType classes upstream uses (ConstRef/MutRef/Optional/Vector).
# ---------------------------------------------------------------------------
from tools.codegen.model import _ensure_torchgen
_ensure_torchgen()
from torchgen.api.types import (
    BaseCType, ConstRefCType, MutRefCType, OptionalCType, VectorCType,
    longT as tg_longT, doubleT as tg_doubleT, boolT as tg_boolT,
    stringT as tg_stringT,
)
from torchgen.api.types import BaseCppType


class _P10(BaseCppType):
    pass


_TENSOR = BaseCType(_P10("", "Tensor"))
_SCALAR = BaseCType(_P10("", "Scalar"))
_DTYPE = BaseCType(_P10("", "DType"))
_DEVICE = BaseCType(_P10("", "Device"))
_GENERATOR = BaseCType(_P10("", "Generator"))


class StdOptionalCType(OptionalCType):
    """upstream emits c10::optional; p10 uses std::optional."""
    def cpp_type(self) -> str:
        return f"std::optional<{self.elem.cpp_type()}>"


class StdVectorCType(VectorCType):
    def cpp_type(self) -> str:
        return f"std::vector<{self.elem.cpp_type()}>"


_RAW_ATOMIC = {
    "Tensor": _P10("", "Tensor"),
    "Scalar": _P10("", "Scalar"),
    "DType": _P10("", "DType"),
    "Device": _P10("", "Device"),
    "Generator": _P10("", "Generator"),
    "int64_t": tg_longT,
    "double": tg_doubleT,
    "bool": tg_boolT,
    "str": _P10("", "std::string"),
    "MemoryFormat": tg_longT,   # enum not exposed yet; ABI-stable int
    "Layout": tg_longT,
}


def p10_ctype(t: Type):
    """Compose the p10 C++ type for a schema Type via torchgen algebra."""
    atom = BaseCType(_RAW_ATOMIC[t.kind])
    if t.is_list:
        return StdVectorCType(atom)
    if t.is_opt:
        return StdOptionalCType(atom)
    if t.is_tensor_like:
        return MutRefCType(atom) if t.mutability else ConstRefCType(atom)
    return atom


# ---------------------------------------------------------------------------
# C++ argument types (function/method declarations)
# ---------------------------------------------------------------------------

_CPP_ARG_TYPES = {
    "Tensor": "const Tensor&",
    "int64_t": "int64_t",
    "double": "double",
    "bool": "bool",
    "str": "std::string",
    "Scalar": "Scalar",
    "DType": "DType",
    "Device": "Device",
    "MemoryFormat": "int64_t",  # TensorPlay does not expose the enum yet;
                                # keep the schema optional and ABI-stable.
    "Layout": "int64_t",
    "Generator": "Generator",
    "void": "void",
}


def cpp_arg_type(t: Type) -> str:
    """C++ type used in public declarations (methods, tpx wrappers)."""
    def _norm(s):
        return s.replace(" &", "&")
    ct = p10_ctype(t)
    if t.is_tensor_like:
        if t.is_mutable_ref:
            return _norm("Tensor&")
        if t.is_opt:
            return _norm(f"const {ct.cpp_type()}&")
        if t.is_list:
            # Mutable lists pass by value so the pybind ABI accepts Python
            # lists while element mutation is preserved.
            return ct.cpp_type() if t.mutability else f"const {ct.cpp_type()}&"
        return _norm(ct.cpp_type())
    if t.is_list and not t.is_opt:
        return _norm(f"const {ct.cpp_type()}&")
    return ct.cpp_type()


def cpp_return_type(f: NativeFunction) -> str:
    kind = f.cpp_return_kind
    if kind == "void":
        return "void"
    if kind == "mut_ref":
        return "Tensor&"
    if kind == "list":
        return "std::vector<Tensor>"
    if kind == "tuple":
        parts = []
        for r in f.returns:
            rt = r.type
            if rt.is_tensor_like and not rt.is_list:
                parts.append("Tensor")
            elif rt.is_list:
                parts.append(f"std::vector<{rt.kind}>")
            else:
                parts.append(_CPP_ARG_TYPES.get(rt.kind, rt.kind))
        return f"std::tuple<{', '.join(parts)}>"
    r = f.returns[0]
    if r.type.is_tensor_like:
        return "Tensor"
    return _CPP_ARG_TYPES.get(r.type.kind, r.type.kind)


def tuple_element_cpp_types(f: NativeFunction) -> list[str]:
    assert f.cpp_return_kind == "tuple"
    out = []
    for r in f.returns:
        rt = r.type
        if rt.is_tensor_like:
            out.append("std::vector<Tensor>" if rt.is_list else "Tensor")
        else:
            out.append(_CPP_ARG_TYPES.get(rt.kind, rt.kind))
    return out


def tuple_element_names(f: NativeFunction) -> list[str]:
    names = [r.name or f"ret{i}" for i, r in enumerate(f.returns)]
    return names


# ---------------------------------------------------------------------------
# Dispatcher stub template arguments (type-erased kernel ABI)
# ---------------------------------------------------------------------------

def stub_arg_type(t: Type) -> str:
    """Dispatcher-stub ABI (type-erased kernel call signature)."""
    if t.is_tensor_like:
        if t.is_mutable_ref:
            return "Tensor&"
        if t.is_opt:
            return "std::optional<Tensor>"
        if t.is_list:
            return ("std::vector<Tensor>" if t.mutability
                    else "const std::vector<Tensor>&")
        return "const Tensor&"
    atom = BaseCType(_RAW_ATOMIC[t.kind])
    if t.is_list:
        vec = StdVectorCType(atom)
        return f"const {vec.cpp_type()}&" if not t.is_opt \
            else StdOptionalCType(vec).cpp_type()
    if t.is_opt:
        return StdOptionalCType(atom).cpp_type()
    return atom.cpp_type()


# ---------------------------------------------------------------------------
# Autograd node member types
# ---------------------------------------------------------------------------

def node_member_type(t: Type) -> str:
    """Saved-variable member type: always owned by value."""
    if t.is_tensor_like:
        if t.is_opt:
            return "std::optional<Tensor>"
        if t.is_list:
            return "std::vector<Tensor>"
        return "Tensor"
    if t.is_list:
        vec = f"std::vector<{t.kind}>"
        return f"std::optional<{vec}>" if t.is_opt else vec
    base = _CPP_ARG_TYPES.get(t.kind, t.kind)
    return f"std::optional<{base}>" if t.is_opt else base


# ---------------------------------------------------------------------------
# Default value translation
# ---------------------------------------------------------------------------

def cpp_default(t: Type, default: str) -> str:
    d = default
    if d == "Float32":
        return "DType::Float32"
    if d == "Int64":
        return "DType::Int64"
    if d == "Undefined":
        return "DType::Undefined"
    if d == "CPU":
        return "Device(DeviceType::CPU)"
    if d == "None":
        return "std::nullopt"
    if d == "true":
        return "true"
    if d == "false":
        return "false"
    if d.startswith("{") or d.startswith("["):
        body = d[1:-1].strip()
        if t.is_list or t.kind.endswith("[]"):
            return "{" + body + "}"
    return d


_PYI_DEFAULT_MAP = {
    "Float32": "DType.float32",
    "DType::Float32": "DType.float32",
    "Int64": "DType.int64",
    "DType::Int64": "DType.int64",
    "Undefined": "DType.undefined",
    "DType::Undefined": "DType.undefined",
    "CPU": "...",
    "Device(DeviceType::CPU)": "...",
    "None": "None",
    "std::nullopt": "None",
    "true": "True",
    "false": "False",
}


def pyi_default(t: Type, default: str) -> str:
    d = _PYI_DEFAULT_MAP.get(default, default)
    if t.is_tensor_like and d == "{}":
        return "None"
    if t.is_list and d.startswith("{") and d.endswith("}"):
        return "(" + d[1:-1] + ")"
    return d


def binding_default(t: Type, default: str) -> str:
    """Default expression inside a pybind11 def()."""
    d = default
    if d == "CPU":
        return "Device(DeviceType::CPU)"
    if d == "Undefined":
        return "DType::Undefined"
    if d == "None":
        return "py::none()"
    if d == "Float32":
        return "DType::Float32"
    if d == "Int64":
        return "DType::Int64"
    if d.startswith("{") or d.startswith("["):
        inner = d[1:-1].strip()
        if t.is_tensor_like:
            return "py::none()"
        return f"std::vector<{t.kind}>{{{inner}}}"
    return d


# ---------------------------------------------------------------------------
# Python typing (.pyi)
# ---------------------------------------------------------------------------

_PYI_TYPES = {
    "int64_t": "int",
    "double": "float",
    "bool": "bool",
    "str": "str",
    "Scalar": "Scalar",
    "DType": "DType",
    "Device": "Device",
    "MemoryFormat": "int",
    "Layout": "int",
    "Generator": "Generator",
    "void": "None",
}


def pyi_type(t: Type) -> str:
    if t.is_tensor_like:
        base = "TensorBase"
    else:
        base = _PYI_TYPES.get(t.kind, t.kind)
    if t.is_list:
        base = {
            "TensorBase": "Sequence[TensorBase]",
            "int": "Sequence[int]",
            "Scalar": "Sequence[Scalar]",
        }.get(base, f"Sequence[{base}]")
    if t.is_opt:
        base += " | None"
    return base


def pyi_return_type(f: NativeFunction) -> str:
    kind = f.cpp_return_kind
    if kind == "void":
        return "None"
    if kind == "mut_ref":
        return "TensorBase"
    if kind == "list":
        return "list[TensorBase]"
    if kind == "tuple":
        parts = [pyi_type(r.type) for r in f.returns]
        parts = ["TensorBase" if p == "TensorBase" else p for p in parts]
        return f"tuple[{', '.join(parts)}]"
    return pyi_type(f.returns[0].type)


# ---------------------------------------------------------------------------
# Functional-Python parameter defaults
# ---------------------------------------------------------------------------

def python_default(t: Type, default: str) -> str:
    return pyi_default(t, default)


# ---------------------------------------------------------------------------
# Misc helpers shared by generators
# ---------------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    return "from_" if name == "from" else name


def autograd_node_name(func_name: str) -> str:
    """Torch-style backward node name (`add.Tensor` -> AddBackward).

    In-place overloads share the functional overload's node: canonicalizing
    `_.` to `.` also prevents duplicate node definitions.
    """
    canonical = func_name.replace("_.", ".")
    clean = "".join(x.title() for x in canonical.replace(".", "_").split("_"))
    return clean + "Backward"


_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


# ---------------------------------------------------------------------------
# Optional-Tensor unwrap boundary
# ---------------------------------------------------------------------------

# Ops whose backend kernels take `const Tensor&` for arguments the canonical
# schema spells as `Tensor?`.  ATen behaves identically: optionality is
# unwrapped in generated glue (nullopt -> undefined Tensor), kernels keep the
# reference ABI.  Generated call sites route through call_arg_expr().
UNWRAP_OPT_TENSOR: dict[str, set[str]] = {
    "conv1d": {"bias"},
    "conv2d": {"bias"},
    "conv3d": {"bias"},
    "conv_transpose1d": {"bias"},
    "conv_transpose2d": {"bias"},
    "conv_transpose3d": {"bias"},
    "nll_loss_backward": {"total_weight"},
}


def _unwrap_targeted(op_base: str, a) -> bool:
    return (a.type.is_opt and a.type.is_tensor_like
            and op_base in UNWRAP_OPT_TENSOR
            and a.name in UNWRAP_OPT_TENSOR[op_base])


def stub_arg_type_for(op_base: str, a) -> str:
    """Dispatcher-stub ABI type honoring the unwrap boundary."""
    if _unwrap_targeted(op_base, a):
        t = make_type("Tensor", False, False, None, False)
        return stub_arg_type(t)
    return stub_arg_type(a.type)


def call_arg_expr(op_base: str, a) -> str:
    """Argument expression passed across the unwrap boundary."""
    if _unwrap_targeted(op_base, a):
        return f"{a.name}.has_value() ? *{a.name} : Tensor()"
    return a.name


# python_defaults support: schema-level defaults stripped during
# canonicalization live on NativeFunction.pydefaults; translate their
# `[a, b]` spelling into each target language here.

def py_default_for(f, a, kind: str):
    """kind: 'python' | 'binding' | 'pyi'. Returns default text or None."""
    raw = (f.pydefaults or {}).get(a.name)
    if raw is None:
        return None
    if kind == "python":
        return "(" + raw + ")"
    if kind == "binding":
        inner = raw[1:-1].strip()
        vec = f"std::vector<{a.type.kind}>{{{inner}}}"
        return f"{vec}" if not a.type.is_opt else f"std::optional<{vec}>{'' }"
    return "(" + raw + ")"
