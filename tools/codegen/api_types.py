"""C++ / stub / pybind / pyi type mapping for the schema model.

`Type` is translated exactly once into each target spelling, and generators
never re-derive C++ signatures from raw strings.
"""

from __future__ import annotations

import ast
import json
import re

from .model import Argument, NativeFunction, Type, make_type

# ---------------------------------------------------------------------------
#
# and reference/value conventions, then every signature composes through the
# Shared CType classes (ConstRef/MutRef/Optional/Vector).
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


def _symbolic_kind(t: Type, symbolic: bool = False) -> str:
    if symbolic and t.symint:
        return "SymInt"
    if symbolic and t.symbool:
        return "SymBool"
    if symbolic and t.symfloat:
        return "SymFloat"
    return t.kind


class StdOptionalCType(OptionalCType):
    """Use std::optional for optional arguments."""
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
    "SymInt": _P10("", "SymInt"),
    "SymBool": _P10("", "SymBool"),
    "SymFloat": _P10("", "SymFloat"),
    "int64_t": tg_longT,
    "double": tg_doubleT,
    "bool": tg_boolT,
    "str": _P10("", "std::string"),
    "MemoryFormat": tg_longT,   # enum not exposed yet; ABI-stable int
    "Layout": tg_longT,
}


def p10_ctype(t: Type, symbolic: bool = False):
    atom = BaseCType(_RAW_ATOMIC[_symbolic_kind(t, symbolic)])
    if t.list_elem_opt:
        atom = StdOptionalCType(atom)
    if t.is_list and t.is_opt:
        # ``int[]?`` composes optional over the vector type.
        return StdOptionalCType(StdVectorCType(atom))
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
    "SymInt": "SymInt",
    "SymBool": "SymBool",
    "SymFloat": "SymFloat",
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


def _cpp_value_type(t: Type) -> str:
    if t.is_tensor_like:
        if t.is_list:
            elem = "std::optional<Tensor>" if t.list_elem_opt else "Tensor"
            value = f"std::vector<{elem}>"
        else:
            value = "Tensor"
    elif t.is_list:
        atom = _CPP_ARG_TYPES.get(t.kind, t.kind)
        elem = f"std::optional<{atom}>" if t.list_elem_opt else atom
        value = f"std::vector<{elem}>"
    else:
        value = _CPP_ARG_TYPES.get(t.kind, t.kind)
    return f"std::optional<{value}>" if t.is_opt else value


def _cpp_symbolic_value_type(t: Type) -> str:
    if t.is_tensor_like:
        return _cpp_value_type(t)
    kind = _symbolic_kind(t, True)
    if t.is_list:
        atom = _CPP_ARG_TYPES.get(kind, kind)
        elem = f"std::optional<{atom}>" if t.list_elem_opt else atom
        value = f"std::vector<{elem}>"
    else:
        value = _CPP_ARG_TYPES.get(kind, kind)
    return f"std::optional<{value}>" if t.is_opt else value


def cpp_return_type(f: NativeFunction) -> str:
    kind = f.cpp_return_kind
    if kind == "void":
        return "void"
    if kind == "mut_ref":
        return "Tensor&"
    if kind == "list":
        return _cpp_symbolic_value_type(f.returns[0].type)
    if kind == "tuple":
        parts = [_cpp_symbolic_value_type(r.type) for r in f.returns]
        return f"std::tuple<{', '.join(parts)}>"
    return _cpp_symbolic_value_type(f.returns[0].type)


def tuple_element_cpp_types(f: NativeFunction) -> list[str]:
    assert f.cpp_return_kind == "tuple"
    out = []
    for r in f.returns:
        out.append(_cpp_symbolic_value_type(r.type))
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
            return f"std::optional<{_cpp_value_type(make_type(t.kind, t.is_list, False, None, t.symint, t.list_elem_opt))}>"
        if t.is_list:
            elem = "std::optional<Tensor>" if t.list_elem_opt else "Tensor"
            vector = f"std::vector<{elem}>"
            return vector if t.mutability else f"const {vector}&"
        return "const Tensor&"
    atom = BaseCType(_RAW_ATOMIC[t.kind])
    if t.list_elem_opt:
        atom = StdOptionalCType(atom)
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
        if t.is_list:
            elem = "std::optional<Tensor>" if t.list_elem_opt else "Tensor"
            value = f"std::vector<{elem}>"
            return f"std::optional<{value}>" if t.is_opt else value
        if t.is_opt:
            return "std::optional<Tensor>"
        return "Tensor"
    if t.is_list:
        elem = f"std::optional<{t.kind}>" if t.list_elem_opt else t.kind
        vec = f"std::vector<{elem}>"
        return f"std::optional<{vec}>" if t.is_opt else vec
    base = _CPP_ARG_TYPES.get(t.kind, t.kind)
    return f"std::optional<{base}>" if t.is_opt else base


# ---------------------------------------------------------------------------
# Default value translation
# ---------------------------------------------------------------------------

# Canonical definition for MemoryFormat's ABI-stable integer values; the
# dispatcher rides the enum as int64_t everywhere ("enum not exposed" in the
# C++ atomics above), so every target language renders defaults from here.
_MEMORY_FORMAT_VALUES = {"Contiguous": 0, "Preserve": 1,
                         "ChannelsLast": 2, "ChannelsLast3d": 3}

_REDUCTION_VALUES = {"None": 0, "Mean": 1, "Sum": 2}

# analog): tensorplay exposes these as IntEnum members, which ARE their
# integer ABI values.
_MEMORY_FORMAT_PY = {
    "Contiguous": "tensorplay.contiguous_format",
    "Preserve": "tensorplay.preserve_format",
    "ChannelsLast": "tensorplay.channels_last",
    "ChannelsLast3d": "tensorplay.channels_last_3d",
}


def _memory_format_name(default: str) -> str | None:
    """Accept both bare and enum-qualified yaml spellings."""
    name = default.split("::")[-1].strip()
    aliases = {
        "contiguous_format": "Contiguous",
        "preserve_format": "Preserve",
        "channels_last": "ChannelsLast",
        "channels_last_3d": "ChannelsLast3d",
    }
    name = aliases.get(name, name)
    return name if name in _MEMORY_FORMAT_VALUES else None


def _string_value(default: str) -> str:
    value = ast.literal_eval(default)
    if not isinstance(value, str):
        raise ValueError(f"string schema default is not a string: {default}")
    return value


def _cpp_string_default(default: str) -> str:
    return json.dumps(_string_value(default), ensure_ascii=False)


def cpp_default(t: Type, default: str) -> str:
    d = default
    if t.kind == "int64_t" and not t.is_list and not t.is_opt and d in _REDUCTION_VALUES:
        return str(_REDUCTION_VALUES[d])
    if d == "Float32":
        return "DType::Float32"
    if d == "Int64":
        return "DType::Int64"
    if d == "long":
        # Schema shorthand for the 64-bit integer dtype.
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
    if t.kind == "str":
        return _cpp_string_default(d)
    if t.kind == "MemoryFormat":
        # MemoryFormat rides the dispatcher as its integer value, and the
        # rendered C++ parameter type is int64_t.
        name = _memory_format_name(d)
        if name is not None:
            return str(_MEMORY_FORMAT_VALUES[name])
    # schema (`int[1] padding=0`, `int[2] dilation=1`, ...).  TensorPlay's
    # public ABI represents every list as std::vector, so retain the exact
    # schema while making the generated C++ default a valid one-element
    # vector.  The CPython bridge deliberately keeps accepting the scalar at
    # the Python boundary, matching PythonArgParser.
    if t.is_list and d.lstrip("+-").isdigit():
        return "{" + d + "}"
    if d.startswith("{") or d.startswith("["):
        body = d[1:-1].strip()
        if t.is_list or t.kind.endswith("[]"):
            return "{" + body + "}"
    if t.is_list or t.kind.endswith("[]"):
        # Scalar default for an array param (e.g. "int[1] padding=0"): brace it
        # so it initializes std::vector / IntArrayRef ("= 0" is not convertible).
        return "{" + d + "}"
    return d


_PYI_DEFAULT_MAP = {
    "Float32": "DType.float32",
    "DType::Float32": "DType.float32",
    "Int64": "DType.int64",
    "DType::Int64": "DType.int64",
    "Undefined": "DType.undefined",
    "DType::Undefined": "DType.undefined",
    "long": "DType.int64",
    "CPU": "...",
    "Device(DeviceType::CPU)": "...",
    "None": "None",
    "std::nullopt": "None",
    "true": "True",
    "false": "False",
}


def pyi_default(t: Type, default: str) -> str:
    if t.kind == "int64_t" and not t.is_list and not t.is_opt and default in _REDUCTION_VALUES:
        return str(_REDUCTION_VALUES[default])
    if t.kind == "MemoryFormat":
        # Bare `Contiguous` would be an undefined name in generated Python;
        name = _memory_format_name(default)
        if name is not None:
            return _MEMORY_FORMAT_PY[name]
    if t.kind == "str" and default != "None":
        return repr(_string_value(default))
    d = _PYI_DEFAULT_MAP.get(default, default)
    if t.is_tensor_like and d == "{}":
        return "None"
    if t.is_list and d.startswith("{") and d.endswith("}"):
        return "(" + d[1:-1] + ")"
    return d


def binding_default(t: Type, default: str) -> str:
    """Default expression inside a pybind11 def()."""
    d = default
    if t.kind == "int64_t" and not t.is_list and not t.is_opt and d in _REDUCTION_VALUES:
        return str(_REDUCTION_VALUES[d])
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
    if d == "long":
        # Schema shorthand for the 64-bit integer dtype.
        return "DType::Int64"
    if t.kind == "str":
        return _cpp_string_default(d)
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
        if base == "TensorBase" and not t.list_elem_opt:
            base = "Sequence[TensorBase]"
        else:
            elem = f"{base} | None" if t.list_elem_opt else base
            base = f"Sequence[{elem}]"
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
    """

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
        t = make_type("Tensor", False, False, None, False, False, False)
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
