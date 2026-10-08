"""Generate CPython C-API fast-path wrappers.

Per op emits one ``METH_FASTCALL | METH_KEYWORDS`` function that parses
positional/keyword args against the schema, unpacks through the
``CPythonBridge.h`` helpers, calls the generated ``tpx::ops`` symbol and
packs the result.  pybind11 overload dispatch is bypassed; the existing
binding surface stays untouched.

Every configured schema signature must have a native bridge mapping.  Code
generation fails when a type is not implemented instead of emitting a second
binding path with different call semantics.
"""

from __future__ import annotations

import ast as _ast
import hashlib as _hashlib
import json as _json

from .api_types import (_MEMORY_FORMAT_VALUES, _memory_format_name,
                        binding_default, cpp_arg_type, cpp_return_type,
                        py_default_for, tuple_element_cpp_types)
from .main import CodegenContext, register_generator

import re as _re

_INT_RE = _re.compile(r"^[+-]?\d+$")
_FLOAT_RE = _re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+[eE][+-]?\d+)$")


def _default_pyobject(a, expr: str) -> str:
    """Turn a schema-default C++ text into a PyObject* producing expression.

    Shared by every METH_FASTCALL generator: C-level defaults are what let
    callers omit trailing/keyword args at the raw layer.  Raises on defaults
    with no CPython-literal form (Device/DType); those are rejected loudly at
    generation time rather than mis-bound silently.
    """
    if expr == "py::none()" or expr == "None":
        return "Py_None"
    if expr in ("true", "True"):
        return "Py_True"
    if expr in ("false", "False"):
        return "Py_False"
    if _INT_RE.match(expr):
        return f"PyLong_FromLongLong({expr}LL)"
    if _FLOAT_RE.match(expr):
        return f"PyFloat_FromDouble({expr})"
    if a.type.kind == "str":
        try:
            value = _ast.literal_eval(expr)
        except (SyntaxError, ValueError) as error:
            raise SystemExit(
                f"unsupported string default {expr!r} "
                "(expected a quoted string literal)") from error
        if not isinstance(value, str):
            raise SystemExit(
                f"unsupported string default {expr!r} "
                "(expected a quoted string literal)")
        return f"PyUnicode_FromString({_json.dumps(value, ensure_ascii=False)})"
    if a.type.is_list:
        return expr  # marker: caller emits a list-builder helper
    if a.type.kind == "DType" and expr.startswith("DType::"):
        return f"tpx_py_wrap_dtype({expr})"
    if a.type.kind == "Device" and expr.startswith("Device("):
        return f"tpx_py_wrap_device({expr})"
    if a.type.kind == "MemoryFormat":
        # MemoryFormat rides the dispatcher as its integer value; accept both
        # bare and enum-qualified spellings from the yaml.
        name = _memory_format_name(expr)
        if name in _MEMORY_FORMAT_VALUES:
            return f"PyLong_FromLongLong({_MEMORY_FORMAT_VALUES[name]}LL)"
    raise SystemExit(
        f"default {expr!r} for argument '{a.name}' of type '{a.type.kind}' "
        "has no CPython-literal mapping; drop the default from the yaml or "
        "extend gen_python_c._default_pyobject")

# Schema C++ type -> native bridge call template.
# Tensor args bind by reference into the Python wrapper's storage: the const
# form skips one refcount pair per argument, the mutable form is what makes
# in-place ops write through to the caller's tensor.
_BRIDGE = {
    "const Tensor&": "tpx_py_tensor_cref({n})",
    "Tensor&": "tpx_py_tensor_mref({n})",
    "Tensor": "tpx_py_tensor_cref({n})",
    "const Scalar&": "tpx_py_scalar({n})",
    "Scalar": "tpx_py_scalar({n})",
    "std::optional<Tensor>": "tpx_py_opt_tensor({n})",
    "int64_t": "tpx_py_int64({n})",
    "double": "tpx_py_double({n})",
    "bool": "tpx_py_bool({n})",
    "std::optional<int64_t>": "tpx_py_opt_int64({n})",
    "std::optional<double>": "tpx_py_opt_double({n})",
    "std::optional<bool>": "tpx_py_opt_bool({n})",
    "std::optional<Scalar>": "tpx_py_opt_scalar({n})",
    "Generator": "tpx_py_generator({n})",
    "std::optional<Generator>": "tpx_py_opt_generator({n})",
    "std::vector<int64_t>": "tpx_py_intlist({n})",
    "std::vector<double>": "tpx_py_doublelist({n})",
    # Schemas bind lists by const-ref at signature level; the unpackers
    # return by value, which binds fine -- keep both spellings claimed.
    "const std::vector<int64_t>&": "tpx_py_intlist({n})",
    "const std::vector<double>&": "tpx_py_doublelist({n})",
    "std::vector<bool>": "tpx_py_boollist({n})",
    "const std::vector<bool>&": "tpx_py_boollist({n})",
    "std::vector<Tensor>": "tpx_py_tensorlist({n})",
    "const std::vector<Tensor>&": "tpx_py_tensorlist({n})",
    "const std::vector<std::optional<Tensor>>&": "tpx_py_opt_tensorlist({n})",
    "std::vector<Scalar>": "tpx_py_scalarlist({n})",
    "const std::vector<Scalar>&": "tpx_py_scalarlist({n})",
    "const std::optional<Tensor>&": "tpx_py_opt_tensor({n})",
    "std::optional<std::vector<int64_t>>": "tpx_py_opt_intlist({n})",
    "std::optional<std::vector<double>>": "tpx_py_opt_doublelist({n})",
    "std::optional<std::string>": "tpx_py_opt_string({n})",
    "std::string": "tpx_py_string({n})",
    "DType": "tpx_py_dtype({n})",
    "std::optional<DType>": "tpx_py_opt_dtype({n})",
    "std::optional<Device>": "tpx_py_opt_device({n})",
    "Device": "tpx_py_device({n})",
    "Storage": "tpx_py_storage({n})",
}

# C++ argument type -> tpx_py_type_kind byte (see CPythonBridge.h).
_KIND_CONST = {
    "const Tensor&": "TPK_TENSOR",
    "Tensor&": "TPK_TENSOR",
    "Tensor": "TPK_TENSOR",
    "const Scalar&": "TPK_NUMBER",
    "Scalar": "TPK_NUMBER",
    "int64_t": "TPK_INT",
    "double": "TPK_FLOAT",
    "bool": "TPK_BOOL",
    "std::vector<int64_t>": "TPK_INTLIST",
    "std::vector<double>": "TPK_FLOATLIST",
    "const std::vector<int64_t>&": "TPK_INTLIST",
    "const std::vector<double>&": "TPK_FLOATLIST",
    "std::vector<bool>": "TPK_BOOLLIST",
    "const std::vector<bool>&": "TPK_BOOLLIST",
    "std::vector<Tensor>": "TPK_TENSORLIST",
    "const std::vector<Tensor>&": "TPK_TENSORLIST",
    "const std::vector<std::optional<Tensor>>&": "TPK_TENSORLIST_OPTIONAL",
    "std::vector<Scalar>": "TPK_SCALARLIST",
    "const std::vector<Scalar>&": "TPK_SCALARLIST",
    "std::string": "TPK_STR",
    "DType": "TPK_DTYPE",
    "Device": "TPK_DEVICE",
    "Generator": "TPK_GENERATOR",
    "Storage": "TPK_STORAGE",
}
_OPT = {
    "std::optional<Tensor>": "TPK_TENSOR",
    "std::optional<int64_t>": "TPK_INT",
    "std::optional<double>": "TPK_FLOAT",
    "std::optional<bool>": "TPK_BOOL",
    "std::optional<Scalar>": "TPK_NUMBER",
    "std::optional<std::string>": "TPK_STR",
    "std::optional<DType>": "TPK_DTYPE",
    "std::optional<Device>": "TPK_DEVICE",
    "const std::optional<Tensor>&": "TPK_TENSOR",
    "std::optional<std::vector<int64_t>>": "TPK_INTLIST",
    "std::optional<std::vector<double>>": "TPK_FLOATLIST",
    "std::optional<Generator>": "TPK_GENERATOR",
}
_KIND_CONST.update({k: v + " | TPK_OPTIONAL" for k, v in _OPT.items()})

_RET_SHAPES = {"void", "value", "tuple", "list", "mut_ref"}

_VMAP_MEMBER_OPS = frozenset({
    "neg", "negative", "abs", "exp", "log", "sin", "cos", "sinh",
    "cosh", "tanh", "sqrt", "rsqrt", "sigmoid", "relu", "floor", "ceil",
    "round", "trunc", "erf", "erfc", "log1p", "expm1", "mul.Tensor",
    "div.Tensor", "logical_or", "logical_xor", "add.Scalar", "sub.Scalar",
    "mul.Scalar", "div.Scalar",
    "add.Tensor", "sub.Tensor", "pow.Tensor_Scalar", "pow.Tensor_Tensor",
    "sum", "sum.dim_IntList", "size", "size.int", "stride", "stride.int",
    "permute", "transpose", "movedim",
    "reshape", "expand", "squeeze", "squeeze.dim", "squeeze.dims", "unsqueeze",
    "contiguous", "slice", "narrow", "index_select",
    "mm", "matmul", "bmm",
})

_VMAP_STATIC_OPS = frozenset({
    "maximum", "minimum", "logical_and", "cat", "stack", "linear",
})


def _schema_tag(f, variant: str, ordinal: int) -> str:
    payload = f"{variant}\0{ordinal}\0{f.schema}".encode("utf-8")
    return _hashlib.sha1(payload).hexdigest()[:10]


def _schema_fallback_doc(schemas: list[str]) -> str:
    """Render the schema-only fallback docstring as inline literals.

    Schema text carries trailing-underscore overload names and bare stars;
    once a method without a dedicated docstring reaches a documentation
    renderer, every summary and body pass parses it as links and emphasis.
    One double-backtick literal per overload is parse-neutral everywhere:
    inline literals suppress both, and a truncated summary still renders.
    """
    return "\\n".join("``" + s + "``" for s in schemas)


def _pack_expr(cpp_type: str, value: str) -> str | None:
    """Return the native Python-object packer for one result value."""
    if cpp_type == "Tensor":
        return f"tpx_py_wrap({value})"
    if cpp_type == "Tensor&":
        return f"tpx_py_wrap({value})"
    if cpp_type == "std::optional<Tensor>":
        return f"tpx_py_wrap_optional_tensor({value})"
    if cpp_type == "Scalar":
        return f"tpx_py_wrap_scalar({value})"
    if cpp_type == "std::optional<Scalar>":
        return f"tpx_py_wrap_optional_scalar({value})"
    if cpp_type == "Generator":
        return f"tpx_py_wrap_generator({value})"
    if cpp_type == "DType":
        return f"tpx_py_wrap_dtype({value})"
    if cpp_type == "Device":
        return f"tpx_py_wrap_device({value})"
    if cpp_type == "SymInt":
        return f"tpx_py_wrap_symint({value})"
    if cpp_type == "SymBool":
        return f"tpx_py_wrap_symbool({value})"
    if cpp_type == "SymFloat":
        return f"tpx_py_wrap_symfloat({value})"
    if cpp_type == "std::optional<SymInt>":
        return f"tpx_py_wrap_optional_symint({value})"
    if cpp_type == "std::optional<SymBool>":
        return f"tpx_py_wrap_optional_symbool({value})"
    if cpp_type == "std::optional<SymFloat>":
        return f"tpx_py_wrap_optional_symfloat({value})"
    if cpp_type == "std::vector<SymInt>":
        return f"tpx_py_wrap_symintlist({value})"
    if cpp_type == "std::vector<SymBool>":
        return f"tpx_py_wrap_symboollist({value})"
    if cpp_type == "std::vector<SymFloat>":
        return f"tpx_py_wrap_symfloatlist({value})"
    if cpp_type == "std::optional<std::vector<SymInt>>":
        return f"tpx_py_wrap_optional_symintlist({value})"
    if cpp_type == "std::optional<std::vector<SymBool>>":
        return f"tpx_py_wrap_optional_symboollist({value})"
    if cpp_type == "std::optional<std::vector<SymFloat>>":
        return f"tpx_py_wrap_optional_symfloatlist({value})"
    if cpp_type == "bool":
        return f"PyBool_FromLong({value})"
    if cpp_type == "std::optional<bool>":
        return f"tpx_py_wrap_optional_bool({value})"
    if cpp_type == "int64_t":
        return f"PyLong_FromLongLong({value})"
    if cpp_type == "std::optional<int64_t>":
        return f"tpx_py_wrap_optional_int64({value})"
    if cpp_type == "double":
        return f"PyFloat_FromDouble({value})"
    if cpp_type == "std::optional<double>":
        return f"tpx_py_wrap_optional_double({value})"
    if cpp_type == "std::string":
        return f"PyUnicode_FromString({value}.c_str())"
    if cpp_type == "std::optional<std::string>":
        return f"tpx_py_wrap_optional_string({value})"
    if cpp_type == "std::vector<Tensor>":
        return f"tpx_py_wrap_list({value})"
    if cpp_type == "std::vector<std::optional<Tensor>>":
        return f"tpx_py_wrap_optional_tensor_list({value})"
    if cpp_type == "std::vector<int64_t>":
        return f"tpx_py_wrap_intlist({value})"
    if cpp_type == "std::vector<double>":
        return f"tpx_py_wrap_doublelist({value})"
    if cpp_type == "std::vector<bool>":
        return f"tpx_py_wrap_boollist({value})"
    if cpp_type == "std::vector<Scalar>":
        return f"tpx_py_wrap_scalarlist({value})"
    if cpp_type == "std::optional<DType>":
        return f"tpx_py_wrap_optional_dtype({value})"
    if cpp_type == "std::optional<Device>":
        return f"tpx_py_wrap_optional_device({value})"
    if cpp_type == "std::optional<Generator>":
        return f"tpx_py_wrap_optional_generator({value})"
    if cpp_type == "std::optional<std::vector<int64_t>>":
        return f"tpx_py_wrap_optional_intlist({value})"
    if cpp_type == "std::optional<std::vector<double>>":
        return f"tpx_py_wrap_optional_doublelist({value})"
    return None


def _require_bridge(cpp_type: str) -> str:
    try:
        return _BRIDGE[cpp_type]
    except KeyError as error:
        raise SystemExit(
            f"native CPython bridge has no converter for C++ type {cpp_type!r}") from error


def _validate_op_support(f, variant: str) -> None:
    for a in f.args:
        cpp_type = cpp_arg_type(a.type)
        _require_bridge(cpp_type)
        if cpp_type not in _KIND_CONST:
            raise SystemExit(
                f"native CPython bridge has no type check for C++ type {cpp_type!r} "
                f"in {f.func_name} ({variant})")
    if f.cpp_return_kind not in _RET_SHAPES:
        raise SystemExit(
            f"native CPython bridge has no return shape for {f.cpp_return_kind!r} "
            f"in {f.func_name} ({variant})")
    if f.cpp_return_kind in {"value", "list"}:
        cpp_type = cpp_return_type(f)
        if _pack_expr(cpp_type, "result") is None:
            raise SystemExit(
                f"native CPython bridge has no packer for C++ type {cpp_type!r} "
                f"in {f.func_name} ({variant})")
    elif f.cpp_return_kind == "tuple":
        for cpp_type in tuple_element_cpp_types(f):
            if _pack_expr(cpp_type, "result") is None:
                raise SystemExit(
                    f"native CPython bridge has no tuple packer for C++ type "
                    f"{cpp_type!r} in {f.func_name} ({variant})")


def _tuple_invoke(f, invoke_expr: str, site_hook: str) -> str | None:
    elements = tuple_element_cpp_types(f)
    packed = [
        _pack_expr(t, f"std::get<{i}>(r)")
        for i, t in enumerate(elements)
    ]
    if any(expr is None for expr in packed):
        return None
    statements = [
        f"PyObject* tpx_tuple_result = PyTuple_New({len(elements)});",
        "if (tpx_tuple_result == nullptr) return nullptr;",
    ]
    for i, expr in enumerate(packed):
        assert expr is not None
        item = f"tpx_tuple_item_{i}"
        statements.extend([
            f"PyObject* {item} = {expr};",
            f"if ({item} == nullptr) {{ Py_DECREF(tpx_tuple_result); return nullptr; }}",
            f"PyTuple_SET_ITEM(tpx_tuple_result, {i}, {item});",
        ])
    statements.append("return tpx_tuple_result;")
    return (site_hook
            + f"auto r = [&]() {{ tpx_py_GilRelease _gil; return {invoke_expr}; }}(); "
            + " ".join(statements))


_PY_SIG_TYPE = {
    "Tensor": "Tensor", "Tensor&": "Tensor", "const Tensor&": "Tensor",
    "std::optional<Tensor>": "Tensor?",
    "DType": "DType", "std::optional<DType>": "DType?",
    "Device": "Device", "std::optional<Device>": "Device?",
    "Scalar": "Number", "std::optional<Scalar>": "Number?",
    "int64_t": "int", "std::optional<int64_t>": "int?",
    "double": "float", "std::optional<double>": "float?",
    "bool": "bool", "std::optional<bool>": "bool?",
    "std::string": "str", "std::optional<std::string>": "str?",
    "const std::vector<int64_t>&": "int[]", "std::vector<int64_t>": "int[]",
    "const std::vector<double>&": "float[]",
    "const std::vector<Tensor>&": "Tensor[]",
    "const std::vector<std::optional<Tensor>>&": "Tensor?[]",
    "const std::vector<bool>&": "bool[]",
    "const std::vector<std::string>&": "str[]",
    "Generator": "Generator", "Storage": "Storage",
    "SymInt": "SymInt", "std::optional<SymInt>": "SymInt?",
    "SymBool": "SymBool", "SymFloat": "SymFloat",
}


_FACTORY_OPTIONS = frozenset({"dtype", "layout", "device", "pin_memory"})


def _takes_requires_grad(f, variant: str) -> bool:
    """Whether the overload takes a ``requires_grad`` keyword of its own.

    A factory spelled with the full set of tensor options (dtype, layout,
    device, pin_memory) creates a fresh leaf; on the Python surface every such
    signature also takes ``requires_grad``, which marks the result.  The flag
    is not part of the schema, so the entry point parses it beside the schema
    arguments and applies it to the returned tensor.
    """
    if variant != "function" or f.out_args or f.cpp_return_kind != "value":
        return False
    if cpp_return_type(f) != "Tensor":
        return False
    names = {a.name for a in f.args if a.kwonly}
    return _FACTORY_OPTIONS <= names and not any(
        a.name == "requires_grad" for a in f.args)


def _py_sig_default(dflt: str | None) -> str:
    if dflt is None:
        return ""
    spelled = {
        "false": "False", "true": "True", "nullptr": "None",
        "DType::Undefined": "None",
    }
    text = spelled.get(dflt, dflt.replace("::", "."))
    if text.endswith("_a") or text.startswith("pydflt_"):
        return ""
    return f"={text}"


def _py_signature(f, variant: str) -> str:
    """Python-style overload signature shown when no candidate matched."""
    is_method = variant == "method" and any(a.name == "self" for a in f.args)
    parts = [a for a in f.args
             if not (is_method and a.name == "self")]
    display = []
    seen_kwonly = False
    for a in parts:
        if a.kwonly and not seen_kwonly:
            display.append("*")
            seen_kwonly = True
        cxx = _PY_SIG_TYPE.get(cpp_arg_type(a.type), cpp_arg_type(a.type))
        display.append(f"{cxx} {a.name}{_py_sig_default(a.default)}")
    if _takes_requires_grad(f, variant):
        display.append("bool requires_grad=False")
    return f"{f.base_name}({', '.join(display)})"


def _op_supported(f, variant: str) -> bool:
    """Validate one overload and report that it has a native entry point."""
    _validate_op_support(f, variant)
    return True


def plan_groups(funcs) -> "dict[tuple[str, str], list]":
    """Group every overload by its exposed variant and public name."""
    groups: "dict[tuple[str, str], list]" = {}
    for f in funcs:
        for variant in f.variants:
            _op_supported(f, variant)
            groups.setdefault((variant, f.cpp_name), []).append(f)
    return groups


def capi_claims(funcs):
    """Names the FASTCALL layer owns: {(variant, cpp_name): [funcs...]}."""
    return plan_groups(funcs)


def claims_variant(claimed, f, variant: str) -> bool:
    return (variant, f.cpp_name) in claimed


def _probe_info(f, variant: str):
    """Positional kind signature for the multi-overload fast probe.

    Returns None when the overload cannot be safely kind-probed (unknown kind
    constant or a trailing IntList splat, whose positional folding changes
    nargs semantics).  Otherwise a dict with the positional arity, the count
    of required positionals, and the per-position kind constants -- consumed
    by the generated dispatcher to pick a candidate overload without raising.
    """
    is_method = variant == "method"
    # The probe works on user arguments; the receiver's `self` slot sits
    # wherever the schema places it, so exclude it by name.
    pos = [a for a in f.args if not (is_method and a.name == "self")
           and not a.kwonly]
    if pos and pos[-1].type.is_list:
        return None                       # splat folding: nargs not comparable
    kinds = [_KIND_CONST.get(cpp_arg_type(a.type)) for a in pos]
    # A call that is only the receiver is a call of a definite arity, so it is
    # probeable: there is nothing to read and nothing to fold.  A group whose
    # shortest candidate takes nothing but the receiver is told apart from the
    # ones that take arguments, which is what makes the choice.
    if kinds and any(k is None for k in kinds):
        return None
    required = sum(1 for a in pos if a.default is None)
    return {"arity": len(pos), "required": required, "kinds": kinds}


def _arity_max(f, variant: str) -> int | None:
    """Delivered-argument ceiling for one group candidate, or None.

    The dispatcher can settle a candidate on the count of delivered
    arguments -- positionals plus keyword values, since keyword values ride
    alongside the positionals -- before running the candidate's parser.  The
    ceiling is the candidate's parameter count: schema positionals (the
    receiver too on the function surface, where it travels inside args[])
    plus keyword-only parameters.  A trailing sequence parameter has no
    ceiling: the variadic fold absorbs any number of extra positionals.

    The ceiling is a superset of what the parser accepts, never a subset: a
    candidate skipped by it would have thrown on the same count inside its
    parser, so dispatch order and error reporting are unchanged, minus one
    thrown exception per skipped candidate.
    """
    is_method = variant == "method"
    pos = [a for a in f.args
           if not (is_method and a.name == "self") and not a.kwonly]
    if pos and pos[-1].type.is_list:
        return None
    kwonly = [a for a in f.args if a.kwonly]
    return len(pos) + len(kwonly) + int(_takes_requires_grad(f, variant))


def _unique_keyword_probes(funcs, variant: str):
    """Return keyword names that identify one overload in a group."""
    names = [
        {a.name for a in f.args
         if not (variant == "method" and a.name == "self")}
        | ({"requires_grad"} if _takes_requires_grad(f, variant) else set())
        for f in funcs
    ]
    probes = []
    for index, own_names in enumerate(names):
        other_names = set().union(*(names[:index] + names[index + 1:]))
        unique = sorted(own_names - other_names)
        if unique:
            probes.append((index, unique))
    return probes


def _arg_type_smaller(t1, t2) -> bool:
    """True when ``t1`` binds a strictly narrower set of Python objects than
    ``t2`` at the same parameter position.

    A zero-dim tensor argument binds both the tensor spellings and the
    number spellings, so a group with a tensor-taking overload and a
    number-taking twin is ambiguous: the number-taking parser would pull the
    argument to the host through the Python number protocol (a device
    round trip) while the tensor-taking kernel reads it on device.  The
    tensor-taking overload therefore has to be tried first.
    """
    s1, s2 = str(t1), str(t2)
    if s1 == "Scalar" and s2 == "Tensor":
        return True
    if s1 == "Scalar?" and s2 == "Tensor?":
        return True
    # A bare whole number binds an unsized whole-number list too, so the
    # list-taking overload goes after the number-taking one whichever way
    # either is spelled; ``tensor_split(x, 3)`` asks for three sections, not
    # for a cut at index three.
    if s1 in ("int64_t[]", "SymInt[]") and s2 in ("int64_t", "int64_t?", "SymInt", "SymInt?"):
        return True
    if s1 == "Tensor[]" and s2.endswith("[]") and s2 != "Tensor[]":
        return True
    if s1 in ("int64_t", "SymInt") and s2 == "Tensor":
        return True
    return False


def _canonical_overload_order(fs, variant: str) -> list:
    """Try tensor-taking candidates before their number-taking twins.

    Part of the call sites a number-taking overload serves are also served
    by its tensor-taking sibling, and which one runs decides whether the
    call stays on device.  This builds the same partial order the schema
    conventions assume (tensor > number at a shared position, tensor lists
    after other lists) and walks a topological sort of it, keeping schema
    order for pairs the order does not relate.
    """
    group_args = [
        [a for a in f.args
         if a.name not in f.out_args
         and not (variant == "method" and a.name == "self")]
        for f in fs
    ]

    def smaller(i1: int, i2: int) -> bool:
        a1, a2 = group_args[i1], group_args[i2]
        if len(a1) != len(a2):
            return False
        equal = all(x.type == y.type for x, y in zip(a1, a2))
        dominated = all(
            x.type == y.type or _arg_type_smaller(x.type, y.type)
            for x, y in zip(a1, a2))
        return dominated and not equal

    n = len(fs)
    larger_than: dict[int, set[int]] = {
        i: {j for j in range(n) if smaller(i, j)} for i in range(n)
    }
    larger_than = {i: rest for i, rest in larger_than.items() if rest}
    if not larger_than:
        return list(fs)
    sorted_ids = [i for i in range(n) if i not in larger_than]
    for _ in range(n):
        if len(sorted_ids) == n:
            break
        for j in sorted(larger_than.keys()):
            larger_than[j].difference_update(sorted_ids)
            if not larger_than[j]:
                del larger_than[j]
                sorted_ids.append(j)
    if len(sorted_ids) != n:
        return list(fs)
    return [fs[i] for i in sorted_ids]


def _trailing_tensorlist(f, variant: str) -> bool:
    """True when the overload's last positional parameter is a tensor list.

    Uses the slot bookkeeping in :func:`_emit_op` so the group-level answer
    and the per-overload fold agree on which parameter the fold would target.
    """
    self_idx = next((i for i, a in enumerate(f.args) if a.name == "self"), None)
    pos = [a for i, a in enumerate(f.args) if i != self_idx and not a.kwonly]
    if not pos or not pos[-1].type.is_list:
        return False
    return "tensorlist" in _BRIDGE.get(cpp_arg_type(pos[-1].type), "")


def _is_variadic_shape_list(f, variant: str) -> bool:
    """Whether a trailing integer shape list accepts positional expansion."""
    self_idx = next((i for i, a in enumerate(f.args) if a.name == "self"), None)
    pos = [a for i, a in enumerate(f.args)
           if i != self_idx and not a.kwonly]
    if not pos:
        return False
    shape = pos[-1].type
    if (not shape.is_list or shape.list_size is not None or
            shape.kind != "int64_t" or shape.list_elem_opt):
        return False
    if len(pos) == 1:
        return True
    return (variant == "function" and len(pos) == 2 and
            pos[0].name == "self" and pos[0].type.is_tensor_like)


def _emit_op(out: list[str], f, variant: str, fn: str,
             own_catch: bool = True, dispatch: bool = True,
             helper_tag: str | None = None,
             splat_singleton: bool = True,
             autograd_ops: set[str] | None = None) -> None:
    """Emit one native overload entry point under ``fn``.

    own_catch=False (multi-overload group members) leaves argument errors
    uncaught so the group dispatcher can fall through to the next candidate.
    splat_singleton=False suppresses the lone-positional fold into a trailing
    tensor list, which a group only permits when no sibling overload could
    have claimed that argument as a plain tensor.
    """
    prelude: list[str] = []
    slots: list[tuple[str, str, str | None]] = []  # (argname, template, dflt)
    for i, a in enumerate(f.args):
        tpl = _require_bridge(cpp_arg_type(a.type))
        dft = py_default_for(f, a, 'binding') or (
            binding_default(a.type, a.default) if a.default is not None else None)
        dflt = None
        if a.default is not None or dft is not None:
            expr = dft if dft is not None else binding_default(a.type, a.default)
            dflt = _default_pyobject(a, expr)
            if dflt == expr and a.type.is_list:
                inner = expr[expr.find("{") + 1:expr.rfind("}")]
                items = [s.strip() for s in inner.split(",") if s.strip()]
                tag = helper_tag or _schema_tag(f, variant, 0)
                helper = f"pydflt_{f.cpp_name}_{variant}_{tag}_{i}"
                prelude.append(f"static PyObject* {helper}() {{")
                prelude.append(f"    PyObject* v = PyList_New({len(items)});")
                for j, item in enumerate(items):
                    prelude.append(
                        f"    PyList_SET_ITEM(v, {j}, PyLong_FromLongLong({item}LL));")
                prelude.extend(["    return v;", "}", ""])
                dflt = f"{helper}()"
        slots.append((a.name, tpl, dflt))
    # The schema arguments the operator is called with; a factory's
    # requires_grad flag rides behind them as one more keyword.
    call_slots = list(slots)
    marks_result = _takes_requires_grad(f, variant)
    if marks_result:
        slots.append(("requires_grad", _BRIDGE["bool"], "Py_False"))

    if f.cpp_return_kind not in _RET_SHAPES:
        raise SystemExit(
            f"native CPython bridge has no return shape for {f.cpp_return_kind!r} "
            f"in {f.func_name} ({variant})")

    nargs = len(slots)
    # The method surface binds the receiver to the schema argument named
    # "self", wherever it sits in the signature (leading `self` is the common
    # case; where.self carries it mid-signature).
    self_idx = next(
        (i for i, (n, _, _) in enumerate(slots) if n == "self"), None)
    is_method = variant == "method" and self_idx is not None
    # Schema names that collide with Python keywords ("from") map to their
    # trailing-underscore spelling everywhere a Python caller can spell them;
    # C++ locals keyed on slots stay on the raw schema name.
    def _py_kw(name: str) -> str:
        return "from_" if name == "from" else name

    if is_method:
        # METH_FASTCALL method descriptors pass the receiver as the first C
        # parameter; args[] holds only the user arguments.  The schema's
        # `self` slot therefore never appears in kwlist.
        kw_names = [_py_kw(n) for i, (n, _, _) in enumerate(slots)
                    if i != self_idx]
        user_pos = sum(1 for i, a in enumerate(f.args)
                       if i != self_idx and not a.kwonly)
    else:
        kw_names = [_py_kw(n) for n, _, _ in slots]
        user_pos = sum(1 for a in f.args if not a.kwonly)
    kwlist = ('static const char* kwlist[] = {'
              + ", ".join(f'"{n}"' for n in kw_names)
              + ', nullptr};') if kw_names else \
             'static const char* kwlist[] = {nullptr};'

    call = ", ".join("s_" + n for n, _, _ in call_slots)
    # A hand-registered operation without a derivative is generated as an inline
    # wrapper that cannot be called through a function pointer, so the call is
    # made on the tensor instead.  Once the operation carries a derivative
    # (select.int is the case today), its wrapper is emitted out-of-line and
    # reaches the same autograd-aware tpx::ops symbol the dispatched path
    # uses; the raw member call would bypass view/gradient bookkeeping.
    has_autograd = autograd_ops is not None and f.func_name in autograd_ops
    use_member_entry = (
        (f.func_name in _VMAP_MEMBER_OPS
         or (f.manual_kernel_registration and not has_autograd))
        and f.args
        and f.args[0].name == "self"
    )
    if use_member_entry:
        method_call = ", ".join("s_" + n for n, _, _ in call_slots[1:])
        # An overload name says which reading of the operation this is; the
        # member it stands for is named without it.
        member = f.base_name if f.overload_name else f.cpp_name
        invoke_expr = f"s_self.{member}({method_call})"
    elif f.func_name in _VMAP_STATIC_OPS:
        invoke_expr = f"Tensor::{f.cpp_name}({call})"
    else:
        op_signature = (f"{cpp_return_type(f)} (*)({', '.join(cpp_arg_type(a.type) for a in f.args)})")
        op = f"static_cast<{op_signature}>(tensorplay::tpx::ops::{f.cpp_name})"
        invoke_expr = f"{op}({call})"
    kind = f.cpp_return_kind
    ret_cpp = cpp_return_type(f)
    # Python call-site capture for the profiler (with_stack): runs under the
    # GIL at binding entry, before the GIL-releasing invoke.  The helper
    # itself re-checks the capture flags, so inactive cost is one load.
    site_hook = "tensorplay::python::tpx_prof_capture_site();\n        "
    # Wrap every dispatch in an unconditional GIL release so kernels
    # run multithreaded.  The lambda restores the GIL before the result is
    # wrapped (all Python C-API stays under the GIL).
    if kind == "void":
        invoke = site_hook + f"[&]() {{ tpx_py_GilRelease _gil; {invoke_expr}; }}(); Py_RETURN_NONE;"
    elif kind == "value":
        if ret_cpp == "bool":
            invoke = (site_hook + f"auto r = [&]() {{ tpx_py_GilRelease _gil; return {invoke_expr}; }}(); "
                      "return PyBool_FromLong(r);")
        elif ret_cpp == "Scalar":
            invoke = (site_hook + f"auto r = [&]() {{ tpx_py_GilRelease _gil; return {invoke_expr}; }}(); "
                      "return tpx_py_wrap_scalar(r);")
        elif ret_cpp == "int64_t":
            invoke = (site_hook + f"auto r = [&]() {{ tpx_py_GilRelease _gil; return {invoke_expr}; }}(); "
                      "return PyLong_FromLongLong(r);")
        elif ret_cpp == "Tensor":
            mark = ("if (s_requires_grad) "
                    "tensorplay::tpx::impl::set_requires_grad(r, true); "
                    if marks_result else "")
            invoke = (site_hook + f"auto r = [&]() {{ tpx_py_GilRelease _gil; return {invoke_expr}; }}(); "
                      f"{mark}return tpx_py_wrap(r);")
        else:
            pack = _pack_expr(ret_cpp, "r")
            if pack is None:
                raise SystemExit(
                    f"native CPython bridge has no packer for C++ type {ret_cpp!r} "
                    f"in {f.func_name} ({variant})")
            invoke = (site_hook + f"auto r = [&]() {{ tpx_py_GilRelease _gil; return {invoke_expr}; }}(); "
                      f"return {pack};")
    elif kind == "tuple":
        invoke = _tuple_invoke(f, invoke_expr, site_hook)
        if invoke is None:
            raise SystemExit(
                f"native CPython bridge has no tuple packer for {f.func_name} "
                f"({variant})")
    elif kind == "list":
        pack = _pack_expr(ret_cpp, "r")
        if pack is None:
            raise SystemExit(
                f"native CPython bridge has no packer for C++ type {ret_cpp!r} "
                f"in {f.func_name} ({variant})")
        invoke = (site_hook + f"auto r = [&]() {{ tpx_py_GilRelease _gil; return {invoke_expr}; }}(); "
                  f"return {pack};")
    else:                                      # mut_ref
        # slots[0] is the raw self PyObject; the s_* locals hold unpacked
        # C++ tensors.
        keep = "tpx_py_keep_alive(slots[0]);" if nargs else ""
        invoke = (site_hook + f"auto& r = [&]() -> auto& {{ tpx_py_GilRelease _gil; "
                  f"return {invoke_expr}; }}(); {keep} return tpx_py_wrap(r);")

    recv = "PyObject* self" if is_method else "PyObject*"
    out.extend(prelude)
    # Python call-site capture for the profiler (with_stack): runs under the
    # GIL at binding entry, before the GIL-releasing invoke.  The helper
    # itself re-checks the capture flags, so inactive cost is one load.
    site_hook = "tensorplay::python::tpx_prof_capture_site();\n        "
    body = [
        f"static PyObject* {fn}({recv}, PyObject* const* args,",
        f"{' ' * len(fn)}                        Py_ssize_t nargs, PyObject* kwnames) {{",
    ]
    if own_catch:
        body.append("    try {")
    body += [f"        {kwlist}"]
    # MSVC rejects zero-length stack arrays; a no-argument op skips the slot
    # array entirely and hands nullptr to the parser, which touches no slots.
    if nargs:
        body.append(f"        PyObject* slots[{nargs}];")
    if dispatch:
        # The three layers are asked only when there is somewhere the call
        # could be sent.  Asking them otherwise is three calls that can only
        # answer "nothing here", and that is nearly every call.
        recv = "self" if is_method else "nullptr"
        meth = "true" if is_method else "false"
        name = f.cpp_name
        body += [
            f"        if (tpx_py_hooks_active({recv}, args, nargs, kwnames)) {{",
            "        PyObject* tpx_dispatch_result = nullptr;",
            f'        const int tpx_mode_status = '
            f'tpx_py_try_function_mode_dispatch("{name}", '
            f" {recv}, {meth}, args, nargs, kwnames, "
            "&tpx_dispatch_result);",
            "        if (tpx_mode_status != 0) return tpx_dispatch_result;",
            f'        const int tpx_function_status = '
            f'tpx_py_try_tensor_function_dispatch("{name}", '
            f" {recv}, {meth}, args, nargs, kwnames, "
            "&tpx_dispatch_result);",
            "        if (tpx_function_status != 0) return tpx_dispatch_result;",
            f'        const int tpx_dispatch_status = '
            f'tpx_py_try_tensor_subclass_dispatch("{name}", '
            f"{recv}, {meth}, args, nargs, kwnames, "
            "&tpx_dispatch_result);",
            "        if (tpx_dispatch_status != 0) return tpx_dispatch_result;",
            "        }",
        ]

    # Fold surplus positionals into a trailing IntList parameter
    # (t.view(2, 3) == t.view([2, 3])).  This keeps the public call form:
    # when the last positional parameter is list-typed, pack args[P-1..]
    # into a tuple before parsing instead of rejecting extra positionals.
    _pos = [a for i, a in enumerate(f.args)
            if i != self_idx and not a.kwonly]
    splat = _is_variadic_shape_list(f, variant)

    if splat:
        P = user_pos
        # The parser reads keyword values from the same array it reads
        # positionals from -- at an + i once buf replaces args -- so the
        # buffer must have room for the keyword values behind the folded
        # positionals.  At most one value per kwlist slot is ever read.
        KW_MAX = len(kw_names)
        body += [
            f"        PyObject* buf[{P + KW_MAX}];",
            # Seed every slot from args up front: the fold branches below may
            # rewrite only the tail slots, and ap=buf must never expose an
            # uninitialized stack value to tpx_py_parse_into.  The call may
            # deliver fewer positionals than P (then it cannot fold and buf
            # goes unread), so the seed stops at what was delivered.
            f"        Py_ssize_t tpx_seed = nargs < {P} ? nargs : (Py_ssize_t){P};",
            "        for (Py_ssize_t i = 0; i < tpx_seed; ++i) buf[i] = args[i];",
            "        PyObject* const* ap = args;",
            "        Py_ssize_t an = nargs;",
            f"        if (nargs > {P}) {{",
            f"            PyObject* folded = PyTuple_New(nargs - {P - 1});",
            f"            for (Py_ssize_t i = 0; i < nargs - {P - 1}; ++i) {{",
            f"                PyObject* it = args[{P - 1} + i];",
            "                Py_INCREF(it);",
            "                PyTuple_SET_ITEM(folded, i, it);",
            "            }",
        ]
        if P > 1:
            body.append(
                f"            for (Py_ssize_t i = 0; i < {P - 1}; ++i) buf[i] = args[i];")
        body += [
            f"            buf[{P - 1}] = folded;",
            "            ap = buf;",
            f"            an = {P};",
            "        }",
        ]
        arg_arr, arg_n = "ap", "an"
        # A bare Tensor passed to a TensorList splat folds to a singleton.
        # In a multi-overload group the fold is only safe when every sibling
        # takes a tensor list in the same slot: otherwise it would misroute a
        # broadcastable Tensor into the list overload (raising a length
        # mismatch instead of falling through to the Tensor overload), and
        # group members rely on argument-shape errors to reach later
        # candidates.
        if (splat_singleton and
                "tensorlist" in _BRIDGE.get(cpp_arg_type(_pos[-1].type), "")):
            body += [
                "        if (an == " + str(P) + " && ap[" + str(P - 1) + "] != nullptr &&",
                "            !PyList_Check(ap[" + str(P - 1) + "]) &&",
                "            !PyTuple_Check(ap[" + str(P - 1) + "])) {",
                "            PyObject* single = PyTuple_New(1);",
                "            Py_INCREF(ap[" + str(P - 1) + "]);",
                "            PyTuple_SET_ITEM(single, 0, ap[" + str(P - 1) + "]);",
                "            buf[" + str(P - 1) + "] = single;",
                "            ap = buf;",
                "        }",
            ]
        if KW_MAX:
            body += [
                # Both rewrites above hand buf to the parser; keyword values
                # still sit at the tail of the original args array, so they
                # are carried behind the folded positionals here.  More
                # keywords than kwlist slots cannot be read by the parser
                # (an unknown name is rejected before its value is read),
                # which bounds the copy.
                "        if (ap != args && kwnames != nullptr) {",
                "            Py_ssize_t tpx_nkw = PyTuple_GET_SIZE(kwnames);",
                f"            if (tpx_nkw > {KW_MAX}) tpx_nkw = {KW_MAX};",
                "            for (Py_ssize_t i = 0; i < tpx_nkw; ++i) {",
                "                buf[an + i] = args[nargs + i];",
                "            }",
                "        }",
            ]
    else:
        arg_arr, arg_n = "args", "nargs"

    if is_method:
        # parse_into owns the fill of a contiguous user-slot array; the
        # receiver's named `self` slot is patched in afterwards.  A method
        # taking only `self` has no user slots: skip the zero-length array
        # (MSVC forbids it) and let the parser run with a null sink.
        #
        # Argument errors quote the Python-visible base name: the overload
        # suffix in the schema name is an internal dispatch identity users
        # never spell, and every entry point registers under the base name.
        user_idx = [i for i in range(nargs) if i != self_idx]
        if user_idx:
            body.append(f"        PyObject* uslots[{nargs - 1}];")
            body.append(
                f'        tpx_py_parse_into({arg_arr}, {arg_n}, kwnames, kwlist, '
                f'{nargs - 1}, "{f.base_name}", uslots);')
        else:
            body.append(
                f'        tpx_py_parse_into({arg_arr}, {arg_n}, kwnames, kwlist, '
                f'0, "{f.base_name}", nullptr);')
        for u, i in enumerate(user_idx):
            body.append(f"        slots[{i}] = uslots[{u}];")
        body.append(f"        slots[{self_idx}] = self;")
        if not splat and user_pos < nargs - 1:
            # std::invalid_argument (not a Python error) so multi-overload
            # dispatch can fall through to the next candidate signature.
            body.append(f"        if (nargs > {user_pos}) {{")
            body.append(f'            throw std::invalid_argument("{f.base_name}: '
                        'too many positional arguments");')
            body.append("        }")
    else:
        sink = "slots" if nargs else "nullptr"
        body.append(
            f'        tpx_py_parse_into({arg_arr}, {arg_n}, kwnames, kwlist, '
            f'{nargs}, "{f.base_name}", {sink});')
        if not splat and user_pos < nargs:
            body.append(f"        if (nargs > {user_pos}) {{")
            body.append(f'            throw std::invalid_argument("{f.base_name}: '
                        'too many positional arguments");')
            body.append("        }")
    out.extend(body)

    # Eager type validation with one static kind table per overload.  For the
    # method surface the table covers the user-argument array (uslots, self
    # excluded); for function overloads the schema's own `self` argument is a
    # real slot, so it must stay in the table to match the checked slot count.
    kind_consts = [_KIND_CONST.get(cpp_arg_type(a.type))
                   for i, a in enumerate(f.args)
                   if not (is_method and i == self_idx)]
    if marks_result:
        kind_consts.append(_KIND_CONST["bool"])
    if any(kind is None for kind in kind_consts):
        missing = next(
            cpp_arg_type(a.type) for i, a in enumerate(f.args)
            if not (is_method and i == self_idx)
            and _KIND_CONST.get(cpp_arg_type(a.type)) is None)
        raise SystemExit(
            f"native CPython bridge has no type check for C++ type {missing!r} "
            f"in {f.func_name} ({variant})")
    if nargs > 1:
        out.append('        static const unsigned char tpx_kinds[] = {'
                   + ", ".join(kind_consts) + '};')
        # The kind table follows the checked array: uslots holds only user
        # arguments (method surface), slots holds every schema argument
        # including self (function surface).
        check_arr = "uslots" if is_method else "slots"
        check_n = nargs - 1 if is_method else nargs
        out.append(
            f'        tpx_py_check_types({check_arr}, {check_n}, '
            f'"{f.base_name}", kwlist, tpx_kinds, {user_pos});')

    first_default = 0
    splat_slot = -1
    if splat:
        splat_name = _pos[-1].name
        splat_slot = next(i for i, (n, _, _) in enumerate(slots) if n == splat_name)
    for i, (name, tpl, dflt) in enumerate(slots):
        src = "slots[%d]" % i
        if i == self_idx:
            out.append(f"        PyObject* r_{i} = {src};")
            out.append(f"        (void)r_{i};")
        elif dflt is not None:
            # Cached default object substitutes a missing slot; without this,
            # omitted kwargs would hand nullptr straight to the unpackers.
            out.append(f"        static PyObject* k{i} = {dflt}; (void)k{i};")
            out.append(
                f"        PyObject* r_{i} = {src} ? {src} : k{i};")
        elif i == splat_slot:
            # trailing list instead of a missing required argument.
            out.append(f"        PyObject* r_{i} = {src} ? {src} : PyTuple_New(0);")
        else:
            # Required argument: a missing slot must raise (invalid_argument
            # reads as TypeError and lets multi-overload groups fall through),
            # never flow into the unpackers -- they would deref null.
            out.append(f"        if ({src} == nullptr) {{")
            out.append(
                f'            throw std::invalid_argument("{f.base_name}: '
                f'missing required argument \\"{name}\\"");')
            out.append("        }")
            out.append(f"        PyObject* r_{i} = {src};")
        out.append(f"        auto&& s_{name} = {tpl.format(n=f'r_{i}')};")
    out.append(f"        {invoke}")
    if own_catch:
        out.extend([
            "    } catch (const std::exception& e) {",
            "        tpx_py_set_error(e);",
            "        return nullptr;",
            "    }",
        ])
    out.extend([
        "}",
        "",
    ])
    return None


@register_generator("PythonCAPI")
def _gen_python_capi(ctx: CodegenContext) -> None:
    autograd_ops = set(ctx.derivatives) if ctx.derivatives else set()
    out: list[str] = [
        "// Generated by tools/codegen/gen_python_c.py -- DO NOT EDIT.",
        "#pragma once",
        "",
        "#include <Python.h>",
        "#include <stdexcept>",
        '#include "CPythonBridge.h"',
        '#include "tensorplay/ops/TPXOpsGenerated.h"',
        "namespace tensorplay { namespace python { "
        "void tpx_prof_capture_site(); } }  // profiler with_stack hook",
        "namespace tensorplay { namespace python_c {",
        "",
    ]
    fn_table: list[str] = []
    meth_table: list[str] = []
    # descriptors), not methods -- their zero-arg method wrappers double as
    # property getters.  T/H are hand-written pybind properties; the
    # transposed-view family (mT, mH) reuses the same shape through the
    # generated getter shim.  adjoint and matrix_H stay methods: their
    # documented and functional surfaces call them with parentheses.
    property_methods = {"real", "imag", "mT", "mH"}
    prop_table: list[str] = []
    claimed = plan_groups(ctx.funcs)
    for (variant, cname), fs in sorted(claimed.items()):
        fs = _canonical_overload_order(fs, variant)
        base = f"pyop_{cname}_{variant}"
        multi = len(fs) > 1
        # A lone positional may fold into the trailing tensor list only when
        # no overload in the group would rather have read it as a tensor.
        fold_singleton = all(_trailing_tensorlist(f, variant) for f in fs)
        docs: list[str] = []
        ovfns: list[str] = []
        for k, f in enumerate(fs):
            ovfn = f"{base}_ov{k}" if multi else base
            _emit_op(out, f, variant, ovfn, own_catch=not multi,
                     dispatch=not multi,
                     helper_tag=_schema_tag(f, variant, k),
                     splat_singleton=fold_singleton,
                     autograd_ops=autograd_ops)
            docs.append(f.schema.replace("\\", "\\\\").replace('"', '\\"'))
            ovfns.append(ovfn)

        # Multi-overload names dispatch by trying candidates in declaration
        # order; only argument-shape mismatches (std::invalid_argument from
        # parse/unpack) fall through -- kernel failures convert immediately,
        # used by the argument parser.
        if multi:
            doc = _schema_fallback_doc(docs)
            probes = [_probe_info(f, variant) for f in fs]
            out.append(
                f"static PyObject* {base}(PyObject* self, PyObject* const* args,"
                " Py_ssize_t nargs, PyObject* kwnames) {")
            dispatch_self = "self" if variant == "method" else "nullptr"
            dispatch_method = "true" if variant == "method" else "false"
            out += [
                "    try {",
                # Asked only when there is somewhere the call could go; see the
                # single-overload entry for why.
                f"        if (tpx_py_hooks_active({dispatch_self}, args, nargs, kwnames)) {{",
                "        PyObject* tpx_dispatch_result = nullptr;",
                f'        const int tpx_mode_status = '
                f'tpx_py_try_function_mode_dispatch("{fs[0].cpp_name}", '
                f" {dispatch_self}, {dispatch_method}, args, nargs, kwnames, "
                "&tpx_dispatch_result);",
                "        if (tpx_mode_status != 0) return tpx_dispatch_result;",
                f'        const int tpx_function_status = '
                f'tpx_py_try_tensor_function_dispatch("{fs[0].cpp_name}", '
                f" {dispatch_self}, {dispatch_method}, args, nargs, kwnames, "
                "&tpx_dispatch_result);",
                "        if (tpx_function_status != 0) return tpx_dispatch_result;",
                f'        const int tpx_dispatch_status = '
                f'tpx_py_try_tensor_subclass_dispatch("{fs[0].cpp_name}", '
                f"{dispatch_self}, {dispatch_method}, args, nargs, kwnames, "
                "&tpx_dispatch_result);",
                "        if (tpx_dispatch_status != 0) return tpx_dispatch_result;",
                "        }",
            ]
            # Kind-probe fast path: for positional-only calls, pick the single
            # compatible overload by argument kind instead of throwing on each
            # mismatched candidate (mul_(1.0) etc.).  Enabled only when every
            # candidate is probeable; a deeper mismatch in the chosen overload
            # (std::invalid_argument) still falls through to full dispatch.
            unique_keyword_probes = _unique_keyword_probes(fs, variant)
            if unique_keyword_probes:
                out.append(
                    "    if (kwnames != nullptr && "
                    "PyTuple_GET_SIZE(kwnames) != 0) {")
                for k, names in unique_keyword_probes:
                    checks = " || ".join(
                        f'tpx_py_kwnames_has(kwnames, "{name}")'
                        for name in names
                    )
                    out.extend([
                        f"        if ({checks}) {{",
                        "            try {",
                        f"                return {ovfns[k]}"
                        "(self, args, nargs, kwnames);",
                        "            } catch (const std::invalid_argument&) {",
                        "                // Continue with ordinary overload checks.",
                        "            } catch (const std::exception& e) {",
                        "                tpx_py_set_error(e);",
                        "                return nullptr;",
                        "            }",
                        "        }",
                    ])
                out.append("    }")
            if all(p is not None for p in probes):
                # The kind probes read the shape of the call -- how many
                # arguments there are and what kind each is -- so they settle
                # the candidate only while the keywords agree with that reading,
                # which is decided per candidate, since each candidate has its
                # own parameter list.  The bounds are on the shape rather than on
                # nargs alone, so a call that spells its arguments out by name
                # lands here too: counting only the positionals would reject it,
                # and sending every keyword call to the candidate-by-candidate
                # path below costs a thrown exception per rejected candidate.
                out.append("    {")
                is_method = variant == "method"
                user_args = [
                    [a for a in f.args
                     if not (is_method and a.name == "self")]
                    for f in fs
                ]
                for k, args_k in enumerate(user_args):
                    names = ", ".join(f'"{a.name}"' for a in args_k)
                    # Receiver excluded, matching the kwlist each candidate's
                    # own parser is handed, so a probe slot and a parser slot
                    # name the same parameter.
                    if names:
                        out.append(
                            f'        static const char* tpx_kwlist_{k}[] = '
                            f"{{{names}, nullptr}};")
                    else:
                        out.append(
                            f'        static const char* tpx_kwlist_{k}[] = '
                            "{{nullptr}};")
                out.append("        int pick = -1;")
                out.append("        int matches = 0;")
                for k, p in enumerate(probes):
                    kinds_c = ", ".join(str(kc) for kc in p["kinds"])
                    if not kinds_c:
                        kinds_c = "0"
                    out.append(
                        f'        static const unsigned char tpx_kinds_{k}[] '
                        f"= {{{kinds_c}}};")
                for k, p in enumerate(probes):
                    out.append(
                        f"        const int tpx_probe_{k} = "
                        f"tpx_py_probe_match(args, nargs, "
                        f"kwnames, tpx_kwlist_{k}, {len(user_args[k])}, "
                        f"tpx_kinds_{k}, {p['arity']});")
                for k, p in enumerate(probes):
                    conds = [f"tpx_probe_{k} >= {p['required']}",
                             f"tpx_probe_{k} <= {p['arity']}"]
                    out.append(f"        if ({' && '.join(conds)})"
                               f" {{ pick = {k}; ++matches; }}")
                out.append("        if (matches == 1) {")
                out.append("            try {")
                out.append("                switch (pick) {")
                for k, ovn in enumerate(ovfns):
                    out.append(f"                    case {k}: return {ovn}"
                               "(self, args, nargs, kwnames);")
                out.append("                }")
                out.append("            } catch (const std::invalid_argument&) {")
                out.append("                // deeper mismatch: full dispatch below")
                out.append("            } catch (const std::exception& e) {")
                out.append("                tpx_py_set_error(e);")
                out.append("                return nullptr;")
                out.append("            }")
                out.append("        }")
                out.append("    }")
            # No candidate accepted the call.  Report every candidate with
            # its Python-visible signature and the reason it rejected the
            # arguments, instead of surfacing whichever overload happened
            # to be tried last.
            #
            # Count gate: a candidate whose parameter count is fixed cannot
            # serve a call delivering more arguments than it has slots, so
            # the dispatcher settles such a candidate on the delivered count
            # instead of entering its parser and paying a thrown exception
            # per rejected candidate.  Candidates ending in a sequence
            # parameter have no ceiling (the variadic fold absorbs any
            # number of extra positionals) and are always tried.  A skipped
            # candidate still contributes its reason line, worded as its
            # parser would have worded it.
            out.append("        std::string tpx_reasons;")
            maxes = [_arity_max(f, variant) for f in fs]
            if any(m is not None for m in maxes):
                out.append("        const Py_ssize_t tpx_supplied = nargs +")
                out.append("            (kwnames == nullptr ? 0"
                           " : PyTuple_GET_SIZE(kwnames));")
            for k, ovn in enumerate(ovfns):
                sig = _py_signature(fs[k], variant).replace(
                    '\\', '\\\\').replace('"', '\\"')
                bound = maxes[k]
                if bound is None:
                    out.append("        try { return " + ovn
                               + "(self, args, nargs, kwnames); }")
                    out.append("        catch (const std::invalid_argument& e) {")
                    out.append(f'            tpx_reasons += "\\n * {sig}: ";')
                    out.append("            tpx_reasons += e.what();")
                    out.append("        }")
                    continue
                op = fs[k].base_name
                out.append(f"        if (tpx_supplied <= {bound}) {{")
                out.append("            try { return " + ovn
                           + "(self, args, nargs, kwnames); }")
                out.append("            catch (const std::invalid_argument& e) {")
                out.append(f'                tpx_reasons += "\\n * {sig}: ";')
                out.append("                tpx_reasons += e.what();")
                out.append("            }")
                out.append("        } else {")
                out.append(f'            tpx_reasons += "\\n * {sig}: ";')
                out.append(f'            tpx_reasons += (nargs > {bound})')
                out.append(f'                ? "{op}: too many positional arguments"')
                out.append(f'                : "{op}: too many arguments";')
                out.append("        }")
            receiver = "self" if variant == "method" else "nullptr"
            out.append(
                '        throw std::invalid_argument(std::string("'
                + f"{fs[0].base_name}()"
                + ' received an invalid combination of arguments - got ")'
                + " + tpx_py_args_desc(args, nargs, kwnames, " + receiver + ")"
                + ' + ", but expected one of:" + tpx_reasons);')
            out.append("    } catch (const std::exception& e) {")
            out.append("        tpx_py_set_error(e);")
            out.append("        return nullptr;")
            out.append("    }")
            out.append("}")
            out.append("")
            entry_fn = base
        else:
            entry_fn = ovfns[0]
            doc = _schema_fallback_doc(docs)
        entry_line = (
            f'    {{"{cname}", (PyCFunction)(void*){entry_fn},'
            f' METH_FASTCALL | METH_KEYWORDS, "{doc}"}},')
        if variant == "method":
            if cname in property_methods and len(fs) == 1:
                # Property getter shim over the zero-arg FASTCALL entry.
                out.append(
                    f"static PyObject* pyprop_{cname}_get(PyObject* self, void*) {{")
                out.append(f"    return {base}(self, nullptr, 0, nullptr);")
                out.append("}")
                out.append("")
                prop_table.append(
                    f'    {{"{cname}", pyprop_{cname}_get, nullptr, nullptr, nullptr}},')
            else:
                meth_table.append(entry_line)
        else:
            fn_table.append(entry_line)

    # Hook-free entry point per overload.  An operator overload object
    # calls exactly one overload: the binding-layer function modes, tensor
    # function hooks and subclass hooks already had their turn when the
    # overload object was reached, and overload selection must not re-run.
    overload_rows: list[str] = []
    seen_overloads: set[str] = set()
    for index, f in enumerate(ctx.funcs):
        if f.func_name in seen_overloads:
            continue
        seen_overloads.add(f.func_name)
        if "function" in f.variants or "method" not in f.variants:
            variant = "function"
        else:
            variant = "method"
        _op_supported(f, variant)
        fn = f"pyovl_{f.func_name.replace('.', '__')}"
        _emit_op(out, f, variant, fn, own_catch=True, dispatch=False,
                 helper_tag=f"ovl{index}", splat_singleton=False,
                 autograd_ops=autograd_ops)
        receiver = "true" if variant == "method" and any(
            a.name == "self" for a in f.args) else "false"
        overload_rows.append(
            f'    {{"{f.func_name}", (PyCFunction)(void*){fn}, {receiver}}},')

    # Not constexpr: the (PyCFunction)(void*) casts in each entry are not a
    # constant expression, so these tables stay dynamically initialized.
    out += [
        "// One hook-free entry per operator overload, keyed by overload name.",
        "// receiver=true entries take the schema `self` as the C receiver.",
        "struct GeneratedOverloadCall {",
        "    const char* name;",
        "    PyCFunction entry;",
        "    bool receiver;",
        "};",
        "inline const GeneratedOverloadCall generated_overload_calls[] = {",
        *overload_rows,
        "    {nullptr, nullptr, false},",
        "};",
        "",
        "// Module-level op functions.",
        f"inline PyMethodDef generated_functions[] = {{",
        *fn_table,
        "    {nullptr, nullptr, 0, nullptr},",
        "};",
        "",
        "// Tensor methods, installed as unbound method descriptors so the",
        "// receiver flows through METH_FASTCALL like a builtin method.",
        f"inline PyMethodDef generated_tensor_methods[] = {{",
        *meth_table,
        "    {nullptr, nullptr, 0, nullptr},",
        "};",
        "",
        "// Tensor.real / Tensor.imag surface as properties.",
        "inline PyGetSetDef generated_tensor_properties[] = {",
        *prop_table,
        "    {nullptr, nullptr, nullptr, nullptr, nullptr},",
        "};",
        "",
        "// Fill-only installation: an entry is skipped whenever its name is",
        "// already bound.  Hand-written pybind11 bindings carry semantics the",
        "// raw layer must not clobber (factory dtype/device resolution,",
        "// requires_grad marking, Union[int, int[]]-style extra overloads),",
        "// so the FASTCALL layer only serves names nothing else defined.",
        "inline int register_generated_cpython_functions(PyObject* module) {",
        "    for (auto* def = generated_functions; def->ml_name != nullptr; ++def) {",
        "        if (PyObject_HasAttrString(module, def->ml_name)) continue;",
        "        PyObject* f = PyCFunction_NewEx(def, nullptr, nullptr);",
        "        if (f == nullptr) return -1;",
        "        int rc = PyObject_SetAttrString(module, def->ml_name, f);",
        "        Py_DECREF(f);",
        "        if (rc != 0) return -1;",
        "    }",
        "    return 0;",
        "}",
        "",
        "inline int register_generated_cpython_methods(PyObject* type_obj) {",
        "    auto* type = reinterpret_cast<PyTypeObject*>(type_obj);",
        "    for (auto* def = generated_tensor_methods; def->ml_name != nullptr;",
        " ++def) {",
        # A name already holding a method descriptor is already what this
        # table would install, so it is left alone.  A name holding something
        # else is left alone too: the wrapper a reflected binding leaves behind
        # reaches the operation by a shorter route than this table does, and a
        # query that only reads a fact about the value -- how many elements it
        # has, which dimension it was asked about -- spends less time being
        # called through it than being asked for the name of the operation it
        # would answer with.  What this table owns is the names nothing else
        # defined, and it serves those the same way.
        "        PyObject* existing = PyDict_GetItemString(type->tp_dict, def->ml_name);",
        "        if (existing != nullptr) continue;",
        "        PyObject* descr = PyDescr_NewMethod(type, def);",
        "        if (descr == nullptr) return -1;",
        "        int rc = PyObject_SetAttrString(type_obj, def->ml_name, descr);",
        "        Py_DECREF(descr);",
        "        if (rc != 0) return -1;",
        "    }",
        "    for (auto* def = generated_tensor_properties; def->name != nullptr;",
        " ++def) {",
        "        PyObject* descr = PyDescr_NewGetSet(type, def);",
        "        if (descr == nullptr) return -1;",
        "        int rc = PyObject_SetAttrString(type_obj, def->name, descr);",
        "        Py_DECREF(descr);",
        "        if (rc != 0) return -1;",
        "    }",
        "    return 0;",
        "}",
        "",
        "} }  // namespace tensorplay::python_c",
        "",
    ]
    ctx.write("TensorCPythonGenerated.h", "\n".join(out))
