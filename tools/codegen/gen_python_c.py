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
    "sum", "sum.dim_IntList", "view", "permute", "transpose", "movedim",
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
    if not kinds or any(k is None for k in kinds):
        return None
    required = sum(1 for a in pos if a.default is None)
    return {"arity": len(pos), "required": required, "kinds": kinds}


def _unique_keyword_probes(funcs, variant: str):
    """Return keyword names that identify one overload in a group."""
    names = [
        {a.name for a in f.args
         if not (variant == "method" and a.name == "self")}
        for f in funcs
    ]
    probes = []
    for index, own_names in enumerate(names):
        other_names = set().union(*(names[:index] + names[index + 1:]))
        unique = sorted(own_names - other_names)
        if unique:
            probes.append((index, unique))
    return probes


def _emit_op(out: list[str], f, variant: str, fn: str,
             own_catch: bool = True, dispatch: bool = True,
             helper_tag: str | None = None) -> None:
    """Emit one native overload entry point under ``fn``.

    own_catch=False (multi-overload group members) leaves argument errors
    uncaught so the group dispatcher can fall through to the next candidate.
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

    call = ", ".join("s_" + n for n, _, _ in slots)
    use_member_entry = (
        f.func_name in _VMAP_MEMBER_OPS
        and f.args
        and f.args[0].name == "self"
    )
    if use_member_entry:
        method_call = ", ".join("s_" + n for n, _, _ in slots[1:])
        invoke_expr = f"s_self.{f.cpp_name}({method_call})"
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
            invoke = (site_hook + f"auto r = [&]() {{ tpx_py_GilRelease _gil; return {invoke_expr}; }}(); "
                      "return tpx_py_wrap(r);")
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
    body += [f"        {kwlist}", f"        PyObject* slots[{nargs}];"]
    if dispatch:
        body += [
            "        PyObject* tpx_dispatch_result = nullptr;",
            f'        const int tpx_mode_status = '
            f'tpx_py_try_function_mode_dispatch("{f.cpp_name}", '
            f' {"self" if is_method else "nullptr"}, '
            f' {"true" if is_method else "false"}, args, nargs, kwnames, '
            "&tpx_dispatch_result);",
            "        if (tpx_mode_status != 0) return tpx_dispatch_result;",
            f'        const int tpx_function_status = '
            f'tpx_py_try_tensor_function_dispatch("{f.cpp_name}", '
            f' {"self" if is_method else "nullptr"}, '
            f' {"true" if is_method else "false"}, args, nargs, kwnames, '
            "&tpx_dispatch_result);",
            "        if (tpx_function_status != 0) return tpx_dispatch_result;",
            f'        const int tpx_dispatch_status = '
            f'tpx_py_try_tensor_subclass_dispatch("{f.cpp_name}", '
            f'{"self" if is_method else "nullptr"}, '
            f'{"true" if is_method else "false"}, args, nargs, kwnames, '
            "&tpx_dispatch_result);",
            "        if (tpx_dispatch_status != 0) return tpx_dispatch_result;",
        ]

    # Fold surplus positionals into a trailing IntList parameter
    # (t.view(2, 3) == t.view([2, 3])).  This keeps the public call form:
    # when the last positional parameter is list-typed, pack args[P-1..]
    # into a tuple before parsing instead of rejecting extra positionals.
    _pos = [a for i, a in enumerate(f.args)
            if i != self_idx and not a.kwonly]
    splat = bool(_pos) and _pos[-1].type.is_list

    if splat:
        P = user_pos
        body += [
            f"        PyObject* buf[{P}];",
            # Seed every slot from args up front: the fold branches below may
            # rewrite only the tail slots, and ap=buf must never expose an
            # uninitialized stack value to tpx_py_parse_into.
            f"        for (Py_ssize_t i = 0; i < {P}; ++i) buf[i] = args[i];",
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
        # A bare Tensor passed to a TensorList splat folds to a singleton
        if "tensorlist" in _BRIDGE.get(cpp_arg_type(_pos[-1].type), ""):
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
    else:
        arg_arr, arg_n = "args", "nargs"

    if is_method:
        # parse_into owns the fill of a contiguous user-slot array; the
        # receiver's named `self` slot is patched in afterwards.
        user_idx = [i for i in range(nargs) if i != self_idx]
        body.append(f"        PyObject* uslots[{nargs - 1}];")
        body.append(
            f'        tpx_py_parse_into({arg_arr}, {arg_n}, kwnames, kwlist, '
            f'{nargs - 1}, "{f.func_name}", uslots);')
        for u, i in enumerate(user_idx):
            body.append(f"        slots[{i}] = uslots[{u}];")
        body.append(f"        slots[{self_idx}] = self;")
        if not splat and user_pos < nargs - 1:
            # std::invalid_argument (not a Python error) so multi-overload
            # dispatch can fall through to the next candidate signature.
            body.append(f"        if (nargs > {user_pos}) {{")
            body.append(f'            throw std::invalid_argument("{f.func_name}: '
                        'too many positional arguments");')
            body.append("        }")
    else:
        body.append(
            f'        tpx_py_parse_into({arg_arr}, {arg_n}, kwnames, kwlist, '
            f'{nargs}, "{f.func_name}", slots);')
        if not splat and user_pos < nargs:
            body.append(f"        if (nargs > {user_pos}) {{")
            body.append(f'            throw std::invalid_argument("{f.func_name}: '
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
            f'"{f.func_name}", kwlist, tpx_kinds, {user_pos});')

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
                f'            throw std::invalid_argument("{f.func_name}: '
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
    # property getters.
    property_methods = {"real", "imag"}
    prop_table: list[str] = []
    claimed = plan_groups(ctx.funcs)
    for (variant, cname), fs in sorted(claimed.items()):
        base = f"pyop_{cname}_{variant}"
        multi = len(fs) > 1
        docs: list[str] = []
        ovfns: list[str] = []
        for k, f in enumerate(fs):
            ovfn = f"{base}_ov{k}" if multi else base
            _emit_op(out, f, variant, ovfn, own_catch=not multi,
                     dispatch=not multi,
                     helper_tag=_schema_tag(f, variant, k))
            docs.append(f.schema.replace("\\", "\\\\").replace('"', '\\"'))
            ovfns.append(ovfn)

        # Multi-overload names dispatch by trying candidates in declaration
        # order; only argument-shape mismatches (std::invalid_argument from
        # parse/unpack) fall through -- kernel failures convert immediately,
        # used by the argument parser.
        if multi:
            doc = " | ".join(docs)
            probes = [_probe_info(f, variant) for f in fs]
            out.append(
                f"static PyObject* {base}(PyObject* self, PyObject* const* args,"
                " Py_ssize_t nargs, PyObject* kwnames) {")
            dispatch_self = "self" if variant == "method" else "nullptr"
            dispatch_method = "true" if variant == "method" else "false"
            out += [
                "    try {",
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
                out.append(
                    "    if (kwnames == nullptr || PyTuple_GET_SIZE(kwnames) == 0) {")
                out.append("        int pick = -1;")
                out.append("        int matches = 0;")
                for k, p in enumerate(probes):
                    conds = [f"nargs >= {p['required']}",
                             f"nargs <= {p['arity']}"]
                    for i, kc in enumerate(p["kinds"]):
                        conds.append(
                            f"(nargs <= {i} || "
                            f"tpx_py_obj_matches_kind(args[{i}], {kc}))")
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
            out.append("        std::exception_ptr arg_err;")
            for ovn in ovfns:
                out.append(f"        try {{ return {ovn}(self, args, nargs, kwnames); }}")
                out.append(
                    "        catch (const std::invalid_argument&) "
                    "{ arg_err = std::current_exception(); }")
            out.append("        std::rethrow_exception(arg_err);")
            out.append("    } catch (const std::exception& e) {")
            out.append("        tpx_py_set_error(e);")
            out.append("        return nullptr;")
            out.append("    }")
            out.append("}")
            out.append("")
            entry_fn = base
        else:
            entry_fn = ovfns[0]
            doc = docs[0]
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

    # Not constexpr: the (PyCFunction)(void*) casts in each entry are not a
    # constant expression, so these tables stay dynamically initialized.
    out += [
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
        "        if (PyObject_HasAttrString(type_obj, def->ml_name)) continue;",
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
