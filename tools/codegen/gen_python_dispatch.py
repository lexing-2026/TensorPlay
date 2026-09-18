"""Python dispatch key kernels.

One kernel per operator handle, registered under ``DispatchKey::Python``
with the same ABI every other kernel of that handle uses (the dispatcher
stub signature, unwrap boundary included).  A kernel converts its arguments
to Python objects, hands them to the innermost dispatch mode through
``python_dispatch::ModeCall`` and converts the mode's result back.
"""

from __future__ import annotations

from .api_types import cpp_return_type, stub_arg_type_for, _unwrap_targeted
from .model import NativeFunction, Type


class UnsupportedPythonDispatch(Exception):
    """An operator whose ABI has no Python conversion yet."""


_SCALAR_WRAP = {
    "int64_t": ("PyLong_FromLongLong({})", "python_c::tpx_py_wrap_optional_int64({})"),
    "double": ("PyFloat_FromDouble({})", "python_c::tpx_py_wrap_optional_double({})"),
    "bool": ("PyBool_FromLong({})", "python_c::tpx_py_wrap_optional_bool({})"),
    "str": ("wrap_string({})", "python_c::tpx_py_wrap_optional_string({})"),
    "Scalar": ("python_c::tpx_py_wrap_scalar({})", "python_c::tpx_py_wrap_optional_scalar({})"),
    "DType": ("python_c::tpx_py_wrap_dtype({})", "python_c::tpx_py_wrap_optional_dtype({})"),
    "Device": ("python_c::tpx_py_wrap_device({})", "python_c::tpx_py_wrap_optional_device({})"),
    "Generator": ("python_c::tpx_py_wrap_generator({})", "python_c::tpx_py_wrap_optional_generator({})"),
    "Storage": ("python_c::tpx_py_wrap_storage({})", None),
    "MemoryFormat": ("wrap_memory_format({})", "wrap_optional_memory_format({})"),
    "Layout": ("wrap_layout({})", "wrap_optional_layout({})"),
}

_LIST_WRAP = {
    # (kind, list_elem_opt, is_opt) -> converter
    ("int64_t", False, False): "python_c::tpx_py_wrap_intlist({})",
    ("int64_t", False, True): "python_c::tpx_py_wrap_optional_intlist({})",
    ("int64_t", True, False): "wrap_optional_int64_list({})",
    ("double", False, False): "python_c::tpx_py_wrap_doublelist({})",
    ("double", False, True): "python_c::tpx_py_wrap_optional_doublelist({})",
    ("bool", False, False): "python_c::tpx_py_wrap_boollist({})",
    ("bool", False, True): "wrap_optional_bool_list({})",
    ("Scalar", False, False): "python_c::tpx_py_wrap_scalarlist({})",
    ("Scalar", False, True): "wrap_optional_scalar_list({})",
    ("str", False, False): "wrap_string_list({})",
}


def _arg_to_py(op_base: str, a) -> str:
    t: Type = a.type
    name = a.name
    if _unwrap_targeted(op_base, a):
        # Optional tensor carried as an undefined Tensor across the kernel
        # ABI: an undefined value is the schema's None.
        return f"wrap_tensor_or_none({name})"
    if t.is_tensor_like:
        if t.is_list:
            if t.list_elem_opt:
                return f"python_c::tpx_py_wrap_optional_tensor_list({name})"
            if t.is_opt:
                return f"wrap_optional_tensor_list({name})"
            return f"python_c::tpx_py_wrap_list({name})"
        if t.is_opt:
            return f"python_c::tpx_py_wrap_optional_tensor({name})"
        return f"python_c::tpx_py_wrap({name})"
    if t.is_list:
        conv = _LIST_WRAP.get((t.kind, t.list_elem_opt, t.is_opt))
        if conv is None:
            raise UnsupportedPythonDispatch(f"list argument {t}")
        return conv.format(name)
    conv = _SCALAR_WRAP.get(t.kind)
    if conv is None:
        raise UnsupportedPythonDispatch(f"argument {t}")
    plain, optional = conv
    if t.is_opt:
        if optional is None:
            raise UnsupportedPythonDispatch(f"optional argument {t}")
        return optional.format(name)
    return plain.format(name)


_SCALAR_PARSE = {
    "Tensor": "python_c::tpx_py_tensor({})",
    "int64_t": "python_c::tpx_py_int64({})",
    "double": "python_c::tpx_py_double({})",
    "bool": "python_c::tpx_py_bool({})",
    "Scalar": "python_c::tpx_py_scalar({})",
    "DType": "python_c::tpx_py_dtype({})",
    "Device": "python_c::tpx_py_device({})",
    "Generator": "python_c::tpx_py_generator({})",
    "Storage": "python_c::tpx_py_storage({})",
    "str": "python_c::tpx_py_string({})",
    "SymInt": "parse_symint({})",
    "SymBool": "parse_symbool({})",
    "SymFloat": "parse_symfloat({})",
}

_OPT_PARSE = {
    "Tensor": "parse_optional_tensor({})",
    "int64_t": "python_c::tpx_py_opt_int64({})",
    "double": "python_c::tpx_py_opt_double({})",
    "bool": "python_c::tpx_py_opt_bool({})",
    "Scalar": "python_c::tpx_py_opt_scalar({})",
    "DType": "python_c::tpx_py_opt_dtype({})",
    "Device": "python_c::tpx_py_opt_device({})",
    "str": "python_c::tpx_py_opt_string({})",
}

_LIST_PARSE = {
    "Tensor": "python_c::tpx_py_tensorlist({})",
    "int64_t": "parse_int64_list({})",
    "double": "python_c::tpx_py_doublelist({})",
    "bool": "python_c::tpx_py_boollist({})",
    "Scalar": "python_c::tpx_py_scalarlist({})",
    "SymInt": "parse_symint_list({})",
}


def _value_parse(t: Type, expr: str) -> str:
    """Parse ``expr`` (a borrowed PyObject*) into the return value type."""

    kind = "Tensor" if t.is_tensor_like else (
        "SymInt" if t.symint else "SymBool" if t.symbool
        else "SymFloat" if t.symfloat else t.kind)
    if t.is_list:
        if t.is_tensor_like and t.list_elem_opt:
            conv = "parse_optional_tensor_list({})"
        else:
            conv = _LIST_PARSE.get(kind)
        if conv is None or (t.list_elem_opt and not t.is_tensor_like):
            raise UnsupportedPythonDispatch(f"list return {t}")
        if t.is_opt:
            return (f"({expr} == Py_None ? std::nullopt : "
                    f"std::make_optional({conv.format(expr)}))")
        return conv.format(expr)
    if t.is_opt:
        conv = _OPT_PARSE.get(kind)
        if conv is None:
            raise UnsupportedPythonDispatch(f"optional return {t}")
        return conv.format(expr)
    conv = _SCALAR_PARSE.get(kind)
    if conv is None:
        raise UnsupportedPythonDispatch(f"return {t}")
    return conv.format(expr)


def _symbol(func_name: str) -> str:
    return func_name.replace(".", "__")


def _cpp_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _kernel_args(f: NativeFunction):
    return [a for a in f.args if a.name != "requires_grad"]


def _emit_kernel(out: list[str], f: NativeFunction) -> str:
    sym = _symbol(f.func_name)
    args = _kernel_args(f)
    ret = cpp_return_type(f)
    params = [f"{stub_arg_type_for(f.base_name, a)} {a.name}" for a in args]
    conversions = [_arg_to_py(f.base_name, a) for a in args]
    kind = f.cpp_return_kind
    if kind == "tuple":
        parts = [
            _value_parse(r.type, f"result_item(o, {len(f.returns)}, {i}, entry_{sym}.name)")
            for i, r in enumerate(f.returns)
        ]
        ret_expr = f"{ret}({', '.join(parts)})"
    elif kind in ("void", "mut_ref"):
        ret_expr = None
    else:
        ret_expr = _value_parse(f.returns[0].type, "o")

    mutable = [a.name for a in args if a.type.is_mutable_ref]
    if kind == "mut_ref" and not mutable:
        raise UnsupportedPythonDispatch("mutable return without a mutable argument")

    names = ", ".join(_cpp_string(a.python_name) for a in args) or "nullptr"
    num_positional = sum(1 for a in args if not a.kwonly)
    out.append(f"const char* const argnames_{sym}[] = {{{names}}};")
    out.append(
        f"OpEntry entry_{sym} = {{{_cpp_string(f.func_name)}, "
        f"{_cpp_string(f.schema)}, argnames_{sym}, {len(args)}, "
        f"{num_positional}, {_cpp_string(','.join(sorted(f.tags)))}, nullptr}};")
    out.append(f"{ret} kernel_{sym}({', '.join(params)}) {{")
    out.append(f"    ModeCall call(entry_{sym});")
    for i, conv in enumerate(conversions):
        out.append(f"    call.set_arg({i}, {conv});")
    if kind == "void":
        out.append("    Py_DECREF(call.invoke());")
    elif kind == "mut_ref":
        # The mode mutates the destination in place; the kernel ABI returns
        # that argument, whatever object the handler returned.
        out.append("    Py_DECREF(call.invoke());")
        out.append(f"    return {mutable[0]};")
    else:
        out.append(
            f"    return convert_result(call.invoke(), "
            f"[&](PyObject* o) -> {ret} {{ return {ret_expr}; }});")
    out.append("}")
    out.append("")
    signature = ", ".join(stub_arg_type_for(f.base_name, a) for a in args)
    return (f'    D.registerKernel({_cpp_string(f.func_name)}, DispatchKey::Python, '
            f'(KernelFunction)static_cast<{ret} (*)({signature})>(&kernel_{sym}));')


def generate_python_dispatch_cpp(funcs: list[NativeFunction]) -> tuple[str, list[str]]:
    """Return the generated translation unit and the skipped handles."""

    out: list[str] = [
        "// Generated by tools/codegen/main.py -- DO NOT EDIT",
        '#include "PythonDispatch.h"',
        '#include "Dispatcher.h"',
        "",
        "namespace tensorplay {",
        "namespace python_dispatch {",
        "namespace {",
        "",
    ]
    registrations: list[str] = []
    entries: list[str] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for f in funcs:
        if f.func_name in seen:
            continue
        seen.add(f.func_name)
        body: list[str] = []
        try:
            registration = _emit_kernel(body, f)
        except UnsupportedPythonDispatch as exc:
            skipped.append(f"{f.func_name}: {exc}")
            continue
        out.extend(body)
        registrations.append(registration)
        entries.append(f"    &entry_{_symbol(f.func_name)},")

    out += [
        "const OpEntry* const all_entries[] = {",
        *entries,
        "};",
        "",
        "}  // namespace",
        "",
        "void register_python_kernels() {",
        "    auto& D = Dispatcher::singleton();",
        *registrations,
        "}",
        "",
        "const OpEntry* const* python_dispatch_entries(int64_t* count) {",
        "    *count = static_cast<int64_t>(sizeof(all_entries) / sizeof(all_entries[0]));",
        "    return all_entries;",
        "}",
        "",
        "}  // namespace python_dispatch",
        "}  // namespace tensorplay",
        "",
    ]
    return "\n".join(out), skipped
