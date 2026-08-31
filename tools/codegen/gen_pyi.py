"""Generation of the ``_C`` type stubs from the schema model.

The stub set follows the layout of a C extension module package:

- ``tensorplay/_C/__init__.pyi`` — the extension surface: hand-declared
  classes, the generated ``Tensor`` method set, and the module-level helper
  functions; the generated op-function surface is re-exported;
- ``tensorplay/_C/_VariableFunctions.pyi`` — the generated op functions, one
  signature per schema grouped by bound name.

Signature conventions shared by both outputs:

- a name bound to more than one schema renders one ``@overload`` candidate
  per schema;
- kwarg-only schema parameters render behind a ``*`` separator;
- the ``out`` variant of an op merges into its functional signature as an
  optional ``out`` keyword, matching the dispatcher-level calling form;
- docstrings come from the runtime documentation module so ``help()`` and the
  stub agree;
- the DType block is derived from the binding source, keeping member names in
  lockstep with the runtime enum.

Templates are rendered with block-aware substitution: a ``${name}`` hole that
starts a line indents every substituted line to the hole's depth, so class
bodies can embed multi-line signatures under a single placeholder.
"""

from __future__ import annotations

import importlib.util
import inspect
import itertools
import os
import re
import sys
import textwrap
from collections import defaultdict
from unittest.mock import Mock, patch
from warnings import warn

from .api_types import py_default_for, sanitize_name
from .model import Argument, NativeFunction, Type

# Output file / template pairs, both relative to the stub and template
# directories handed in by the caller.
_PYI_OUTPUTS: tuple[tuple[str, str], ...] = (
    ("_C/__init__.pyi", "_C/__init__.pyi.in"),
    ("_C/_VariableFunctions.pyi", "_C/_VariableFunctions.pyi.in"),
    ("functional.pyi", "functional.pyi.in"),
)

# The functional layer hand-defines wrappers for ops that only exist as
# tensor methods in the schema set, and re-signatures a few others (factory
# kwargs, reduction dim merging, dispatcher-routed families).  Their hint
# bodies live in the functional template; this set tells the generator which
# names it must not emit.
_FUNCTIONAL_HAND_NAMES = frozenset([
    "abs_", "add", "add_", "absolute_", "addcmul_", "addcdiv_", "arange",
    "bernoulli_", "cauchy_", "clone", "contiguous", "copy_",
    "div", "div_",
    "empty", "empty_like", "expand", "expand_as", "exponential_",
    "fill_", "full", "full_like", "geometric_",
    "item",
    "kaiser_window",
    "linear", "lerp_", "log_normal_", "logspace", "linspace",
    "matmul", "mm", "mul", "mul_",
    "neg_", "normal_",
    "rand", "rand_like", "randint", "randint_like", "randn", "randn_like",
    "random_", "repeat", "resize_", "rsqrt_",
    "select", "slice", "sqrt_", "squeeze_copy", "sub", "sub_",
    "trapz",
    "unique_consecutive", "uniform_",
    "view",
    "where",
    "zeros", "zeros_like", "ones", "ones_like", "zero_",
])

# ---------------------------------------------------------------------------
# Python typing (.pyi)
# ---------------------------------------------------------------------------

# Canonical atom spellings.  Composite aliases (_size, Device, Number, ...)
# live in tensorplay.types and absorb their optional forms.
_PYI_ATOMS = {
    "Tensor": "TensorBase",
    "int64_t": "_int",
    "double": "_float",
    "bool": "_bool",
    "str": "str",
    "Scalar": "Number | _complex",
    "DType": "_dtype",
    "Device": "Device",
    "MemoryFormat": "MemoryFormat",
    "Layout": "int",
    "Generator": "Generator",
    "SymInt": "SymInt",
    "SymBool": "SymBool",
    "SymFloat": "SymFloat",
}

# Return types prefer the concrete spelling: argument unions widen the input
# surface, while a returned Device is always the runtime class.
_RETURN_DEVICE = "_device"

_INT_DEFAULT_RE = re.compile(r"^[+-]?\d+$")


def _append_optional(type_str: str) -> str:
    # Append `| None`, collapsing a doubled clause if the type already carries
    # one (Device and MemoryFormat aliases already include None).
    return f"{type_str} | None".replace(" | None | None", " | None")


def pyi_type(t: Type, *, list_default: str | None = None) -> str:
    if t.is_list:
        if t.kind == "Tensor":
            ret = "tuple[TensorBase, ...] | list[TensorBase]"
        elif t.kind == "int64_t":
            # A scalar default means the bridge accepts the bare int, so the
            # hint widens to the sized spelling.
            ret = "_int | _size" if (list_default and _INT_DEFAULT_RE.match(list_default)) else "_size"
        elif t.kind == "double":
            ret = "Sequence[_float]"
        else:
            ret = f"Sequence[{_PYI_ATOMS.get(t.kind, t.kind)}]"
    else:
        ret = _PYI_ATOMS.get(t.kind, t.kind)
    if t.is_opt:
        ret = _append_optional(ret)
    return ret


def _return_elem(t: Type) -> str:
    if t.kind == "Device":
        return _RETURN_DEVICE
    if t.symint:
        return "SymInt"
    if t.symbool:
        return "SymBool"
    if t.symfloat:
        return "SymFloat"
    return _PYI_ATOMS.get(t.kind, t.kind)


def pyi_return_type(f: NativeFunction) -> str:
    kind = f.cpp_return_kind
    if kind == "void":
        return "None"
    if kind == "mut_ref":
        return "TensorBase"
    if kind == "list":
        return f"tuple[{_return_elem(f.returns[0].type)}, ...]"
    if kind == "tuple":
        parts = []
        for r in f.returns:
            if r.type.is_list:
                parts.append(f"tuple[{_return_elem(r.type)}, ...]")
            else:
                parts.append(_return_elem(r.type))
        return f"tuple[{', '.join(parts)}]"
    return _return_elem(f.returns[0].type)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

_PYI_DEFAULTS = {
    "None": "None",
    "nullptr": "None",
    "{}": "None",
    "true": "True",
    "false": "False",
    "Contiguous": "contiguous_format",
    "Preserve": "preserve_format",
    "ChannelsLast": "channels_last",
    "ChannelsLast3d": "channels_last_3d",
    "c10::MemoryFormat::Contiguous": "contiguous_format",
    "c10::MemoryFormat::Preserve": "preserve_format",
    "Float32": "DType.float32",
    "Int64": "DType.int64",
    "Undefined": "DType.undefined",
    "CPU": "...",
}


def pyi_default(t: Type, default: str) -> str:
    d = _PYI_DEFAULTS.get(default, default)
    if t.is_list and d.startswith(("{", "[")):
        inner = ", ".join(x.strip() for x in d[1:-1].split(","))
        return "(" + inner + ")"
    return d


# ---------------------------------------------------------------------------
# Signature rendering
# ---------------------------------------------------------------------------

def _format_signature(name: str, formals: list[str], ret: str) -> str:
    sig = f"def {name}({', '.join(formals)}) -> {ret}: ..."
    if len(sig) <= 80 or not formals or formals == ["self"]:
        return sig
    lines = [
        f"def {name}(",
        *(f"    {arg}," for arg in formals),
        f") -> {ret}: ...",
    ]
    sig = "\n".join(lines)
    if all(len(line) <= 80 for line in lines):
        return sig
    # Formatters need the trailing compound statement on its own line; the
    # skip marker keeps auto-formatters from re-flowing the body.
    return sig.removesuffix(" ...") + "  # fmt: skip\n    ..."


def _foreach_return_fix(f: NativeFunction) -> str | None:
    # In-place foreach ops mutate the input list in place and hand it back;
    # the hint keeps the container spellings so either view type-checks.
    if (f.base_name.startswith("_foreach_") and f.base_name.endswith("_")
            and f.cpp_return_kind == "list"):
        return "tuple[TensorBase, ...] | list[TensorBase]"
    return None


def _out_arg(f: NativeFunction) -> Argument | None:
    for a in f.args:
        if a.name == "out" and a.type.is_mutable_ref:
            return a
    return None


def _args_equal(a: Argument, b: Argument) -> bool:
    return (a.name == b.name and a.type == b.type and a.kwonly == b.kwonly
            and a.default == b.default)


def _signature(f: NativeFunction, *, method: bool,
               merged_out: Type | None = None,
               functional: bool = False) -> str:
    formals: list[str] = []
    args = list(f.args)
    start = 0
    if method:
        # The bound instance plays the first schema slot whatever its name;
        # a later stray ``self`` slot renames to the wrapper-level spelling
        # so the formal list stays legal Python.
        formals.append("self")
        start = 1
    positional = 0
    for a in args[start:]:
        if merged_out is not None and a.name == "out" and a.type.is_mutable_ref:
            continue  # rendered as the optional keyword below
        name = sanitize_name(a.name)
        if name == "self":
            if functional:
                name = "input"  # the functional wrapper layer renames it
            elif method:
                # stray self slot behind the bound instance
                name = "input"
            # function-variant bindings keep the schema spelling
        s = f"{name}: {pyi_type(a.type, list_default=a.default)}"
        dft = py_default_for(f, a, "pyi")
        if dft is None and a.default is not None:
            dft = pyi_default(a.type, a.default)
        if dft is not None:
            s += f" = {dft}"
        formals.append(s)
        if not a.kwonly:
            positional += 1
    if merged_out is not None:
        formals.append(f"out: {_append_optional(pyi_type(merged_out))} = None")
    lead = 1 if method else 0
    if len(formals) > positional + lead:
        formals.insert(positional + lead, "*")
    ret = _foreach_return_fix(f) or pyi_return_type(f)
    return _format_signature(f.cpp_name, formals, ret)


def _vararg_signature(f: NativeFunction, *, method: bool) -> str | None:
    # Alternate spelling for leading int-list parameters: the call form
    # ``f(1, 2, 3)`` unrolls the list into positional ints.
    args = list(f.args)
    start = 0
    if method:
        if not args or args[0].name != "self":
            return None
        start = 1
    rest = args[start:]
    if len(rest) == 0:
        return None
    first = rest[0]
    if not (first.type.is_list and not first.type.is_opt
            and first.type.kind in ("int64_t",) and not first.default):
        return None
    if any(not a.kwonly for a in rest[1:]):
        return None
    formals = (["self"] if method else []) + [
        f"*{sanitize_name(first.name)}: _int"
    ]
    for a in rest[1:]:
        s = f"{sanitize_name(a.name)}: {pyi_type(a.type, list_default=a.default)}"
        if a.default is not None:
            dft = py_default_for(f, a, "pyi") or pyi_default(a.type, a.default)
            s += f" = {dft}"
        formals.append(s)
    ret = _foreach_return_fix(f) or pyi_return_type(f)
    return _format_signature(f.cpp_name, formals, ret)


# ---------------------------------------------------------------------------
# Docstrings
# ---------------------------------------------------------------------------

def gather_docstrs(docs_path: str) -> dict[str, str]:
    """Collect runtime docstrings without importing the package.

    The documentation module attaches its strings through the extension's
    ``_add_docstr`` hook; stub generation substitutes a recorder and executes
    the module in isolation.
    """
    docstrs: dict[str, str] = {}

    def mock_add_docstr(func: Mock, docstr: str) -> None:
        docstrs[func._extract_mock_name()] = docstr.strip()

    if not os.path.exists(docs_path):
        warn(f"docstring source not found, skipping docs in stub: {docs_path}")
        return docstrs
    with patch.dict(sys.modules):
        sys.modules["tensorplay"] = Mock(name="tensorplay")
        sys.modules["tensorplay._C"] = Mock(_add_docstr=mock_add_docstr)
        try:
            spec = importlib.util.spec_from_file_location("_tp_stub_docs",
                                                          docs_path)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_tp_stub_docs"] = mod
            spec.loader.exec_module(mod)
        except Exception as e:  # noqa: BLE001 - stubs degrade to no docs
            warn(f"failed to collect docstrings, skipping docs in stub: {e!r}")
    return docstrs


def _add_docstr_to_hint(docstr: str, hint: str) -> str:
    if "'''" in docstr or '"""' in docstr:
        return hint
    docstr = inspect.cleandoc(docstr).strip()
    if "..." in hint:  # function or method
        if not hint.endswith("..."):
            raise AssertionError(f"Hint `{hint}` does not end with '...'")
        hint = hint.removesuffix("...").rstrip()
        content = hint + "\n" + textwrap.indent(f'r"""\n{docstr}\n"""',
                                                prefix="    ")
        return "\n".join(map(str.rstrip, content.splitlines())).rstrip()
    return f'{hint}\nr"""{docstr}"""'


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _group_hints(funcs: list[NativeFunction], *, method: bool = False,
                 docstrs: dict[str, str] | None = None,
                 functional: bool = False) -> dict[str, list[str]]:
    variant = "method" if method else "function"
    groups: dict[str, list[NativeFunction]] = defaultdict(list)
    for f in funcs:
        if variant not in f.variants:
            continue
        if functional and _out_arg(f) is not None:
            # The functional layer carries no out overloads.
            continue
        groups[f.cpp_name].append(f)

    out: dict[str, list[str]] = {}
    for name in sorted(groups):
        grp = groups[name]
        base = next((x for x in grp if x.overload_name == ""), None)
        outv = None if functional else next(
            (x for x in grp if _out_arg(x) is not None), None)
        pairs: list[tuple[NativeFunction, Type | None]] = []
        if (base is not None and outv is not None and base is not outv
                and _out_arg(base) is None):
            rest_out = [a for a in outv.args
                        if not (a.name == "out" and a.type.is_mutable_ref)]
            if (len(base.args) == len(rest_out)
                    and all(_args_equal(a, b)
                            for a, b in zip(base.args, rest_out))):
                pairs.append((base, _out_arg(outv).type))
                grp = [x for x in grp if x is not base and x is not outv]
        for f in grp:
            oa = _out_arg(f)
            pairs.append((f, oa.type if oa else None))
        sigs = [_signature(f, method=method, merged_out=ot,
                           functional=functional) for f, ot in pairs]
        if not method and not functional:
            vararg = next((_vararg_signature(f, method=method)
                           for f, _ in pairs if f.overload_name == ""), None)
            if vararg is not None and vararg not in sigs:
                sigs.append(vararg)
        if len(sigs) > 1:
            sigs = ["@overload\n" + s for s in sigs]
        if docstrs and not functional:
            doc = docstrs.get(f"tensorplay.{name}")
            if doc is not None:
                sigs = [_add_docstr_to_hint(doc, s) for s in sigs]
        out[name] = sigs
    return out


# ---------------------------------------------------------------------------
# DType block
# ---------------------------------------------------------------------------

_SHORT_ALIAS_BLOCKLIST = {"bool", "float", "int"}  # would shadow builtins


def _dtype_block(binding_path: str | None, header_path: str | None) -> str:
    members: list[str] = []
    aliases: list[str] = []
    if binding_path and os.path.exists(binding_path):
        text = open(binding_path).read()
        seen: set[str] = set()
        for m in re.finditer(r'\.value\("(\w+)",\s*DType::\w+\)', text):
            if m.group(1) not in seen:
                seen.add(m.group(1))
                members.append(m.group(1))
        for m in re.finditer(r'm\.attr\("(\w+)"\)\s*=\s*DType::\w+;', text):
            aliases.append(m.group(1))
    elif header_path:
        for d in _parse_dtypes(header_path):
            members.append(d["py_name"])
        aliases = list(members)

    lines = ["class DType(enum.Enum):"]
    lines += [f"    {name} = auto()" for name in members]
    lines += [
        "",
        "    def __str__(self) -> str: ...",
        "    def __repr__(self) -> str: ...",
        "    is_floating_point: bool",
        "    is_complex: bool",
        "    is_signed: bool",
        "    itemsize: int",
        "",
    ]
    for name in aliases:
        if name in _SHORT_ALIAS_BLOCKLIST:
            continue
        lines.append(f"{name}: DType = DType.{name}")
        lines.append("")
    return "\n".join(lines)


def _parse_dtypes(header_path: str):
    if not os.path.exists(header_path):
        return []
    content = open(header_path).read()
    m = re.search(r'enum class ScalarType\s*:\s*\w+\s*\{(.*?)\};', content, re.DOTALL)
    if not m:
        return []
    dtypes, val = [], 0
    for line in m.group(1).split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        line = line.rstrip(',')
        if '=' in line:
            name, _, vs = line.partition('=')
            name = name.strip()
            try:
                val = int(vs.strip())
            except ValueError:
                pass
        else:
            name = line
        py_name = name.lower()
        dtypes.append({'name': name, 'py_name': py_name, 'val': val})
        val += 1
    return dtypes


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

# A hole that starts a line (``${name}`` at column 0 or after whitespace) is a
# block hole: every substituted line inherits the hole's indentation and
# trailing whitespace is stripped.  Mid-line holes substitute verbatim.
_SUBST_RE = re.compile(r"(^[^\n\S]*)?\$\{(\w+)\}", re.MULTILINE)

# Keep a blank line between a closing docstring and the following definition
# so the generated stub stays formattable.
_DOCSTRING_GAP_RE = re.compile(
    r'''(""")\n+((\s*@.+\n)*\s*(class|def))''',
    re.VERBOSE,
)


def _substitute(template: str, env: dict[str, str]) -> str:
    def replace(m: re.Match[str]) -> str:
        indent, key = m.group(1), m.group(2)
        if key not in env:
            raise KeyError(f"template variable not provided: {key}")
        if indent is None:
            return env[key]
        content = "\n".join(itertools.chain.from_iterable(
            env[key].splitlines() for _ in (0,)))
        content = textwrap.indent(content, indent)
        return "\n".join(line.rstrip() for line in content.splitlines()).rstrip()

    out = _SUBST_RE.sub(replace, template)
    return _DOCSTRING_GAP_RE.sub(r"\g<1>\n\n\g<2>", out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _functional_def_names(template_dir: str) -> list[str]:
    """Public def names of the generated functional wrapper module."""
    path = os.path.join(template_dir, "functional.py")
    if not os.path.exists(path):
        warn(f"functional module not found, functional stub is empty: {path}")
        return []
    text = open(path).read()
    return [n for n in re.findall(r"^def (\w+)\(", text, re.M)
            if not n.startswith("_")]


def generate_pyi(funcs: list[NativeFunction], template_dir: str,
                 dtype_header_path: str | None = None,
                 dtype_binding_path: str | None = None) -> dict[str, str]:
    """Render every stub output; returns {output file name: content}."""
    docs_path = os.path.join(template_dir, "_docs.py")
    docstrs = gather_docstrs(docs_path)

    methods = _group_hints(funcs, method=True, docstrs=docstrs)
    functions = _group_hints(funcs, method=False, docstrs=docstrs)
    functional = _group_hints(funcs, functional=True)

    names = sorted({f.cpp_name for f in funcs if "function" in f.variants})
    all_directive = "__all__ = [\n" + "".join(f'    "{n}",\n' for n in names) + "]"

    # The functional surface: hand-wrapped names live in the template, every
    # other def name is forwarded and renders from its schema.
    functional_defs = _functional_def_names(template_dir)
    forwarded = [n for n in sorted(functional_defs)
                 if n not in _FUNCTIONAL_HAND_NAMES]
    uncovered = [n for n in forwarded if n not in functional]
    if uncovered:
        warn(f"functional defs without schema hints: {uncovered[:10]}")
    forwarded = [n for n in forwarded if n in functional]
    imported_hints = "\n".join(
        hint for n in forwarded for hint in functional[n])
    hand = [n for n in _FUNCTIONAL_HAND_NAMES if n in functional_defs]
    uncovered_hand = sorted(_FUNCTIONAL_HAND_NAMES - set(functional_defs))
    if uncovered_hand:
        warn(f"functional hand names absent from the module: {uncovered_hand}")
    extra_all = ("__all__ += [\n"
                 + "".join(f'    "{n}",\n' for n in sorted(
                     set(forwarded) | set(hand)))
                 + "]")

    env = {
        "generated_dtypes": _dtype_block(dtype_binding_path,
                                         dtype_header_path),
        "generated_methods": "\n".join(
            hint for n in sorted(methods) for hint in methods[n]),
        "generated_functions": "\n".join(
            hint for n in sorted(functions) for hint in functions[n]),
        "all_directive": all_directive,
        "imported_hints": imported_hints,
        "dispatched_hints": "",
        "extra_functional___all__": extra_all,
    }

    outputs: dict[str, str] = {}
    for out_name, tmpl_name in _PYI_OUTPUTS:
        tmpl_path = os.path.join(template_dir, tmpl_name)
        with open(tmpl_path) as fh:
            template = fh.read()
        env["generated_comment"] = (
            f"@generated by tools/codegen/gen_pyi.py from {tmpl_name}")
        outputs[out_name] = _substitute(template, env)
    return outputs
