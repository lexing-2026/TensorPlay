"""Generation of the _C.pyi typing stub."""

from __future__ import annotations

import os
import re

from .api_types import pyi_default, pyi_return_type, pyi_type, sanitize_name
from .model import NativeFunction


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


def _dtype_pyi(dtypes) -> str:
    lines = ['class DType(enum.Enum):',
             '    def __str__(self) -> str: ...',
             '    def __repr__(self) -> str: ...',
             '']
    for d in dtypes:
        lines += [f'    {d["py_name"]} = {d["val"]}', '']
    for d in dtypes:
        lines += [f'{d["py_name"]}: DType = DType.{d["py_name"]}', '']
    return '\n'.join(lines)


def generate_pyi(funcs: list[NativeFunction], template_path: str,
                 dtype_header_path: str | None = None) -> str:
    template = open(template_path).read()

    methods_lines, functions_lines = [], []
    seen: set[str] = set()
    for f in funcs:
        if f.cpp_name in seen:
            continue
        # Prefer the function variant's signature for stub purposes when both
        # exist; the method variant differs only by the implicit self.
        ret = pyi_return_type(f)
        start = 0
        if 'method' in f.variants and 'function' not in f.variants:
            if f.args and f.args[0].name == 'self':
                start = 1
        arg_strs = []
        for a in f.args[start:]:
            s = f'{sanitize_name(a.name)}: {pyi_type(a.type)}'
            if a.default:
                s += f' = {pyi_default(a.type, a.default)}'
            arg_strs.append(s)
        sig = f'def {f.cpp_name}({", ".join(arg_strs)}) -> {ret}: ...'
        if 'method' in f.variants and 'function' not in f.variants:
            methods_lines.append(f'    {sig}')
        else:
            functions_lines.append(sig)
        seen.add(f.cpp_name)

    template = template.replace('${generated_methods}', '\n'.join(methods_lines))
    template = template.replace('${generated_functions}', '\n'.join(functions_lines))
    if dtype_header_path:
        template = template.replace('${generated_dtypes}',
                                    _dtype_pyi(_parse_dtypes(dtype_header_path)))
    return template
