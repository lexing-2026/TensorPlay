#!/usr/bin/env python3
"""Shared preprocessing for the audit tools: expand the function-like
registration macros (e.g. NAME(#OP) bodies emitting m.impl("_x_" #OP)) that
appear inside TENSORPLAY_LIBRARY_IMPL blocks, so plain regex scans can see the
generated registrations.

The expander supports exactly what the kernel sources use:
  - single-line macros joined across "\" line continuations,
  - stringization (#PARAM) and token pasting (PARAM##suffix),
  - adjacent C string literals merged after substitution.
"""
import re


def load_function_macros(text):
    """Collect `#define NAME(p1, p2) body` definitions (continuations joined)."""
    joined = text.replace("\\\n", "  ")
    macros = {}
    pattern = re.compile(
        r"^[ \t]*#define\s+([A-Za-z_]\w*)\(\s*([A-Za-z_]\w*"
        r"(?:\s*,\s*[A-Za-z_]\w*)*)\s*\)(.*)$", re.M)
    for m in pattern.finditer(joined):
        params = [p.strip() for p in m.group(2).split(",")]
        macros[m.group(1)] = (params, m.group(3).strip())
    return macros, joined


def _merge_adjacent_strings(text):
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r'"([^"\\]*)"\s*"([^"\\]*)"', r'"\1\2"', text)
    return text


def expand_function_macros(text, macros, rounds=8):
    """Replace invocation sites with macro bodies; drop the #define lines."""
    for name in macros:
        text = re.sub(
            r"^[ \t]*#define\s+" + re.escape(name) + r"\([^\n]*$", "",
            text, flags=re.M)
    for _ in range(rounds):
        changed = False
        for name, (params, body) in macros.items():
            call = re.compile(r"\b" + re.escape(name) +
                              r"\(\s*([A-Za-z_]\w*)\s*\)")

            def repl(mo):
                nonlocal changed
                arg = mo.group(1)
                out = body
                # Token pasting first (## disappears, args concatenate), then
                # stringization, then plain parameter substitution.
                for p in params:
                    out = re.sub(
                        r"\b" + re.escape(p) + r"\s*##\s*", lambda _m, a=arg: a, out)
                    out = re.sub(
                        r"##\s*" + re.escape(p) + r"\b", lambda _m, a=arg: a, out)
                out = out.replace("##", "")
                for p in params:
                    out = out.replace("#" + p, '"' + arg + '"')
                for p in params:
                    out = re.sub(r"\b" + re.escape(p) + r"\b",
                                 lambda _m, a=arg: a, out)
                if out != mo.group(0):
                    changed = True
                return out

            text = call.sub(repl, text)
        if not changed:
            break
    return _merge_adjacent_strings(text)


def preprocess(text):
    macros, joined = load_function_macros(text)
    return expand_function_macros(joined, macros)
