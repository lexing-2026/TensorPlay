#!/usr/bin/env python3
"""Audit: which ops declare CPU/CUDA dispatch in native_functions.yaml but have
no kernel symbol defined anywhere under p10/ (or tpx/).

Registration macros are expanded first (see audit_macro_expander) so kernels
defined only through NAME(#OP)-style macros are recognized."""
import os
from collections import defaultdict
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_macro_expander import preprocess

YAML = "config/native_functions.yaml"

text = open(YAML).read().splitlines()

# Collect (op_name, func_line, {backend: kernel}) blocks.
entries = []  # (func_line, backend, symbol)
current_func = None
pending_dispatch = None
for i, line in enumerate(text):
    m = re.match(r"- func:\s*(.+)", line)
    if m:
        current_func = m.group(1).strip()
        pending_dispatch = None
        continue
    # dispatch can be inline: `dispatch: {CPU: foo_cpu, CUDA: foo_cuda}`
    m = re.search(r"dispatch:\s*\{([^}]*)\}", line)
    if m and current_func:
        for pair in m.group(1).split(","):
            if ":" in pair:
                k, v = pair.split(":", 1)
                entries.append((current_func, k.strip(), v.strip()))
        continue
    # or block: `dispatch:\n    CPU: foo_cpu\n    CUDA: foo_cuda`
    if re.match(r"\s+dispatch:\s*$", line) and current_func:
        pending_dispatch = True
        continue
    if pending_dispatch:
        m = re.match(r"\s+(CPU|CUDA|CompositeExplicitAutograd|Autograd|AutogradOther|CompositeImplicitAutograd|BackendSelect|MPS|Meta):\s*(\S+)", line)
        if m:
            entries.append((current_func, m.group(1), m.group(2)))
        elif re.match(r"- func:", line) or re.match(r"\S", line):
            pending_dispatch = None

# Build corpus of defined symbols: grep each candidate symbol once via a bulk
# fixed-string search over p10/ and tpx/ source files.
symbols = sorted({sym for (_, backend, sym) in entries
                  if backend in ("CPU", "CUDA")})

# Gather all source text once; tokenize identifiers into a set (single pass).
files = subprocess.run(
    ["bash", "-c",
     "find p10 tpx -type f \\( -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' \\) 2>/dev/null"],
    capture_output=True, text=True).stdout.split()
tok_re = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
identifiers = set()
impl_re = re.compile(r'm\.impl\(\s*"([^"]+)"')
reg_re = re.compile(r"TENSORPLAY_LIBRARY_IMPL\(\s*([A-Za-z0-9_]+)\s*,")
registered = defaultdict(set)  # dispatch key -> {opname}
for f in files:
    try:
        content = open(f, encoding="utf-8", errors="ignore").read()
    except OSError:
        continue
    pre = preprocess(content)
    identifiers.update(tok_re.findall(pre))
    # A yaml dispatch symbol is satisfied when the op itself is registered on
    # that backend, even if the live kernel has a different (macro-generated)
    # name than the one documented in the yaml dispatch field.
    for m in reg_re.finditer(pre):
        key = m.group(1)
        i = pre.find("{", m.end())
        if i < 0:
            continue
        depth = 0
        j = i
        while j < len(pre):
            if pre[j] == "{":
                depth += 1
            elif pre[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        for im in impl_re.finditer(pre[i:j]):
            registered[key].add(im.group(1))

missing = {sym for sym in symbols if sym not in identifiers}

# Report per-op which backends are missing.
from collections import defaultdict
by_op = defaultdict(dict)
for func, backend, sym in entries:
    if backend in ("CPU", "CUDA"):
        by_op[func][backend] = sym


def op_base(func):
    """yaml `func:` line -> base op name (e.g. `sub.Tensor(Tensor self...)`)."""
    m = re.match(r"([A-Za-z0-9_]+(?:\.[A-Za-z0-9_.]+)*)\(", func)
    return m.group(1) if m else func


def op_registered(func, backend):
    name = op_base(func)
    keys = {backend, "Composite"} if backend == "CPU" else {backend, "Composite"}
    return any(name in registered[k] for k in keys)


print("=== ops with dispatch entries whose kernel symbol never appears in p10/tpx ===")
for func in sorted(by_op):
    backends = by_op[func]
    miss = [b for b, sym in backends.items()
            if sym in missing and not op_registered(func, b)]
    if miss:
        print(f"{func}\n    missing: {miss}  syms: {[backends[b] for b in miss]}")
print(f"\ntotal ops with CPU/CUDA dispatch: {len(by_op)}, symbols checked: {len(symbols)}, "
      f"fully-absent symbols: {len(missing)}")
