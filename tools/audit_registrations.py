#!/usr/bin/env python3
"""Audit v2: compare yaml op schemas against actual per-backend registrations
(TENSORPLAY_LIBRARY_IMPL blocks with m.impl("<op>", ...)) under p10/ and tpx/.
Report ops that would raise MissingDeviceKernel on CPU and/or CUDA."""
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_macro_expander import preprocess

ROOTS = ["p10/src", "tpx/src"]

impl_re = re.compile(r'm\.impl\(\s*"([^"]+)"')
reg_re = re.compile(r"TENSORPLAY_LIBRARY_IMPL\(\s*([A-Za-z0-9_]+)\s*,")
registered = defaultdict(set)  # key -> {opname}

for root in ROOTS:
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith((".cpp", ".cu", ".cc")):
                continue
            path = os.path.join(dirpath, fn)
            try:
                text = open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            text = preprocess(text)
            for m in reg_re.finditer(text):
                key = m.group(1)
                i = text.find("{", m.end())
                if i < 0:
                    continue
                depth = 0
                j = i
                while j < len(text):
                    if text[j] == "{":
                        depth += 1
                    elif text[j] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                body = text[i:j]
                for im in impl_re.finditer(body):
                    registered[key].add(im.group(1))

# ---- collect yaml op names ----
yaml_ops = set()
for line in open("config/native_functions.yaml").read().splitlines():
    m = re.match(r"- func:\s*([A-Za-z0-9_.]+)\s*\(", line)
    if m:
        yaml_ops.add(m.group(1))

cpu_have = registered["CPU"] | registered["Composite"]
cuda_have = registered["CUDA"] | registered["Composite"]

missing_cpu = sorted(o for o in yaml_ops if o not in cpu_have)
missing_cuda = sorted(o for o in yaml_ops if o not in cuda_have)

print(f"yaml ops: {len(yaml_ops)}; registered CPU: {len(registered['CPU'])}, "
      f"CUDA: {len(registered['CUDA'])}, Composite: {len(registered['Composite'])}")
print(f"missing on CPU: {len(missing_cpu)}; missing on CUDA: {len(missing_cuda)}\n")
print("=== missing on CPU ===")
for o in missing_cpu:
    print(" ", o)
print("\n=== missing on CUDA but present on CPU ===")
for o in missing_cuda:
    if o in cpu_have:
        print(" ", o)
