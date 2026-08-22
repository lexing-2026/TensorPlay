"""Migrate config/native_functions.yaml schemas to ATen canonical spelling.

For every TensorPlay operator that also exists upstream, adopt ATen's exact
`- func:` string (argument names, kwarg-only markers, defaults, tuple-return
spelling) whenever the mapped C++ ABI is identical -- same argument types in
the same order and the same return kind.  Entries whose ABI differs from
ATen (kernel signatures were written against the TP dialect) are kept as-is
and reported, so backend work can catch up op by op.

Usage:
    python3 tools/codegen/migrate_schemas.py [--apply]
Dry-run by default; prints a report and writes nothing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve()
for _p in ROOT.parents:
    if (_p / "config" / "native_functions.yaml").exists():
        ROOT = _p
        break
else:
    raise SystemExit("cannot locate TensorPlay repo root")
TP_YAML = ROOT / "config" / "native_functions.yaml"
ATEN_YAML = (ROOT / "third_party" / "pytorch" /
             "aten" / "src" / "ATen" / "native" / "native_functions.yaml")

# Repo root must WIN over any vendored tree that also ships a `tools`
# package (e.g. the pytorch checkout under third_party/).
sys.path.insert(0, str(ROOT))

from tools.codegen.api_types import cpp_arg_type  # noqa: E402
from tools.codegen.model import parse_schema       # noqa: E402


def abi_signature(schema_str: str) -> tuple | None:
    """C++ ABI fingerprint: ordered arg types + return kind."""
    try:
        f = parse_schema(schema_str)
    except ValueError:
        return None
    args = tuple(
        (cpp_arg_type(a.type), a.default is not None)
        for a in f.args if a.name != "requires_grad"
    )
    ret = f.cpp_return_kind
    if ret == "tuple":
        ret += ":" + ",".join(cpp_arg_type(r.type) for r in f.returns)
    return args, ret


def load_aten_schemas() -> dict[str, str]:
    data = yaml.safe_load(ATEN_YAML.read_text())
    out: dict[str, str] = {}
    for item in data or []:
        func = item.get("func")
        if not func:
            continue
        name = func.split("(", 1)[0]
        # First definition wins, matching torchgen's duplicate rejection.
        out.setdefault(name, func)
    return out



# ---------------------------------------------------------------------------
# Pure spelling canonicalization (no semantic/ABI change beyond the
# documented Tensor-x={} -> Tensor?-x=None sugar, which matches ATen).
# ---------------------------------------------------------------------------

_TUPLE_RE = re.compile(r"->\s*std::tuple<([^>]*)>\s*$")


_LIST_DEFAULT_RE = re.compile(
    r"\b((?:int|float|SymInt)\[\](?:\?)?)\s+([A-Za-z_]\w*)=\{([^}]*)\}")


def _canonicalize_schema(func: str, extras: dict | None = None) -> str:
    """Rewrite one `- func:` string into ATen-canonical grammar.

    Non-empty list defaults (`int[] stride={1, 1}`) are not representable in
    upstream's grammar (only broadcast scalars or `[]`); they are moved to the
    entry-level `python_defaults` table so the Python/pybind surface keeps
    torch-compatible defaults while kernels keep receiving full vectors.
    """
    # 0) spelling normalization FIRST so later stages see ATen names
    #    (ATen: int == int64, float == double, ScalarType is the enum)
    func = re.sub(r"\bint64_t(\??)", r"int\1", func)
    func = re.sub(r"(?<=[\s(])double(?=\s+[A-Za-z_])", "float", func)
    func = re.sub(r"\bDType(\??)", r"ScalarType\1", func)

    # 1) non-empty list defaults -> [] + python_defaults entry
    if extras is None:
        extras = {}
    def _sub_list_default(m):
        typ, name, body = m.group(1), m.group(2), m.group(3).strip()
        extras.setdefault("python_defaults", {})[name] = "[" + body + "]"
        return f"{typ} {name}=[]"
    func = _LIST_DEFAULT_RE.sub(_sub_list_default, func)

    # 2) multi-mutable in-place ops: only `self` keeps its write annotation
    #     (torchgen requires returns[0].annotation == self.annotation).
    if re.search(r"Tensor\([a-z]!\)\s+growth_tracker", func):
        func = re.sub(r"Tensor\([a-z]!\)(\s+growth_tracker)", r"Tensor\1", func)

    # 3) tuple return spelling
    m = _TUPLE_RE.search(func)
    if m:
        inner = ", ".join(t.strip() for t in m.group(1).split(","))
        func = func[:m.start()] + "-> (" + inner + ")"
    # 4) undefined-Tensor default sugar
    func = re.sub(r"\bTensor\s+([A-Za-z_]\w*)=\{\}", r"Tensor? \1=None", func)
    return func


def canonicalize(args, apply):
    tp_docs = yaml.safe_load(TP_YAML.read_text())
    changed = 0
    for it in tp_docs:
        new = _canonicalize_schema(it["func"])
        if new != it["func"]:
            it["func"] = new
            changed += 1
    print(f"canonicalized schemas: {changed}/{len(tp_docs)}")
    # derivatives.yaml carries its own `- name:` schema copies in the same
    # grammar; canonicalize those lines too (formula bodies are untouched).
    dpath = ROOT / "config" / "derivatives.yaml"
    if dpath.exists():
        dlines = dpath.read_text().split("\n")
        dchanged = 0
        out = []
        for line in dlines:
            stripped = line.lstrip()
            if stripped.startswith("- name:"):
                indent = line[: len(line) - len(line.lstrip())]
                old_name = line.split(":", 1)[1].strip()
                new_name = _canonicalize_schema(old_name)
                if new_name != old_name:
                    dchanged += 1
                out.append(f"{indent}- name: {new_name}")
            else:
                out.append(line)
        if apply:
            dpath.write_text("\n".join(out))
        print(f"derivatives schemas canonicalized: {dchanged}")

    if apply and changed:
        TP_YAML.write_text(TP_YAML.read_text())  # placeholder; surgical below
        # Surgical line rewrite preserving comments/structure.
        lines = TP_YAML.read_text().split("\n")
        by_old = {}
        # map old func string -> new
        # rebuild mapping from in-memory docs order is unreliable after load;
        # instead do a second pass over lines directly.
        out, i = [], 0
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            if stripped.startswith("- func:") or stripped.startswith("func:"):
                indent = line[: len(line) - len(line.lstrip())]
                prefix = "- func:" if "- func:" in line else "func:"
                old_func = line.split(":", 1)[1].strip()
                new_func = _canonicalize_schema(old_func)
                out.append(f"{indent}{prefix} {new_func}")
                i += 1
                continue
            out.append(line)
            i += 1
        TP_YAML.write_text("\n".join(out))
        print(f"WROTE {TP_YAML}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--canonicalize", action="store_true")
    args = ap.parse_args()

    if args.canonicalize:
        canonicalize(args, args.apply)
        return

    aten = load_aten_schemas()
    tp_docs = yaml.safe_load(TP_YAML.read_text())

    adopted, kept_same, mismatched, missing_aten, unparseable = [], [], [], [], []

    for item in tp_docs:
        func = item.get("func")
        name = func.split("(", 1)[0]
        tp_abi = abi_signature(func)
        if tp_abi is None:
            unparseable.append(name)
            continue
        if func == aten.get(name):
            kept_same.append(name)
            continue
        if name not in aten:
            missing_aten.append(name)
            continue
        aten_abi = abi_signature(aten[name])
        if aten_abi is not None and aten_abi == tp_abi:
            item["func"] = aten[name]
            adopted.append((name, aten[name]))
        else:
            mismatched.append((name, tp_abi is None, aten_abi is None))

    print(f"total ops          : {len(tp_docs)}")
    print(f"already canonical  : {len(kept_same)}")
    print(f"adopted ATen schema: {len(adopted)}")
    print(f"ABI mismatch (keep): {len(mismatched)}")
    print(f"not in ATen (TP-only): {len(missing_aten)}")
    print(f"unparseable        : {len(unparseable)}")
    for n in adopted[:15]:
        print("  ADOPT", n)
    for n, _, _ in mismatched[:15]:
        print("  DIFF ", n)
    for n in missing_aten[:10]:
        print("  TONLY", n)

    if args.apply and adopted:
        header = "# Migrated to ATen canonical spelling by tools/codegen/migrate_schemas.py\n"
        text = yaml.safe_dump(tp_docs, sort_keys=False,
                              allow_unicode=True, width=100)
        TP_YAML.write_text(header + text)
        print(f"WROTE {TP_YAML} ({len(adopted)} schemas updated)")


if __name__ == "__main__":
    main()
