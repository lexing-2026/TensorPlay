#!/usr/bin/env python3
"""Audit the destination variants the runtime carries against the reference.

A schema may carry a destination in two ways.  It may declare one outright,
as an entry whose parameter list names a destination; or it may declare the
in-place form and let the runtime derive the destination form from it.  The
second is spelled in the reference's `autogen:` line, and the derived
spelling is a destination variant the runtime carries.

This audit reads both schema files, collects the declared destination
variants and the derived ones, and reports the derived spellings the
project does not carry.

Usage:
    python tools/audit_out_overloads.py --reference PATH [--project PATH]

The reference schema file is named on the command line; this tool does not
look for it, so it stays clear of where that file lives.
"""
import argparse
import os
import sys
from collections import defaultdict

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml ships with the runtime
    yaml = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_PROJECT = os.path.join(ROOT, "config", "native_functions.yaml")


def load_entries(path):
    """Every entry of a schema file as a plain dict."""
    if yaml is None:
        raise SystemExit("pyyaml is required to read the schema files")
    entries = []
    with open(path, encoding="utf-8") as handle:
        document = yaml.compose(handle)
    for node in document.value:
        entry = {}
        for key, value in node.value:
            if isinstance(value, yaml.ScalarNode):
                entry[key.value] = value.value
            elif isinstance(value, yaml.SequenceNode):
                if all(isinstance(v, yaml.ScalarNode) for v in value.value):
                    entry[key.value] = [v.value for v in value.value]
            elif isinstance(value, yaml.MappingNode):
                entry[key.value] = {k.value: v.value for k, v in value.value}
        entries.append(entry)
    return entries


def split_schema(schema):
    """``sigmoid.out(Tensor self, ...) -> ...`` into ``("sigmoid", "out")``.

    A schema without a dot names its operation alone and carries no
    overload spelling.
    """
    base = schema.split("(", 1)[0]
    if "." in base:
        operation, spelling = base.rsplit(".", 1)
        return operation, spelling
    return base, ""


def is_destination_spelling(spelling):
    """Whether an overload spelling names a destination variant."""
    return spelling == "out" or spelling.endswith("_out")


def derived_destination_spellings(entries):
    """The destination spellings the runtime derives from the in-place forms.

    The reference writes them in an ``autogen:`` line; the value names one
    spelling or a comma-separated list of them.
    """
    derived = defaultdict(list)
    for entry in entries:
        autogen = entry.get("autogen")
        if not autogen:
            continue
        names = autogen if isinstance(autogen, list) else [autogen]
        for full in names:
            if not isinstance(full, str):
                continue
            for piece in full.split(","):
                piece = piece.strip()
                if "." not in piece:
                    continue
                operation, spelling = piece.rsplit(".", 1)
                if is_destination_spelling(spelling):
                    derived[(operation, spelling)].append(
                        entry["func"].split("(", 1)[0])
    return derived


def declared_pairs(entries):
    """Every (operation, overload spelling) the file carries."""
    return {split_schema(entry["func"]) for entry in entries
            if isinstance(entry.get("func"), str)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True,
                        help="the schema file the derived spellings are read from")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    args = parser.parse_args()

    for path in (args.reference, args.project):
        if not os.path.exists(path):
            raise SystemExit(f"no such schema file: {path}")

    reference = load_entries(args.reference)
    project = load_entries(args.project)

    derived = derived_destination_spellings(reference)
    carried = declared_pairs(project)

    missing = sorted(pair for pair in derived if pair not in carried)

    print(f"reference derived destination spellings: {len(derived)}")
    print(f"  carried by the project: {len(derived) - len(missing)}")
    print(f"  missing from the project: {len(missing)}")
    if missing:
        print()
        print("missing destination spellings:")
        for operation, spelling in missing:
            print(f"  {operation}.{spelling}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
