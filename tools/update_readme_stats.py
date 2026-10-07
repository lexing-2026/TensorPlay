#!/usr/bin/env python3
"""Refresh the measured statistics embedded in README.md and README.zh.md.

The suite size is measured, never typed by hand: this tool counts the test
functions and methods the tree defines and rewrites the regions delimited by

    <!-- tp-stats:<name>:begin -->  ...  <!-- tp-stats:<name>:end -->

so a hand edit inside a region is overwritten on the next run and `--check`
fails while the committed text disagrees with the measurement. Regions:

    tests-badge     the shields.io badge in the header block
    tests-collected the number inside the benchmark paragraph

Counting is a scan, not a test run. Collecting through pytest instead would
make the published number depend on where it was measured: the count moves
with the compiled extension, the GPU and every optional package a module
guards with importorskip, so the same tree yields a different number per
machine. Scanning needs none of that and lands on the same answer anywhere.

The scan follows the collector's default rules, which this repository does not
override: a module-level function whose name starts with `test`, and methods
of a class whose name starts with `Test`. A parametrized function counts once
even though it expands into several items at run time, so the number is a
property of the source, not of a run.

Usage:
    python tools/update_readme_stats.py                 # rewrite in place
    python tools/update_readme_stats.py --check         # verify, no writes
    python tools/update_readme_stats.py --test-path ../some/test
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_READMES = ("README.md", "README.zh.md")
DEFAULT_TEST_PATH = "test"

TEST_PREFIX = "test"
CLASS_PREFIX = "Test"
TEST_FALSE = "__test__"

BADGE_STYLE = "style=flat-square&labelColor=11B5D1&logo=pytest&logoColor=white"
BADGE_LINK = "https://github.com/lexing-2026/TensorPlay/actions/workflows/trunk.yml"


def is_test(node: ast.AST) -> bool:
    """A test definition: a plain function or coroutine named test*."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return node.name.startswith(TEST_PREFIX)


def is_excluded(node: ast.ClassDef | ast.FunctionDef) -> bool:
    """Honour the `__test__ = False` opt-out the collector honours."""
    return any(
        isinstance(child, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == TEST_FALSE for t in child.targets)
        and isinstance(child.value, ast.Constant)
        and child.value.value is False
        for child in node.body
    )


def scan_module(tree: ast.Module) -> tuple[int, int]:
    """Return (test functions, test classes) for one parsed module.

    Only module-level functions and direct methods of a class count: a
    definition nested in a function or in a method is not a test the collector
    would pick up.
    """
    functions = 0
    classes = 0
    for node in tree.body:
        if is_test(node):
            functions += 0 if is_excluded(node) else 1
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith(CLASS_PREFIX) or is_excluded(node):
                continue
            methods = [m for m in node.body if is_test(m) and not is_excluded(m)]
            if methods:
                classes += 1
                functions += len(methods)
    return functions, classes


def count_tests(test_path: str, cwd: Path) -> tuple[int, int, list[str]]:
    """Scan the test tree and return (functions, classes, unreadable files).

    A module that cannot be parsed would contribute nothing to the count, so it
    is reported instead of silently dropped.
    """
    root = Path(test_path)
    if not root.is_absolute():
        root = cwd / root
    files = sorted(root.rglob("*.py")) if root.is_dir() else [root]
    if not files:
        raise RuntimeError(f"no Python files found under {root}")

    functions = classes = 0
    unreadable: list[str] = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        except SyntaxError as exc:
            unreadable.append(f"{path}: {exc}")
            continue
        file_functions, file_classes = scan_module(tree)
        functions += file_functions
        classes += file_classes
    return functions, classes, unreadable


def render_tests_badge(count: int) -> str:
    """The badge reports what was measured: test functions defined.

    The body carries its own indentation and no trailing newline: the newline
    and indent before the closing marker belong to the file, not to the value.
    """
    url = f"https://img.shields.io/badge/tests-{count}%20defined-23347A?{BADGE_STYLE}"
    return (
        f'    <a href="{BADGE_LINK}">\n'
        f'        <img src="{url}" alt="Tests">\n'
        "    </a>"
    )


def render_tests_collected(count: int) -> str:
    """Thousands-separated, matching the prose style of both READMEs."""
    return f"{count:,}"


# Region name -> renderer. Renderers receive the measured count.
REGIONS = {
    "tests-badge": render_tests_badge,
    "tests-collected": render_tests_collected,
}


def patch_regions(text: str, count: int, origin: str) -> str:
    """Return `text` with every managed region replaced by its rendered value.

    A renderer owns the layout of the value it emits, so only the line break
    that separates a block region from its markers is left alone (and the
    inline case keeps none at all). Indentation is deliberately not carried
    over from the surrounding text: reusing it would append a fresh indent on
    every run and walk the block sideways.
    """
    for name, render in REGIONS.items():
        pattern = re.compile(
            rf"(<!-- tp-stats:{name}:begin -->)([ \t]*\n)?(.*?)(\s*)(<!-- tp-stats:{name}:end -->)",
            re.DOTALL,
        )
        found = pattern.findall(text)
        if len(found) != 1:
            raise RuntimeError(
                f"{origin}: expected exactly one tp-stats:{name} region, found {len(found)}"
            )

        def replacement(match: re.Match, rendered: str = render(count)) -> str:
            return (
                match.group(1)
                + (match.group(2) or "")
                + rendered
                + match.group(4)
                + match.group(5)
            )

        text = pattern.sub(replacement, text, count=1)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--test-path",
        default=DEFAULT_TEST_PATH,
        help="test tree to scan (default: %(default)s)",
    )
    parser.add_argument(
        "--readme",
        action="append",
        dest="readmes",
        metavar="FILE",
        help="README to refresh; repeatable (default: README.md and README.zh.md)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when a README disagrees with the measurement, write nothing",
    )
    args = parser.parse_args()

    cwd = REPO_ROOT if args.test_path == DEFAULT_TEST_PATH else Path.cwd()

    try:
        functions, classes, unreadable = count_tests(args.test_path, cwd)
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if unreadable:
        print(f"error: {len(unreadable)} module(s) could not be parsed:", file=sys.stderr)
        for line in unreadable:
            print(f"  {line}", file=sys.stderr)
        return 2
    print(f"scanned {functions} test functions in {classes} test classes under {args.test_path}")

    stale = []
    for name in args.readmes or DEFAULT_READMES:
        path = REPO_ROOT / name
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2
        original = path.read_text(encoding="utf-8")
        try:
            updated = patch_regions(original, functions, name)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if updated == original:
            print(f"{name}: up to date")
            continue
        stale.append(name)
        if args.check:
            print(f"{name}: stale", file=sys.stderr)
            continue
        path.write_text(updated, encoding="utf-8")
        print(f"{name}: rewritten")

    if stale and args.check:
        print(
            "run `python tools/update_readme_stats.py` to refresh: " + ", ".join(stale),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())