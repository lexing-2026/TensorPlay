"""Static checks for the TensorPlay Python package layout."""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_LEGACY_MODULE_FILES = (
    "fft.py",
    "sparse.py",
    "special.py",
    "linalg.py",
    "export.py",
    "distributed/distributed_c10d.py",
    "nn/modules/fold.py",
)
_LEGACY_MODULE_REFERENCES = (
    "tensorplay.distributed.distributed_c10d",
    "tensorplay.nn.modules.fold",
)


@dataclass(frozen=True)
class Issue:
    path: Path
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def _target_names(node: ast.AST) -> Iterable[str]:
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for item in node.elts:
            yield from _target_names(item)


def _literal_names(node: ast.AST) -> list[str] | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        return None
    return list(value)


def _module_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(_target_names(target))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name.split(".", 1)[0])
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.rsplit(".", 1)[-1])
    return names


def _replaces_module_object(tree: ast.Module) -> bool:
    return any(
        (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
            and node.attr == "modules"
        )
        or (
            isinstance(node, (ast.Import, ast.ImportFrom))
            and any(alias.name == "*" for alias in node.names)
        )
        for node in ast.walk(tree)
    )


def _static_all(tree: ast.Module) -> tuple[list[str], bool]:
    names: list[str] = []
    dynamic = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            is_all = any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
            if is_all:
                values = _literal_names(node.value)
                if values is None:
                    dynamic = True
                else:
                    names = values
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            values = _literal_names(node.value)
            if values is None:
                dynamic = True
            else:
                names = values
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            values = _literal_names(node.value)
            if values is None:
                dynamic = True
            else:
                names.extend(values)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "__all__"
                and call.func.attr == "extend"
            ):
                if len(call.args) != 1:
                    dynamic = True
                else:
                    values = _literal_names(call.args[0])
                    if values is None:
                        dynamic = True
                    else:
                        names.extend(values)
    return names, dynamic


def _check_collisions(root: Path) -> list[Issue]:
    issues: list[Issue] = []
    for path in root.rglob("*.py"):
        if path.name != "__init__.py" and path.with_suffix("").is_dir():
            issues.append(Issue(path, "module file conflicts with a package directory"))
    return issues


def _check_legacy_paths(root: Path) -> list[Issue]:
    issues: list[Issue] = []
    for relative in _LEGACY_MODULE_FILES:
        path = root / relative
        if path.exists():
            issues.append(Issue(path, "legacy module path is still present"))

    repository_root = root.parent
    text_roots = [root, repository_root / "docs" / "source"]
    for text_root in text_roots:
        if not text_root.is_dir():
            continue
        paths = text_root.rglob("*.py") if text_root == root else text_root.rglob("*")
        for path in paths:
            if not path.is_file():
                continue
            try:
                contents = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for reference in _LEGACY_MODULE_REFERENCES:
                if re.search(rf"(?<![\w.]){re.escape(reference)}(?![\w.])", contents):
                    issues.append(Issue(path, f"legacy module reference is still present: {reference}"))
    return issues


def _check_all_exports(root: Path) -> list[Issue]:
    issues: list[Issue] = []
    for path in sorted(root.rglob("__init__.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            issues.append(Issue(path, f"cannot parse package initializer: {exc}"))
            continue
        names, dynamic = _static_all(tree)
        if not names or dynamic:
            continue
        missing = sorted(set(names) - _module_names(tree))
        if missing and not _replaces_module_object(tree):
            issues.append(Issue(path, f"__all__ names are not defined: {', '.join(missing)}"))
        if len(names) != len(set(names)):
            issues.append(Issue(path, "__all__ contains duplicate names"))
    return issues


def audit_package(root: str | Path = "tensorplay") -> list[Issue]:
    package_root = Path(root)
    if not package_root.is_dir():
        return [Issue(package_root, "package directory does not exist")]
    return (
        _check_collisions(package_root)
        + _check_all_exports(package_root)
        + _check_legacy_paths(package_root)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="tensorplay")
    args = parser.parse_args()
    issues = audit_package(args.root)
    for issue in issues:
        print(issue)
    if issues:
        return 1
    print(f"python package audit passed: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
