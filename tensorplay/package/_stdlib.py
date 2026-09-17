# mypy: allow-untyped-defs
"""Identify Python standard library modules.

There is no fully reliable way to tell whether a module is part of the
standard library, so the interpreter's own canonical list of top-level
standard library names is used as the source of truth.
"""

import sys


def is_stdlib_module(module: str) -> bool:
    base_module = module.partition(".")[0]
    return base_module in _get_stdlib_modules()


def _get_stdlib_modules():
    return sys.stdlib_module_names
