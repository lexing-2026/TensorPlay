"""Whether the device compiler is here, and what to do when it is not.

Generated code that hands work to the device compiler has to know whether the
compiler is installed before it names anything from it, because naming a module
that is not there is a failure at import -- which for a generated file means the
whole region fails to load, rather than the one path that needed the compiler
falling back.

So the question is asked through here, and asked by import rather than by
looking for a file: a package can be present and still unusable, and the only
way to know is to import it.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

log = logging.getLogger(__name__)

#: Whether the device compiler could be imported, settled once per process.  A
#: second attempt would cost an import to learn the same answer, and the answer
#: cannot change while the process runs: an installed package stays installed.
_compiler_module: Any = None
_probed = False


def has_triton_package() -> bool:
    """Whether the device compiler can be imported here.

    False is a normal answer, not an error: a machine without the compiler runs
    the regions that do not need one, and a caller asking this is deciding
    which of two paths to take rather than reporting a failure.
    """

    global _compiler_module, _probed
    if _probed:
        return _compiler_module is not None
    _probed = True
    try:
        _compiler_module = importlib.import_module("triton")
    except Exception:  # noqa: BLE001 - any failure to import means "not here"
        log.debug("the device compiler is not importable", exc_info=True)
        _compiler_module = None
    return _compiler_module is not None


def compiler_module() -> Any:
    """The device compiler module, or nothing if it is not here.

    For a caller that needs the module itself rather than the answer, so that it
    does not import a name that may not exist.
    """

    has_triton_package()
    return _compiler_module


__all__ = ["compiler_module", "has_triton_package"]
