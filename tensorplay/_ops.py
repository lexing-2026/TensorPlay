"""``tensorplay.ops`` namespace, mirroring ``torch._ops``.

Two kinds of entries resolve here:

1. Python-registered operators (:mod:`tensorplay.library`):
   ``tensorplay.ops.mylib.add(x, y)`` returns the :class:`CustomOpDef` and
   calling it runs the normal dispatch path (autograd, capture awareness).
2. Natively loaded extension libraries: ``tensorplay.ops.load_library(path)``
   dlopens a shared object whose static registrars feed the p10 dispatcher
   (the ``TENSORPLAY_LIBRARY_IMPL`` macro family) and attaches the module
   under this namespace, exactly like ``torch.ops.load_library``.
"""

from __future__ import annotations

import types
from typing import Any

import tensorplay
import tensorplay._C as _C


class _OpNamespace(types.ModuleType):
    """Attribute-access packet for one operator namespace (``ns``)."""

    def __init__(self, ns: str) -> None:
        super().__init__(f"tensorplay.ops.{ns}")
        self.ns = ns

    def __getattr__(self, opname: str) -> Any:
        # Native extension modules registered via load_library win: they are
        # real submodules placed on this namespace.
        own = self.__dict__.get(opname)
        if own is not None:
            return own
        full_name = f"{self.ns}::{opname}"
        if tensorplay.library.has_op(full_name):
            return tensorplay.library.get_op(full_name)
        raise AttributeError(
            f"No operator {full_name!r} is registered; define it with "
            f"tensorplay.library.custom_op(\"{full_name}\") or load its "
            "extension library via tensorplay.ops.load_library"
        )


class _Ops(types.ModuleType):
    """The ``tensorplay.ops`` root namespace."""

    __file__ = "_ops.py"

    def __getattr__(self, name: str) -> _OpNamespace:
        if name.startswith("_"):
            raise AttributeError(name)
        namespace = _OpNamespace(name)
        setattr(self, name, namespace)
        return namespace

    @property
    def load_library(self) -> Any:
        return _C.ops.load_library

    @property
    def loaded_libraries(self) -> Any:
        return getattr(_C.ops, "loaded_libraries")


ops = _Ops("tensorplay.ops")
