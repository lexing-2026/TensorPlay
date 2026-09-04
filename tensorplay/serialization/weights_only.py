"""Restricted pickle reader used for checkpoint metadata."""

from __future__ import annotations

import pickle


class WeightsOnlyUnpickler(pickle.Unpickler):
    """Resolve only globals explicitly admitted by the caller or policy."""

    def __init__(self, file, *, persistent_load, resolve_global, **kwargs):
        super().__init__(file, **kwargs)
        self._persistent_load_fn = persistent_load
        self._resolve_global = resolve_global

    def persistent_load(self, saved_id):
        return self._persistent_load_fn(saved_id)

    def find_class(self, module, name):
        if module in {"os", "posix", "nt", "subprocess", "sys"}:
            raise pickle.UnpicklingError(
                f"Unsupported global: GLOBAL {module}.{name} was not allowlisted"
            )
        resolved = self._resolve_global(module, name)
        if resolved is None:
            from .policy import _get_safe_global

            resolved = _get_safe_global(module, name)
        if resolved is not None:
            return resolved
        raise pickle.UnpicklingError(
            f"Unsupported global: GLOBAL {module}.{name} was not allowlisted by "
            "the TensorPlay weights-only loader."
        )


_WeightsOnlyUnpickler = WeightsOnlyUnpickler

__all__ = ["WeightsOnlyUnpickler"]
