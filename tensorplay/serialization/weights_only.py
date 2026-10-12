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


class _MetadataObject(dict):
    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def append(self, value):
        pass

    def extend(self, values):
        pass

    def add(self, value):
        pass


class StorageMetadataUnpickler(pickle._Unpickler):
    """Discover storage references without constructing checkpoint objects."""

    find_class = WeightsOnlyUnpickler.find_class

    def __init__(self, file, *, persistent_load, resolve_global, **kwargs):
        super().__init__(file, **kwargs)
        self.persistent_load = persistent_load
        self._resolve_global = resolve_global

    def load_reduce(self):
        from codecs import encode
        args = self.stack.pop()
        func = self.stack[-1]
        self.stack[-1] = encode(*args) if func is encode else _MetadataObject()

    def load_newobj(self):
        self.stack.pop()
        self.stack[-1] = _MetadataObject()

    def load_newobj_ex(self):
        self.stack.pop()
        self.stack.pop()
        self.stack[-1] = _MetadataObject()

    def load_build(self):
        self.stack.pop()

    def load_obj(self):
        self.pop_mark()
        self.append(_MetadataObject())

    def load_inst(self):
        module = self.readline()[:-1].decode('ascii')
        name = self.readline()[:-1].decode('ascii')
        self.find_class(module, name)
        self.pop_mark()
        self.append(_MetadataObject())

    dispatch = pickle._Unpickler.dispatch.copy()
    dispatch[pickle.REDUCE[0]] = load_reduce
    dispatch[pickle.NEWOBJ[0]] = load_newobj
    dispatch[pickle.NEWOBJ_EX[0]] = load_newobj_ex
    dispatch[pickle.BUILD[0]] = load_build
    dispatch[pickle.OBJ[0]] = load_obj
    dispatch[pickle.INST[0]] = load_inst

__all__ = ["WeightsOnlyUnpickler"]
