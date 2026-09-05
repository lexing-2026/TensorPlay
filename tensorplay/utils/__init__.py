import copyreg
import weakref

import tensorplay

from . import data


def set_module(obj, mod):
    """Set the module label used when displaying ``obj``."""
    if not isinstance(mod, str):
        raise TypeError("the module label must be a string")
    obj.__module__ = mod


def swap_tensors(first, second):
    """Exchange tensor storage while retaining both Python identities."""
    if weakref.getweakrefs(first):
        raise RuntimeError("cannot swap the first tensor while it has weakrefs")
    if weakref.getweakrefs(second):
        raise RuntimeError("cannot swap the second tensor while it has weakrefs")

    first_slots = set(copyreg._slotnames(first.__class__) or ())
    second_slots = set(copyreg._slotnames(second.__class__) or ())
    if first_slots != second_slots:
        raise RuntimeError("cannot swap tensors with different slots")

    def swap_attribute(name):
        first_value = getattr(first, name)
        second_value = getattr(second, name)
        setattr(first, name, second_value)
        setattr(second, name, first_value)

    swap_attribute("__class__")
    swap_attribute("__dict__")
    for slot in first_slots:
        first_has = hasattr(first, slot)
        second_has = hasattr(second, slot)
        if first_has and second_has:
            swap_attribute(slot)
        elif first_has:
            setattr(second, slot, getattr(first, slot))
            delattr(first, slot)
        elif second_has:
            setattr(first, slot, getattr(second, slot))
            delattr(second, slot)

    tensorplay._C._swap_tensor_impl(first, second)

def __getattr__(name):
    if name == "viz":
        from . import viz
        return viz
    if name == "tensorboard":
        from . import tensorboard
        return tensorboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
