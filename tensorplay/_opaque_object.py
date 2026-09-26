"""Values that are not tensors, and how one is written into generated source.

Most values in a compiled region are tensors, and what the generated code does
with one is arithmetic on its memory.  Some are not: a placement, a sparsity
pattern, a scalar some other system defined.  Such a value has no shape and no
element type, so it cannot be a buffer, and what the generated source has to
reconstruct is the value itself -- which means writing it as source text and
saying what has to be in scope for that text to evaluate.

That is what a value provides for itself: a method returning the text and the
names that text mentions.  This module is the machinery around that: the
registry of which types are values of this kind, and the two questions the code
generators ask -- whether a type is one, and how a value is written.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, NamedTuple


class OpaqueTypeInfo(NamedTuple):
    """What one registered value type is, and how it is written.

    ``opaque_typ`` says which of the two kinds it is.  A constant is a value
    known when the code was written, so the generated source reconstructs it
    from its own text.  A symbolic one is only known at run time, so it arrives
    as a value and the generated source refers to it rather than rebuilding it.
    """

    class_name: str
    opaque_typ: str


#: Registered value types, by name.  A type is a value of this kind by being
#: registered here, which is what makes the question answerable at all: a type
#: that is not registered is not one, rather than being one this build has not
#: heard of.
_OPAQUE_TYPES_BY_NAME: dict[str, OpaqueTypeInfo] = {}


def register_opaque_type(
    class_name: str,
    opaque_typ: str,
    class_obj: type | None = None,
) -> type:
    """Register a type as a value of this kind, and return it.

    Usable as a decorator, so a type says what it is next to its definition
    rather than in a table somewhere else.  Registering the same name twice is
    an error: two definitions of one name would leave the registry saying which
    of them a generated source means.
    """

    if opaque_typ not in ("constant", "symbolic"):
        raise ValueError(
            f"a value is either a constant or symbolic, not {opaque_typ!r}"
        )
    if class_name in _OPAQUE_TYPES_BY_NAME:
        raise ValueError(f"{class_name!r} is already registered as a value type")
    _OPAQUE_TYPES_BY_NAME[class_name] = OpaqueTypeInfo(class_name, opaque_typ)
    if class_obj is None:
        return lambda cls: register_opaque_type(class_name, opaque_typ, cls)
    return class_obj


def is_opaque_type(cls: type[Any] | str) -> bool:
    """Whether a type is a registered value rather than a tensor."""

    name = cls if isinstance(cls, str) else getattr(cls, "__name__", None)
    return name is not None and name in _OPAQUE_TYPES_BY_NAME


def is_opaque_constant_type(cls: type[Any] | str) -> bool:
    """Whether a type is a value known when the code was written.

    The distinction decides what the generated source does with it: a constant
    is written out as its own text, where a symbolic value is referred to by the
    name it arrives under.
    """

    if not is_opaque_type(cls):
        return False
    name = cls if isinstance(cls, str) else getattr(cls, "__name__", "")
    return _OPAQUE_TYPES_BY_NAME[name].opaque_typ == "constant"


def opaque_type_info(cls: type[Any] | str) -> OpaqueTypeInfo | None:
    """What is registered about a type, or nothing if it is not registered."""

    name = cls if isinstance(cls, str) else getattr(cls, "__name__", None)
    if name is None:
        return None
    return _OPAQUE_TYPES_BY_NAME.get(name)


def get_opaque_obj_repr(obj: Any) -> tuple[str, dict[str, type]]:
    """How a value is written into generated source, and what that text needs.

    Returns the text and the names it mentions, so that a caller emitting the
    text can also emit whatever has to be in scope for it to evaluate.  A value
    that cannot write itself has no honest spelling here, so this raises rather
    than falling back to ``repr``: a fallback would put text in the generated
    source that names an object the source cannot rebuild, and the failure would
    surface as a name error far from its cause.
    """

    # An enumeration member is written as the member rather than as a
    # constructed value, so the generated source says which member it is
    # instead of rebuilding it from its value.
    if isinstance(obj, Enum):
        cls = type(obj)
        return f"{cls.__name__}.{obj.name}", {cls.__name__: cls}

    writer = getattr(obj, "__fx_repr__", None)
    if writer is None:
        raise TypeError(
            f"a value of type {type(obj).__name__} is expected to write itself "
            "into generated source, and does not: it has no method that returns "
            "the text and the names that text mentions. Add one, or register "
            "the type with a spelling of its own."
        )
    repr_str, globals_dict = writer()
    return repr_str, dict(globals_dict)


__all__ = [
    "OpaqueTypeInfo",
    "get_opaque_obj_repr",
    "is_opaque_constant_type",
    "is_opaque_type",
    "opaque_type_info",
    "register_opaque_type",
]
