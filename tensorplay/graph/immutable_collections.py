"""Immutable containers used for graph argument structures."""

from __future__ import annotations

from typing import Any, NoReturn, TypeVar

from ._compatibility import compatibility
from ._pytree import (
    _list_flatten_spec,
    _list_flatten_spec_exact_match,
    register_pytree_flatten_spec,
    register_pytree_node,
)

__all__ = ["immutable_dict", "immutable_list"]

_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def _no_mutation(self: Any, *args: Any, **kwargs: Any) -> NoReturn:
    del self, args, kwargs
    raise TypeError("immutable graph argument containers do not support mutation")


@compatibility(is_backward_compatible=True)
class immutable_list(list[_T]):
    """List-shaped container whose contents cannot be changed in place."""

    __delitem__ = _no_mutation
    __iadd__ = _no_mutation
    __imul__ = _no_mutation
    __setitem__ = _no_mutation
    append = _no_mutation
    clear = _no_mutation
    extend = _no_mutation
    insert = _no_mutation
    pop = _no_mutation
    remove = _no_mutation
    reverse = _no_mutation
    sort = _no_mutation

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __reduce__(self):
        return type(self), (tuple(self),)


@compatibility(is_backward_compatible=True)
class immutable_dict(dict[_KT, _VT]):
    """Dict-shaped container whose entries cannot be changed in place."""

    __delitem__ = _no_mutation
    __ior__ = _no_mutation
    __setitem__ = _no_mutation
    clear = _no_mutation
    pop = _no_mutation
    popitem = _no_mutation
    setdefault = _no_mutation
    update = _no_mutation

    def __hash__(self) -> int:
        return hash(frozenset(self.items()))

    def __reduce__(self):
        return type(self), (tuple(self.items()),)


def _immutable_list_flatten(value: immutable_list[_T]):
    return list(value), None


def _immutable_list_unflatten(values, context):
    del context
    return immutable_list(values)


def _immutable_dict_flatten(value: immutable_dict[_KT, _VT]):
    return list(value.values()), tuple(value.keys())


def _immutable_dict_unflatten(values, context):
    return immutable_dict(zip(context, values))


register_pytree_node(immutable_list, _immutable_list_flatten, _immutable_list_unflatten)
register_pytree_node(immutable_dict, _immutable_dict_flatten, _immutable_dict_unflatten)
register_pytree_flatten_spec(
    immutable_list, _list_flatten_spec, _list_flatten_spec_exact_match
)
register_pytree_flatten_spec(
    immutable_dict,
    lambda value, spec: [value[key] for key in spec.context],
    lambda value, spec: len(value) == spec.num_children
    and tuple(value) == tuple(spec.context),
)
