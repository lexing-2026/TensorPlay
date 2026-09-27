"""Generic container (pytree) flattening.

A *pytree* is a nested structure built out of containers -- lists, tuples,
dicts, namedtuples, and anything registered here -- whose non-container
leaves are the values a caller cares about.  Flattening a pytree yields its
leaves in a fixed order plus a :class:`TreeSpec` describing the structure, so
a transform can operate on the leaves and rebuild the original shape
afterwards.

This is the general-purpose implementation.  Consumers that need it (the
function transforms, tracing, serialization) build on the API here rather
than growing their own flatteners.
"""
from collections import OrderedDict, defaultdict, deque, namedtuple
from dataclasses import dataclass, field
from collections.abc import Iterator
from typing import Any, Callable, Iterable, Optional, TypeVar, Union

__all__ = [
    "Context",
    "FlattenFn",
    "GetAttrKey",
    "LeafSpec",
    "PyTree",
    "SUPPORTED_NODES",
    "TreeSpec",
    "UnflattenFn",
    "register_pytree_node",
    "tree_all",
    "tree_all_only",
    "tree_any",
    "tree_any_only",
    "flatten_up_to",
    "tree_flatten",
    "tree_leaves",
    "tree_map",
    "tree_map_",
    "tree_map_only",
    "tree_map_only_",
    "tree_structure",
    "tree_unflatten",
    "treespec_pprint",
]

PyTree = Any
Context = Any
_T = TypeVar("_T")

FlattenFn = Callable[[PyTree], tuple[list[Any], Context]]
UnflattenFn = Callable[[Iterable[Any], Context], PyTree]
FlattenWithKeysFn = Callable[[PyTree], tuple[list[Any], Context]]


@dataclass(frozen=True)
class GetAttrKey:
    """Identifies one attribute of a flattened container object."""

    name: str

    def __str__(self) -> str:
        return f".{self.name}"

    def get(self, obj: Any) -> Any:
        return getattr(obj, self.name)


@dataclass(frozen=True)
class SequenceKey:
    """Identifies one position of a flattened sequence."""

    idx: int

    def __str__(self) -> str:
        return f"[{self.idx!r}]"

    def get(self, sequence):
        return sequence[self.idx]


@dataclass(frozen=True)
class NodeDef:
    """How one registered container type is taken apart and put back together."""

    type: type[Any]
    flatten_fn: FlattenFn
    unflatten_fn: UnflattenFn
    serialized_type_name: Optional[str] = None
    flatten_with_keys_fn: Optional[FlattenWithKeysFn] = None
    to_dumpable_context: Optional[Any] = None
    from_dumpable_context: Optional[Any] = None


SUPPORTED_NODES: dict[type[Any], NodeDef] = {}


def register_pytree_node(
    cls: type[Any],
    flatten_fn: FlattenFn,
    unflatten_fn: UnflattenFn,
    *,
    serialized_type_name: Optional[str] = None,
    flatten_with_keys_fn: Optional[FlattenWithKeysFn] = None,
    to_dumpable_context: Optional[Any] = None,
    from_dumpable_context: Optional[Any] = None,
) -> None:
    """Registers ``cls`` as a container type.

    ``flatten_fn`` returns ``(children, context)``; ``unflatten_fn`` takes
    those back and rebuilds an instance.  The context carries whatever the
    children do not -- dict keys, a namedtuple's class, a defaultdict's
    factory.  ``serialized_type_name`` is the fully qualified name recorded
    when the tree spec is serialized; it must uniquely identify ``cls``.
    ``flatten_with_keys_fn`` optionally supplies a key-aware flatten; the
    dumpable-context pair optionalizes the spec serialization hooks.
    """
    if cls in SUPPORTED_NODES:
        raise ValueError(f"{cls} is already registered as a pytree node type")
    SUPPORTED_NODES[cls] = NodeDef(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name,
        flatten_with_keys_fn,
        to_dumpable_context,
        from_dumpable_context,
    )


def _deregister_pytree_node(cls: type[Any]) -> None:
    SUPPORTED_NODES.pop(cls, None)


# ---------------------------------------------------------------------------
# Tree specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TreeSpec:
    """The shape of a flattened pytree: enough to rebuild it from its leaves."""

    type: Any
    context: Context = None
    children_specs: tuple["TreeSpec", ...] = ()
    num_leaves: int = field(default=0, compare=False)

    def __post_init__(self) -> None:
        count = sum(child.num_leaves for child in self.children_specs)
        object.__setattr__(self, "num_leaves", count)

    def is_leaf(self) -> bool:
        return self.type is None

    def __repr__(self) -> str:
        return treespec_pprint(self)


class LeafSpec(TreeSpec):
    """The spec of a single leaf."""

    def __init__(self) -> None:
        super().__init__(None, None, ())
        object.__setattr__(self, "num_leaves", 1)

    def __repr__(self) -> str:
        return "*"


_LEAF_SPEC = LeafSpec()


def treespec_pprint(spec: TreeSpec) -> str:
    """Compact human-readable rendering of a spec, for error messages."""
    if spec.is_leaf():
        return "*"
    children = ", ".join(treespec_pprint(child) for child in spec.children_specs)
    name = getattr(spec.type, "__name__", str(spec.type))
    if spec.context is None:
        return f"{name}({children})"
    return f"{name}({spec.context}: {children})"


# ---------------------------------------------------------------------------
# Flatten / unflatten
# ---------------------------------------------------------------------------

def _get_node_type(tree: PyTree) -> Any:
    node_type = type(tree)
    if node_type in SUPPORTED_NODES:
        return node_type
    if _is_namedtuple_instance(tree):
        return namedtuple
    return None


def _is_namedtuple_instance(tree: PyTree) -> bool:
    typ = type(tree)
    if not issubclass(typ, tuple):
        return False
    fields = getattr(typ, "_fields", None)
    return isinstance(fields, tuple) and all(isinstance(f, str) for f in fields)


def tree_is_leaf(
    tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = None
) -> bool:
    """True when ``tree`` should be treated as a leaf rather than descended into."""
    if is_leaf is not None and is_leaf(tree):
        return True
    return _get_node_type(tree) is None


def tree_flatten(
    tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = None
) -> tuple[list[Any], TreeSpec]:
    """Returns the leaves of ``tree`` in order plus the spec to rebuild it."""
    if tree_is_leaf(tree, is_leaf):
        return [tree], _LEAF_SPEC

    node_type = _get_node_type(tree)
    children, context = SUPPORTED_NODES[node_type].flatten_fn(tree)

    leaves: list[Any] = []
    child_specs: list[TreeSpec] = []
    for child in children:
        child_leaves, child_spec = tree_flatten(child, is_leaf)
        leaves.extend(child_leaves)
        child_specs.append(child_spec)
    return leaves, TreeSpec(node_type, context, tuple(child_specs))


def tree_unflatten(leaves: Iterable[Any], spec: TreeSpec) -> PyTree:
    """Rebuilds the pytree ``spec`` describes from ``leaves``."""
    if not isinstance(spec, TreeSpec):
        raise TypeError(f"tree_unflatten: expected a TreeSpec, got {type(spec)}")
    leaves = list(leaves)
    if len(leaves) != spec.num_leaves:
        raise ValueError(
            f"tree_unflatten: expected {spec.num_leaves} leaves for spec "
            f"{treespec_pprint(spec)}, got {len(leaves)}"
        )
    return _unflatten(iter(leaves), spec)


def _unflatten(leaves: Any, spec: TreeSpec) -> PyTree:
    if spec.is_leaf():
        return next(leaves)
    children = [_unflatten(leaves, child) for child in spec.children_specs]
    return SUPPORTED_NODES[spec.type].unflatten_fn(children, spec.context)


def tree_iter(
    tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = None
) -> Iterator[Any]:
    """The leaves of ``tree``, one at a time.

    Same order as :func:`tree_flatten`, without building the list of them or
    the description of how to put them back -- which is what a caller that
    only wants to look at each one in turn should not pay for.
    """

    if tree_is_leaf(tree, is_leaf):
        yield tree
        return
    for child in _children_of(tree):
        yield from tree_iter(child, is_leaf)


def _children_of(tree: PyTree) -> list[Any]:
    """The sub-structures of a container, in the order they are flattened.

    Taken from the same place :func:`tree_flatten` takes them, so that a walk
    over the leaves of a tree and a flatten of it agree on what the leaves
    are and in what order -- which is the whole of what this is for.
    """

    flatten_fn = SUPPORTED_NODES[_get_node_type(tree)].flatten_fn
    return list(flatten_fn(tree)[0])


def tree_leaves(
    tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = None
) -> list[Any]:
    """The leaves of ``tree``, in flatten order."""
    return tree_flatten(tree, is_leaf)[0]


def arg_tree_leaves(*args: PyTree, **kwargs: PyTree) -> list[Any]:
    """The leaves of a call's arguments, in the order they were passed.

    Same answer as ``tree_leaves((args, kwargs))``, reached without building
    the pair and taking it apart again -- which is the whole of the
    difference, and is worth having on the paths that ask this question about
    every call they look at.
    """

    leaves: list[Any] = []
    for value in args:
        leaves.extend(tree_flatten(value)[0])
    for value in kwargs.values():
        leaves.extend(tree_flatten(value)[0])
    return leaves


def tree_structure(
    tree: PyTree, is_leaf: Optional[Callable[[PyTree], bool]] = None
) -> TreeSpec:
    """The spec of ``tree``, discarding the leaves."""
    return tree_flatten(tree, is_leaf)[1]


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

def flatten_up_to(tree: PyTree, spec: TreeSpec) -> list[Any]:
    """Flattens ``tree`` only as deep as ``spec``.

    Where ``spec`` has a leaf, the whole corresponding subtree of ``tree`` is
    taken as one element.  This is what makes a shallower first argument work
    as a prefix: ``tree_map(f, 0, nested)`` calls ``f`` once per leaf of
    ``spec`` with the matching subtree.
    """
    out: list[Any] = []

    def helper(subtree: PyTree, subspec: TreeSpec) -> None:
        if subspec.is_leaf():
            out.append(subtree)
            return
        node_type = _get_node_type(subtree)
        if node_type is not subspec.type:
            raise ValueError(
                f"expected a subtree of type {subspec.type}, got {type(subtree)}"
            )
        children, context = SUPPORTED_NODES[node_type].flatten_fn(subtree)
        if context != subspec.context:
            raise ValueError(
                f"expected subtree context {subspec.context}, got {context}"
            )
        if len(children) != len(subspec.children_specs):
            raise ValueError(
                f"expected {len(subspec.children_specs)} children, got {len(children)}"
            )
        for child, child_spec in zip(children, subspec.children_specs):
            helper(child, child_spec)

    helper(tree, spec)
    return out


def _flatten_up_to(rests: tuple[PyTree, ...], spec: TreeSpec) -> list[list[Any]]:
    return [flatten_up_to(rest, spec) for rest in rests]


def tree_map(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    """Applies ``func`` leafwise across ``tree`` and any same-shaped ``rests``."""
    leaves, spec = tree_flatten(tree, is_leaf)
    if not rests:
        return tree_unflatten([func(leaf) for leaf in leaves], spec)
    rest_leaves = _flatten_up_to(rests, spec)
    return tree_unflatten(
        [func(*args) for args in zip(leaves, *rest_leaves)], spec
    )


def tree_map_(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    """Like :func:`tree_map` but for side effects: returns ``tree`` unchanged."""
    leaves, spec = tree_flatten(tree, is_leaf)
    if not rests:
        for leaf in leaves:
            func(leaf)
        return tree
    rest_leaves = _flatten_up_to(rests, spec)
    for args in zip(leaves, *rest_leaves):
        func(*args)
    return tree


def _type_predicate(type_or_types: Any) -> Callable[[Any], bool]:
    if callable(type_or_types) and not isinstance(type_or_types, (type, tuple)):
        return type_or_types
    return lambda x: isinstance(x, type_or_types)


def tree_map_only(
    type_or_types: Any,
    func: Callable[[Any], Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    """Maps ``func`` over the leaves matching ``type_or_types``, leaving the rest."""
    matches = _type_predicate(type_or_types)
    return tree_map(
        lambda x, *r: func(x, *r) if matches(x) else x, tree, *rests, is_leaf=is_leaf
    )


def tree_map_only_(
    type_or_types: Any,
    func: Callable[[Any], Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    """In-place counterpart of :func:`tree_map_only`."""
    matches = _type_predicate(type_or_types)

    def wrapped(x, *r):
        if matches(x):
            func(x, *r)

    return tree_map_(wrapped, tree, *rests, is_leaf=is_leaf)


def tree_all(
    pred: Callable[[Any], bool],
    tree: PyTree,
    *,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    return all(pred(leaf) for leaf in tree_leaves(tree, is_leaf))


def tree_any(
    pred: Callable[[Any], bool],
    tree: PyTree,
    *,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    return any(pred(leaf) for leaf in tree_leaves(tree, is_leaf))


def tree_all_only(
    type_or_types: Any,
    pred: Callable[[Any], bool],
    tree: PyTree,
    *,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    matches = _type_predicate(type_or_types)
    return all(pred(x) for x in tree_leaves(tree, is_leaf) if matches(x))


def tree_any_only(
    type_or_types: Any,
    pred: Callable[[Any], bool],
    tree: PyTree,
    *,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> bool:
    matches = _type_predicate(type_or_types)
    return any(pred(x) for x in tree_leaves(tree, is_leaf) if matches(x))


def _broadcast_to_and_flatten(
    tree: PyTree,
    spec: TreeSpec,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> Optional[list[Any]]:
    """Expands a prefix tree to ``spec``'s leaf count, or None if incompatible.

    ``tree`` may stop short of ``spec``: wherever it has a leaf, that leaf is
    repeated once per leaf of the corresponding subtree.  This is what lets
    ``in_dims=0`` stand in for a whole nested argument structure.
    """
    full_tree = tree_unflatten([0] * spec.num_leaves, spec)
    result: list[Any] = []

    def add_leaves(value: Any, subtree: PyTree) -> None:
        result.extend([value] * tree_structure(subtree, is_leaf).num_leaves)

    try:
        tree_map_(add_leaves, tree, full_tree, is_leaf=is_leaf)
    except ValueError:
        return None
    return result


# ---------------------------------------------------------------------------
# Built-in container types
# ---------------------------------------------------------------------------

def _tuple_flatten(d):
    return list(d), None


def _tuple_unflatten(values, context):
    return tuple(values)


def _list_flatten(d):
    return list(d), None


def _list_unflatten(values, context):
    return list(values)


def _dict_flatten(d):
    return list(d.values()), list(d.keys())


class MappingKey:
    """A dictionary key carried inside a key path.

    Wraps the key so flattened-with-keys results stay addressable through
    ``get`` regardless of whether the key is hashable-stable across trees.
    """

    __slots__ = ("key",)

    def __init__(self, key):
        self.key = key

    def __eq__(self, other):
        return isinstance(other, MappingKey) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __str__(self):
        return f"[{self.key!r}]"

    def get(self, mapping):
        return mapping[self.key]


def key_get(obj, kp):
    """Resolve a key path (a sequence of key entries) against an object."""
    for k in kp:
        obj = k.get(obj)
    return obj


def _dict_flatten_with_keys(d):
    values, context = _dict_flatten(d)
    return [(MappingKey(k), v) for k, v in zip(context, values)], context


def _dict_unflatten(values, context):
    return dict(zip(context, values))


def _ordereddict_flatten(d):
    return list(d.values()), list(d.keys())


def _ordereddict_unflatten(values, context):
    return OrderedDict(zip(context, values))


def _defaultdict_flatten(d):
    return list(d.values()), (d.default_factory, list(d.keys()))


def _defaultdict_unflatten(values, context):
    default_factory, keys = context
    return defaultdict(default_factory, zip(keys, values))


def _namedtuple_flatten(d):
    return list(d), type(d)


def _namedtuple_unflatten(values, context):
    return context(*values)


def _deque_flatten(d):
    return list(d), d.maxlen


def _deque_unflatten(values, context):
    return deque(values, maxlen=context)


def _tuple_flatten_with_keys(d):
    values, context = _tuple_flatten(d)
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


def _list_flatten_with_keys(d):
    values, context = _list_flatten(d)
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


def _dict_flatten_with_keys(d):
    values, context = _dict_flatten(d)
    return [(MappingKey(k), v) for k, v in zip(context, values)], context


def _namedtuple_flatten_with_keys(d):
    values, context = _namedtuple_flatten(d)
    # The context carries the class rather than the field names, so the names
    # come off the value; they are the keys that address its fields.
    return [(GetAttrKey(f), v) for f, v in zip(d._fields, values)], context


def _ordereddict_flatten_with_keys(d):
    values, context = _ordereddict_flatten(d)
    return [(MappingKey(k), v) for k, v in zip(context, values)], context


def _defaultdict_flatten_with_keys(d):
    values, context = _defaultdict_flatten(d)
    _, dict_context = context
    return [(MappingKey(k), v) for k, v in zip(dict_context, values)], context


def _deque_flatten_with_keys(d):
    values, context = _deque_flatten(d)
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


register_pytree_node(tuple, _tuple_flatten, _tuple_unflatten,
                    flatten_with_keys_fn=_tuple_flatten_with_keys)
register_pytree_node(list, _list_flatten, _list_unflatten,
                    flatten_with_keys_fn=_list_flatten_with_keys)
register_pytree_node(dict, _dict_flatten, _dict_unflatten,
                    flatten_with_keys_fn=_dict_flatten_with_keys)
register_pytree_node(OrderedDict, _ordereddict_flatten, _ordereddict_unflatten,
                    flatten_with_keys_fn=_ordereddict_flatten_with_keys)
register_pytree_node(defaultdict, _defaultdict_flatten, _defaultdict_unflatten,
                    flatten_with_keys_fn=_defaultdict_flatten_with_keys)
register_pytree_node(deque, _deque_flatten, _deque_unflatten,
                    flatten_with_keys_fn=_deque_flatten_with_keys)
register_pytree_node(namedtuple, _namedtuple_flatten, _namedtuple_unflatten,
                    flatten_with_keys_fn=_namedtuple_flatten_with_keys)


def keystr(kp) -> str:
    """Given a key path, return a pretty-printed representation."""
    return "".join([str(k) for k in kp])


def _generate_key_paths(
    key_path: tuple,
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
):
    """Each leaf of a tree, with the keys that lead to it.

    A tree flattened without keys says what the leaves are but not where
    each one was, and "where" is what a reader comparing two trees needs: a
    difference is only useful if it names the place it happened.
    """

    if is_leaf and is_leaf(tree):
        yield key_path, tree
        return

    node_type = _get_node_type(tree)
    handler = SUPPORTED_NODES.get(node_type)
    if not handler:
        # This is a leaf
        yield key_path, tree
        return

    flatten_with_keys = handler.flatten_with_keys_fn
    if flatten_with_keys:
        key_children, _ = flatten_with_keys(tree)
        for k, c in key_children:
            yield from _generate_key_paths((*key_path, k), c, is_leaf)
    else:
        # We registered this pytree but didn't add a flatten_with_keys_fn, complain.
        raise ValueError(
            f"Did not find a flatten_with_keys_fn for type: {node_type}. "
            "Please pass a flatten_with_keys_fn argument to register_pytree_node."
        )


def tree_flatten_with_path(
    tree: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> tuple[list, TreeSpec]:
    """Flattens a pytree like :func:`tree_flatten`, but also returns each leaf's key path."""

    _, treespec = tree_flatten(tree, is_leaf)
    return list(_generate_key_paths((), tree, is_leaf)), treespec


def tree_map_with_path(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> PyTree:
    """Like :func:`tree_map`, but the provided callable takes an additional key path argument.

    The callable is given the keys that lead to the leaf before the leaf
    itself, so that a value it produces can be labelled with where it came
    from rather than only with what it is.
    """

    keypath_leaves, treespec = tree_flatten_with_path(tree, is_leaf)
    keypath_leaves = list(zip(*keypath_leaves, strict=True))
    # The rests are documented to have this tree's structure, so their leaves
    # line up with its leaves by position; the tree is flattened only to find
    # out how many there are.
    all_keypath_leaves = keypath_leaves + [tree_flatten(r)[0] for r in rests]
    return tree_unflatten(
        (func(*xs) for xs in zip(*all_keypath_leaves, strict=True)), treespec
    )
