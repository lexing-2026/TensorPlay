from collections import defaultdict

from tensorplay.graph import Graph, Node, map_arg
from tensorplay.graph.experimental.sympy_functions import OrderedSet


class BitsetAncestors:
    """Precomputed transitive ancestor sets using Python arbitrary-precision ints.

    Each node gets an index i (0..N-1) from topological order. A node's
    ancestor set is a single Python ``int`` where bit j is set iff node j
    is a transitive ancestor. Python ``int`` is arbitrary-precision, so for
    N nodes each int is ~N/8 bytes internally (~N/64 machine words).

    The key advantage over ``dict[Node, OrderedSet[Node]]`` is that
    CPython implements ``int |= int`` as a C-level loop over machine words,
    making the transitive closure O(N^2 / 64) instead of O(N^2) in
    Python-level set operations.

    Example -- diamond graph (edges point downward)::

            a          index 0
           / \\
          b   c        index 1, 2
           \\ /
            d          index 3

    Ancestor bitsets::

        a: 0b0000 = 0   (no ancestors)
        b: 0b0001 = 1   (bit 0 -> a)
        c: 0b0001 = 1   (bit 0 -> a)
        d: 0b0111 = 7   (bits 0,1,2 -> a, b, c)

    How d's bitset is built (d has parents b and c)::

        b = 0
        # parent b (idx=1): set bit 1, merge b's ancestors
        b |= (1 << 1) | bits[1]   # b = 0b0010 | 0b0001 = 0b0011
        # parent c (idx=2): set bit 2, merge c's ancestors
        b |= (1 << 2) | bits[2]   # b = 0b0011 | 0b0100 | 0b0001 = 0b0111
        bits[3] = 0b0111

    Querying::

        ancestors = BitsetAncestors(nodes)
        ancestors.is_ancestor(a, d)    # (bits[3] >> 0) & 1 -> True
        ancestors.has_dep(a, d)  # True (a is ancestor of d)
        list(ancestors.iter_ancestors(d))  # [a, b, c] via bit-scan

    Iteration uses the x & -x trick to isolate the lowest set bit::

        bits = 0b0111
        bits & -bits = 0b0001  -> idx 0 -> yield a, clear: 0b0110
        bits & -bits = 0b0010  -> idx 1 -> yield b, clear: 0b0100
        bits & -bits = 0b0100  -> idx 2 -> yield c, clear: 0b0000

    Args:
        nodes: topologically sorted list of FX nodes.
        extra_inputs: optional additional edges beyond the FX graph
            (e.g. hiding-interval deps in the overlap bucketer).
    """

    def __init__(
        self,
        nodes: list[Node],
        extra_inputs: dict[Node, OrderedSet[Node]] | None = None,
    ):
        n = len(nodes)
        node_to_idx: dict[Node, int] = {nd: i for i, nd in enumerate(nodes)}
        bits = [0] * n
        extra = extra_inputs or {}

        for i, node in enumerate(nodes):
            b = 0
            for inp in node._input_nodes:
                j = node_to_idx.get(inp)
                if j is not None:
                    b |= (1 << j) | bits[j]
            for inp in extra.get(node, ()):
                j = node_to_idx.get(inp)
                if j is not None:
                    b |= (1 << j) | bits[j]
            bits[i] = b

        self._bits = bits
        self._node_to_idx = node_to_idx
        self._idx_to_node = nodes

    def is_ancestor(self, ancestor: Node, descendant: Node) -> bool:
        """O(1) test: is ``ancestor`` a transitive ancestor of ``descendant``?"""
        anc_idx = self._node_to_idx.get(ancestor)
        desc_idx = self._node_to_idx.get(descendant)
        if anc_idx is None or desc_idx is None:
            return False
        return bool((self._bits[desc_idx] >> anc_idx) & 1)

    def has_dep(self, n1: Node, n2: Node) -> bool:
        """Check if either node is an ancestor of the other."""
        return self.is_ancestor(n1, n2) or self.is_ancestor(n2, n1)

    def get_ancestor_bits(self, node: Node) -> int:
        """Return the raw ancestor bitset for ``node``."""
        return self._bits[self._node_to_idx[node]]

    def node_bit(self, node: Node) -> int:
        """Return the single-bit mask ``1 << idx`` for ``node``."""
        return 1 << self._node_to_idx[node]

    def ancestors_intersect(self, node: Node, mask: int) -> bool:
        """Check if any ancestor of ``node`` is set in ``mask``."""
        idx = self._node_to_idx.get(node)
        if idx is None:
            return False
        return bool(self._bits[idx] & mask)

    def iter_ancestors(self, node: Node):
        """Yield all ancestors of ``node`` via lowest-bit scan."""
        bits = self._bits[self._node_to_idx[node]]
        idx_to_node = self._idx_to_node
        while bits:
            idx = (bits & -bits).bit_length() - 1
            yield idx_to_node[idx]
            bits &= bits - 1
from collections import defaultdict


# ---------------------------------------------------------------------------
# ordering a span of a graph
# ---------------------------------------------------------------------------


def _get_flat_args(
    node: Node, node_to_additional_deps: dict[Node, "OrderedSet[Node]"]
) -> list[Node]:
    """The nodes a node reads, including any read only for ordering."""
    args: list[Node] = []
    map_arg((node.args, node.kwargs), args.append)
    if node in node_to_additional_deps:
        args.extend(node_to_additional_deps[node])
    return args


def _get_flat_args_unique(
    node: Node, node_to_additional_deps: dict[Node, "OrderedSet[Node]"]
) -> "OrderedSet[Node]":
    """The nodes a node reads, once each, including any read only for ordering."""
    args: OrderedSet[Node] = OrderedSet()
    map_arg((node.args, node.kwargs), args.add)
    if node in node_to_additional_deps:
        args.update(node_to_additional_deps[node])
    return args


def _get_flat_args(
    node: Node, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> list[Node]:
    args = list[Any]()
    map_arg((node.args, node.kwargs), args.append)
    if node in node_to_additional_deps:
        args.extend(node_to_additional_deps[node])
    return args


def _get_flat_args_unique(
    node: Node, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> OrderedSet[Node]:
    args = OrderedSet[Node]()
    map_arg((node.args, node.kwargs), args.add)
    if node in node_to_additional_deps:
        args.update(node_to_additional_deps[node])
    return args


def _stable_topological_sort_impl(
    graph: Graph,
    node_to_additional_deps: dict[Node, OrderedSet[Node]],
    do_sort: bool = True,
    region: OrderedSet[Node] | None = None,
) -> bool:
    # Nodes are in exactly one of these four collections:

    # - Nodes in `pending` are waiting to be processed (in reverse order):
    pending = list(reversed(region or graph.nodes))

    # - Nodes in `ready` have been processed and are already in the correct
    #   order.  When sorting a region, nodes outside the region are
    #   implicitly ready (filtered out in the waiting_for check below).
    ready: set[Node] = set()

    # - `waiting` is a mapping from a dependency to nodes which depend on that
    #   dependency.
    waiting = defaultdict(list)

    # - `outputs` are always at the end of the graph
    outputs = OrderedSet[Node]()

    has_additional_deps = bool(node_to_additional_deps)

    # The cursor indicates the last processed node so we can add new nodes
    # after it.
    cursor = None
    while pending:
        node = pending.pop()

        if node.op == "output":
            outputs.add(node)
            if node.users:
                raise AssertionError("output nodes should have no users")
            continue

        # node._input_nodes is maintained by FX and already contains the
        # unique set of input nodes — avoid rebuilding it via map_arg.
        if has_additional_deps:
            deps = _get_flat_args_unique(node, node_to_additional_deps)
        else:
            deps = node._input_nodes

        last_unready = None
        for x in deps:
            if x not in ready and (region is None or x in region):
                last_unready = x
        if last_unready is not None:
            # We have unprocessed input nodes. Wait for the last unready
            # arg so an already sorted list will only recheck this node once.
            waiting[last_unready].append(node)
        else:
            ready.add(node)
            if cursor and cursor.next is not node and do_sort:
                cursor.append(node)
            cursor = node
            # Mark the nodes that have been waiting for this node to finish as
            # ready to check again.
            pending.extend(reversed(waiting.pop(node, ())))

    ready.update(outputs)
    expected_len = len(region) if region is not None else len(graph.nodes)
    return not waiting and len(ready) == expected_len


def _stable_topological_sort_region(
    graph: Graph,
    region: OrderedSet[Node],
) -> None:
    if not _stable_topological_sort_impl(graph, {}, region=region):
        raise AssertionError("stable topological sort of region failed")


def _has_cycle(
    graph: Graph,
    node_to_additional_deps: dict[Node, OrderedSet[Node]],
) -> bool:
    return not _stable_topological_sort_impl(
        graph, node_to_additional_deps, do_sort=False
    )
