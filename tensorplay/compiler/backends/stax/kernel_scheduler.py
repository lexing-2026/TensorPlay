"""Group computed buffers into kernels.

Every computed buffer is one loop nest over ``(xnumel, rnumel)`` (rnumel is
1 for pointwise work).  Two nests can share a kernel when their iteration
spaces line up:

* two reductions over the same ``(xnumel, rnumel)``;
* two pointwise nests over the same number of elements;
* a pointwise nest and a reduction when the pointwise covers the whole
  ``xnumel * rnumel`` space (it runs inside the reduction loops) or just
  ``xnumel`` (it runs on the reduced values).

A consumer may only join its producer when every element it reads from the
producer is one the same program has already computed: the element at the
very position it is storing, or a reduction result of its own row.  Those
reads are then served from registers, and a buffer read by no one else is
never written to memory at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .loops import (
    Buffer,
    ComputedBuffer,
    Const,
    ExternKernel,
    ExternOutput,
    Pointwise,
    Reduction,
    ReinterpretView,
    Symbol,
    Value,
    as_index,
    dtype_name,
    floordiv,
    free_symbols,
    iter_values,
    modular_indexing,
    prod,
    record_body,
    set_graph,
    simplify_index,
    simplify_loops,
    substitute,
)

XINDEX = Symbol("xindex")
RINDEX = Symbol("rindex")

#: Nodes one kernel may hold before fusion stops growing it.
MAX_FUSION_SIZE = 64

_ITEMSIZE = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "int64": 8, "int32": 4, "bool": 1, "uint8": 1, "int8": 1}


@dataclass
class LoopNode:
    """One computed buffer (or one welford nest with its sibling outputs)."""

    index: int
    buffers: list  # ComputedBuffer(s) this nest writes
    body: Any  # LoopBody
    is_reduction: bool
    xnumel: int
    rnumel: int
    reads: set = field(default_factory=set)

    @property
    def names(self):
        return [b.name for b in self.buffers]

    @property
    def data(self):
        return self.buffers[0].data


@dataclass
class ExternNode:
    index: int
    kernel: ExternKernel
    reads: set = field(default_factory=set)

    @property
    def names(self):
        return [o.name for o in self.kernel.outputs]


@dataclass
class FusedGroup:
    nodes: list  # LoopNode in dependency order
    xnumel: int
    rnumel: int

    @property
    def is_reduction(self):
        return self.rnumel > 1

    @property
    def names(self):
        return [n for node in self.nodes for n in node.names]

    @property
    def reads(self):
        written = set(self.names)
        return {r for node in self.nodes for r in node.reads if r not in written}


# ---------------------------------------------------------------------------
# iteration-space mapping
# ---------------------------------------------------------------------------


def _split_dims(sizes, boundary):
    """Split dims so a prefix multiplies to ``boundary``; None if impossible.

    Returns (x_parts, r_parts, substitution) where the parts are
    (symbol, extent) lists and ``substitution`` rewrites the original dim
    symbols (one dim may be split into an x-part and an r-part).
    """

    x_parts, r_parts, subst = [], [], {}
    acc = 1
    for sym, extent in sizes:
        if acc == boundary:
            r_parts.append((sym, extent))
            continue
        if acc * extent <= boundary and boundary % (acc * extent) == 0:
            x_parts.append((sym, extent))
            acc *= extent
            continue
        need = boundary // acc
        if boundary % acc or extent % need:
            return None
        inner = extent // need
        hi = Symbol(f"{sym.name}h")
        lo = Symbol(f"{sym.name}l")
        subst[sym] = hi * inner + lo
        x_parts.append((hi, need))
        r_parts.append((lo, inner))
        acc = boundary
    if acc != boundary:
        return None
    return x_parts, r_parts, subst


def _dim_exprs(parts, flat):
    """Express each (symbol, extent) of ``parts`` through the flat index."""

    out = {}
    stride = prod(extent for _, extent in parts)
    first = True
    for sym, extent in parts:
        stride //= max(extent, 1)
        if extent == 1:
            out[sym] = Const(0)
        elif first:
            out[sym] = floordiv(flat, Const(stride)) if stride != 1 else flat
        else:
            out[sym] = modular_indexing(flat, stride, extent)
        first = False if extent != 1 else first
    return out


class NodePlacement:
    """How one loop node's variables map onto a kernel's (xindex, rindex)."""

    def __init__(self, node: LoopNode, xnumel: int, rnumel: int):
        self.node = node
        body = node.body
        self.full = False  # runs inside the reduction loops
        if node.is_reduction:
            xs = list(zip(body.vars, body.sizes))
            rs = list(zip(body.rvars, body.rsizes))
            subst = {}
            self.full = True
        else:
            sizes = list(zip(body.vars, body.sizes))
            numel = prod(body.sizes)
            if rnumel > 1 and numel == xnumel * rnumel:
                split = _split_dims(sizes, xnumel)
                if split is None:
                    raise _NoFit()
                xs, rs, subst = split
                self.full = True
            elif numel == xnumel:
                xs, rs, subst = sizes, [], {}
            else:
                raise _NoFit()
        mapping = {}
        mapping.update(_dim_exprs(xs, XINDEX))
        mapping.update(_dim_exprs(rs, RINDEX))
        # A split dim is rewritten into its two parts first, then every part
        # (and every unsplit dim) into the kernel's flat indices.
        self.split = dict(subst)
        self.mapping = mapping
        self.ranges = {XINDEX: xnumel, RINDEX: max(rnumel, 1)}

    def to_kernel(self, expr):
        expr = as_index(expr)
        if self.split:
            expr = substitute(expr, self.split)
        return simplify_index(substitute(expr, self.mapping), self.ranges)

    def store_index(self, buffer: Buffer):
        return self.to_kernel(self.node.body.stores[buffer.name])


class _NoFit(Exception):
    pass


def placement(node: LoopNode, xnumel: int, rnumel: int) -> NodePlacement | None:
    try:
        return NodePlacement(node, xnumel, rnumel)
    except _NoFit:
        return None


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------


class KernelScheduler:
    def __init__(self, graph):
        self.graph = graph
        self.nodes: list = []
        self.producer: dict[str, Any] = {}
        with set_graph(graph):
            self._build()

    def _build(self):
        for position, op in enumerate(self.graph.operations):
            if isinstance(op, ExternKernel):
                node = ExternNode(position, op, {b.name for b in op.input_buffers()})
            else:
                body = record_body(op.data)
                data = op.data
                is_red = isinstance(data, Reduction)
                buffers = [op] + list(op.welford_siblings)
                for b in buffers:
                    body.stores[b.name] = b.layout.indexer(body.vars)
                body = simplify_loops(body)
                node = LoopNode(
                    position, buffers, body, is_red,
                    prod(body.sizes), prod(body.rsizes) if is_red else 1,
                    {v.args[0] for v in body.loads},
                )
            self.nodes.append(node)
            for name in node.names:
                self.producer[name] = node
        self.output_names = set()
        for out in self.graph.graph_outputs:
            if isinstance(out, ReinterpretView):
                self.output_names.add(out.buffer.name)
            elif isinstance(out, Buffer):
                self.output_names.add(out.name)
        self.users: dict[str, set] = {}
        for node in self.nodes:
            for name in node.reads:
                self.users.setdefault(name, set()).add(id(node))

    # -- dependency helpers ---------------------------------------------
    def _deps(self, node) -> set:
        return {id(self.producer[r]) for r in node.reads if r in self.producer}

    def fuse(self) -> list:
        """Return an ordered list of FusedGroup / ExternNode."""

        groups: dict[int, Any] = {}
        group_of: dict[int, int] = {}
        for node in self.nodes:
            if isinstance(node, LoopNode):
                g = FusedGroup([node], node.xnumel, node.rnumel)
            else:
                g = node
            groups[id(g)] = g
            group_of[id(node)] = id(g)
        order = {id(node): node.index for node in self.nodes}

        def gnodes(g):
            return g.nodes if isinstance(g, FusedGroup) else [g]

        def gdeps(g):
            deps = set()
            for n in gnodes(g):
                for d in self._deps(n):
                    gd = group_of[d]
                    if gd != id(g):
                        deps.add(gd)
            return deps

        for _round in range(10):
            changed = False
            candidates = []
            gid_list = list(groups)
            # pairs sharing a buffer: producer->consumer or sibling readers
            readers: dict[str, set] = {}
            for gid, g in groups.items():
                if not isinstance(g, FusedGroup):
                    continue
                for n in g.nodes:
                    for r in n.reads:
                        readers.setdefault(r, set()).add(gid)
            seen = set()
            for gid, g in groups.items():
                if not isinstance(g, FusedGroup):
                    continue
                partners = set()
                for name in g.names:
                    partners |= readers.get(name, set())
                for n in g.nodes:
                    for r in n.reads:
                        partners |= readers.get(r, set())
                        p = self.producer.get(r)
                        if p is not None:
                            partners.add(group_of[id(p)])
                partners.discard(gid)
                for other in partners:
                    key = (min(gid, other), max(gid, other))
                    if key in seen:
                        continue
                    seen.add(key)
                    og = groups.get(other)
                    if not isinstance(og, FusedGroup):
                        continue
                    score = self._score(g, og)
                    candidates.append((score, key))
            candidates.sort(key=lambda item: -item[0])
            fused_away = set()
            for score, (a, b) in candidates:
                if a in fused_away or b in fused_away or a not in groups or b not in groups:
                    continue
                ga, gb = groups[a], groups[b]
                first, second = (ga, gb) if min(order[id(n)] for n in ga.nodes) <= min(order[id(n)] for n in gb.nodes) else (gb, ga)
                merged = self._try_fuse(first, second, groups, group_of, gdeps)
                if merged is None:
                    continue
                for n in merged.nodes:
                    group_of[id(n)] = id(merged)
                del groups[id(first)]
                del groups[id(second)]
                groups[id(merged)] = merged
                fused_away.update({id(first), id(second)})
                changed = True
            if not changed:
                break
        # topological order of the groups
        ordered = []
        done = set()
        pending = sorted(groups.values(), key=lambda g: min(order[id(n)] for n in gnodes(g)))
        while pending:
            progressed = False
            for g in list(pending):
                if gdeps(g) <= done:
                    ordered.append(g)
                    done.add(id(g))
                    pending.remove(g)
                    progressed = True
                    break
            if not progressed:
                raise RuntimeError("kernel schedule has a dependency cycle")
        return ordered

    def _score(self, a: FusedGroup, b: FusedGroup) -> int:
        shared = 0
        names_a = set(a.names)
        names_b = set(b.names)
        reads_a = {r for n in a.nodes for r in n.reads}
        reads_b = {r for n in b.nodes for r in n.reads}
        for name in (names_a & reads_b) | (names_b & reads_a) | (reads_a & reads_b):
            buf = self.graph.buffers.get(name)
            if buf is not None:
                shared += prod(buf.get_size()) * _ITEMSIZE.get(dtype_name(buf.get_dtype()), 4)
        return shared

    def _try_fuse(self, first: FusedGroup, second: FusedGroup, groups, group_of, gdeps):
        if len(first.nodes) + len(second.nodes) > MAX_FUSION_SIZE:
            return None
        # iteration space of the merged kernel
        red = [g for g in (first, second) if g.is_reduction]
        if len(red) == 2:
            if (first.xnumel, first.rnumel) != (second.xnumel, second.rnumel):
                return None
            xnumel, rnumel = first.xnumel, first.rnumel
        elif len(red) == 1:
            xnumel, rnumel = red[0].xnumel, red[0].rnumel
        else:
            if first.xnumel != second.xnumel:
                return None
            xnumel, rnumel = first.xnumel, 1
        nodes = first.nodes + second.nodes
        placements = {}
        for n in nodes:
            p = placement(n, xnumel, rnumel)
            if p is None:
                return None
            placements[id(n)] = p
        # no path first -> (outside) -> second
        if self._reaches_through_outside(first, second, groups, group_of, gdeps):
            return None
        if self._reaches_through_outside(second, first, groups, group_of, gdeps):
            return None
        # every in-kernel read must be thread- or row-local
        written = {}
        for n in nodes:
            for b in n.buffers:
                written[b.name] = (n, b)
        for n in nodes:
            p = placements[id(n)]
            for load in n.body.loads:
                name, index, _mask = load.args
                if name not in written:
                    continue
                producer, buffer = written[name]
                if producer is n:
                    return None
                pp = placements[id(producer)]
                store = pp.store_index(buffer)
                read = p.to_kernel(index)
                if store != read:
                    return None
                if producer.is_reduction and RINDEX in free_symbols(read):
                    return None
                if producer.is_reduction and p.full:
                    # A nest that runs inside the reduction loops cannot read
                    # the row result: it only exists once the loops are done.
                    return None
                if not producer.is_reduction and pp.full != p.full and p.full:
                    # an x-only value read inside the loops: fine (broadcast)
                    pass
                if not producer.is_reduction and pp.full and not p.full:
                    return None
        ordered = sorted(nodes, key=lambda n: n.index)
        return FusedGroup(ordered, xnumel, rnumel)

    def _reaches_through_outside(self, src, dst, groups, group_of, gdeps) -> bool:
        """Does ``dst`` depend on ``src`` through some other group?"""

        src_id = None
        for gid, g in groups.items():
            if g is src:
                src_id = gid
        dst_deps = [d for d in gdeps(dst) if d != src_id]
        seen = set()
        stack = list(dst_deps)
        while stack:
            gid = stack.pop()
            if gid in seen:
                continue
            seen.add(gid)
            if gid == src_id:
                return True
            g = groups.get(gid)
            if g is None:
                continue
            stack.extend(gdeps(g))
        return False


__all__ = ["ExternNode", "FusedGroup", "KernelScheduler", "LoopNode", "NodePlacement", "RINDEX", "XINDEX", "placement"]
