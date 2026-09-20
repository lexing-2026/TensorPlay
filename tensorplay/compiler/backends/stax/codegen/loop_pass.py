"""Loop-level optimization passes for the generated Triton kernels.

Three transformations over the emission-time loop structure of a kernel,
each a plan-time decision the autotuner searches rather than an always-on
rewrite:

* **vectorize** — a pointwise kernel packs ``VEC`` consecutive elements per
  lane (a two-dimensional ``[XBLOCK, VEC]`` iteration space with a
  row-uniform mask), so every load/store covers one contiguous ``VEC``-wide
  segment and the backend emits vector memory instructions.  Applies when
  the element count divides the width and the widened access stays within
  one 16-byte vector.
* **unroll** — a looped axis-reduction r-loop is emitted with an explicit
  unroll factor, trading code size for loop-overhead amortization and ILP.
  Applies only when the loop exists (the reduction space exceeds one tile).
* **split** — a looped r-loop whose tile does not exactly divide the
  reduction space is split into a main part over the full tiles (no
  remainder predication: every lane in-bounds, mask-free loads) plus one
  masked tail tile after the loop.

The candidate builders below are pure functions of the kernel's geometry so
tests can pin applicability; the codegen consumes the chosen knob values
from the tail of the fixed-config tuple (pointwise ``(XBLOCK, warps, VEC)``
and axis-reduction ``(XBLOCK, warps, RBLOCK, stages, unroll, split)``).
"""

from __future__ import annotations

from typing import Sequence, Tuple

# Vectorize widths in element units.  2 and 4 give 8/16-byte accesses for
# 32-bit dtypes; 8-wide pairs only fit 16 bytes for 16-bit dtypes, which the
# applicability check admits through the itemsize gate.
VECTOR_WIDTHS: Tuple[int, ...] = (1, 2, 4, 8)

# Reduction r-loop unroll factors.
UNROLL_FACTORS: Tuple[int, ...] = (1, 2, 4)

#: One vector memory instruction spans at most 16 bytes on the targets the
#: backend ships for.
_MAX_VECTOR_BYTES = 16


def vectorize_applies(xnumel: int, itemsize: int, vec: int) -> bool:
    """True when a pointwise kernel may pack ``vec`` elements per lane.

    The row-uniform mask contract needs the element count to divide the
    width (a row is then either fully in-bounds or fully out), and the
    widened access must stay within one vector instruction.
    """

    if vec <= 1:
        return True
    if xnumel % vec != 0:
        return False
    return vec * max(int(itemsize), 1) <= _MAX_VECTOR_BYTES


def unroll_applies(rnumel: int, rblock: int, unroll: int) -> bool:
    """True when the looped r-loop may carry ``unroll``.

    No loop exists once one tile covers the whole reduction space (the
    persistent form), so unrolling applies only to genuinely looped shapes;
    a factor larger than the trip count only bloats the body.
    """

    if unroll <= 1:
        return True
    if rnumel <= rblock:
        return False
    return unroll <= (rnumel + rblock - 1) // rblock


def split_applies(rnumel: int, rblock: int) -> bool:
    """True when the r-loop has a remainder tile worth peeling.

    An exact tiling already runs mask-free; splitting is only meaningful
    when the last tile is partial (and a loop exists at all).
    """

    if rnumel <= rblock:
        return False
    return rnumel % rblock != 0


def pointwise_loop_candidates(
    xnumel: int,
    itemsize: int,
    base: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[int, ...], ...]:
    """Extend a pointwise ``(XBLOCK, warps)`` table with vectorize widths.

    The neutral width 1 keeps the plain 2-tuple emission (and the existing
    records stay valid members); wider widths append as a third element.
    """

    extended: list[Tuple[int, ...]] = []
    for xblock, warps in base:
        for vec in VECTOR_WIDTHS:
            if vectorize_applies(xnumel, itemsize, vec):
                if vec == 1:
                    extended.append((xblock, warps))
                else:
                    extended.append((xblock, warps, vec))
    return tuple(extended)


def dims_loop_candidates(
    rnumel: int,
    base: Sequence[Tuple[int, ...]],
) -> Tuple[Tuple[int, ...]]:
    """Extend an axis-reduction table with loop-pass variants.

    Every geometry becomes a flat six-tuple ``(X, W, RB, stages, unroll,
    split)``; 3-tuple geometries (which carry no explicit RBLOCK) take the
    shape-derived one so the emission reads a uniform layout.  Variants the
    geometry cannot use (unrolling a persistent form, splitting an exact
    tiling) are pruned so the tuner never benches them.
    """

    from .triton import _PERSISTENT_RNUMEL_MAX, _next_power_of_two

    derived_rblock = min(
        _next_power_of_two(max(rnumel, 1)), _PERSISTENT_RNUMEL_MAX
    )

    extended: list[Tuple[int, ...]] = []
    for entry in base:
        if len(entry) > 3:
            xblock, warps, rblock, stages = entry[:4]
        else:
            xblock, warps, stages = entry
            rblock = derived_rblock
        for unroll in UNROLL_FACTORS:
            if not unroll_applies(rnumel, rblock, unroll):
                continue
            for split in (0, 1):
                if split and not split_applies(rnumel, rblock):
                    continue
                extended.append(
                    (xblock, warps, rblock, stages, unroll, split)
                )
    return tuple(extended)
