"""What this route can do with each operator, declared rather than discovered.

Coverage used to be a fact about the code: an operator was lowered if
someone had written a lowering for it, and library-called otherwise, and the
only way to find out which was to run something and read the exception.  That
makes the gap invisible until it is hit, and it makes a gap impossible to
close on purpose.

So the route now says what it can do, per operator, in one table:

  LOWERED       the loop writes it out element by element
  TEMPLATE      a template measures a choice, the operator being the floor
  LIBRARY_CALL  it runs as a single fused library call (one step, whole
                extents in one launch, no per-element traffic)
  NOT_COVERED   nothing here can run it; reaching it raises, on purpose

``NOT_COVERED`` is a declaration, not an omission: it is the work list.  The
table is checked against the real lowering tables by
``test_operator_coverage_matches_lowering_tables``, so it cannot claim a
capability that was deleted, and the low coverage it reports is measured
rather than guessed.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class Coverage(Enum):
    """How far this route carries an operator."""

    LOWERED = "lowered"
    TEMPLATE = "template"
    LIBRARY_CALL = "library_call"
    NOT_COVERED = "not_covered"

    @property
    def is_covered(self) -> bool:
        return self is not Coverage.NOT_COVERED


#: Operators that run as a single library call, named for the reason they are
#: not being taken apart.  A reduction over a large extent, a kernel with its
#: own tiling and its own gradients, a fused attention -- writing any of these
#: out element by element would be more launches and more memory traffic, not
#: fewer, so the library call is the right lowering and staying is not a gap.
#: Convolutions do not appear here even though they run as a library call: they
#: are answered from the live template table instead, because a template that
#: measures a choice is a stronger claim than a library call, and it is the one
#: that keeps the operator as its floor.  For a one-by-one convolution the table
#: says the arithmetic is a product and measures it as one; for a wider kernel
#: there is no tile kernel to measure, so the operator stands.
LIBRARY_CALLS: dict[str, str] = {
    "_fused_sdp_choice.default": "one launch, and its own backward",
    "_scaled_dot_product_attention.default": "one launch, and its own backward",
    "_scaled_dot_product_attention_with_lse.default": "one launch, and its own backward",
    "_scaled_dot_product_attention_backward.default": "one launch, and its own backward",
    "_scaled_dot_product_attention_with_lse_backward.default": "one launch, and its own backward",
    "_scaled_dot_product_attention_backward_with_lse.default": "one launch, and its own backward",
    "efficient_attention_forward.default": "one launch, and its own backward",
    "scaled_dot_product_attention.default": "one launch, and its own backward",
    "scaled_dot_product_efficient_attention.default": "one launch, and its own backward",
    "nll_loss_forward.default": "one launch, and its own backward",
    "nll_loss_backward.default": "one launch, and its own backward",
    "_log_softmax_backward_data.default": "one launch over a large extent",
    "_softmax_backward_data.default": "one launch over a large extent",
    "log_softmax_backward.default": "one launch over a large extent",
    "softmax_backward.default": "one launch over a large extent",
    "nll_loss2d_forward.default": "one launch over a large extent",
    "nll_loss2d_backward.default": "one launch over a large extent",
    "nll_loss3d_forward.default": "one launch over a large extent",
    "nll_loss3d_backward.default": "one launch over a large extent",
    "max_pool2d_with_indices.default": "one launch, and it returns the indices",
    "max_pool2d_with_indices_backward.default": "one launch, and it reads the indices",
    "max_pool3d_with_indices.default": "one launch, and it returns the indices",
    "max_pool3d_with_indices_backward.default": "one launch, and it reads the indices",
    "max_pool2d_with_indices_backward_cuda2d.default": "one launch, and it reads the indices",
    "max_pool3d_with_indices_backward_cuda3d.default": "one launch, and it reads the indices",
}

#: Operators this route can write out but has not, in the order they are worth
#: doing.  Each one is a real gap: reaching it raises, so it is a work list
#: rather than a silent library call that quietly costs more than it looks.
#:
#: The first group shares one cause.  A pool or an upsample is a reduction
#: whose extents are the *other* side's -- the input needs the output's windows
#: laid over it, and the output needs the input's pixels scattered back -- and
#: writing one means writing the other, so they are listed as a set.
NOT_COVERED: dict[str, str] = {
    "avg_pool2d_backward.default": "reduction over the input, driven by the output's windows",
    "avg_pool3d_backward.default": "reduction over the input, driven by the output's windows",
    "max_pool2d_backward.default": "scatter to the input, driven by the saved indices",
    "max_pool3d_backward.default": "scatter to the input, driven by the saved indices",
    "upsample_nearest2d_backward.default": "scatter to the input, driven by the source map",
    "upsample_nearest3d_backward.default": "scatter to the input, driven by the source map",
    "upsample_bilinear2d_backward.default": "scatter to the input, driven by the source map",
    "index_add.default": "the decompositions' scatter, which this route cannot write",
    "index_copy.default": "the decompositions' scatter, which this route cannot write",
    "scatter_add.default": "the decompositions' scatter, which this route cannot write",
}

#: Operators the decompositions produce that this route also has to write, or
#: a decomposition is not usable here however correct it is.
NEEDED_BY_DECOMPOSITIONS: dict[str, str] = {
    "index_add.default": "several decompositions scatter through it",
    "index_copy.default": "the gather-free path through it",
    "nonzero.default": "the dynamic-shape path through it",
    "item.default": "control flow on a value, which a loop cannot do",
    "arange.default": "a data-dependent extent, which a loop cannot do",
}


def _tables() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """The real registries, imported late so this module stays importable."""

    from .graph_lowering import _TEMPLATE_OPERATORS
    from .op_lowerings import LOWERINGS

    return LOWERINGS, _TEMPLATE_OPERATORS, {}


def declared_coverage() -> dict[str, Coverage]:
    """Every operator this route names, with what it does with it."""

    table: dict[str, Coverage] = {}
    for name in LIBRARY_CALLS:
        table[name] = Coverage.LIBRARY_CALL
    for name in NOT_COVERED:
        table[name] = Coverage.NOT_COVERED
    for name in NEEDED_BY_DECOMPOSITIONS:
        table.setdefault(name, Coverage.NOT_COVERED)
    return table


def coverage_of(name: str) -> Coverage:
    """What this route does with one operator.

    Lowered and templated operators are answered from the live tables, so a
    lowering that was added shows up here without this file being told.  The
    declared tables answer for everything else, and an operator nobody has
    named is reported as not covered, which is the truth: there is no
    capability behind it.
    """

    lowerings, templates, _ = _tables()
    # A templated operator is declared by its template table before the
    # lazy template import registers a lowering under the same name; the
    # template is the stronger claim, so it is answered first.
    if name in templates:
        return Coverage.TEMPLATE
    if name in lowerings:
        return Coverage.LOWERED
    return declared_coverage().get(name, Coverage.NOT_COVERED)


def is_covered(name: str) -> bool:
    """Can this route run the operator at all?"""

    return coverage_of(name).is_covered


def work_list() -> list[tuple[str, str]]:
    """The operators this route cannot run, with why they are worth doing.

    Ordered so the ones that unblock others come first: a scatter op is worth
    more than any single reduction, because several decompositions need it.
    """

    entries = [(name, why) for name, why in NOT_COVERED.items()]
    entries.sort(key=lambda e: (e[0] not in NEEDED_BY_DECOMPOSITIONS, e[0]))
    return entries


def summary(names: Iterable[str]) -> dict[str, int]:
    """Count operators by what happens to them, for a coverage report."""

    counts = {kind.value: 0 for kind in Coverage}
    for name in names:
        counts[coverage_of(name).value] += 1
    return counts
