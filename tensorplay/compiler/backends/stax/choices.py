"""Whether two pieces may be joined, where the answer is a matter of policy.

Joining pieces is always correct and never necessary: it produces the same
values either way.  What it changes is how much is held while the work runs and
how many launches there are, and how much of that is worth trading away is a
decision about the program rather than a fact about it.  So the questions asked
here are not whether a join would be right but whether it is worth it, and the
answer is allowed to differ between two programs that do the same arithmetic.

Nothing here is needed for the answer to be correct.  A pair refused here is one
that would have worked, so what is lost is a possible saving; that is why the
reasons are recorded, since a surprising refusal is worth being able to read.
"""

from __future__ import annotations

import logging

from . import config
from .ir import OrderedSet
from .loops import V
from .memory import estimate_peak_memory

log = logging.getLogger(__name__)


def can_fuse(scheduler, node1, node2, shared_data_score: int) -> bool:
    """Whether joining these is worth doing.

    Three things are asked.  Do the two share a value at all -- if not, the join
    saves nothing, and joining purely to have one launch fewer would make the
    program bigger for no reason.  Would the join be so large that the code
    stops fitting in what a program may hold?  And would it mean holding more
    memory than before, which is a cost paid for a saving and may not be worth
    it?
    """

    if shared_data_score == 0 and (
        not config.aggressive_fusion or node1.is_reduction() or node2.is_reduction()
    ):
        # Nothing is shared, so there is nothing to save.  Where a value is
        # read by both but at different places, the reason is worth having
        # since it looks like a pair that should have joined.
        common_buf_names: OrderedSet = (
            node1.read_writes.buffer_names() & node2.read_writes.buffer_names()
        )
        if len(common_buf_names) > 0:
            _why(node1, node2)("no shared data due to indexing mismatch")
        else:
            _why(node1, node2)("no shared data")
        return False

    if (
        not node1.is_foreach()
        and not node2.is_foreach()
        and len(node1.get_nodes()) + len(node2.get_nodes()) > config.max_fusion_size
    ):
        _why(node1, node2)("exceeds max fusion")
        return False

    if scheduler.can_fusion_increase_peak_memory(node1, node2):
        _why(node1, node2)("Fusion will increase peak memory")
        return False

    if (
        config.max_fusion_unique_io_buffers is not None
        and scheduler.fusion_prevent_too_many_reads_and_writes(
            node1,
            node2,
            config.max_fusion_unique_io_buffers,
        )
    ):
        _why(node1, node2)("fusion_prevent_too_many_reads_and_writes")
        return False

    return True


def can_fuse_vertical(
    scheduler, node1, node2, shared_data_score: int
) -> bool:
    """Whether one piece may be written as part of another that feeds it.

    Whether the two fit is a property of the machine and is asked of the
    backend; what is left to ask here is whether it is worth it, and for this
    shape there is nothing further to say.
    """

    return True


def can_fuse_horizontal(
    scheduler, node1, node2, shared_data_score: int
) -> bool:
    """Whether two pieces walking the same axes may become one.

    Unlike the other shape, this one is worth a bound: two pieces that read and
    write a great deal relative to what they share would be joined into a kernel
    that holds far more open than either did alone, and a kernel that does not
    fit in what a program may hold is worse than two that each fit.
    """

    from .kernel_scheduler import MixOrderReduction

    if MixOrderReduction.can_fuse(node1, node2):
        # The two share every element of the value, so neither what they share
        # nor how far apart they are says anything.
        return True
    if shared_data_score < config.score_fusion_memory_threshold:
        _why(node1, node2)("score_fusion_memory_threshold")
        return False
    if scheduler.are_long_distant_nodes(node1, node2):
        _why(node1, node2)(
            "Nodes are too far away. Fusing them may increase peak memory."
        )
        return False

    return True


def _why(node1, node2):
    from .kernel_scheduler import WhyNoFuse

    return WhyNoFuse(node1, node2)


