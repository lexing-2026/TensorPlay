"""Settings for how collectives are moved relative to the compute around them.

A collective is not decided when it is written but when it runs, and how long
it takes is not known until then.  So the question these answer is what to do
while that is still unknown: whether to move a collective earlier or later at
all, whether to let the ranks disagree about the answer, and how much extra
memory and overlap to spend on being wrong.

Two of the settings below are deliberately unsafe.  They let the ranks reorder
collectives differently from one another, which is faster and deadlocks when
they do, so they are only correct when the reordering has already been made
identical everywhere by the setting above them.
"""

import os

#: Whether the time a collective takes is taken from the library that performs
#: it rather than measured here.
runtime_estimations_use_nccl_lib_estimations: bool = False

#: Whether the time a collective takes is agreed on across every rank.
#:
#: Without this each rank reorders from what it saw alone, and a reorder that
#: differs between ranks hangs: one rank waits for a collective another has
#: moved past.
runtime_estimations_align_across_all_distributed_ranks: bool = False

#: Left in for narrowing a reorder to a few nodes while working on it.
reorder_iterative_debug_memory_recompute: bool = False

#: How many nodes a reorder is limited to while working on it.
reorder_iterative_debug_limit_to_reorder: int | None = (
    None
    if (env_str := os.getenv("TP_REORDER_COLLECTIVES_LIMIT")) is None
    else int(env_str)
)

#: How many waits a sink reorder is limited to while working on it.
sink_waits_iterative_debug_limit_to_sink: int | None = (
    None if (env_str := os.getenv("TP_SINK_WAITS_LIMIT")) is None else int(env_str)
)

#: Whether a reorder is decided from how long a collective actually took
#: rather than from a guess.  Costs a measurement per collective.
reorder_iterative_use_runtime_estimations: bool = False
sink_iterative_use_runtime_estimations: bool = False

#: Whether the measured times are made identical across ranks by performing
#: the collective for real between them.  Required when the measurements
#: themselves are not deterministic and the ranks must still agree.
reorder_for_compute_comm_overlap_broadcast_runtime_estimations: bool = False

#: How much a measured collective is scaled by, to work around the measured
#: time being short of what it really costs.
reorder_sink_runtime_estimations_comm_mult: float = 2.0

#: How much a measured compute is scaled by, for the same reason.
reorder_sink_runtime_estimations_non_comm_mult: float = 1.0

#: How much more compute than communication has to overlap before a reorder
#: stops, as a fraction of the measured communication.  Higher means more
#: aggressive.
reorder_iterative_extra_comm_comp_overlap: float = 0.5
sink_iterative_extra_comm_comp_overlap: float = 0.5

#: How much the peak memory a reorder may reach is allowed to rise, as a
#: fraction of what it was before.  Moving a collective earlier holds what it
#: produces longer, and this is what pays for that.
reorder_iterative_peak_memory_budget: float = 0.2
sink_iterative_peak_memory_budget: float = 0.2

#: Whether the ranks may reorder collectives differently from one another.
#: Faster, and hangs when they do; only correct together with agreeing on the
#: measured times above.
reorder_iterative_unsafe_collectives_reorder: bool = True
sink_waits_iterative_unsafe_collectives_reorder: bool = True

#: Whether a collective may be moved along with the group of nodes around it
#: rather than on its own.
reorder_iterative_group_with_collectives: bool = False
sink_iterative_unsafe_swap_with_collectives: bool = False

#: Whether to say what a reorder decided and why.
reorder_sink_verbose_logging: bool = False
