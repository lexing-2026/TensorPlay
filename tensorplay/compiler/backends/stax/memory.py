"""When each piece of memory is needed, and when it stops being needed.

Every buffer this compilation produces is held in memory from the point some
operation writes it until the last operation that reads it has run.  Working
out that window for each buffer is what makes it possible to say how much
memory is held at once, and it is also what makes it possible to hand a buffer
over to a later operation to write into once nobody is reading it.

A buffer whose memory is somebody else's -- one that came in from outside the
program -- is counted differently: it was already there before anything ran, and
it is still there afterwards unless the program says it is finished with it.  So
what is tracked for those is only the point after which nothing reads them.
"""

from __future__ import annotations

import collections
import dataclasses
import heapq
import logging
from typing import TYPE_CHECKING, TypedDict

from . import config
from .ir import MultiOutputLayout, NoneLayout, OrderedSet, is_nonfreeable_buffers
from .loops import V
from .utils import get_dtype_size
from .dependencies import WeakDep

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .dependencies import Dep
    from .scheduler import BaseSchedulerNode, SchedulerBuffer


@dataclasses.dataclass
class PeakMemoryResult:
    """One ordering of the work, and how much memory it was estimated to need."""

    order: list
    peak_memory: int
    method_name: str


@dataclasses.dataclass
class MemoryPlanningInfoForBuffer:
    """What is known about when a buffer is needed and when it is not.

    The two sets of successors answer two different questions, and both are
    needed: one is everything that comes after this buffer and constrains the
    order, and the other is only what still needs the memory, which is what
    decides when the memory can be handed to something else.
    """

    size_alloc: int = 0
    size_free: int = 0
    #: Who still needs the memory, which is what decides when it can be reused.
    succ_nodes: OrderedSet = dataclasses.field(default_factory=OrderedSet)
    #: Who comes after, which is what decides the order work is done in.
    succ_nodes_for_ordering: OrderedSet = dataclasses.field(default_factory=OrderedSet)

    def __post_init__(self) -> None:
        if len(self.succ_nodes) > len(self.succ_nodes_for_ordering):
            raise AssertionError(
                f"succ_nodes must be a subset of succ_nodes_for_ordering. "
                f"len(succ_nodes)={len(self.succ_nodes)}, len(succ_nodes_for_ordering)={len(self.succ_nodes_for_ordering)}",
            )


@dataclasses.dataclass
class MemoryPlanningInfoForNode:
    """What is known about one piece of work: what it waits for, and what waits for it."""

    index: int = 0
    size: int = 0
    pred_buffers: OrderedSet = dataclasses.field(default_factory=OrderedSet)
    pred_nodes: OrderedSet = dataclasses.field(default_factory=OrderedSet)
    succ_nodes: OrderedSet = dataclasses.field(default_factory=OrderedSet)


@dataclasses.dataclass
class FreeableInputBuffer:
    """A buffer that came in from outside and may be released once read."""

    name: str
    mpi_buffer: MemoryPlanningInfoForBuffer = dataclasses.field(
        default_factory=MemoryPlanningInfoForBuffer
    )

    def get_name(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)


def get_freeable_input_buf(nodes, graph_inputs) -> dict:
    """Every buffer that came in from outside and that nothing else needs hold on to.

    A buffer that came in from outside was already there, so it is not this
    program's to release -- unless nothing needs it after some point, in which
    case saying so is what lets it be released there.  What must not be released
    this way is anything the program did not produce and does not own, and
    anything that exists only to say an order rather than to be read.
    """

    def _dep_size_hint(dep) -> int:
        return V.graph.get_dep_size_hint(dep)

    # Who still needs each input's memory, and who merely comes after it.
    dep_name_to_succ_nodes: dict = collections.defaultdict(OrderedSet)
    dep_name_to_succ_nodes_for_ordering: dict = collections.defaultdict(OrderedSet)
    dep_name_to_size: dict = {}

    for node in nodes:
        for dep in node.read_writes.reads:
            if dep.name in graph_inputs:
                if not is_nonfreeable_buffers(dep):
                    # Everything constrains the order, but an order-only
                    # dependency does not keep the memory alive.
                    dep_name_to_succ_nodes_for_ordering[dep.name].add(node)
                    dep_name_to_size[dep.name] = _dep_size_hint(dep)
                    if not (isinstance(dep, WeakDep) and dep.is_fake):
                        dep_name_to_succ_nodes[dep.name].add(node)

    name_to_freeable_input_buf: dict = {}
    for dep_name in dep_name_to_succ_nodes_for_ordering:
        name_to_freeable_input_buf[dep_name] = FreeableInputBuffer(
            dep_name,
            MemoryPlanningInfoForBuffer(
                size_free=dep_name_to_size[dep_name],
                succ_nodes=dep_name_to_succ_nodes[dep_name],
                succ_nodes_for_ordering=dep_name_to_succ_nodes_for_ordering[dep_name],
            ),
        )
    return name_to_freeable_input_buf


def compute_size_for_scheduler_buffer(name_to_buf) -> dict:
    """How much memory each buffer costs to make, and how much it gives back.

    A buffer holding several results has no shape of its own, so its size is the
    total of the results taken from it, and giving one of those results back
    gives back only that one's share.  That is why both numbers are recorded
    per result as well as per buffer: a result taken from a several-result
    buffer costs nothing to make and gives back its own size.

    A buffer that was renamed because something wrote over it shares its memory
    with the name it came from, so it neither costs anything nor gives anything
    back, and whatever was waiting on the old name is transferred to the new one
    -- otherwise the memory would be given away while something was still
    reading it.
    """

    from .ir import MultiOutput
    from .scheduler import OutputNode

    sched_buf_to_size: dict = {}

    def _compute_and_update_buf_size(sched_buf, user_of_MultiOutputLayout: bool = False):
        if sched_buf.get_name() in V.graph.scheduler.mutation_real_name:
            sched_buf_to_size[sched_buf.get_name()] = (0, 0)
            return 0
        elif isinstance(sched_buf.node.layout, NoneLayout):
            sched_buf_to_size[sched_buf.get_name()] = (0, 0)
            return 0
        elif isinstance(sched_buf.node.layout, MultiOutputLayout):
            size_alloc = 0
            for user in sched_buf.users:
                if isinstance(user.node, OutputNode):
                    continue
                for buf in user.node.get_outputs():
                    if isinstance(buf.node, MultiOutput):
                        size_alloc += _compute_and_update_buf_size(buf, True)
            sched_buf_to_size[sched_buf.get_name()] = (
                0 if user_of_MultiOutputLayout else size_alloc,
                0,
            )
            return size_alloc
        else:
            buf_size = V.graph.sizevars.optimization_hint(
                sched_buf.node.get_numel(), fallback=0
            ) * get_dtype_size(sched_buf.node.get_dtype())
            sched_buf_to_size[sched_buf.get_name()] = (
                0 if user_of_MultiOutputLayout else buf_size,
                buf_size,
            )
            return buf_size

    for sched_buf in name_to_buf.values():
        # A buffer already accounted for as one of several results of another
        # buffer must not be counted a second time.
        if sched_buf.get_name() not in sched_buf_to_size:
            _compute_and_update_buf_size(sched_buf)

    return sched_buf_to_size


def assign_memory_planning_info_for_scheduler_buffers(nodes, name_to_buf) -> None:
    """Record on each buffer what it costs and who is waiting on it.

    A buffer can be given away once the last thing waiting on it has run, so
    what is recorded is who that is.
    """

    sched_buf_to_size = compute_size_for_scheduler_buffer(name_to_buf)

    dep_name_to_succ_nodes: dict = collections.defaultdict(OrderedSet)
    dep_name_to_succ_nodes_for_ordering: dict = collections.defaultdict(OrderedSet)
    for node in nodes:
        for dep in node.unmet_dependencies:
            # Everything constrains the order; only real reads keep memory alive.
            dep_name_to_succ_nodes_for_ordering[dep.name].add(node)
            if not (isinstance(dep, WeakDep) and dep.is_fake):
                dep_name_to_succ_nodes[dep.name].add(node)

    # Going backwards picks up dependencies that are only reachable through a
    # renamed buffer, so that a rename does not lose what was waiting on it.
    for mutating_buf_name, real_buf_name in reversed(
        V.graph.scheduler.mutation_real_name.items()
    ):
        dep_name_to_succ_nodes[real_buf_name] |= dep_name_to_succ_nodes[
            mutating_buf_name
        ]
        dep_name_to_succ_nodes_for_ordering[real_buf_name] |= (
            dep_name_to_succ_nodes_for_ordering[mutating_buf_name]
        )

    # A buffer nothing waits on -- a result of the program, for one -- is not in
    # the mapping, and still has to be told its size.
    for buf_name in name_to_buf:
        name_to_buf[buf_name].mpi_buffer = MemoryPlanningInfoForBuffer(
            size_alloc=sched_buf_to_size[buf_name][0],
            size_free=sched_buf_to_size[buf_name][1],
            succ_nodes=dep_name_to_succ_nodes[buf_name],
            succ_nodes_for_ordering=dep_name_to_succ_nodes_for_ordering[buf_name],
        )


def assign_memory_planning_info_for_scheduler_nodes(
    nodes, name_to_fused_node, name_to_buf, name_to_freeable_input_buf
) -> None:
    """Record on each piece of work what it waits for and what waits for it."""

    node_to_pred_nodes: dict = collections.defaultdict(OrderedSet)
    node_to_succ_nodes: dict = {}
    node_to_pred_buffers: dict = collections.defaultdict(OrderedSet)

    # Who follows what, taken from what was already recorded about the buffers:
    # anything waiting on a buffer this work produces follows this work.
    for node in nodes:
        succ_nodes = OrderedSet(
            succ_node
            for buffer in node.get_outputs()
            for succ_node in buffer.mpi_buffer.succ_nodes_for_ordering
        )
        node_to_succ_nodes[node] = succ_nodes

        for succ_node in succ_nodes:
            node_to_pred_nodes[succ_node].add(node)

        # What this work is responsible for giving back.  What is waited on
        # rather than what constrains the order, since this is about how long
        # memory is held rather than what may be done first.
        for buffer in node.get_outputs():
            for succ_node in buffer.mpi_buffer.succ_nodes:
                node_to_pred_buffers[succ_node].add(buffer)

    for freeable_buffer in name_to_freeable_input_buf.values():
        for succ_node in freeable_buffer.mpi_buffer.succ_nodes:
            node_to_pred_buffers[succ_node].add(freeable_buffer)

    # A second pass, now that everything is known, so that each work item gets
    # its position in the order as well.
    for index, node in enumerate(nodes):
        size_alloc = sum(buffer.mpi_buffer.size_alloc for buffer in node.get_outputs())
        succ_nodes = node_to_succ_nodes[node]
        pred_nodes = node_to_pred_nodes[node]

        # Nothing can be waiting on itself.
        succ_nodes.discard(node)
        pred_nodes.discard(node)

        node.mpi_node = MemoryPlanningInfoForNode(
            index=index,
            size=size_alloc,
            pred_buffers=node_to_pred_buffers[node],
            pred_nodes=node_to_pred_nodes[node],
            succ_nodes=succ_nodes,
        )


@dataclasses.dataclass
class BufferInfo:
    """One buffer, what it costs, and the span of the work over which it is held."""

    buffer: object
    size_alloc: int
    size_free: int
    start_step: int
    end_step: int


def compute_memory_timeline(nodes, name_to_freeable_input_buf, graph_outputs):
    """Put every buffer on the timeline: where it starts being held and where it stops.

    A buffer that is a result of the program is never given back, which is what
    an end step of -1 says.  A buffer nothing waits on is held only as long as
    the work that made it takes, since nothing else refers to it.
    """

    node_to_step: dict = {node: step for step, node in enumerate(nodes)}

    buf_info_list: list = []
    buf_to_snode_last_use: dict = {}

    def _get_end_step_and_snode(buf):
        max_step: int = -1
        max_step_snode = None
        succ_nodes = buf.mpi_buffer.succ_nodes
        if succ_nodes:
            for succ_node in succ_nodes:
                step = node_to_step[succ_node]
                if step > max_step:
                    max_step = step
                    max_step_snode = succ_node
            if max_step_snode is None:
                raise AssertionError("expected max_step_snode to be set")
        return max_step, max_step_snode

    # 1. for buffers that came in from outside
    for buf_name, input_buf in name_to_freeable_input_buf.items():
        end_step = -1
        if buf_name not in graph_outputs:
            end_step, end_step_snode = _get_end_step_and_snode(input_buf)
            if end_step_snode is None:
                raise AssertionError("expected end_step_snode to be set")
            buf_to_snode_last_use[input_buf] = end_step_snode

        buf_info_list.append(
            BufferInfo(
                input_buf,
                input_buf.mpi_buffer.size_free,
                input_buf.mpi_buffer.size_free,
                0,
                end_step,
            )
        )

    # 2. for buffers this program produced
    for step, node in enumerate(nodes):
        for sched_buf in node.get_outputs():
            # A buffer can have nothing waiting on it and still exist, when
            # everything that read it was folded into the work that made it.  It
            # is then held for exactly that work.
            buf_name = sched_buf.get_name()
            end_step = -1
            if buf_name not in graph_outputs:
                end_step, end_step_snode = _get_end_step_and_snode(sched_buf)
                if end_step == -1:
                    end_step = step
                    buf_to_snode_last_use[sched_buf] = node
                else:
                    if end_step_snode is None:
                        raise AssertionError("expected end_step_snode to be set")
                    buf_to_snode_last_use[sched_buf] = end_step_snode

            buf_info_list.append(
                BufferInfo(
                    sched_buf,
                    sched_buf.mpi_buffer.size_alloc,
                    sched_buf.mpi_buffer.size_free,
                    step,
                    end_step,
                )
            )

    return buf_info_list, node_to_step, buf_to_snode_last_use


def peak_memory_from_buf_info_list(buf_info_list, num_steps: int):
    """The most memory held at once, and how much is held at each step.

    A buffer that is a result of the program is not given back, so its giving
    back is not counted: doing so would cancel out the taking that created it.
    """

    delta = [0] * (num_steps + 1)
    for bi in buf_info_list:
        delta[bi.start_step] += bi.size_alloc
        if bi.end_step != -1:
            delta[bi.end_step + 1] -= bi.size_free

    max_memory = 0
    cur_memory = 0
    memories_at_nodes = [0] * (num_steps + 1)
    for t in range(num_steps + 1):
        cur_memory += delta[t]
        memories_at_nodes[t] = cur_memory
        if cur_memory > max_memory:
            max_memory = cur_memory
    return max_memory, memories_at_nodes


def live_memory_before_steps_from_buf_info_list(buf_info_list, num_steps: int):
    """How much is held immediately before each step runs."""

    delta = [0] * (num_steps + 1)
    cur_memory = 0
    for bi in buf_info_list:
        if isinstance(bi.buffer, FreeableInputBuffer):
            cur_memory += bi.size_alloc
        else:
            delta[bi.start_step + 1] += bi.size_alloc

        if bi.end_step != -1:
            delta[bi.end_step + 1] -= bi.size_free

    live_before = [0] * (num_steps + 1)
    live_before[0] = cur_memory
    for t in range(1, num_steps + 1):
        cur_memory += delta[t]
        live_before[t] = cur_memory
    return live_before


def estimate_peak_memory(nodes, name_to_freeable_input_buf, graph_outputs):
    """How much memory this order of the work needs, and where the peak is.

    The answer is found by following the order and adding up what each step
    takes and gives back, which is exact for this order rather than an estimate
    of what some other order would need.
    """

    buf_info_list, _, _ = compute_memory_timeline(
        nodes, name_to_freeable_input_buf, graph_outputs
    )
    return peak_memory_from_buf_info_list(buf_info_list, len(nodes))


def estimate_region_peak_memory(
    nodes_in_window,
    *,
    region_start: int,
    region_end: int,
    step_of,
    graph_outputs,
    cur_memory: int = 0,
) -> int:
    """The most memory held inside a span, for an order that has not happened yet.

    Used to compare candidate orderings, so the span is walked as it would be
    walked if that ordering were adopted: what each step takes, and what it
    gives back once the last thing waiting on a buffer has run.
    """

    R = region_end - region_start + 1
    region = [SNodeMemory(0, 0) for _ in range(R)]

    for node in nodes_in_window:
        s = step_of(node)
        slot = s - region_start
        if not (0 <= slot < R):
            raise AssertionError(f"expected 0 <= slot < {R}, got {slot}")

        for buf in node.get_outputs():
            bi = buf.mpi_buffer
            region[slot].size_alloc += bi.size_alloc
            name = buf.get_name()
            if name in graph_outputs:
                continue
            succ_steps = [step_of(n) for n in bi.succ_nodes]
            if not succ_steps:
                region[slot].size_free += bi.size_free

        for pb in node.mpi_node.pred_buffers:
            name = pb.get_name()
            if name in graph_outputs:
                continue
            succ_steps = [step_of(n) for n in pb.mpi_buffer.succ_nodes]
            if not succ_steps:
                raise AssertionError("expected non-empty succ_steps")
            if max(succ_steps) == s:
                region[slot].size_free += pb.mpi_buffer.size_free

    cur = cur_memory
    peak = cur
    for af in region:
        cur += af.size_alloc
        if cur > peak:
            peak = cur
        cur -= af.size_free
    return peak


@dataclasses.dataclass
class SNodeMemory:
    """What one step takes and what it gives back."""

    size_alloc: int
    size_free: int


def estimate_peak_memory_allocfree(
    nodes, name_to_freeable_input_buf, graph_outputs
):
    """The peak, and where it is, treating each step as taking then giving back.

    Collapsing each step into one number loses the difference between the moment
    the taking has happened and the moment the giving back has, and that
    difference is exactly when the peak is.  So both moments are kept.
    """

    buf_info_list, _, buf_to_snode_last_use = compute_memory_timeline(
        nodes, name_to_freeable_input_buf, graph_outputs
    )

    step_idx_allocfree = [SNodeMemory(0, 0) for _ in range(len(nodes))]

    for buf_info in buf_info_list:
        step_idx_allocfree[buf_info.start_step].size_alloc += buf_info.size_alloc
        if buf_info.end_step != -1:
            step_idx_allocfree[buf_info.end_step].size_free += buf_info.size_free

    snodes_allocfree = {}
    for i, node in enumerate(nodes):
        snodes_allocfree[node] = step_idx_allocfree[i]

    max_memory = 0
    cur_memory = 0
    snodes_curr_memory = []
    for t in range(len(nodes)):
        alloc = step_idx_allocfree[t].size_alloc
        free = step_idx_allocfree[t].size_free
        cur_memory += alloc
        post_alloc = cur_memory
        max_memory = max(max_memory, cur_memory)
        cur_memory -= free
        post_free = cur_memory
        snodes_curr_memory.append((post_alloc, post_free))

    return (
        max_memory,
        snodes_curr_memory,
        snodes_allocfree,
        buf_to_snode_last_use,
    )


def topological_sort_lpmf(nodes, name_to_freeable_input_buf, name_to_buf, graph_outputs):
    """An order that takes the piece needing least memory first.

    The idea is from a paper on buffer memory optimization for a video codec:
    at each step, for everything that could run, work out how much memory it
    needs while it runs and how much it gives back afterwards, and then take
    the one whose need stays under the high-water mark and leaves the least
    behind; where nothing fits under the mark, take whichever needs least.
    """

    class NodeInfo(TypedDict):
        indegree: int
        memory_to_free: int

    class BufferInfo(TypedDict):
        outdegree: int

    node_info: dict = {}
    buf_info: dict = {}

    # How much each piece is still waiting for, and so which are ready now.
    nodes_to_schedule: OrderedSet = OrderedSet()
    for node in nodes:
        node_info[node] = {
            "indegree": len(node.mpi_node.pred_nodes),
            "memory_to_free": 0,
        }
        if node_info[node]["indegree"] == 0:
            nodes_to_schedule.add(node)

    # How many things are still waiting on each buffer, which is what says when
    # it can be given away.
    for buf in list(name_to_buf.values()) + list(name_to_freeable_input_buf.values()):
        buf_info[buf] = {
            "outdegree": len(buf.mpi_buffer.succ_nodes)
            + (1 if buf.get_name() in graph_outputs else 0)
        }

    live_memory = sum(
        input_buf.mpi_buffer.size_free
        for input_buf in name_to_freeable_input_buf.values()
    )

    # The results have to exist, so their memory is a floor under the peak.
    output_memory = 0
    for buf_name in graph_outputs:
        if buf_name in name_to_buf:
            output_memory += name_to_buf[buf_name].mpi_buffer.size_free
        elif buf_name in name_to_freeable_input_buf:
            output_memory += name_to_freeable_input_buf[buf_name].mpi_buffer.size_free
    max_memory = max(live_memory, output_memory)
    memory_gap = max_memory - live_memory

    # What each piece would take and would give back.
    for node in nodes:
        # 1. a buffer this piece reads for the last time
        for buf in node.mpi_node.pred_buffers:
            if buf_info[buf]["outdegree"] == 1:
                node_info[node]["memory_to_free"] += buf.mpi_buffer.size_free
        # 2. a buffer this piece writes that nothing will read
        for buf in node.get_outputs():
            if buf_info[buf]["outdegree"] == 0:
                node_info[node]["memory_to_free"] += buf.mpi_buffer.size_free

    schedule: list = []
    size_threshold = config.size_threshold_for_succ_based_strategy
    num_iters: int = 0
    while num_iters < len(nodes) and nodes_to_schedule:
        # What to run next.  Once everything is big, fitting under the
        # high-water mark no longer distinguishes anything, so the piece whose
        # successors come soonest is taken instead.
        if (
            size_threshold > 0
            and min(node.mpi_node.size for node in nodes_to_schedule) > size_threshold
        ):
            selected_node = min(
                nodes_to_schedule,
                key=lambda node: min(
                    (
                        succ_node.mpi_node.index
                        for succ_node in node.mpi_node.succ_nodes
                    ),
                    default=len(nodes),
                ),
            )
        else:
            selected_node = min(
                nodes_to_schedule,
                key=lambda node: (
                    node.mpi_node.size if node.mpi_node.size > memory_gap else 0,
                    node.mpi_node.size - node_info[node]["memory_to_free"],
                    node.mpi_node.index,
                ),
            )
        nodes_to_schedule.remove(selected_node)
        schedule.append(selected_node)
        num_iters += 1

        live_memory += selected_node.mpi_node.size
        max_memory = max(max_memory, live_memory)
        live_memory -= node_info[selected_node]["memory_to_free"]
        memory_gap = max_memory - live_memory

        for succ_node in selected_node.mpi_node.succ_nodes:
            if node_info[succ_node]["indegree"] <= 0:
                raise AssertionError(
                    f"expected positive indegree, got {node_info[succ_node]['indegree']}"
                )
            node_info[succ_node]["indegree"] -= 1
            if node_info[succ_node]["indegree"] == 0:
                nodes_to_schedule.add(succ_node)

        for buf in selected_node.mpi_node.pred_buffers:
            if buf_info[buf]["outdegree"] <= 0:
                raise AssertionError(
                    f"expected positive outdegree, got {buf_info[buf]['outdegree']}"
                )
            buf_info[buf]["outdegree"] -= 1
            if buf_info[buf]["outdegree"] == 1:
                for succ_node in buf.mpi_buffer.succ_nodes:
                    node_info[succ_node]["memory_to_free"] += buf.mpi_buffer.size_free

    if num_iters > len(nodes):
        raise RuntimeError("Failed to schedule, while loop ran too long for lpmf")

    return schedule


def topological_sort_bfs(nodes):
    """An order that lets each piece run as soon as its own inputs are done with.

    A piece becomes ready when everything it reads has been read, and among the
    ready ones the one whose own inputs were finished earliest is taken.  That
    is what keeps a buffer from being held longer than it has to be: it is
    given away the moment the last reader is done, and the last reader is
    reached early.
    """

    class NodeInfo(TypedDict):
        indegree: int
        order: int

    node_info: dict = {}

    @dataclasses.dataclass
    class NodeWithPriority:
        priority: list
        node: object

        def __lt__(self, other: "NodeWithPriority") -> bool:
            if self.priority == other.priority:
                return self.node.mpi_node.index < other.node.mpi_node.index
            return self.priority < other.priority

    def _node_priority(node) -> list:
        # The priority is when the things it reads were themselves run.
        if node_info[node]["indegree"] != 0:
            raise AssertionError(
                f"expected zero indegree, got {node_info[node]['indegree']}"
            )
        exec_orders = sorted(
            OrderedSet(
                node_info[pred_node]["order"] for pred_node in node.mpi_node.pred_nodes
            )
        )
        return exec_orders

    nodes_to_schedule: list = []
    for node in nodes:
        node_info[node] = {"indegree": len(node.mpi_node.pred_nodes), "order": -1}
        if node_info[node]["indegree"] == 0:
            heapq.heappush(
                nodes_to_schedule, NodeWithPriority(_node_priority(node), node)
            )

    schedule: list = []
    num_iters: int = 0
    while num_iters < len(nodes) and nodes_to_schedule:
        selected_node = heapq.heappop(nodes_to_schedule).node
        node_info[selected_node]["order"] = len(schedule)
        schedule.append(selected_node)
        num_iters += 1

        for succ_node in selected_node.mpi_node.succ_nodes:
            if node_info[succ_node]["indegree"] <= 0:
                raise AssertionError(
                    f"expected positive indegree, got {node_info[succ_node]['indegree']}"
                )
            node_info[succ_node]["indegree"] -= 1
            if node_info[succ_node]["indegree"] == 0:
                heapq.heappush(
                    nodes_to_schedule,
                    NodeWithPriority(_node_priority(succ_node), succ_node),
                )

    if num_iters > len(nodes):
        raise RuntimeError("Failed to schedule, while loop ran too long for bfs")

    return schedule


def topological_sort_dfs(nodes):
    """An order that takes the smallest pieces first, following what they need.

    The outer loop visits pieces in order of how much memory they touch rather
    than in the order they were made, so the small ones are settled before the
    large ones, and each is preceded by whatever it reads.
    """

    seen: OrderedSet = OrderedSet()
    name_to_node: dict = {}
    result: list = []
    size_with_reads: dict = {}

    def visit(n) -> None:
        if n not in seen:
            seen.add(n)
            dep_nodes = [
                name_to_node[dep.name]
                for dep in n.unmet_dependencies
                if dep.name in name_to_node
            ]
            for node in sorted(
                dep_nodes, key=lambda n: (size_with_reads[n], n.mpi_node.index)
            ):
                visit(node)
            result.append(n)

    for node in nodes:
        for name in node.get_buffer_names():
            name_to_node[name] = node

    for node in nodes:
        size_with_reads[node] = node.mpi_node.size + sum(
            pred_buf.mpi_buffer.size_free for pred_buf in node.mpi_node.pred_buffers
        )
    for node in sorted(nodes, key=lambda n: (size_with_reads[n], n.mpi_node.index)):
        visit(node)

    return result


def validate_graph_acyclic(nodes) -> None:
    """Check that nothing is waiting on something that waits on it.

    Three colours, as in any cycle check: not yet looked at, being looked at
    now, and finished.  Meeting something being looked at now means the walk
    has come back to where it started, which is a cycle.
    """

    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(nodes, WHITE)
    path: list = []

    def dfs_visit(node) -> None:
        if color[node] == BLACK:
            return

        if color[node] == GRAY:
            path.append(node)
            path_info = " -> ".join([node.get_name() for node in path])

            raise RuntimeError(
                f"Cycle detected in memory planning graph"
                f"Path containing cycle (i -> j: j is a dependency of i): {path_info} "
                f"This indicates invalid dependency relationships in the scheduler graph"
            )

        color[node] = GRAY
        path.append(node)

        for pred_node in node.mpi_node.pred_nodes:
            if pred_node == node:
                raise AssertionError("expected pred_node != node (self-loop)")
            dfs_visit(pred_node)

        path.pop()
        color[node] = BLACK

    for node in nodes:
        if color[node] == WHITE:
            dfs_visit(node)


def validate_unique_buffer_names(nodes, name_to_buf, name_to_freeable_input_buf) -> None:
    """Check that a name refers to one buffer and not to two.

    A name that resolved to two buffers would let one piece read what another
    wrote, and a name used for both a produced and an input buffer would let a
    produced value overwrite something the program still holds.
    """

    for node in nodes:
        for buf in node.get_outputs():
            buf_name = buf.get_name()

            if buf_name not in name_to_buf:
                raise RuntimeError(
                    f"{buf_name} from {node.get_name()} is not found in name_to_buf mapping."
                    f" This indicates a missing buffer mapping."
                )

            if name_to_buf[buf_name] != buf:
                raise RuntimeError(
                    f"Buffer name mapping is incorrect for '{buf_name}'."
                    f"Expected name_to_buf['{buf_name}'] to be {buf.debug_str()}"
                    f"but got {name_to_buf[buf_name].debug_str()}"
                    f"This indicates some buffers share the same name"
                )

            if buf_name in name_to_freeable_input_buf:
                raise RuntimeError(
                    f"Buffer name conflict detected: '{buf_name}' from node {node.get_name()} "
                    f"is also used as a freeable input buffer name. "
                )


def prepare_planning_info(
    nodes, name_to_buf, name_to_fused_node, graph_inputs, graph_outputs
):
    """Work out, for the order as it stands, when each buffer is needed.

    Returns the peak for this order and the input buffers that may be given
    back, which are the two things every reordering method needs.
    """

    name_to_freeable_input_buf = get_freeable_input_buf(nodes, graph_inputs)
    assign_memory_planning_info_for_scheduler_buffers(nodes, name_to_buf)
    assign_memory_planning_info_for_scheduler_nodes(
        nodes, name_to_fused_node, name_to_buf, name_to_freeable_input_buf
    )

    estimated_peak_memory, _ = estimate_peak_memory(
        nodes, name_to_freeable_input_buf, graph_outputs
    )

    return estimated_peak_memory, name_to_freeable_input_buf


def reorder_for_peak_memory(
    nodes,
    name_to_buf,
    name_to_fused_node,
    graph_inputs,
    graph_outputs,
    methods=None,
):
    """Try several orders and keep the one that needs least memory.

    Each method proposes an order; the peak is worked out for each and the
    smallest wins.  A method that cannot produce an order is not allowed to
    decide anything, so a failure in one is not silently ignored -- it is
    reported, because it means the order actually used is not the one that was
    measured to be best.
    """

    torch_log = logging.getLogger(__name__)

    if methods is None:
        methods = [
            topological_sort_lpmf,
            topological_sort_bfs,
            topological_sort_dfs,
        ]

    torch_log.info("Reordering for peak memory -- %d nodes", len(nodes))

    estimated_peak_memory, name_to_freeable_input_buf = prepare_planning_info(
        nodes,
        name_to_buf,
        name_to_fused_node,
        graph_inputs,
        graph_outputs,
    )

    try:
        validate_graph_acyclic(nodes)
        validate_unique_buffer_names(nodes, name_to_buf, name_to_freeable_input_buf)
    except RuntimeError:
        torch_log.exception("Memory planning validation failed")
        raise

    peak_memory_diff_methods: list = []
    peak_memory_diff_methods.append(
        PeakMemoryResult(nodes, estimated_peak_memory, "baseline")
    )
    torch_log.info("Baseline peak memory: %d", estimated_peak_memory)

    for method in methods:
        try:
            if method is topological_sort_lpmf:
                order = method(
                    nodes, name_to_freeable_input_buf, name_to_buf, graph_outputs
                )
            else:
                order = method(nodes)
            if len(order) != len(nodes):
                raise AssertionError(
                    f"expected order length {len(nodes)}, got {len(order)}"
                )
            peak_memory, _ = estimate_peak_memory(
                order, name_to_freeable_input_buf, graph_outputs
            )
            peak_memory_diff_methods.append(
                PeakMemoryResult(order, peak_memory, method.__name__)
            )
            torch_log.info("%s peak memory: %d", method.__name__, peak_memory)
        except Exception:
            torch_log.exception("Failed to reorder for %s", method.__name__)
            raise

    best_result = min(peak_memory_diff_methods, key=lambda x: x.peak_memory)

    return best_result.order


def export_graph_for_simulator(
    nodes, name_to_freeable_input_buf, name_to_fused_node, graph_inputs, graph_outputs
) -> None:
    """Write down what this order does, for a simulator to be run against.

    A simulator replays the order against a model of the hardware and says what
    it would have cost, which is a check that does not depend on running the
    thing for real.
    """

    import json
    import os

    class ORMBuffer(TypedDict):
        name: str
        size_alloc: int
        size_free: int
        size: int
        is_input: bool
        is_output: bool
        deps: list
        unmet_deps: list

    class ORMNode(TypedDict):
        name: str
        buffer_names: list

    class ORMGraph(TypedDict):
        nodes: list
        buffers: list

    orm_buffers: list = []
    orm_nodes: list = []

    for buf_name, input_buf in name_to_freeable_input_buf.items():
        orm_buf_input_buffer: ORMBuffer = {
            "name": buf_name,
            "size_alloc": input_buf.mpi_buffer.size_free,
            "size_free": input_buf.mpi_buffer.size_free,
            "size": input_buf.mpi_buffer.size_free,
            "is_input": True,
            "is_output": buf_name in graph_outputs,
            "deps": [],
            "unmet_deps": [],
        }
        orm_buffers.append(orm_buf_input_buffer)

    # The mapping has to be worked out again, since the work may have been
    # pruned since it was first built.
    name_to_buf: dict = {
        buf.get_name(): buf for node in nodes for buf in node.get_outputs()
    }
    for buf_name, sched_buf in name_to_buf.items():
        if sched_buf.defining_op is None:
            continue
        deps = [
            pred_buf.get_name()
            for pred_buf in name_to_fused_node[
                sched_buf.defining_op.get_name()
            ].mpi_node.pred_buffers
        ]
        orm_buf_scheduler_buffer: ORMBuffer = {
            "name": buf_name,
            "size_alloc": sched_buf.mpi_buffer.size_alloc,
            "size_free": sched_buf.mpi_buffer.size_free,
            "size": sched_buf.mpi_buffer.size_free,
            "is_input": False,
            "is_output": buf_name in graph_outputs,
            "deps": deps,
            "unmet_deps": [
                buf_name for buf_name in deps if buf_name not in graph_inputs
            ],
        }
        orm_buffers.append(orm_buf_scheduler_buffer)

    for node in nodes:
        orm_node: ORMNode = {
            "name": node.get_name(),
            "buffer_names": list(node.get_buffer_names()),
        }
        orm_nodes.append(orm_node)

    g: ORMGraph = {
        "nodes": orm_nodes,
        "buffers": orm_buffers,
    }

    name = os.path.splitext("fused_graph")[0] + "_fused"
    g_str = json.dumps(g, indent=2)

    if config.reorder_for_peak_memory_debug:
        out_dir = os.environ.get("TP_GRAPH_DUMP_DIR", ".")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            f.write(g_str)
