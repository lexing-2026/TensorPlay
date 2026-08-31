from __future__ import annotations

import math
import os
import re
from enum import Enum, auto
from typing import Any, Generic, NamedTuple, TypeVar

from .fr_logger import FlightRecorderLogger

__all__ = [
    "Ref", "TypeInfo", "MatchState", "MatchInfo", "Group", "Membership",
    "Traceback", "Collective", "NCCLCall", "Database", "EntryState", "Op",
    "MatchStateRecord",
]

logger = FlightRecorderLogger()
T = TypeVar("T", bound=NamedTuple)


class Ref(Generic[T]):
    pass


class TypeInfo(NamedTuple):
    name: str
    fields: list[tuple[str, type]]

    @classmethod
    def from_type(cls, value: T) -> "TypeInfo":
        name = getattr(value, "__name__", str(value))
        annotations = getattr(value, "__annotations__", {})
        return cls(name, [(field, annotations.get(field, object)) for field in value._fields])


class MatchState(Enum):
    FULLY_MATCHED = auto()
    COLLECTIVE_TYPE_MISMATCH = auto()
    SIZE_OR_SYNTAX_MISMATCH = auto()
    COLLECTIVE_STATE_MISMATCH = auto()
    COLLECTIVE_DTYPE_MISMATCH = auto()
    UNDECIDED = auto()


class MatchInfo:
    def __init__(self, state: MatchState, culprit: str | None = None) -> None:
        self._state = state
        self.culprit = culprit

    @property
    def state(self) -> MatchState:
        return self._state

    def __str__(self) -> str:
        detail = f", {self.culprit}" if self.culprit else ""
        return f"Error type: {self._state.name}{detail}"


class Group(NamedTuple):
    id: str
    desc: str
    size: int


class Membership(NamedTuple):
    group_id: str
    global_rank: int


class Traceback(NamedTuple):
    id: int
    frames: str


class Collective(NamedTuple):
    id: int
    group_id: str
    pass_check: bool
    collective_seq_id: int
    p2p_seq_id: int
    record_id: int
    pg_desc: str
    collective_name: str
    input_sizes: list[list[int]] | None
    output_sizes: list[list[int]] | None
    expected_ranks: set[int]
    collective_state: str
    collective_frames: list[dict[str, str]]
    input_numel: int | None = None
    output_numel: int | None = None
    missing_ranks: set[int] | None = None
    mismatch_collectives: dict[int, "Collective"] | None = None
    type_of_mismatch: MatchInfo | None = None


class NCCLCall(NamedTuple):
    id: int
    collective_id: Ref[Collective] | int | None
    group_id: str
    global_rank: int
    traceback_id: Ref[Traceback] | int
    collective_type: str
    sizes: list[list[int]] | None


class Database(NamedTuple):
    groups: list[Group]
    memberships: list[Membership]
    tracebacks: list[Traceback]
    collectives: list[Collective]
    ncclcalls: list[NCCLCall]


types = [TypeInfo.from_type(item) for item in (Database, NCCLCall, Collective, Traceback, Membership, Group)]
COLLECTIVES = {
    "broadcast", "_broadcast_oop", "reduce", "_reduce_oop", "all_gather", "all_gather_single",
    "all_gather_v", "all_reduce", "_all_gather_base",
    "reduce_scatter", "reduce_scatter_single", "reduce_scatter_v",
    "_reduce_scatter_base", "gather", "scatter", "all_to_all", "all_to_all_single",
    "all_to_all_v_single", "all_reduce_barrier", "barrier", "split", "new_window",
    "allreduce_coalesced", "allgather_coalesced", "ALLGATHER_coalesced", "REDUCE_SCATTER_coalesced",
}
_UNEVEN_COLLECTIVES = {"all_to_all", "all_to_all_single"}
P2P = {"send", "recv"}


def _shape(value) -> list[list[int]] | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], int):
        return [list(value)]
    return [list(item) for item in value] if isinstance(value, list) else None


class EntryState:
    def __init__(self, entry: dict[str, Any], expected_ranks: set[int]) -> None:
        process_group = entry.get("process_group", (entry.get("pg", "default"), "undefined"))
        if isinstance(process_group, str):
            process_group = (process_group, "undefined")
        self.pg_name, self.desc = process_group[0], process_group[1]
        self.pg_desc = self.pg_name if self.desc == "undefined" else f"{self.pg_name}:{self.desc}"
        self.profiling_name = entry.get("profiling_name", entry.get("collective", "tp:unknown"))
        if ":" not in self.profiling_name:
            self.profiling_name = f"tp:{self.profiling_name}"
        self.collective_seq_id = int(entry.get("collective_seq_id", entry.get("id", 0)))
        self.p2p_seq_id = int(entry.get("p2p_seq_id", self.collective_seq_id))
        self.record_id = int(entry.get("record_id", entry.get("id", 0)))
        self.input_sizes = _shape(entry.get("input_sizes"))
        self.output_sizes = _shape(entry.get("output_sizes"))
        self.collective_state = str(entry.get("state", "completed"))
        self.collective_frames = list(entry.get("frames", []))
        self.expected_ranks = set(expected_ranks)

    def log(self, logger_obj: FlightRecorderLogger, logger_msg: str, frame_formatter, total_numel=None, errors=None, missing_ranks=None) -> None:
        logger_obj.info(logger_msg, self.collective_seq_id)
        logger_obj.info("record id: %s; group: %s; operation: %s", self.record_id, self.pg_desc, self.profiling_name)
        if total_numel:
            logger_obj.info("input numel: %s; output numel: %s", total_numel[0], total_numel[1])
        if missing_ranks:
            logger_obj.info("missing ranks: %s", sorted(missing_ranks))
        if errors:
            logger_obj.info("mismatches: %s", ", ".join(f"rank {rank}: {info}" for rank, info in errors))
        logger_obj.info("frames:\n%s", frame_formatter(self.collective_frames))

    def to_collective(self, id: int, errors=None, idx_map=None, all_entries=None) -> Collective:
        mismatch = None
        if errors:
            mismatch = {}
            for rank, info in errors:
                source = all_entries.get(rank, [])[idx_map[rank]] if all_entries and idx_map and rank in idx_map else None
                if source is None:
                    continue
                state = EntryState(source, self.expected_ranks)
                mismatch[rank] = state.to_collective(id, errors=None, idx_map=None, all_entries=None)._replace(pass_check=False, type_of_mismatch=info)
        return Collective(
            id=id,
            group_id=self.pg_name,
            pass_check=not bool(errors),
            collective_seq_id=self.collective_seq_id,
            p2p_seq_id=self.p2p_seq_id,
            record_id=self.record_id,
            pg_desc=self.pg_desc,
            collective_name=self.profiling_name,
            input_sizes=self.input_sizes,
            output_sizes=self.output_sizes,
            expected_ranks=self.expected_ranks,
            collective_state=self.collective_state,
            collective_frames=self.collective_frames,
            input_numel=getattr(self, "input_numel", None),
            output_numel=getattr(self, "output_numel", None),
            missing_ranks=getattr(self, "missing_ranks", None),
            mismatch_collectives=mismatch,
        )

    def to_nccl_call(self, all_entries, idx_map, nccl_call_id: int, collective_id) -> list[NCCLCall]:
        calls = []
        for rank, index in sorted(idx_map.items()):
            entries = all_entries.get(rank, [])
            if 0 <= index < len(entries):
                entries.pop(index)
            calls.append(NCCLCall(nccl_call_id, collective_id, self.pg_name, int(rank), 0, self.profiling_name, self.input_sizes))
            nccl_call_id += 1
        return calls


class Op:
    def __init__(self, event: dict[Any, Any], memberships: dict[str, set[Any]], pg_name: str):
        self.event = event
        self.profiling_name = event.get("profiling_name", event.get("collective", "tp:unknown"))
        backend, separator, name = self.profiling_name.rpartition(":")
        if not separator:
            backend, name = "tp", self.profiling_name
        self.backend = backend
        parts = name.split(" ")
        self.type = parts[0]
        if self.type not in COLLECTIVES | P2P | {"coalesced"}:
            self.type = name
        self.state = str(event.get("state", "completed"))
        self.pg_name = pg_name
        group = event.get("process_group", (pg_name, "undefined"))
        self.original_pg_name = group[0] if isinstance(group, (tuple, list)) else pg_name
        self.pg_desc = group[1] if isinstance(group, (tuple, list)) and len(group) > 1 else "undefined"
        self._src = None
        self._dst = -1
        if len(parts) > 1 and self.type in P2P:
            meta = parts[1]
            if self.type == "send" and "->" in meta:
                src, dst = meta.split("->", 1)
                self._src, self._dst = int(src), int(dst)
            elif self.type == "recv" and "<-" in meta:
                dst, src = meta.split("<-", 1)
                self._dst, self._src = int(dst), None if src == "?" else int(src)
        ranks = sorted(memberships.get(pg_name, set()))
        self._src_g = ranks[self._src] if self._src is not None and 0 <= self._src < len(ranks) else self._src
        self._dst_g = ranks[self._dst] if 0 <= self._dst < len(ranks) else self._dst
        self.pg_size = len(ranks)
        self.input_sizes = _shape(event.get("input_sizes"))
        self.output_sizes = _shape(event.get("output_sizes"))
        self.input_dtypes = list(event.get("input_dtypes", []))
        self.output_dtypes = list(event.get("output_dtypes", []))
        self.collective_seq_id = int(event.get("collective_seq_id", event.get("id", 0)))
        self.p2p_seq_id = int(event.get("p2p_seq_id", self.collective_seq_id))
        self.stack_id = int(event.get("stack_id", -1))
        self.time_created_ns = int(event.get("time_created_ns", event.get("ts", 0) * 1e9 if isinstance(event.get("ts"), (int, float)) else 0))
        self.collective_frames = list(event.get("frames", []))
        self.is_verbose = os.getenv("FR_TRACE_VERBOSE_OUTPUT", "0") == "1"

    @property
    def src(self) -> int | None:
        if self.type not in P2P:
            raise AssertionError("src is defined only for point-to-point operations")
        return self._src

    @property
    def dst(self) -> int:
        if self.type not in P2P:
            raise AssertionError("dst is defined only for point-to-point operations")
        return self._dst

    def __repr__(self) -> str:
        peer = f"s={self._src_g} d={self._dst_g}, " if self.type in P2P else ""
        return f"{self.type}({peer}input_sizes={self.input_sizes}, state={self.state})"

    def dtype_mismatch(self, other: "Op") -> bool:
        return bool(self.input_dtypes and other.input_dtypes and set(self.input_dtypes) != set(other.input_dtypes)) or bool(self.output_dtypes and other.output_dtypes and set(self.output_dtypes) != set(other.output_dtypes))

    def _gathered_numel_matches(self, gathered, shard) -> bool:
        if not gathered or not shard or not self.pg_size:
            return True
        try:
            return sum(math.prod(item) for item in gathered) == sum(math.prod(item) for item in shard) * self.pg_size
        except (TypeError, ValueError):
            return False

    def match(self, other: "Op") -> MatchInfo:
        src_ok = self._src is None or other._src is None or self._src == other._src
        if self.type == "send":
            ok = other.type == "recv" and src_ok and self._dst == other._dst and self.input_sizes == other.output_sizes
            return MatchInfo(MatchState.FULLY_MATCHED if ok else MatchState.SIZE_OR_SYNTAX_MISMATCH)
        if self.type == "recv":
            ok = other.type == "send" and src_ok and self._dst == other._dst and self.output_sizes == other.input_sizes
            return MatchInfo(MatchState.FULLY_MATCHED if ok else MatchState.SIZE_OR_SYNTAX_MISMATCH)
        if self.type in COLLECTIVES:
            if self.type != other.type:
                return MatchInfo(MatchState.COLLECTIVE_TYPE_MISMATCH, f"{self.type} versus {other.type}")
            if self.type not in _UNEVEN_COLLECTIVES and self.input_sizes != other.input_sizes:
                return MatchInfo(MatchState.SIZE_OR_SYNTAX_MISMATCH, "input sizes differ")
            if self.type not in _UNEVEN_COLLECTIVES and self.output_sizes != other.output_sizes:
                return MatchInfo(MatchState.SIZE_OR_SYNTAX_MISMATCH, "output sizes differ")
            if self.dtype_mismatch(other):
                return MatchInfo(MatchState.COLLECTIVE_DTYPE_MISMATCH, "dtypes differ")
            if self.state != other.state:
                return MatchInfo(MatchState.COLLECTIVE_STATE_MISMATCH, "states differ")
            return MatchInfo(MatchState.UNDECIDED if self.type in _UNEVEN_COLLECTIVES else MatchState.FULLY_MATCHED)
        return MatchInfo(MatchState.FULLY_MATCHED if self.type == other.type else MatchState.SIZE_OR_SYNTAX_MISMATCH)


class MatchStateRecord:
    def __init__(self, expected_ranks: set[int], other_ranks: list[int], entry_state: EntryState, candidate_ranks: set[int], candidate_idx: dict[int, int], found_ranks: set[int], found_idx: dict[int, int], errors: set[tuple[int, MatchInfo]]) -> None:
        self.expected_ranks = expected_ranks
        self.other_ranks = other_ranks
        self.entry_state = entry_state
        self.candidate_ranks = candidate_ranks
        self.candidate_idx = candidate_idx
        self.found_ranks = found_ranks
        self.found_idx = found_idx
        self.errors = errors
        self.has_undecided_case = False

    def reset_for_coalesced(self, entry_state: EntryState, candidate_ranks: set[int]) -> None:
        self.entry_state = entry_state
        self.candidate_ranks = set(candidate_ranks)
        self.candidate_idx = {}
        self.found_ranks = set()
        self.found_idx = {}
        self.errors = set()
