from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Any

from .fr_logger import FlightRecorderLogger
from .types import (
    Collective,
    EntryState,
    Group,
    MatchInfo,
    MatchState,
    MatchStateRecord,
    Membership,
    Op,
    P2P,
)

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

__all__ = [
    "add_stack_id_in_entries", "align_trace_from_beginning", "check_current_entry_match",
    "check_no_missing_dump_files", "check_version", "error_analysis", "find_coalesced_group",
    "find_coalesced_group_with_non_p2p", "get_version_detail", "just_print_entries",
    "match_coalesced_groups_with_non_p2p", "match_coalesced_groups", "format_frame",
    "format_frames", "match_one_event", "check_size_alltoall",
]

logger = FlightRecorderLogger()


def format_frame(frame: dict[str, str]) -> str:
    return f"{frame.get('name', '<unknown>')} at {frame.get('filename', '<unknown>')}:{frame.get('line', '?')}"


def format_frames(frames: list[dict[str, str]]) -> str:
    return "\n".join(format_frame(frame) for frame in frames)


def match_one_event(event_a: dict[Any, Any], event_b: dict[Any, Any], memberships: dict[str, set[Any]], pg_name: str) -> MatchInfo:
    return Op(event_a, memberships, pg_name).match(Op(event_b, memberships, pg_name))


def _first_op(events: list[tuple[int, dict[str, Any]]], memberships, pg_guids, rank):
    if not events:
        return None
    index, event = events[0]
    key = (event.get("process_group", (event.get("pg", "default"),))[0], rank)
    return index, event, Op(event, memberships, pg_guids.get(key, key[0]))


def match_coalesced_groups(all_rank_events: dict[Any, list[tuple[int, dict[str, Any]]]], group_size: int, groups: dict[str, Group], memberships: dict[str, set[Any]], _pg_guids: dict[tuple[str, int], str]) -> bool:
    del group_size, groups
    pending = {rank: list(events) for rank, events in all_rank_events.items()}
    for rank, events in pending.items():
        if events and Op(events[-1][1], memberships, _pg_guids.get((events[-1][1].get("process_group", ("default",))[0], rank), "default")).type == "coalesced":
            events.pop()
    while pending:
        rank = next(iter(pending))
        if not pending[rank]:
            pending.pop(rank)
            continue
        _, event, op = _first_op(pending[rank], memberships, _pg_guids, rank)
        if op is None:
            pending.pop(rank)
            continue
        if op.type not in P2P:
            return False
        peer = op._dst_g
        peer_events = pending.get(peer, [])
        match_index = next((i for i, (_, other_event) in enumerate(peer_events) if op.match(Op(other_event, memberships, _pg_guids.get((other_event.get("process_group", ("default",))[0], peer), op.pg_name))).state == MatchState.FULLY_MATCHED), None)
        if match_index is None:
            return False
        pending[rank].pop(0)
        peer_events.pop(match_index)
    return True


def match_coalesced_groups_with_non_p2p(all_rank_events: dict[Any, list[tuple[int, dict[str, Any]]]], pg_info: tuple[str, str], memberships: dict[str, set[Any]], _pg_guids: dict[tuple[str, int], str], mismatch: dict[str, int], dumps_ranks: set[int], version: str, collectives: list[Collective], match_record: MatchStateRecord) -> bool:
    del pg_info, mismatch, dumps_ranks, version, collectives, match_record
    return match_coalesced_groups(all_rank_events, len(all_rank_events), {}, memberships, _pg_guids)


def check_size_alltoall(alltoall_cases: list[dict[str, Any]]) -> tuple[bool, int, int]:
    input_numel = sum(math.prod((item.get("input_sizes") or [[0]])[0]) for item in alltoall_cases)
    output_numel = sum(math.prod((item.get("output_sizes") or [[0]])[0]) for item in alltoall_cases)
    return input_numel != output_numel, input_numel, output_numel


def _group_name(event: dict[str, Any]) -> str:
    group = event.get("process_group", event.get("pg", "default"))
    return group[0] if isinstance(group, (tuple, list)) else str(group)


def _group_desc(event: dict[str, Any]) -> str:
    group = event.get("process_group", ("default", "undefined"))
    return str(group[1]) if isinstance(group, (tuple, list)) and len(group) > 1 else "undefined"


def _same_group(event: dict[str, Any], group_name: str, desc: str) -> bool:
    return _group_name(event) == group_name and _group_desc(event) == desc


def check_current_entry_match(all_entries: dict[int, list[dict[str, Any]]], _pg_guids: dict[tuple[str, int], str], pg_info: tuple[str, str], current_entry: dict[str, Any], _memberships: dict[str, set[Any]], mismatch: dict[str, int], match_record: MatchStateRecord) -> None:
    group_name, desc = pg_info
    sequence = int(current_entry.get("collective_seq_id", current_entry.get("id", 0)))
    for rank in sorted(match_record.expected_ranks.intersection(match_record.other_ranks)):
        for index, event in enumerate(all_entries.get(rank, [])):
            if _same_group(event, group_name, desc) and int(event.get("collective_seq_id", event.get("id", 0))) == sequence:
                group_id = _pg_guids.get((_group_name(event), rank), group_name)
                info = match_one_event(current_entry, event, _memberships, group_id)
                if info.state in {MatchState.FULLY_MATCHED, MatchState.UNDECIDED}:
                    match_record.found_ranks.add(rank)
                    match_record.found_idx[rank] = index
                    match_record.has_undecided_case |= info.state == MatchState.UNDECIDED
                else:
                    match_record.candidate_ranks.add(rank)
                    match_record.candidate_idx[rank] = index
                    match_record.errors.add((rank, info))
                break


def error_analysis(all_entries: dict[int, list[dict[str, Any]]], match_record: MatchStateRecord, dumps_ranks: set[int], first_rank: int, current_entry: dict[str, Any], mismatch: dict[str, int], version: tuple[int, int], pg_name: str) -> None:
    del all_entries, first_rank, current_entry, version
    matched = match_record.found_ranks | match_record.candidate_ranks
    if matched != match_record.expected_ranks:
        missing = match_record.expected_ranks - matched
        if missing <= dumps_ranks:
            mismatch[pg_name] += 1
            match_record.entry_state.missing_ranks = missing
        return
    if match_record.errors:
        mismatch[pg_name] += 1
        return
    match_record.found_ranks.update(match_record.candidate_ranks)
    match_record.found_idx.update(match_record.candidate_idx)
    match_record.candidate_ranks.clear()
    match_record.candidate_idx.clear()


def _coalesced(pg_name: str, entries: list[dict[str, Any]], _pg_guids: dict[tuple[str, int], str], rank: int, allow_non_p2p: bool) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    seq = None
    for index, entry in enumerate(entries):
        key = (_group_name(entry), rank)
        if _pg_guids.get(key, pg_name) != pg_name:
            continue
        event_seq = entry.get("p2p_seq_id") if entry.get("is_p2p") else entry.get("collective_seq_id", entry.get("id", 0))
        if seq is None:
            seq = event_seq
            selected.append((index, entry))
            continue
        if event_seq == seq:
            selected.append((index, entry))
            continue
        break
    if len(selected) > 1:
        name = selected[-1][1].get("profiling_name", "")
        if allow_non_p2p or name.endswith("coalesced") or name == "coalesced":
            return selected
    return []


def find_coalesced_group(pg_name: str, entries: list[dict[str, Any]], _pg_guids: dict[tuple[str, int], str], rank: int) -> list[tuple[int, dict[str, Any]]]:
    return _coalesced(pg_name, entries, _pg_guids, rank, False)


def find_coalesced_group_with_non_p2p(pg_name: str, entries: list[dict[str, Any]], _pg_guids: dict[tuple[str, int], str], rank: int) -> list[tuple[int, dict[str, Any]]]:
    return _coalesced(pg_name, entries, _pg_guids, rank, True)


def just_print_entries(all_entries: dict[int, list[dict[str, Any]]], _groups: dict[str, Group], _memberships: dict[str, set[Any]], _pg_guids: dict[tuple[str, int], str], args: argparse.Namespace, stack_id_trace_map: dict[str, int]) -> None:
    selected = set(args.selected_ranks) if args.selected_ranks is not None else set(all_entries)
    rows = []
    while any(all_entries.get(rank) for rank in selected):
        row = []
        for rank in sorted(selected):
            if not all_entries.get(rank):
                row.append("")
                continue
            entry = all_entries[rank].pop(0)
            group = _pg_guids.get((_group_name(entry), rank), _group_name(entry))
            if args.pg_filters and not ({_group_name(entry), _group_desc(entry)} & set(args.pg_filters)):
                row.append("")
            else:
                row.append(str(Op(entry, _memberships, group)))
        rows.append(row)
    output = tabulate(rows, headers=[f"Rank {rank}" for rank in sorted(selected)]) if tabulate else "\n".join(" | ".join(map(str, row)) for row in rows)
    logger.info(output)
    if stack_id_trace_map and args.print_stack_trace:
        logger.info("%s", stack_id_trace_map)


def check_no_missing_dump_files(entries: dict[int, Any], memberships: list[Membership]) -> None:
    expected = {int(item.global_rank) for item in memberships}
    actual = {int(rank) for rank in entries}
    if expected - actual:
        raise AssertionError(f"missing trace files for ranks {sorted(expected - actual)}")


def check_version(version_by_ranks: dict[str, str], version: str) -> None:
    mismatch = {rank: value for rank, value in version_by_ranks.items() if str(value) != str(version)}
    if mismatch:
        raise AssertionError(f"trace versions differ: expected {version}, found {mismatch}")


def get_version_detail(version: str) -> tuple[int, int]:
    parts = str(version).split(".")
    if len(parts) < 2:
        raise AssertionError(f"invalid trace version {version}")
    return int(parts[0]), int(parts[1])


def add_stack_id_in_entries(entries: dict[int, list[dict[str, Any]]]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, int]]:
    trace_to_id: dict[str, int] = {}
    for rank_entries in entries.values():
        for entry in rank_entries:
            frames = entry.get("frames", [])
            key = str(frames)
            if not frames:
                entry["stack_id"] = -1
            else:
                entry["stack_id"] = trace_to_id.setdefault(key, len(trace_to_id))
    return entries, trace_to_id


def align_trace_from_beginning(entries: dict[int, list[dict[str, Any]]]) -> dict[int, list[dict[str, Any]]]:
    starts = [int(rank_entries[0].get("record_id", rank_entries[0].get("id", 0))) for rank_entries in entries.values() if rank_entries]
    if not starts:
        return entries
    start = max(starts)
    for rank, rank_entries in entries.items():
        entries[rank] = [entry for entry in rank_entries if int(entry.get("record_id", entry.get("id", 0))) >= start]
    return entries
