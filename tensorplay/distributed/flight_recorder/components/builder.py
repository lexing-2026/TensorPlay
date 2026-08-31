from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import os
from collections import defaultdict
from typing import Any

from .fr_logger import FlightRecorderLogger
from .types import Collective, Database, EntryState, Group, Membership, NCCLCall, Traceback
from .utils import (
    add_stack_id_in_entries,
    align_trace_from_beginning,
    check_no_missing_dump_files,
    check_version,
    get_version_detail,
    just_print_entries,
    match_one_event,
)

try:
    from tabulate import tabulate as _tabulate
except ImportError:
    def _tabulate(data, headers=None):
        del headers
        return data

__all__ = ["build_groups_memberships", "build_collectives", "transform_ft", "build_db"]
logger = FlightRecorderLogger()


def _pg_parts(value: str, desc: str = "undefined") -> tuple[str, str]:
    if isinstance(value, (tuple, list)):
        return str(value[0]), str(value[1] if len(value) > 1 else desc)
    return str(value), desc


def _ranks(value) -> list[int]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = [item for item in value.split(",") if item]
    return [int(item) for item in value]


def build_groups_memberships(pg_config: Any) -> tuple[list[Group], dict[Any, Group], list[Membership], dict[str, set[Any]], dict[tuple[str, int], str]]:
    groups: list[Group] = []
    group_map: dict[str, Group] = {}
    memberships: list[Membership] = []
    membership_map: dict[str, set[Any]] = {}
    pg_guids: dict[tuple[str, int], str] = {}
    for rank, configs in pg_config.items():
        rank = int(rank)
        for pg_uid, config in configs.items():
            if isinstance(config, dict):
                desc = str(config.get("desc", "undefined"))
                ranks = _ranks(config.get("ranks", [rank]))
            else:
                desc, ranks = "undefined", _ranks(config)
            digest = hashlib.sha1(",".join(map(str, sorted(ranks))).encode()).hexdigest()[:12]
            group_id = f"{pg_uid}:{digest}"
            pg_guids[(str(pg_uid), rank)] = group_id
            if group_id not in group_map:
                group = Group(group_id, desc, len(ranks))
                groups.append(group)
                group_map[group_id] = group
                membership_map[group_id] = set(ranks)
                memberships.extend(Membership(group_id, member) for member in ranks)
            elif group_map[group_id].desc != desc or membership_map[group_id] != set(ranks):
                raise AssertionError(f"inconsistent group definition for {group_id}")
    return groups, group_map, memberships, membership_map, pg_guids


def _event_group(event: dict[str, Any]) -> tuple[str, str]:
    value = event.get("process_group", event.get("pg", "default"))
    return _pg_parts(value)


def _normalize_entry(event: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(event)
    group, desc = _event_group(event)
    normalized.setdefault("process_group", (group, desc))
    normalized.setdefault("profiling_name", f"tp:{event.get('collective', 'unknown')}")
    normalized.setdefault("collective_seq_id", event.get("id", 0))
    normalized.setdefault("p2p_seq_id", normalized["collective_seq_id"])
    normalized.setdefault("record_id", event.get("id", 0))
    normalized.setdefault("state", "completed")
    normalized.setdefault("input_sizes", [])
    normalized.setdefault("output_sizes", [])
    normalized.setdefault("input_dtypes", [])
    normalized.setdefault("output_dtypes", [])
    return normalized


def _derive_pg_config(entries: dict[int, list[dict[str, Any]]]) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {}
    known: dict[str, set[int]] = defaultdict(set)
    descs: dict[str, str] = {}
    for rank, rank_entries in entries.items():
        for raw in rank_entries:
            event = _normalize_entry(raw)
            group, desc = _event_group(event)
            known[group].add(int(rank))
            descs[group] = desc
    for rank, rank_entries in entries.items():
        result[int(rank)] = {}
        seen = set()
        for raw in rank_entries:
            group, _ = _event_group(_normalize_entry(raw))
            if group in seen:
                continue
            seen.add(group)
            result[int(rank)][group] = {"desc": descs[group], "ranks": repr(sorted(known[group]))}
    return result


def build_collectives(all_entries: dict[int, list[dict[str, Any]]], groups: dict[str, Group], memberships: dict[str, set[Any]], pg_guids: dict[tuple[str, int], str], version: str, mismatch_cap: int = 10) -> tuple[list[Traceback], list[Collective], list[NCCLCall]]:
    entries = {int(rank): [_normalize_entry(event) for event in rank_entries] for rank, rank_entries in all_entries.items()}
    tracebacks: list[Traceback] = []
    traceback_ids: dict[str, int] = {}
    collectives: list[Collective] = []
    calls: list[NCCLCall] = []
    mismatches: defaultdict[str, int] = defaultdict(int)
    while any(entries.values()):
        first_rank = next(rank for rank in sorted(entries) if entries[rank])
        current = entries[first_rank][0]
        group_name, desc = _event_group(current)
        group_id = pg_guids.get((group_name, first_rank), group_name)
        expected = set(memberships.get(group_id, {first_rank}))
        current_seq = int(current.get("collective_seq_id", current.get("id", 0)))
        selected: dict[int, tuple[int, dict[str, Any], object]] = {first_rank: (0, current, None)}
        errors: set[tuple[int, object]] = set()
        for rank in sorted(expected - {first_rank}):
            candidate = None
            for index, event in enumerate(entries.get(rank, [])):
                ev_group, ev_desc = _event_group(event)
                if ev_group != group_name or ev_desc != desc:
                    continue
                if int(event.get("collective_seq_id", event.get("id", 0))) != current_seq:
                    continue
                other_group_id = pg_guids.get((ev_group, rank), group_id)
                info = match_one_event(current, event, memberships, other_group_id)
                candidate = (index, event, info)
                if info.state.name in {"FULLY_MATCHED", "UNDECIDED"}:
                    break
            if candidate is None:
                continue
            selected[rank] = candidate
            info = candidate[2]
            if info is not None and info.state.name not in {"FULLY_MATCHED", "UNDECIDED"}:
                errors.add((rank, info))
        entry_state = EntryState(current, expected)
        missing = expected - set(selected)
        if missing:
            entry_state.missing_ranks = missing
        if errors:
            mismatches[group_id] += 1
        collective = entry_state.to_collective(len(collectives), errors=errors, idx_map={rank: item[0] for rank, item in selected.items()}, all_entries=entries)
        collectives.append(collective)
        frames = str(current.get("frames", []))
        if frames and frames not in traceback_ids:
            traceback_ids[frames] = len(tracebacks)
            tracebacks.append(Traceback(traceback_ids[frames], frames))
        for rank, (index, _, _) in sorted(selected.items()):
            if 0 <= index < len(entries[rank]):
                entries[rank].pop(index)
            calls.append(NCCLCall(len(calls), collective.id if not errors else None, group_id, rank, traceback_ids.get(frames, 0), entry_state.profiling_name, entry_state.input_sizes))
        if mismatches[group_id] > mismatch_cap:
            break
    return tracebacks, collectives, calls


def transform_ft(details: dict[str, dict[str, Any]], group_world_size: int) -> dict[str, dict[str, Any]]:
    if group_world_size <= 0:
        raise ValueError("group_world_size must be positive")
    transformed = copy.deepcopy(details)
    for dump in transformed.values():
        rank = int(dump["rank"])
        for config in dump.get("pg_config", {}).values():
            if config.get("desc") != "default_pg":
                continue
            ranks = _ranks(config.get("ranks", []))
            base = rank // group_world_size * group_world_size
            config["ranks"] = repr([base + item for item in ranks])
    return transformed


def build_db(details: dict[str, dict[str, Any]], args: argparse.Namespace, version: str) -> Database:
    entries: dict[int, list[dict[str, Any]]] = {}
    pg_config: dict[int, Any] = {}
    versions: dict[str, str] = {}
    for dump in details.values():
        rank = int(dump["rank"])
        entries[rank] = [_normalize_entry(item) for item in dump.get("entries", [])]
        versions[str(rank)] = str(dump.get("version", version))
    derived_pg_config = _derive_pg_config(entries)
    for dump in details.values():
        rank = int(dump["rank"])
        pg_config[rank] = dump.get("pg_config") or derived_pg_config.get(rank, {})
    check_version(versions, version)
    entries = align_trace_from_beginning(entries)
    if getattr(args, "just_print_entries", False):
        entries, stack_map = add_stack_id_in_entries(entries)
        groups, _, memberships, membership_map, pg_guids = build_groups_memberships(pg_config)
        just_print_entries(entries, {group.id: group for group in groups}, membership_map, pg_guids, args, stack_map)
        return Database(groups, memberships, [], [], [])
    groups, _, memberships, membership_map, pg_guids = build_groups_memberships(pg_config)
    if not getattr(args, "allow_incomplete_ranks", False):
        check_no_missing_dump_files(entries, memberships)
    tracebacks, collectives, calls = build_collectives(entries, {group.id: group for group in groups}, membership_map, pg_guids, version, getattr(args, "mismatch_cap", 10))
    if getattr(args, "verbose", False):
        logger.debug("groups=%s", _tabulate(groups, headers=Group._fields))
    return Database(groups, memberships, tracebacks, collectives, calls)
