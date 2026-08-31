from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .fr_logger import FlightRecorderLogger

__all__ = ["read_dump", "read_dir"]

logger = FlightRecorderLogger()
_suffix = re.compile(r"([\w\-]*?)(\d+)$")


def read_dump(prefix: str, filename: str) -> dict[str, Any]:
    basename = os.path.basename(filename)
    rank = int(basename[len(prefix):])
    try:
        with open(filename, "rb") as stream:
            payload = pickle.load(stream)
    except (pickle.UnpicklingError, UnicodeDecodeError, EOFError, ValueError, AttributeError, TypeError):
        with open(filename, encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        payload = {"entries": rows, "version": "0.0", "pg_config": {}}
    result = dict(payload)
    result.setdefault("host_name", f"host_rank{rank}")
    result["rank"] = rank
    result.setdefault("entries", [])
    result.setdefault("version", "0.0")
    result.setdefault("pg_config", {})
    return result


def _determine_prefix(files: list[str]) -> str:
    possible: defaultdict[str, set[int]] = defaultdict(set)
    for filename in files:
        match = _suffix.search(filename)
        if match:
            possible[match.group(1)].add(int(match.group(2)))
    if len(possible) != 1:
        raise ValueError("unable to infer a common trace filename prefix")
    return next(iter(possible))


@contextmanager
def _disable_gc() -> Iterator[None]:
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def read_dir(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], str]:
    trace_dir = getattr(args, "trace_dir", None)
    if not trace_dir or not os.path.isdir(trace_dir):
        raise AssertionError(f"trace directory does not exist: {trace_dir}")
    prefix = getattr(args, "prefix", None)
    details: dict[str, dict[str, Any]] = {}
    version = ""
    started = time.monotonic()
    with _disable_gc():
        for root, _, files in os.walk(trace_dir):
            local_prefix = prefix or _determine_prefix(files)
            for filename in files:
                if not filename.startswith(local_prefix):
                    continue
                suffix = filename[len(local_prefix):]
                if not suffix.isdigit():
                    continue
                path = os.path.join(root, filename)
                item = read_dump(local_prefix, path)
                details[path] = item
                version = version or str(item["version"])
    if not details:
        raise AssertionError(f"no trace files found in {trace_dir}")
    logger.debug("loaded %d trace files in %.3fs", len(details), time.monotonic() - started)
    return details, version
