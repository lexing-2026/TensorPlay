import hashlib
import os
import sys

import pytest

# Make the in-repo package (including tensorplay.testing) importable when
# running against a source checkout rather than an installed wheel.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def pytest_addoption(parser):
    group = parser.getgroup("sharding", "split one run across machines")
    group.addoption(
        "--shard-id", type=int, default=None,
        help="1-indexed shard to execute in this run (requires --num-shards)")
    group.addoption(
        "--num-shards", type=int, default=None,
        help="total number of shards the run is split into")


def _bucket(nodeid: str, num_shards: int) -> int:
    # A stable hash of the node id spreads tests evenly and without
    # coordination across shards, and re-sharding after adding tests only
    # moves the new tests.
    digest = hashlib.sha256(nodeid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def pytest_collection_modifyitems(config, items):
    shard_id = config.getoption("--shard-id")
    num_shards = config.getoption("--num-shards")
    if shard_id is None and num_shards is None:
        return
    if shard_id is None or num_shards is None:
        raise pytest.UsageError("--shard-id and --num-shards must be given together")
    if not 1 <= shard_id <= num_shards:
        raise pytest.UsageError(f"--shard-id must be between 1 and {num_shards}")

    keep, drop = [], []
    for item in items:
        (keep if _bucket(item.nodeid, num_shards) == shard_id - 1
         else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
    items[:] = keep
