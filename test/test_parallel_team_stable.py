"""The intra-op worker team is created once and kept.

Loops of different lengths ask for different numbers of chunks.  The team
that serves them must stay the same set of threads: a team resized per loop
retires its surplus workers and spawns new ones at every change, which costs
thread creation on the hot path and lands the fresh threads on arbitrary
CPUs.
"""
import os
import sys

import pytest

import tensorplay as tp

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="reads /proc/self/task")


def _thread_ids():
    return set(os.listdir("/proc/self/task"))


def _mixed_workload():
    # Chunk counts from one up to the full team: the grain size is 32768
    # elements, so these lengths split into 1, 2, 3 and many chunks.
    for n in (1000, 40000, 70000, 100000, 4000000, 70000, 40000, 4000000):
        x = tp.ones(n)
        y = (x * 2.0 + 1.0).sum()
        assert float(y) == 3.0 * n


def test_worker_threads_are_not_respawned_between_loops():
    if tp.get_num_threads() < 3:
        pytest.skip("needs a team of at least three threads")
    _mixed_workload()          # brings the team up
    before = _thread_ids()
    for _ in range(20):
        _mixed_workload()
    after = _thread_ids()
    assert after == before


def test_chunks_cover_the_range_exactly_once():
    # Sums over lengths around the chunking boundaries: an element visited
    # twice or skipped changes the total.
    for n in (1, 2, 32767, 32768, 32769, 65537, 98305, 1000003):
        x = tp.arange(n, dtype=tp.float64)
        assert float(x.sum()) == n * (n - 1) / 2
        assert float((x + 1.0).sum()) == n * (n + 1) / 2
