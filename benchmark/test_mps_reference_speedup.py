"""Regression checks for the MPS speedup gate and the paired case inventory."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import bench_mps_operator
from bench_mps_operator import _ReferenceBackend, _TreeBackend, build_cases
from check_mps_reference_speedup import evaluate, summarize

# Cases above this footprint are skipped by the inventory check: the pair of
# runtimes materializes both sides at once, which is more than a CI shell wants
# to hold for a correctness check that only needs the argument order to agree.
MAX_CASE_BYTES = 512 << 20


def payload(pairs, reference=True):
    measurements = []
    for name, ours, theirs in pairs:
        item = {
            "name": name,
            "category": "binary",
            "input_shapes": [[1024]],
            "seconds": ours,
        }
        if theirs is not None:
            item["reference_seconds"] = theirs
            item["speedup"] = theirs / ours
        measurements.append(item)
    body = {
        "schema_version": 1,
        "benchmark": "mps-operator",
        "suite": "short",
        "dtype": "f32",
        "device": "mps",
        "measurements": measurements,
    }
    if reference:
        body["reference"] = {"runtime": "reference", "version": "1.0",
                             "device": "mps"}
    return body


class TestSpeedupGate(unittest.TestCase):
    def test_faster_overall_passes(self):
        summary = summarize(payload([("a", 1.0, 2.0), ("b", 1.0, 2.0)]))
        self.assertEqual(evaluate(summary, 1.0, 0.0, None), [])
        self.assertAlmostEqual(summary["geomean"], 2.0)

    def test_slower_overall_fails(self):
        summary = summarize(payload([("a", 2.0, 1.0), ("b", 2.0, 1.0)]))
        violations = evaluate(summary, 1.0, 0.0, None)
        self.assertEqual(len(violations), 1)
        self.assertIn("overall speedup", violations[0])

    def test_mixed_cases_average_geometrically(self):
        summary = summarize(payload([("a", 1.0, 4.0), ("b", 4.0, 1.0)]))
        self.assertAlmostEqual(summary["geomean"], 1.0)
        self.assertAlmostEqual(summary["win_rate"], 0.5)

    def test_win_rate_floor_is_enforced(self):
        summary = summarize(payload([("a", 1.0, 4.0), ("b", 2.0, 1.0)]))
        violations = evaluate(summary, 0.0, 0.75, None)
        self.assertEqual(len(violations), 1)
        self.assertIn("win rate", violations[0])

    def test_unpaired_run_cannot_pass(self):
        summary = summarize(payload([("a", 1.0, None)], reference=False))
        self.assertEqual(summary["compared"], 0)
        violations = evaluate(summary, 0.0, 0.0, None)
        self.assertEqual(len(violations), 1)
        self.assertIn("--reference", violations[0])

    def test_per_case_floor_reports_the_slow_case(self):
        summary = summarize(payload([("a", 1.0, 1.0), ("b", 3.0, 1.0)]))
        violations = evaluate(summary, 0.0, 0.0, 1.0)
        self.assertEqual(len(violations), 1)
        self.assertIn("b:", violations[0])

    def test_broken_timings_are_ignored(self):
        body = payload([("a", 1.0, 2.0)])
        body["measurements"].append(
            {"name": "b", "category": "binary", "input_shapes": [[1024]],
             "seconds": 0.0, "reference_seconds": 1.0}
        )
        summary = summarize(body)
        self.assertEqual(summary["compared"], 1)


def _primary(result):
    """The value a case produces, for tuple-returning reductions."""
    if isinstance(result, tuple) and not hasattr(result, "shape"):
        return result[0]
    return result


def _to_numpy(value):
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _input_bytes(specs, itemsize=4):
    return sum(int(np.prod(spec[1])) * itemsize for spec in specs)


class TestCaseInventory(unittest.TestCase):
    """One inventory drives both timed sides, so the pair has to agree.

    Both backends run on the host device here: the check is that a case names
    the same operation, arguments and dtype on each side, which a shape and
    value comparison catches before either number is published.
    """

    def test_paired_cases_compute_the_same_thing(self):
        for suite in ("short", "long"):
            ours = build_cases(_TreeBackend("cpu"), suite, "f32")
            theirs = build_cases(_ReferenceBackend("cpu"), suite, "f32")
            self.assertEqual([case[0] for case in ours],
                             [case[0] for case in theirs])
            for (name, category, factory, specs, _), (_, _, other, _, _) in \
                    zip(ours, theirs):
                if category == "dropout":
                    continue
                if _input_bytes(specs) > MAX_CASE_BYTES:
                    continue
                with self.subTest(suite=suite, case=name):
                    mine = _primary(factory()())
                    ref = _primary(other()())
                    if "argsort" in name:
                        # Equal values may be ordered differently by two sorts;
                        # each result still has to be a permutation of the axis.
                        width = _to_numpy(mine).shape[-1]
                        expected = np.tile(np.arange(width), (1024, 1))
                        for candidate in (mine, ref):
                            self.assertTrue(np.array_equal(
                                np.sort(_to_numpy(candidate), axis=1),
                                expected,
                            ))
                        continue
                    mine = _to_numpy(mine)
                    ref = _to_numpy(ref)
                    self.assertEqual(mine.shape, ref.shape)
                    if mine.dtype.kind in "iub" or ref.dtype.kind in "iub":
                        self.assertTrue(np.array_equal(mine, ref))
                    else:
                        np.testing.assert_allclose(
                            mine, ref, rtol=3e-3, atol=3e-3, equal_nan=True)



class TestPairedRun(unittest.TestCase):
    """The whole script, on the host device, with the reference side enabled.

    Only the device changes; argument handling, sampling, the JSON shape and
    the gate all run exactly as they do for a Metal run.
    """

    def setUp(self):
        self._device = bench_mps_operator._DEVICE
        bench_mps_operator._DEVICE = "cpu"

    def tearDown(self):
        bench_mps_operator._DEVICE = self._device

    def test_reference_run_records_paired_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "mps-operator.json"
            with contextlib.redirect_stdout(io.StringIO()):
                bench_mps_operator.main([
                    "--suite", "short",
                    "--reps", "1",
                    "--min-run-time", "0",
                    "--warmup", "1",
                    "--rounds", "1",
                    "--reference",
                    "--json-out", str(out),
                ])
            body = json.loads(out.read_text(encoding="utf-8"))
            summary = summarize(body)
            self.assertIn("reference", body)
            self.assertEqual(body["device"], "cpu")
            self.assertGreater(summary["compared"], 50)
            self.assertGreater(summary["geomean"], 0.0)
            self.assertTrue(all(row["speedup"] > 0 for row in summary["rows"]))


if __name__ == "__main__":
    unittest.main()
