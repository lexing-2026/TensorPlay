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
from check_mps_reference_speedup import evaluate, join, load, summarize

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

    def test_duplicate_measurement_names_are_rejected(self):
        body = payload([("a", 1.0, 2.0), ("a", 1.0, 2.0)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.json"
            path.write_text(json.dumps(body), encoding="utf-8")
            with self.assertRaises(ValueError):
                load(path)

    def test_unexpected_unpaired_case_fails_the_gate(self):
        ours, theirs = _two_sides(["a", "b"], ["a"])
        summary = summarize(join(ours, theirs))
        violations = evaluate(summary, 0.0, 0.0, None)
        self.assertEqual(len(violations), 1)
        self.assertIn("b", violations[0])
        self.assertIn("no recorded skip", violations[0])

    def test_declared_skip_is_not_a_gate_violation(self):
        ours, theirs = _two_sides(["a", "b"], ["a"], requested=["b"])
        summary = summarize(join(ours, theirs))
        self.assertEqual(evaluate(summary, 0.0, 0.0, None), [])
        self.assertEqual(summary["unpaired"], ["b"])

    def test_runtime_unavailable_case_is_not_a_gate_violation(self):
        ours, theirs = _two_sides(["a", "b"], ["a"], unavailable=["b"])
        summary = summarize(join(ours, theirs))
        self.assertEqual(evaluate(summary, 0.0, 0.0, None), [])
        self.assertEqual(summary["unpaired"], ["b"])


def _two_sides(ours_names, reference_names, requested=(), unavailable=()):
    ours = payload([(name, 1.0, None) for name in ours_names],
                   reference=False)
    ours["side"] = "ours"
    theirs = payload([(name, 1.0, None) for name in reference_names],
                     reference=False)
    theirs["side"] = "reference"
    theirs["skipped"] = {"requested": list(requested),
                         "unavailable": {name: "probe failed"
                                         for name in unavailable}}
    return ours, theirs


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

    def test_case_names_are_unique(self):
        for suite in ("short", "long"):
            names = [case[0] for case in
                     build_cases(_TreeBackend("cpu"), suite, "f32")]
            self.assertEqual(len(names), len(set(names)), suite)

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



class TestTwoPassRun(unittest.TestCase):
    """Both passes, each on the host device, joined the way the gate joins."""

    def setUp(self):
        self._device = bench_mps_operator._DEVICE
        bench_mps_operator._DEVICE = "cpu"

    def tearDown(self):
        bench_mps_operator._DEVICE = self._device

    def _run(self, side, out):
        with contextlib.redirect_stdout(io.StringIO()):
            bench_mps_operator.main([
                "--side", side,
                "--suite", "short",
                "--reps", "1",
                "--min-run-time", "0",
                "--warmup", "1",
                "--rounds", "1",
                "--json-out", str(out),
            ])
        return json.loads(out.read_text(encoding="utf-8"))

    def test_two_passes_join_into_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            ours = self._run("ours", Path(tmp) / "ours.json")
            theirs = self._run("reference", Path(tmp) / "ref.json")
        self.assertEqual(ours["side"], "ours")
        self.assertEqual(theirs["side"], "reference")
        self.assertEqual([item["name"] for item in ours["measurements"]],
                         [item["name"] for item in theirs["measurements"]])
        summary = summarize(join(ours, theirs))
        self.assertGreater(summary["compared"], 50)
        self.assertGreater(summary["geomean"], 0.0)
        self.assertLessEqual(summary["win_rate"], 1.0)

    def test_unknown_skip_case_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ours.json"
            # faulthandler needs a real descriptor, so stderr goes to a file.
            with open(Path(tmp) / "stderr.log", "w+",
                      encoding="utf-8") as err:
                with contextlib.redirect_stderr(err):
                    with self.assertRaises(SystemExit):
                        bench_mps_operator.main([
                            "--suite", "short",
                            "--reps", "1",
                            "--min-run-time", "0",
                            "--warmup", "1",
                            "--rounds", "1",
                            "--skip-cases", "not-a-case",
                            "--json-out", str(out),
                        ])

    def test_mismatched_suites_are_rejected(self):
        ours = {
            "schema_version": 1, "benchmark": "mps-operator", "suite": "short",
            "dtype": "f32", "device": "cpu", "side": "ours",
            "measurements": [{"name": "a", "category": "binary",
                              "input_shapes": [[2]], "seconds": 1.0}],
        }
        theirs = dict(ours, side="reference", suite="long")
        with self.assertRaises(ValueError):
            join(ours, theirs)


class TestUnavailableReference(unittest.TestCase):
    """A device the second runtime cannot provision leaves the run alive."""

    def setUp(self):
        self._device = bench_mps_operator._DEVICE
        self._tensor = bench_mps_operator._ReferenceBackend.tensor
        bench_mps_operator._DEVICE = "cpu"

        def refuse(self, array, dtype_key=None):
            raise RuntimeError("device cannot provision the tensor")

        bench_mps_operator._ReferenceBackend.tensor = refuse

    def tearDown(self):
        bench_mps_operator._DEVICE = self._device
        bench_mps_operator._ReferenceBackend.tensor = self._tensor

    def test_run_records_skips_instead_of_aborting(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "mps-operator.json"
            with contextlib.redirect_stdout(io.StringIO()):
                bench_mps_operator.main([
                    "--side", "reference",
                    "--suite", "short",
                    "--reps", "1",
                    "--min-run-time", "0",
                    "--warmup", "1",
                    "--rounds", "1",
                    "--json-out", str(out),
                ])
            body = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(body["side"], "reference")
        self.assertEqual(body["measurements"], [])
        summary = summarize(body)
        self.assertEqual(summary["compared"], 0)
        self.assertEqual(len(evaluate(summary, 1.0, 0.0, None)), 1)


if __name__ == "__main__":
    unittest.main()
