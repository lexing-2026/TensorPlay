"""Regression checks for MPS latency comparisons."""

import unittest

from check_mps_performance_regression import compare, median_baseline


def results(seconds):
    return {
        "benchmark": "mps-operator",
        "suite": "short",
        "dtype": "f32",
        "device": "mps",
        "measurements": [
            {"name": f"case-{i}", "category": "binary",
             "input_shapes": [[1024]], "seconds": value}
            for i, value in enumerate(seconds)
        ],
    }


def reference(seconds):
    payload = results(seconds)
    payload["benchmark"] = "mps-operator-reference"
    payload["side"] = "reference"
    return payload


def history_run(ours, theirs):
    return {"suites": {"mps-operator": results(ours),
                       "mps-operator-reference": reference(theirs)}}


class TestMPSPerformanceRegression(unittest.TestCase):
    def test_overall_speedup_allows_unchanged_cases(self):
        baseline = results([1.0] * 5)
        current = results([0.5, 0.5, 0.5, 0.9, 1.0])
        self.assertEqual(compare(current, baseline, 0.30), [])

    def test_overall_speedup_still_detects_actual_regression(self):
        baseline = results([1.0] * 5)
        current = results([0.5, 0.5, 0.5, 1.0, 1.4])
        violations = compare(current, baseline, 0.30)
        self.assertEqual(len(violations), 1)
        self.assertIn("case-4", violations[0])

    def test_slower_runner_still_detects_relative_regression(self):
        baseline = results([1.0] * 5)
        current = results([2.0, 2.0, 2.0, 2.0, 2.7])
        violations = compare(current, baseline, 0.30)
        self.assertEqual(len(violations), 1)
        self.assertIn("case-4", violations[0])

    def test_uniform_runner_shift_is_normalized(self):
        self.assertEqual(compare(results([2.0] * 5), results([1.0] * 5), 0.30), [])

    def test_history_baseline_is_scaled_by_same_run_reference(self):
        current = results([0.15] * 5)
        baseline = median_baseline(
            current, [history_run([0.05] * 5, [0.1] * 5)], {f"case-{i}": 0.3
                                                           for i in range(5)})
        self.assertAlmostEqual(baseline["measurements"][0]["seconds"], 0.15)
        self.assertEqual(compare(current, baseline, 0.30), [])

    def test_calibrated_baseline_still_detects_regression(self):
        current = results([0.15, 0.15, 0.15, 0.15, 0.2])
        baseline = median_baseline(
            current, [history_run([0.05] * 5, [0.1] * 5)], {f"case-{i}": 0.3
                                                           for i in range(5)})
        violations = compare(current, baseline, 0.30)
        self.assertEqual(len(violations), 1)
        self.assertIn("case-4", violations[0])

    def test_baseline_without_reference_stays_raw(self):
        current = results([0.15] * 5)
        baseline = median_baseline(current, [history_run([0.05] * 5,
                                                         [0.1] * 5)], {})
        self.assertAlmostEqual(baseline["measurements"][0]["seconds"], 0.05)


if __name__ == "__main__":
    unittest.main()
