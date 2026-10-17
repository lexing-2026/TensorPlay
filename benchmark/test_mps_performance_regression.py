"""Regression checks for MPS latency comparisons."""

import unittest

from check_mps_performance_regression import compare


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


if __name__ == "__main__":
    unittest.main()
