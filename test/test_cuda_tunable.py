"""CUDA GEMM tuning context: enable switches, measurement, persistence.

The tuning context is process-global and intentionally keeps its database
across calls, so every test uses its own GEMM shape (signatures never
collide across tests) and compares database snapshots rather than assuming
an empty start.
"""

import os
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp
from tensorplay.cuda import tunable


class TestCudaTunable(unittest.TestCase):
    def setUp(self):
        if not tp.cuda.is_available():
            self.skipTest("CUDA not available")
        # Known starting state for the switches; the results database
        # itself is only observed through before/after snapshots.
        tunable.disable()
        tunable.tuning_enable()
        tunable.record_untuned_disable()
        tunable.set_verbose(False)
        self.addCleanup(tunable.disable)
        self.addCleanup(tunable.tuning_enable)
        self.addCleanup(tunable.record_untuned_disable)
        self.addCleanup(lambda: tunable.set_verbose(False))

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.results_path = os.path.join(self.tmp.name, "results.csv")
        # Point persistence at the sandbox so nothing lands in the CWD.
        tunable.set_filename(self.results_path)

        self.dev = tp.cuda.current_device()

    def test_enable_disable(self):
        self.assertFalse(tunable.is_enabled())
        tunable.enable()
        self.assertTrue(tunable.is_enabled())
        tunable.disable()
        self.assertFalse(tunable.is_enabled())
        tunable.enable(False)
        self.assertFalse(tunable.is_enabled())

    def test_knob_roundtrips(self):
        tunable.set_max_tuning_duration(7)
        self.assertEqual(tunable.get_max_tuning_duration(), 7)
        tunable.set_max_tuning_duration(0)
        self.assertEqual(tunable.get_max_tuning_duration(), 0)
        tunable.set_max_tuning_samples(11)
        self.assertEqual(tunable.get_max_tuning_samples(), 11)
        tunable.set_max_tuning_samples(0)
        self.assertEqual(tunable.get_max_tuning_samples(), 0)
        tunable.set_verbose(True)
        self.assertTrue(tunable.is_verbose())
        tunable.set_verbose(False)
        self.assertFalse(tunable.is_verbose())
        tunable.tuning_disable()
        self.assertFalse(tunable.tuning_is_enabled())
        tunable.tuning_enable()
        self.assertTrue(tunable.tuning_is_enabled())
        tunable.record_untuned_enable()
        self.assertTrue(tunable.record_untuned_is_enabled())
        tunable.record_untuned_disable()
        self.assertFalse(tunable.record_untuned_is_enabled())

    def test_filename_ordinal(self):
        tunable.set_filename("tune.csv", insert_device_ordinal=True)
        self.assertEqual(tunable.get_filename(), f"tune{self.dev}.csv")
        tunable.set_filename("tune%d.csv", insert_device_ordinal=True)
        self.assertEqual(tunable.get_filename(), f"tune{self.dev}.csv")
        tunable.set_filename("plain.csv")
        self.assertEqual(tunable.get_filename(), "plain.csv")

    def _tuned_matmul(self, m, k, n):
        a = tp.randn(m, k, device="cuda")
        b = tp.randn(k, n, device="cuda")
        tunable.set_max_tuning_samples(2)
        tunable.set_max_tuning_duration(50)
        tunable.enable()
        return a, b, a @ b

    def test_search_records_and_persists(self):
        a, b, c = self._tuned_matmul(128, 256, 96)

        self.assertTrue(os.path.exists(self.results_path))
        with open(self.results_path) as f:
            content = f.read()
        self.assertIn("Validator,TP_TUNABLEOP_FORMAT", content)
        self.assertIn("Validator,CUBLASLT_VERSION", content)
        self.assertIn("Validator,CUDA_DEVICE", content)
        self.assertIn("GemmTunableOp_Float32", content)
        self.assertIn(f"taN_m128_n96_k256_bias0_dev{self.dev}", content)

        results = tunable.get_results()
        matching = [r for r in results if "taN_m128_n96_k256_bias0" in r[1]]
        self.assertEqual(len(matching), 1)
        # The winner is either the heuristic default or a serialized
        # algorithm configuration.
        self.assertTrue(matching[0][2] == "Default" or
                        matching[0][2].startswith("lt_"))
        self.assertGreaterEqual(matching[0][3], 0.0)

    def test_tuned_result_is_correct(self):
        # The first call runs the measurement pass, whose trials write the
        # product with beta = 0 before the real execution; the result must
        # still match the untuned reference.
        a, b, c = self._tuned_matmul(72, 90, 60)
        tunable.disable()
        ref = a @ b
        self.assertTrue(tp.allclose(c, ref, rtol=1e-4, atol=1e-4))

        # fp64: reduction-order differences stay far below these bounds.
        x = tp.randn(64, 128, dtype=tp.float64, device="cuda")
        w = tp.randn(96, 128, dtype=tp.float64, device="cuda")
        bias = tp.randn(96, dtype=tp.float64, device="cuda")
        tunable.enable()
        fused = tp.addmm(bias, x, w.t())
        tunable.disable()
        plain = x @ w.t() + bias
        self.assertTrue(tp.allclose(fused, plain, rtol=1e-10, atol=1e-10))

    def test_replay_from_file(self):
        a, b, c1 = self._tuned_matmul(100, 120, 80)

        # Simulate a later run: no search, results come from the file.
        tunable.disable()
        tunable.tuning_disable()
        self.assertTrue(tunable.read_file(self.results_path))
        tunable.enable()
        c2 = a @ b
        self.assertTrue(tp.allclose(c1, c2, rtol=1e-5, atol=1e-5))

        # A second read of the same file must not duplicate entries.
        tunable.read_file(self.results_path)
        matching = [r for r in tunable.get_results()
                    if "taN_m100_n80_k120_bias0" in r[1]]
        self.assertEqual(len(matching), 1)

    def test_read_file_rejects_mismatched_validators(self):
        before = tunable.get_results()
        a, b, c = self._tuned_matmul(96, 128, 64)

        bad = os.path.join(self.tmp.name, "bad.csv")
        with open(bad, "w") as f:
            f.write("Validator,TP_TUNABLEOP_FORMAT,999\n")
            f.write("Validator,CUBLASLT_VERSION,0\n")
            f.write("Validator,CUDA_DEVICE,0.0:none\n")
            f.write(f"GemmTunableOp_Float32,taN_m1_n1_k1_bias0_dev{self.dev},Default,0\n")
        self.assertFalse(tunable.read_file(bad))
        after_bad = tunable.get_results()
        self.assertEqual(len(after_bad), len(before) + 1)  # only the tuned one

        unknown = os.path.join(self.tmp.name, "unknown.csv")
        with open(unknown, "w") as f:
            f.write("Validator,SOME_OTHER_KEY,1\n")
        self.assertFalse(tunable.read_file(unknown))
        self.assertEqual(tunable.get_results(), after_bad)

    def test_write_file_dumps_everything(self):
        a, b, c = self._tuned_matmul(84, 66, 54)
        other = os.path.join(self.tmp.name, "dump.csv")
        tunable.set_filename(other)
        tunable.write_file()
        self.assertTrue(os.path.exists(other))
        with open(other) as f:
            content = f.read()
        self.assertIn("Validator,TP_TUNABLEOP_FORMAT", content)
        self.assertIn("GemmTunableOp_Float32", content)
        self.assertIn(f"taN_m84_n54_k66_bias0_dev{self.dev}", content)
        # The dump is a valid input again.
        self.assertTrue(tunable.read_file(other))

    def test_record_untuned_logs_signatures(self):
        before = tunable.get_results()
        untuned = os.path.join(
            self.tmp.name, f"tunableop_untuned{self.dev}.csv")
        old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        try:
            tunable.tuning_disable()
            tunable.record_untuned_enable()
            tunable.enable()
            a = tp.randn(32, 64, device="cuda")
            b = tp.randn(64, 48, device="cuda")
            _ = a @ b
        finally:
            os.chdir(old_cwd)
        self.assertTrue(os.path.exists(untuned))
        with open(untuned) as f:
            content = f.read()
        self.assertIn("GemmTunableOp_Float32", content)
        self.assertIn(f"taN_m32_n48_k64_bias0_dev{self.dev}", content)
        # Collection mode tunes nothing and persists no results.
        self.assertFalse(os.path.exists(self.results_path))
        self.assertEqual(tunable.get_results(), before)


class TestCudaTunableWithoutCuda(unittest.TestCase):
    def test_enable_requires_cuda_build(self):
        if tp.cuda.is_available():
            self.skipTest("covered by the CUDA tests above")
        with self.assertRaises(RuntimeError):
            tunable.enable()
        self.assertFalse(tunable.is_enabled())


if __name__ == "__main__":
    unittest.main()
