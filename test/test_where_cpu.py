import numpy as np

import tensorplay as tp

from tensorplay.testing._internal.common_utils import TestCase, run_tests
from tensorplay.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
)


class TestWhere(TestCase):
    @onlyCPU
    def test_select_matches_the_condition_byte_for_byte(self, device):
        rng = np.random.RandomState(7)
        for dtype, np_dtype in ((tp.float32, "float32"), (tp.float64, "float64"),
                                (tp.int64, "int64")):
            # 1 << 20 crosses the grain the flat select is split over, and the
            # three densities keep both sides of the select populated.
            for density in (0.0, 1.0 / 3.0, 1.0):
                n = 1 << 18
                a = rng.standard_normal(n) * 4
                b = rng.standard_normal(n) * 4
                if np_dtype == "int64":
                    a = np.round(a).astype("int64")
                    b = np.round(b).astype("int64")
                else:
                    a = a.astype(np_dtype)
                    b = b.astype(np_dtype)
                cond = rng.random_sample(n) < density
                expected = np.where(cond, a, b)
                got = tp.where(tp.tensor(cond, device=device),
                               tp.tensor(a.copy(), device=device),
                               tp.tensor(b.copy(), device=device))
                np.testing.assert_array_equal(got.numpy(), expected)

    @onlyCPU
    def test_select_reads_a_strided_condition_and_operands(self, device):
        rng = np.random.RandomState(9)
        base = rng.standard_normal((512, 512)).astype("float32")
        other = rng.standard_normal((512, 512)).astype("float32")
        cond = (base > 0).astype(bool)
        expected = np.where(cond.T, base.T, other.T)
        got = tp.where(tp.tensor(cond.T.copy(), device=device),
                       tp.tensor(base.T.copy(), device=device),
                       tp.tensor(other.T.copy(), device=device))
        np.testing.assert_array_equal(got.numpy(), expected)


instantiate_device_type_tests(TestWhere, globals())

if __name__ == "__main__":
    run_tests()
