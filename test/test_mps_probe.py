"""Throwaway probe: prints what each single dispatch hands back on the macOS 14
runner, one operation at a time, and marks the case that ends the process.

Written with plain prints so the run is readable even when a case kills it: the
lane runs with captured output disabled, so everything before the death is in
the log.
"""

import platform
import sys
import traceback
import unittest

import tensorplay as tp


def _check(label, got, want):
    print(f"{label}: got={got!r} want={want!r} "
          f"{'OK' if got == want else 'BAD'}", flush=True)


def _run(label, thunk, want):
    try:
        value = thunk()
    except Exception:
        print(f"{label}: RAISED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        return
    _check(label, value, want)


class ProbeAll(unittest.TestCase):
    def test_01_report(self):
        print(f"host={platform.platform()} mac={platform.mac_ver()[0]} "
              f"machine={platform.machine()} "
              f"mps={tp.mps.is_available()}", flush=True)

    def test_02_tensor_tensor(self):
        a = tp.full((2, 3), 6, device="mps", dtype=tp.float32)
        b = tp.full((2, 3), 4, device="mps", dtype=tp.float32)
        _run("add", lambda: (a + b).cpu().tolist(), [[10.0] * 3] * 2)
        _run("sub", lambda: (a - b).cpu().tolist(), [[2.0] * 3] * 2)
        _run("mul", lambda: (a * b).cpu().tolist(), [[24.0] * 3] * 2)
        _run("div", lambda: (a / b).cpu().tolist(), [[1.5] * 3] * 2)
        _run("pow_tensors", lambda: (a ** b).cpu().tolist(),
             [[float(6 ** 4)] * 3] * 2)
        _run("remainder", lambda: (a % b).cpu().tolist(), [[2.0] * 3] * 2)
        _run("fmod", lambda: tp.fmod(a, b).cpu().tolist(), [[2.0] * 3] * 2)
        _run("maximum", lambda: tp.maximum(a, b).cpu().tolist(), [[6.0] * 3] * 2)
        _run("minimum", lambda: tp.minimum(a, b).cpu().tolist(), [[4.0] * 3] * 2)

    def test_03_scalar_operand(self):
        a = tp.full((2, 3), 6, device="mps", dtype=tp.float32)
        _run("add_scalar", lambda: (a + 2).cpu().tolist(), [[8.0] * 3] * 2)
        _run("sub_scalar", lambda: (a - 2).cpu().tolist(), [[4.0] * 3] * 2)
        _run("mul_scalar", lambda: (a * 2).cpu().tolist(), [[12.0] * 3] * 2)
        _run("div_scalar", lambda: (a / 2).cpu().tolist(), [[3.0] * 3] * 2)
        _run("pow_scalar", lambda: tp.pow(a, 2).cpu().tolist(), [[36.0] * 3] * 2)

    def test_04_inplace(self):
        base = tp.full((2, 3), 6, device="mps", dtype=tp.float32)
        for label, apply, want in (("add_", lambda x: x.add_(2), [[8.0] * 3] * 2),
                                   ("sub_", lambda x: x.sub_(2), [[4.0] * 3] * 2),
                                   ("mul_", lambda x: x.mul_(2), [[12.0] * 3] * 2),
                                   ("div_", lambda x: x.div_(2), [[3.0] * 3] * 2)):
            def make(apply=apply):
                x = base.clone()
                apply(x)
                return x.cpu().tolist()
            _run(label, make, want)

    def test_05_addc(self):
        x = tp.full((2, 3), 1, device="mps", dtype=tp.float32)
        y = tp.full((2, 3), 2, device="mps", dtype=tp.float32)
        z = tp.full((2, 3), 3, device="mps", dtype=tp.float32)
        _run("addcmul", lambda: tp.addcmul(x, y, z).cpu().tolist(),
             [[7.0] * 3] * 2)
        _run("addcdiv", lambda: tp.addcdiv(x, y, z).cpu().tolist(),
             [[1 + 2 / 3.0] * 3] * 2)

    def test_06_unary(self):
        four = tp.full((2, 2), 4, device="mps", dtype=tp.float32)
        zero = tp.zeros((2, 2), device="mps")
        one = tp.ones((2, 2), device="mps")
        _run("abs", lambda: tp.abs(-four).cpu().tolist(), [[4.0] * 2] * 2)
        _run("neg", lambda: tp.neg(four).cpu().tolist(), [[-4.0] * 2] * 2)
        _run("square", lambda: tp.square(four).cpu().tolist(), [[16.0] * 2] * 2)
        _run("sqrt", lambda: tp.sqrt(four).cpu().tolist(), [[2.0] * 2] * 2)
        _run("rsqrt", lambda: tp.rsqrt(four).cpu().tolist(), [[0.5] * 2] * 2)
        _run("reciprocal", lambda: tp.reciprocal(four).cpu().tolist(),
             [[0.25] * 2] * 2)
        _run("exp", lambda: tp.exp(zero).cpu().tolist(), [[1.0] * 2] * 2)
        _run("log", lambda: tp.log(one).cpu().tolist(), [[0.0] * 2] * 2)
        _run("floor", lambda: tp.floor(tp.full((2, 2), 3.7, device="mps")).cpu().tolist(),
             [[3.0] * 2] * 2)
        _run("ceil", lambda: tp.ceil(tp.full((2, 2), 3.2, device="mps")).cpu().tolist(),
             [[4.0] * 2] * 2)
        _run("round", lambda: tp.round(tp.full((2, 2), 3.5, device="mps")).cpu().tolist(),
             [[4.0] * 2] * 2)
        _run("trunc", lambda: tp.trunc(tp.full((2, 2), -3.7, device="mps")).cpu().tolist(),
             [[-3.0] * 2] * 2)
        _run("relu", lambda: tp.relu(-four).cpu().tolist(), [[0.0] * 2] * 2)

    def test_07_integer_and_select(self):
        ints = tp.full((2,), 0, device="mps", dtype=tp.int32)
        _run("bitwise_not", lambda: tp.bitwise_not(ints).cpu().tolist(), [-1, -1])
        cond = tp.full((2, 3), 1, device="mps", dtype=tp.float32) > 0
        x = tp.full((2, 3), 1, device="mps", dtype=tp.float32)
        y = tp.full((2, 3), 2, device="mps", dtype=tp.float32)
        _run("where", lambda: tp.where(cond, x, y).cpu().tolist(), [[1.0] * 3] * 2)
        five = tp.full((4,), 5, device="mps", dtype=tp.float32)
        _run("clamp", lambda: tp.clamp(five, 1, 3).cpu().tolist(), [3.0] * 4)
        _run("clamp_min", lambda: tp.clamp_min(five, 2).cpu().tolist(), [5.0] * 4)
        _run("clamp_max", lambda: tp.clamp_max(five, 2).cpu().tolist(), [2.0] * 4)

    def test_08_relations(self):
        a = tp.full((2, 3), 2, device="mps", dtype=tp.float32)
        b = tp.full((2, 3), 3, device="mps", dtype=tp.float32)
        _run("lt", lambda: (a < b).cpu().tolist(), [[True] * 3] * 2)
        _run("le", lambda: (a <= b).cpu().tolist(), [[True] * 3] * 2)
        _run("eq", lambda: (a == a).cpu().tolist(), [[True] * 3] * 2)
        _run("ne", lambda: (a != b).cpu().tolist(), [[True] * 3] * 2)
        _run("gt", lambda: (a > b).cpu().tolist(), [[False] * 3] * 2)
        _run("ge", lambda: (a >= b).cpu().tolist(), [[False] * 3] * 2)

    def test_09_reduce_shapes(self):
        x = tp.arange(24, dtype=tp.float32, device="mps").reshape(2, 3, 4)
        _run("sum_dim1", lambda: tp.sum(x, dim=1).cpu().tolist(),
             [[12.0, 15.0, 18.0, 21.0], [48.0, 51.0, 54.0, 57.0]])
        _run("mean_dim1", lambda: tp.mean(x, dim=1).cpu().tolist(),
             [[4.0, 5.0, 6.0, 7.0], [16.0, 17.0, 18.0, 19.0]])
        _run("mean_all", lambda: tp.mean(x).item(), 11.5)
        _run("max_all", lambda: tp.max(x).item(), 23.0)

    def test_10_mean_multi_axis(self):
        # The case that ends the process on this runner, so it comes last.
        x = tp.arange(24, dtype=tp.float32, device="mps").reshape(2, 3, 4)
        _run("mean_dim0", lambda: tp.mean(x, dim=(0,)).cpu().tolist(),
             [[6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0],
              [14.0, 15.0, 16.0, 17.0]])
        _run("mean_dims02", lambda: tp.mean(x, dim=(0, 2)).cpu().tolist(),
             [10.0, 11.0, 12.0, 13.0])


if __name__ == "__main__":
    unittest.main()
