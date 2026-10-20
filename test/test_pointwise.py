import unittest
import tensorplay as tp
import math

class TestPointwiseOps(unittest.TestCase):
    def test_unary_ops(self):
        # Test basic unary ops
        x = tp.tensor([-1.0, 0.0, 1.0, 2.0])
        
        # abs
        y = x.abs()
        self.assertEqual(y.tolist(), [1.0, 0.0, 1.0, 2.0])
        
        # neg
        y = x.neg()
        self.assertEqual(y.tolist(), [1.0, -0.0, -1.0, -2.0])
        
        # square
        y = x.square()
        self.assertEqual(y.tolist(), [1.0, 0.0, 1.0, 4.0])
        
        # sign
        y = x.sign()
        self.assertEqual(y.tolist(), [-1.0, 0.0, 1.0, 1.0])
        
        # exp
        y = x.exp()
        expected = [math.exp(v) for v in x.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)
            
        # log (avoid log(0) and log(-1))
        x_pos = tp.tensor([1.0, 2.0, math.e])
        y = x_pos.log()
        expected = [math.log(v) for v in x_pos.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)
            
        # sqrt
        y = x_pos.sqrt()
        expected = [math.sqrt(v) for v in x_pos.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)

    def test_trig_ops(self):
        x = tp.tensor([0.0, math.pi/2, math.pi])
        
        # sin
        y = x.sin()
        expected = [math.sin(v) for v in x.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)
            
        # cos
        y = x.cos()
        expected = [math.cos(v) for v in x.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)
            
        # tan
        x_tan = tp.tensor([0.0, math.pi/4])
        y = x_tan.tan()
        expected = [math.tan(v) for v in x_tan.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)
            
        # inverse trig (acos, asin, atan)
        x_inv = tp.tensor([-0.5, 0.0, 0.5])
        
        y = x_inv.asin()
        expected = [math.asin(v) for v in x_inv.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)
            
        y = x_inv.acos()
        expected = [math.acos(v) for v in x_inv.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)
            
        y = x_inv.atan()
        expected = [math.atan(v) for v in x_inv.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)

    def test_hyperbolic_ops(self):
        x = tp.tensor([-1.0, 0.0, 1.0])
        
        # sinh
        y = x.sinh()
        expected = [math.sinh(v) for v in x.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)
            
        # cosh
        y = x.cosh()
        expected = [math.cosh(v) for v in x.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)
            
        # tanh
        y = x.tanh()
        expected = [math.tanh(v) for v in x.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)
            
    def test_sigmoid(self):
        x = tp.tensor([-1.0, 0.0, 1.0])
        y = x.sigmoid()
        # sigmoid(x) = 1 / (1 + exp(-x))
        expected = [1.0 / (1.0 + math.exp(-v)) for v in x.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)
            
    def test_other_math_ops(self):
        # rsqrt
        x = tp.tensor([1.0, 4.0, 9.0])
        y = x.rsqrt()
        expected = [1.0 / math.sqrt(v) for v in x.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)
            
        # angle (for real numbers: 0 if >=0, pi if <0)
        x_angle = tp.tensor([1.0, -1.0, 0.0])
        y = x_angle.angle()
        expected = [0.0, math.pi, 0.0]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)
            
    def test_clamp(self):
        x = tp.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
        
        # clamp(min=0)
        y = x.clamp(min=0.0)
        self.assertEqual(y.tolist(), [0.0, 0.0, 1.0, 2.0, 3.0])
        
        # clamp(max=1)
        y = x.clamp(max=1.0)
        self.assertEqual(y.tolist(), [-1.0, 0.0, 1.0, 1.0, 1.0])
        
        # clamp(min=0, max=2)
        y = x.clamp(min=0.0, max=2.0)
        self.assertEqual(y.tolist(), [0.0, 0.0, 1.0, 2.0, 2.0])

    def test_pow(self):
        x = tp.tensor([1.0, 2.0, 3.0])
        
        # pow(scalar)
        y = x.pow(2)
        self.assertEqual(y.tolist(), [1.0, 4.0, 9.0])
        
        y = x.pow(0.5)
        self.assertAlmostEqual(y.tolist()[1], math.sqrt(2.0), places=5)

    def test_softmax(self):
        x = tp.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        # Softmax along dim 1
        y = x.softmax(dim=1)
        
        # Check sum is 1
        s = y.sum(dim=[1])
        for v in s.tolist():
            self.assertAlmostEqual(v, 1.0, places=5)
            
        # Check values manually for one row
        # e^1, e^2, e^3
        e1 = math.exp(1)
        e2 = math.exp(2)
        e3 = math.exp(3)
        total = e1 + e2 + e3
        expected = [e1/total, e2/total, e3/total]
        
        row0 = y.tolist()[0]
        for a, b in zip(row0, expected):
            self.assertAlmostEqual(a, b, places=5)

    def test_other_ops(self):
        # ceil, floor, round
        x = tp.tensor([1.1, 1.5, 1.9, -1.1])
        
        self.assertEqual(x.ceil().tolist(), [2.0, 2.0, 2.0, -1.0])
        self.assertEqual(x.floor().tolist(), [1.0, 1.0, 1.0, -2.0])
        self.assertEqual(x.round().tolist(), [1.0, 2.0, 2.0, -1.0])

    def test_pow_tensor(self):
        x = tp.tensor([1.0, 2.0, 3.0])
        exp = tp.tensor([2.0, 3.0, 2.0])
        
        y = x.pow(exp)
        self.assertEqual(y.tolist(), [1.0, 8.0, 9.0])
        
        # Broadcasting
        x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
        exp = tp.tensor([2.0, 1.0]) # Broadcasts to [[2, 1], [2, 1]]
        y = x.pow(exp)
        self.assertEqual(y.tolist(), [[1.0, 2.0], [9.0, 4.0]])

    def test_lerp(self):
        start = tp.tensor([0.0, 0.0])
        end = tp.tensor([10.0, 10.0])
        
        # lerp with scalar weight
        y = start.lerp(end, 0.5)
        self.assertEqual(y.tolist(), [5.0, 5.0])
        
        # lerp with tensor weight
        weight = tp.tensor([0.2, 0.8])
        y = start.lerp(end, weight)
        self.assertEqual(y.tolist(), [2.0, 8.0])

    def test_activation_ops(self):
        x = tp.tensor([-1.0, 0.0, 1.0, 2.0])
        
        # relu
        y = x.relu()
        self.assertEqual(y.tolist(), [0.0, 0.0, 1.0, 2.0])
        
        # gelu
        # GELU(0) = 0
        # GELU(1) = 0.5 * 1 * (1 + erf(1/sqrt(2))) approx 0.8413
        # GELU(-1) = -0.5 * (1 + erf(-1/sqrt(2))) approx -0.1587
        y = x.gelu()
        # Check values
        def gelu_ref(v):
            return 0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0)))
            
        expected = [gelu_ref(v) for v in x.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)
            
        # silu
        # SiLU(x) = x * sigmoid(x)
        y = x.silu()
        def silu_ref(v):
            return v / (1.0 + math.exp(-v))
            
        expected = [silu_ref(v) for v in x.tolist()]
        for a, b in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, b, places=5)

    def test_clamp_reduced_precision(self):
        # Every value and bound below is exact in both reduced widths, so a
        # wrong result cannot hide inside a rounding allowance: the width only
        # selects.  The length (30) fills several 8-wide vectors and leaves a
        # tail, so the tail has to agree with the vectors it follows.
        inf = float('inf')
        nan = float('nan')
        vals = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, -inf, inf, nan]

        def same(got, want):
            return all(g == w or (g != g and w != w) for g, w in zip(got, want))

        for dt in (tp.float16, tp.bfloat16):
            x = tp.tensor(vals * 3, dtype=dt)
            self.assertTrue(same(tp.clamp(x, -1.0, 1.0).tolist(),
                                 [-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0,
                                  -1.0, 1.0, nan] * 3))
            self.assertTrue(same(tp.clamp(x, min=-1.0).tolist(),
                                 [-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0,
                                  -1.0, inf, nan] * 3))
            self.assertTrue(same(tp.clamp(x, max=1.0).tolist(),
                                 [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0,
                                  -inf, 1.0, nan] * 3))
            self.assertTrue(same(tp.clamp(x, -0.5, -0.5).tolist(),
                                 ([-0.5] * 9 + [nan]) * 3))
            # A reversed pair stays with the per-element loop, which raises the
            # floor before lowering the ceiling: everything lands on the
            # ceiling, and a value that is not a number stays one.
            self.assertTrue(same(tp.clamp(x, 1.0, -1.0).tolist(),
                                 [-1.0] * 9 + [nan] + [-1.0] * 9 + [nan] +
                                 [-1.0] * 9 + [nan]))

    def test_clamp_reduced_precision_strided(self):
        # A transposed source is not contiguous, so it takes the path that
        # walks the storage instead of the flat one.  Clamping has to commute
        # with the transpose either way.
        seq = [-2.0, -0.5, 0.5, 2.0]
        vals = [[seq[(r + c) % 4] for c in range(6)] for r in range(5)]
        for dt in (tp.float16, tp.bfloat16):
            grid = tp.tensor(vals, dtype=dt)
            self.assertEqual(tp.clamp(grid.t(), -1.0, 1.0).tolist(),
                             tp.clamp(grid, -1.0, 1.0).t().tolist())
            self.assertEqual(tp.clamp(grid.t(), -1.0, 1.0)[0].tolist(),
                             [-1.0, -0.5, 0.5, 1.0, -1.0])

    def test_compare_reduced_precision(self):
        # The comparison runs on values widened out of the storage, so every
        # operand below is exact in both reduced widths and the answer is a
        # plain ordering of those reals: a Python float comparison of the same
        # numbers is the expected bit pattern, with no allowance anywhere.
        # The grid carries the two infinities, a pair of zeros that differ only
        # in sign (equal once widened, unequal as raw storage), subnormals that
        # only the converted path reads correctly, and not-a-number.
        inf = float('inf')
        nan = float('nan')
        inf = float('inf')
        nan = float('nan')
        vals = [-inf, -32768.0, -2.0, -0.5, -0.0, 2 ** -24, 3 * 2 ** -24,
                0.0, 0.5, 2.0, 32768.0, inf, nan]
        ops = [('eq', lambda p, q: p == q, tp.eq),
               ('ne', lambda p, q: p != q, tp.ne),
               ('lt', lambda p, q: p < q, tp.lt),
               ('le', lambda p, q: p <= q, tp.le),
               ('gt', lambda p, q: p > q, tp.gt),
               ('ge', lambda p, q: p >= q, tp.ge)]

        for dt in (tp.float16, tp.bfloat16):
            x = tp.tensor(vals * 2, dtype=dt)
            # Rotated by one, so each lane answers for a different pair and the
            # grid meets itself in both orders across the two copies.
            rot = vals[1:] + vals[:1] + vals[1:] + vals[:1]
            y = tp.tensor(rot, dtype=dt)
            for name, ref, fn in ops:
                got = fn(x, y)
                self.assertEqual(got.dtype, tp.bool, name)
                self.assertEqual(got.tolist(), [ref(a, b) for a, b in zip(vals * 2, rot)],
                                 '{} {} vs {}'.format(dt, name, 'tensor'))
            # A single-element right operand is one value read into every lane.
            half = tp.tensor([0.5], dtype=dt)
            for name, ref, fn in ops:
                want = [ref(a, 0.5) for a in vals * 2]
                self.assertEqual(fn(x, half).tolist(), want,
                                 '{} {} vs storage'.format(dt, name))
                self.assertEqual(fn(x, 0.5).tolist(), want,
                                 '{} {} vs scalar'.format(dt, name))
            # An operand widened to the common float32 leaves the narrow path;
            # the ordering is still the same one.
            wide = tp.tensor([-2.0], dtype=tp.float32)
            for name, ref, fn in ops:
                self.assertEqual(fn(x, wide).tolist(),
                                 [ref(a, -2.0) for a in vals * 2],
                                 '{} {} vs float32'.format(dt, name))

    def test_compare_reduced_precision_strided(self):
        # A transposed pair is not contiguous, so the walk over the storage and
        # the flat compare have to agree with the same compare run before the
        # transpose.
        seq = [-2.0, -0.5, 0.5, 2.0, float('nan'), float('inf')]
        a = [[seq[(r + c) % 6] for c in range(6)] for r in range(5)]
        b = [[seq[(r * 2 + c) % 6] for c in range(6)] for r in range(5)]
        for dt in (tp.float16, tp.bfloat16):
            ga, gb = tp.tensor(a, dtype=dt), tp.tensor(b, dtype=dt)
            self.assertEqual(tp.lt(ga.t(), gb.t()).tolist(),
                             tp.lt(ga, gb).t().tolist())
            self.assertEqual(tp.ge(ga.t(), gb.t()).tolist(),
                             tp.ge(ga, gb).t().tolist())

if __name__ == '__main__':
    unittest.main()
