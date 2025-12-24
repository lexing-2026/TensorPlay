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

if __name__ == '__main__':
    unittest.main()
