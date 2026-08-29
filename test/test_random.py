import tensorplay as tp
import unittest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

SEED = 1234567890123456789

class TestRandom(unittest.TestCase):
    def test_reproducible_rand(self):
        tp.manual_seed(SEED)
        a = tp.rand([100])
        tp.manual_seed(SEED)
        b = tp.rand([100])
        self.assertEqual(a.tolist(), b.tolist())

    def test_reproducible_randn(self):
        tp.manual_seed(SEED)
        a = tp.randn([128])
        tp.manual_seed(SEED)
        b = tp.randn([128])
        self.assertEqual(a.tolist(), b.tolist())

    def test_different_seeds_differ(self):
        tp.manual_seed(1)
        a = tp.rand([100])
        tp.manual_seed(2)
        b = tp.rand([100])
        self.assertNotEqual(a.tolist(), b.tolist())

    def test_initial_seed_roundtrip(self):
        tp.manual_seed(SEED)
        self.assertEqual(tp.initial_seed(), SEED)

    def test_get_set_rng_state(self):
        tp.manual_seed(SEED)
        state = tp.get_rng_state()
        self.assertEqual(state.dtype, tp.uint8)
        a = tp.randn([50])
        # Drawing advanced the state; restoring must replay the same stream.
        tp.set_rng_state(state)
        b = tp.randn([50])
        self.assertEqual(a.tolist(), b.tolist())

    def test_rng_state_survives_normal_cache(self):
        # Box-Muller caches every second sample inside the generator; the
        # cache must round-trip through get/set_state.
        tp.manual_seed(SEED)
        first = tp.randn([3])  # small path: double Box-Muller with cache
        state = tp.get_rng_state()
        second = tp.randn([3])
        tp.set_rng_state(state)
        second_again = tp.randn([3])
        self.assertEqual(second.tolist(), second_again.tolist())
        del first

    def test_generator_object(self):
        g = tp.Generator(SEED)
        self.assertEqual(g.initial_seed(), SEED)
        s1 = g.get_state()
        v1 = tp.randn([10], generator=g) if self._supports_generator_kwarg() else None
        g.set_state(s1)
        if v1 is not None:
            v2 = tp.randn([10], generator=g)
            self.assertEqual(v1.tolist(), v2.tolist())

    def _supports_generator_kwarg(self):
        try:
            tp.randn([2], generator=tp.Generator(SEED))
            return True
        except TypeError:
            return False

    def test_seed_returns_nondeterministic_value(self):
        s1 = tp.seed()
        s2 = tp.seed()
        self.assertIsInstance(s1, int)
        self.assertNotEqual(s1, s2)

    def test_fork_rng(self):
        tp.manual_seed(SEED)
        before = tp.get_rng_state()
        with tp.fork_rng(devices=[]):
            tp.rand([1000])
            tp.randn([1000])
        after = tp.get_rng_state()
        self.assertEqual(before.tolist(), after.tolist())

    def test_bernoulli(self):
        t = tp.rand([100])
        b = tp.bernoulli(t)
        # Check values are 0 or 1
        flat = b.view([-1])
        for i in range(flat.numel()):
            val = flat[i].item()
            self.assertIn(val, [0.0, 1.0])

    def test_normal(self):
        mean = tp.full([100], 0.0, dtype=tp.float32)
        std = tp.full([100], 1.0, dtype=tp.float32)
        n = tp.normal(mean, std)
        self.assertEqual(n.shape, [100])

    def test_poisson(self):
        rates = tp.full([50], 5.0, dtype=tp.float32)
        p = rates.poisson()
        flat = p.view([-1])
        for i in range(flat.numel()):
            val = flat[i].item()
            self.assertTrue(val >= 0)
            self.assertEqual(val, int(val))

    def test_rand(self):
        r = tp.rand([50])
        flat = r.view([-1])
        for i in range(flat.numel()):
            val = flat[i].item()
            self.assertTrue(0.0 <= val < 1.0)

    def test_rand_like(self):
        x = tp.ones([10, 10])
        r = tp.rand_like(x)
        self.assertEqual(r.shape, x.shape)
        # Check one value
        self.assertTrue(0.0 <= r[0,0].item() < 1.0)

    def test_randint(self):
        r = tp.randint(0, 10, [50])
        flat = r.view([-1])
        for i in range(flat.numel()):
            val = flat[i].item()
            self.assertTrue(0 <= val < 10)
            self.assertIsInstance(val, int)

    def test_randint_like(self):
        x = tp.ones([5, 5], dtype=tp.int32)
        r = tp.randint_like(x, 0, 10)
        self.assertEqual(r.shape, x.shape)
        self.assertTrue(0 <= r[0,0].item() < 10)

    def test_randn(self):
        r = tp.randn([50])
        self.assertEqual(r.shape, [50])

    def test_randn_like(self):
        x = tp.ones([5, 5])
        r = tp.randn_like(x)
        self.assertEqual(r.shape, x.shape)

    def test_randperm(self):
        n = 10
        r = tp.randperm(n)
        self.assertEqual(r.numel(), n)
        # Check range
        flat = r.view([-1])
        vals = []
        for i in range(flat.numel()):
            val = flat[i].item()
            self.assertTrue(0 <= val < n)
            vals.append(val)
        # Check uniqueness
        self.assertEqual(len(set(vals)), n)


@unittest.skipUnless(HAS_TORCH, "reference package not installed")
class TestTorchParity(unittest.TestCase):
    """
    algorithms are pure integer/bit operations (no transcendental functions)."""

    def _tp_torch(self, fn_tp, fn_torch, exact=True, tol=0.0):
        tp.manual_seed(SEED)
        a = fn_tp()
        torch.manual_seed(SEED)
        b = fn_torch()
        if exact:
            self.assertEqual(a.tolist(), b.tolist())
        else:
            self.assertTrue(a.allclose(b, rtol=1e-5, atol=tol), f"{a[:4].tolist()} vs {b[:4].tolist()}")

    def test_rand_parity(self):
        self._tp_torch(lambda: tp.rand([100]), lambda: torch.rand(100))

    def test_uniform_parity(self):
        def tp_fn():
            t = tp.empty([100])
            return t.uniform_(-3.5, 7.25)
        def th_fn():
            t = torch.empty(100)
            return t.uniform_(-3.5, 7.25)
        self._tp_torch(tp_fn, th_fn)

    def test_random_parity(self):
        def tp_fn():
            t = tp.empty([100], dtype=tp.int64)
            return t.random_(10, 100)
        def th_fn():
            t = torch.empty(100, dtype=torch.int64)
            return t.random_(10, 100)
        self._tp_torch(tp_fn, th_fn)

    def test_randint_parity(self):
        self._tp_torch(lambda: tp.randint(5, 50, [77]),
                       lambda: torch.randint(5, 50, (77,)))

    def test_randperm_parity(self):
        self._tp_torch(lambda: tp.randperm(37), lambda: torch.randperm(37))

    def test_bernoulli_parity(self):
        tp.manual_seed(SEED)
        p = tp.rand([200])
        tp.manual_seed(SEED)
        a = tp.bernoulli(p)
        torch.manual_seed(SEED)
        pt = torch.rand(200)
        torch.manual_seed(SEED)
        b = torch.bernoulli(pt)
        self.assertEqual(a.tolist(), b.tolist())

    def test_exponential_parity(self):
        def tp_fn():
            t = tp.empty([100])
            return t.exponential_(1.5)
        def th_fn():
            t = torch.empty(100)
            return t.exponential_(1.5)
        self._tp_torch(tp_fn, th_fn)

    def test_geometric_parity(self):
        def tp_fn():
            t = tp.empty([100])
            return t.geometric_(0.3)
        def th_fn():
            t = torch.empty(100)
            return t.geometric_(0.3)
        self._tp_torch(tp_fn, th_fn)

    def test_cauchy_parity(self):
        def tp_fn():
            t = tp.empty([100])
            return t.cauchy_(2.0, 1.5)
        def th_fn():
            t = torch.empty(100)
            return t.cauchy_(2.0, 1.5)
        self._tp_torch(tp_fn, th_fn)

    def test_log_normal_parity(self):
        def tp_fn():
            t = tp.empty([100])
            return t.log_normal_(1.0, 0.5)
        def th_fn():
            t = torch.empty(100)
            return t.log_normal_(1.0, 0.5)
        self._tp_torch(tp_fn, th_fn)

    def test_randn_small_parity(self):
        # < 16 elements takes the serial double Box-Muller path.
        self._tp_torch(lambda: tp.randn([15]), lambda: torch.randn(15))

    def test_randn_large_parity(self):
        # >= 16 elements uses the AVX2 normal_fill path with the same
        self._tp_torch(lambda: tp.randn([1024]), lambda: torch.randn(1024))

    def test_rng_state_cross_compatible(self):
        # sequence (both use the same 5056-byte POD layout).  The state must
        # be captured BEFORE drawing the reference sequence.
        tp.manual_seed(SEED)
        state = tp.get_rng_state()
        expected = tp.rand([8]).tolist()
        gen = torch.Generator()
        raw = state.numpy().tobytes()
        gen.set_state(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
        actual = torch.rand(8, generator=gen).tolist()
        self.assertEqual(expected, actual)

    def test_randn_half_bfloat16_parity(self):
        # Native CPU Half/BFloat16 path uses storage-dtype uniforms and
        self._tp_torch(lambda: tp.randn([1024], dtype=tp.float16),
                       lambda: torch.randn(1024, dtype=torch.float16))
        self._tp_torch(lambda: tp.randn([1024], dtype=tp.bfloat16),
                       lambda: torch.randn(1024, dtype=torch.bfloat16))

    def test_rand_half_parity(self):
        self._tp_torch(lambda: tp.rand([256], dtype=tp.float16),
                       lambda: torch.rand(256, dtype=torch.float16))

    def test_uniform_half_parity(self):
        def tp_fn():
            t = tp.empty([100], dtype=tp.float16)
            return t.uniform_(-2.5, 4.5)
        def th_fn():
            t = torch.empty(100, dtype=torch.float16)
            return t.uniform_(-2.5, 4.5)
        self._tp_torch(tp_fn, th_fn)

    def test_random_wide_int_parity(self):
        # bounds for bool"), so bool is not checked in this case.
        for tp_dt, th_dt in [(tp.int8, torch.int8), (tp.uint8, torch.uint8),
                             (tp.int16, torch.int16), (tp.uint16, torch.uint16),
                             (tp.uint32, torch.uint32)]:
            with self.subTest(dtype=th_dt):
                def tp_fn(dt=tp_dt):
                    t = tp.empty([100], dtype=dt)
                    return t.random_(0, 100)
                def th_fn(dt=th_dt):
                    t = torch.empty(100, dtype=dt)
                    return t.random_(0, 100)
                self._tp_torch(tp_fn, th_fn)

    def test_geometric_wide_int_parity(self):
        def tp_fn():
            t = tp.empty([100], dtype=tp.int32)
            return t.geometric_(0.3)
        def th_fn():
            t = torch.empty(100, dtype=torch.int32)
            return t.geometric_(0.3)
        self._tp_torch(tp_fn, th_fn)

    def test_exponential_half_parity(self):
        def tp_fn():
            t = tp.empty([100], dtype=tp.float16)
            return t.exponential_(1.5)
        def th_fn():
            t = torch.empty(100, dtype=torch.float16)
            return t.exponential_(1.5)
        self._tp_torch(tp_fn, th_fn)

    def test_randperm_int32_parity(self):
        self._tp_torch(lambda: tp.randperm(37, dtype=tp.int32),
                       lambda: torch.randperm(37, dtype=torch.int32))

if __name__ == '__main__':
    unittest.main()
