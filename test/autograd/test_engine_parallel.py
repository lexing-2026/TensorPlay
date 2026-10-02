import gc
import threading
import unittest

import tensorplay as tp


class TestEngineParallel(unittest.TestCase):
    """Multithreaded engine: correctness on wide graphs and hook support."""

    def _build_wide_graph(self, device, width=16):
        leaves = [tp.randn([8, 8], device=device, requires_grad=True) for _ in range(width)]
        # Diamond-ish fan-in/fan-out: every leaf feeds several sums that are
        # then reduced together, creating a deep dependency graph with many
        # ready tasks at once (the case the multithreaded engine exists for).
        partials = [leaf * float(i + 1) for i, leaf in enumerate(leaves)]
        partials += [p.t() for p in partials[: width // 2]]
        out = tp.zeros([8, 8], device=device)
        for p in partials:
            out = out + p
        return leaves, out

    def test_wide_backward_matches_single_thread(self):
        if not tp.cuda.is_available():
            self.skipTest("CUDA unavailable")
        device = "cuda"
        half = 8
        leaves, out = self._build_wide_graph(device)
        out.sum().backward()
        for i, leaf in enumerate(leaves):
            # Direct term (i+1); leaves in the first half also feed a
            # transposed twin with the same scale, so their total is doubled.
            expected = float(i + 1) * (2.0 if i < half else 1.0)
            self.assertTrue(leaf.grad is not None)
            self.assertTrue(
                tp.allclose(leaf.grad, tp.full_like(leaf.grad, expected), rtol=1e-5, atol=1e-5),
                f"leaf {i} grad mismatch",
            )

    def test_backward_under_threaded_caller(self):
        # backward() initiated from a non-main thread must work with the
        # engine's worker pool.
        errors = []

        def run():
            try:
                x = tp.randn([4, 4], requires_grad=True)
                y = (x * 3.0).sum()
                y.backward()
                assert x.grad is not None
                assert tp.allclose(x.grad, tp.full([4, 4], 3.0, dtype=tp.float32))
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_node_hooks(self):
        x = tp.ones([2, 2], requires_grad=True)
        y = x * 2.0

        seen = {}

        def pre_hook(grads):
            seen["pre"] = len(grads)
            return [g * 10.0 for g in grads]

        def post_hook(_inputs, outputs):
            seen["post"] = len(outputs)
            return [o + 1.0 for o in outputs]

        node = y.grad_fn
        node.add_pre_hook(pre_hook)
        node.add_post_hook(post_hook)

        y.sum().backward()
        self.assertEqual(seen.get("pre"), 1)
        self.assertEqual(seen.get("post"), 1)
        # grad 1 -> pre-hook x10 -> apply(*2) -> post-hook +1 => 21
        self.assertTrue(tp.allclose(x.grad, tp.full([2, 2], 21.0, dtype=tp.float32)))

    def test_no_leak_after_many_iterations(self):
        if not tp.cuda.is_available():
            self.skipTest("CUDA unavailable")
        tp.cuda.synchronize()
        tp.cuda.empty_cache()
        baseline = tp.cuda.memory_allocated()
        for _ in range(50):
            x = tp.ones([4, 4], device="cuda", requires_grad=True)
            y = (x * 2.0).sum()
            y.backward()
        del x, y
        gc.collect()
        tp.cuda.synchronize()
        self.assertEqual(tp.cuda.memory_allocated(), baseline)


if __name__ == "__main__":
    unittest.main()
