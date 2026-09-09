"""Zero-stride broadcast views flowing into pointwise kernels.

A view produced by ``expand`` carries a zero stride along every broadcast
dimension.  Elementwise consumers must honor that stride (re-reading the
same source element), never iterate the storage linearly.  These tests pin
that contract directly on the iterator -- independent of any autograd
machinery -- and then through the vjp/jacobian chains that used to expose
the failure on one platform only.
"""

import math
import unittest

import tensorplay as tp


def _row_broadcast_mul(row_seed, source):
    # (N,) -> (R, N): every row of the view repeats row_seed, i.e. the zero
    # stride sits on the OUTER dimension (strides (1, 0) after unsqueeze(1)).
    view = row_seed.unsqueeze(1).expand(source.shape)
    return source * view


class TestExpandViewPointwise(unittest.TestCase):
    SHAPES = [(2, 2), (4, 4), (3, 5)]

    def test_mul_broadcast_inner_dim(self):
        # Broadcast along the LAST dimension (zero stride innermost after
        # the iterator's reorder): view[r][c] = col_seed[c].
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                seed = tp.arange(shape[1], dtype=tp.float64) + 1.0
                source = tp.rand(*shape, dtype=tp.float64)
                view = seed.expand(source.shape)
                out = source * view
                sd = seed.detach()
                xd = source.detach()
                for r in range(shape[0]):
                    for c in range(shape[1]):
                        self.assertAlmostEqual(
                            out[r][c].item(), sd[c].item() * xd[r][c].item(),
                            places=12, msg=f"shape={shape} at ({r},{c})")

    def test_mul_broadcast_outer_dim(self):
        # Broadcast along the FIRST dimension (zero stride outermost in
        # source layout): view[r][c] = row_seed[r].
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                seed = tp.arange(shape[0], dtype=tp.float64) + 1.0
                source = tp.rand(*shape, dtype=tp.float64)
                out = _row_broadcast_mul(seed, source)
                sd = seed.detach()
                xd = source.detach()
                for r in range(shape[0]):
                    for c in range(shape[1]):
                        self.assertAlmostEqual(
                            out[r][c].item(), sd[r].item() * xd[r][c].item(),
                            places=12, msg=f"shape={shape} at ({r},{c})")

    def test_mul_one_hot_row_seed(self):
        # One-hot row seeds are the discriminating pattern: a linear
        # (stride-ignoring) read marks only the first column, while a
        # correct broadcast repeats the seed across the full row.
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                source = tp.rand(*shape, dtype=tp.float64)
                xd = source.detach()
                for r in range(shape[0]):
                    seed = tp.zeros(shape[0], dtype=tp.float64)
                    seed[r] = 1.0
                    out = _row_broadcast_mul(seed, source)
                    for r2 in range(shape[0]):
                        for c in range(shape[1]):
                            expected = xd[r2][c].item() if r2 == r else 0.0
                            self.assertAlmostEqual(
                                out[r2][c].item(), expected,
                                places=12,
                                msg=f"shape={shape} row {r2} col {c}")

    def test_add_broadcast_zero_stride(self):
        seed = tp.tensor([1.0, 2.0], dtype=tp.float64)
        source = tp.rand(2, 2, dtype=tp.float64)
        view = seed.unsqueeze(1).expand(source.shape)
        out = source + view
        sd = seed.detach()
        xd = source.detach()
        for r in range(2):
            for c in range(2):
                self.assertAlmostEqual(
                    out[r][c].item(), sd[r].item() + xd[r][c].item(), places=12)

    def test_sum_dim_backward_matches_reference(self):
        # The sum(dim) backward broadcasts grad over the reduced dim; the
        # result must be the broadcast VALUES (row-constant), whether the
        # engine materializes it or returns a view.
        import tensorplay._C as C
        y = tp.rand(4, 4, dtype=tp.float64)
        grad = tp.tensor([1.0, 0.0, 0.5, 0.0], dtype=tp.float64)
        e = C._sum_dim_backward(grad, y, [1], False)
        gd = grad.detach()
        for r in range(4):
            for c in range(4):
                self.assertAlmostEqual(e[r][c].item(), gd[r].item(), places=12,
                                       msg=f"row {r} col {c}")


class TestBroadcastAutogradChain(unittest.TestCase):
    def test_grad_of_row_sum_matches_row(self):
        # grad(out[i], x) must equal row i of exp(x): exercises
        # select_backward -> sum(dim) backward (zero-stride view) ->
        # exp backward mul.
        x = tp.rand(4, 4, dtype=tp.float64, requires_grad=True)
        outs = x.exp().sum(dim=1)
        xd = x.detach()
        for i in range(4):
            g = tp.autograd.grad(outs[i], x, retain_graph=True)[0]
            for j in range(4):
                self.assertAlmostEqual(g[i][j].item(), math.exp(xd[i][j].item()),
                                       places=10, msg=f"({i},{j})")
                for r in range(4):
                    if r != i:
                        self.assertEqual(g[r][j].item(), 0.0)


class TestBackwardSeedPieces(unittest.TestCase):
    """Direct checks of the two pieces the grad chain composes, without
    running the engine: the one-hot seed from select_backward, and the
    stride metadata of the unsqueeze+expand broadcast.  A failure here
    names the broken stage; a pass shifts the search downstream."""

    def test_expand_view_strides(self):
        seed = tp.tensor([1.0, 2.0], dtype=tp.float64)
        view = seed.unsqueeze(1).expand(2, 3)
        self.assertEqual(tuple(view.stride()), (1, 0))
        self.assertFalse(view.is_contiguous())
        sd = seed.detach()
        for r in range(2):
            for c in range(3):
                self.assertEqual(view[r][c].item(), sd[r].item())

    def test_select_backward_one_hot(self):
        import tensorplay._C as C
        vec = tp.rand(4, dtype=tp.float64)
        for i in range(4):
            gi = C.select_backward(tp.tensor(1.0, dtype=tp.float64),
                                   vec, 0, i)
            expected = [1.0 if r == i else 0.0 for r in range(4)]
            actual = [gi[r].item() for r in range(4)]
            self.assertEqual(actual, expected,
                             msg=f"select_backward index {i}")


class TestForwardSumDim(unittest.TestCase):
    """The grad chain inherits its broadcast axis from the forward
    reduction's recorded dims.  Pin the forward sum(dim) values so a
    mis-bound dim argument fails here, at the source, instead of
    surfacing as a transposed backward broadcast."""

    def test_sum_dim_values_2d(self):
        x = tp.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float64)
        s1 = x.sum(dim=1)
        self.assertEqual(tuple(s1.shape), (2,))
        self.assertAlmostEqual(s1[0].item(), 3.0, places=10)
        self.assertAlmostEqual(s1[1].item(), 7.0, places=10)
        s0 = x.sum(dim=0)
        self.assertEqual(tuple(s0.shape), (2,))
        self.assertAlmostEqual(s0[0].item(), 4.0, places=10)
        self.assertAlmostEqual(s0[1].item(), 6.0, places=10)

    def test_sum_dim_values_3d(self):
        # dims=[0, 2] of (2, 3, 4) -> shape (3,); values sum over the
        # outer and inner axes only.
        x = tp.rand(2, 3, 4, dtype=tp.float64)
        xd = x.detach()
        s = x.sum(dim=[0, 2])
        self.assertEqual(tuple(s.shape), (3,))
        for j in range(3):
            acc = 0.0
            for r in range(2):
                for k in range(4):
                    acc += xd[r][j][k].item()
            self.assertAlmostEqual(s[j].item(), acc, places=10,
                                   msg=f"index {j}")

    def test_sum_dim_negative_and_keepdim(self):
        x = tp.rand(2, 3, 4, dtype=tp.float64)
        xd = x.detach()
        s = x.sum(dim=-1)
        self.assertEqual(tuple(s.shape), (2, 3))
        for r in range(2):
            for j in range(3):
                acc = sum(xd[r][j][k].item() for k in range(4))
                self.assertAlmostEqual(s[r][j].item(), acc, places=10,
                                       msg=f"({r},{j})")
        sk = x.sum(dim=1, keepdim=True)
        self.assertEqual(tuple(sk.shape), (2, 1, 4))
        for r in range(2):
            for k in range(4):
                acc = sum(xd[r][j][k].item() for j in range(3))
                self.assertAlmostEqual(sk[r][0][k].item(), acc, places=10,
                                       msg=f"({r},{k})")


if __name__ == "__main__":
    unittest.main()
