import gc
import unittest

import tensorplay as tp


class TestVersionCounter(unittest.TestCase):

    def test_inplace_op_bumps_version(self):
        x = tp.ones([2, 2])
        v0 = x._version
        x.add_(1.0)
        self.assertEqual(x._version, v0 + 1)
        x.mul_(2.0)
        self.assertEqual(x._version, v0 + 2)

    def test_out_of_place_does_not_bump(self):
        x = tp.ones([2, 2])
        v0 = x._version
        y = x + 1.0
        self.assertEqual(x._version, v0)
        # The result of an out-of-place op starts fresh.
        self.assertEqual(y._version, 0)

    def test_view_shares_version_with_base(self):
        base = tp.ones([4, 4], requires_grad=False)
        v0 = base._version
        view = base.as_strided([2, 4], [4, 1], 0) if hasattr(base, "as_strided") else base[0:2]
        # Mutating through the view must be visible on the base counter and
        # vice versa (they share one VersionCounter).
        view.add_(1.0)
        self.assertEqual(base._version, v0 + 1)
        base.add_(1.0)
        self.assertEqual(view._version, v0 + 2)

    def test_transpose_view_shares_version(self):
        base = tp.ones([3, 5])
        t = base.transpose(0, 1)
        v0 = t._version
        base.mul_(2.0)
        self.assertEqual(t._version, v0 + 1)

    def test_clone_gets_fresh_counter(self):
        x = tp.ones([2, 2])
        x.add_(1.0)
        c = x.clone()
        self.assertEqual(c._version, 0)
        c.add_(1.0)
        self.assertEqual(x._version, 1)
        self.assertEqual(c._version, 1)

    def test_reshape_view_shares_version(self):
        x = tp.ones([2, 6])
        r = x.reshape([3, 4])
        v0 = r._version
        x.fill_(0.0)
        self.assertEqual(r._version, v0 + 1)

    def test_fill_and_zero_bump(self):
        x = tp.ones([2, 2])
        v0 = x._version
        x.zero_()
        self.assertEqual(x._version, v0 + 1)
        x.fill_(3.0)
        self.assertEqual(x._version, v0 + 2)


if __name__ == "__main__":
    unittest.main()
