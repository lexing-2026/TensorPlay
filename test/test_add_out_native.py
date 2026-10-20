import numpy as np
import tensorplay as tp


def test_add_out_reuses_matching_output():
    left = tp.tensor([1.0, 2.0])
    right = tp.tensor([3.0, 4.0])
    out = tp.zeros_like(left)

    result = tp._C.add(left, right, out=out)

    assert result is out
    assert out.tolist() == [4.0, 6.0]


def test_add_out_resizes_output():
    left = tp.tensor([[1.0, 2.0]])
    right = tp.tensor([[3.0], [4.0]])
    out = tp.zeros([4, 4])

    result = tp._C.add(left, right, out=out)

    assert result is out
    assert out.shape == (2, 2)
    assert out.tolist() == [[4.0, 5.0], [5.0, 6.0]]


def test_add_out_dense_pair_matches_add_at_every_alpha():
    # A length that is not a multiple of any vector width, so the answer pins
    # the loop's tail as firmly as its body, over both float widths.
    left = tp.tensor([float(i) * 0.5 - 3.0 for i in range(37)])
    right = tp.tensor([float(i) * 1.25 for i in range(37)])
    for dtype in (tp.float32, tp.float64):
        a = left.to(dtype)
        b = right.to(dtype)
        for alpha in (1, -1, 2.5, 0):
            out = tp.zeros(37, dtype=dtype)
            tp._C.add(a, b, alpha=alpha, out=out)
            assert out.tolist() == tp.add(a, b, alpha=alpha).tolist()


def test_add_out_writes_an_operand_destination_in_place():
    # The destination may be one of the operands when its elements sit at the
    # same addresses as that operand's: the loop reads an element and writes it
    # back to the address it came from.  A length that is not a multiple of any
    # vector width pins the tail of the loop as firmly as its body.
    xs = [float(i) * 0.5 - 3.0 for i in range(37)]
    ys = [float(i) * 1.25 for i in range(37)]
    for dtype in (tp.float32, tp.float64, tp.float16, tp.bfloat16, tp.int32):
        alphas = (1, -1) if dtype == tp.int32 else (1, -1, 2.5)
        for alpha in alphas:
            left = tp.tensor(xs, dtype=dtype)
            right = tp.tensor(ys, dtype=dtype)
            want = tp.add(left, right, alpha=alpha).tolist()

            as_left = tp.tensor(xs, dtype=dtype)
            tp._C.add(as_left, right, alpha=alpha, out=as_left)
            assert as_left.tolist() == want

            as_right = tp.tensor(ys, dtype=dtype)
            tp._C.add(left, as_right, alpha=alpha, out=as_right)
            assert as_right.tolist() == want

    both = tp.tensor(xs, dtype=tp.float32)
    want = tp.add(both, both).tolist()
    tp._C.add(both, both, out=both)
    assert both.tolist() == want


def test_add_out_refuses_a_destination_it_cannot_write():
    # A destination that overlaps an operand at another offset writes over an
    # element some later read still needs, so only a pass that finishes the
    # result before storing it answers this.
    base = tp.tensor([float(i) for i in range(20)])
    head = base[0:16]
    tail = base[2:18]
    want = tp.add(head, tail).tolist()
    tp._C.add(head, tail, out=head)
    assert head.tolist() == want

    # Two views handed out by separate wrappers over one buffer address the same
    # bytes without naming the same allocation, so the overlap has to be caught
    # by the bytes each view reaches.
    shared = np.array([float(i) for i in range(20)], dtype=np.float32)
    first = tp.from_numpy(shared[0:16])
    second = tp.from_numpy(shared[2:18])
    want = tp.add(first, second).tolist()
    tp._C.add(first, second, out=first)
    assert first.tolist() == want

    # A transposed destination is dense in the other order, and the result has
    # to land in the parent storage at the transposed offsets.
    grid = tp.tensor([[float(r * 6 + c) for c in range(6)] for r in range(4)])
    shift = tp.tensor([[float(100 + r * 6 + c) for c in range(6)] for r in range(4)])
    buffer = tp.zeros(24)
    view = buffer.reshape(4, 6).t()
    tp._C.add(grid.t(), shift.t(), out=view)
    assert view.tolist() == tp.add(grid.t(), shift.t()).tolist()
    assert buffer.tolist() == tp.add(grid, shift).reshape(-1).tolist()

    # A destination that is the transposed form of an operand reaches the same
    # rule: the two address the parent storage in different orders.
    parent = tp.tensor([[float(r * 6 + c) for c in range(6)] for r in range(4)])
    other = tp.tensor([[float(-c) for c in range(4)] for r in range(6)])
    want = tp.add(parent.t(), other).tolist()
    tp._C.add(parent.t(), other, out=parent.t())
    assert parent.t().tolist() == want
