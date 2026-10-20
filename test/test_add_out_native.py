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


def test_add_out_rejects_a_destination_it_cannot_write():
    # The destination is also an operand, so a pass that writes as it reads
    # would feed its own output back in as input.
    a = tp.tensor([float(i) for i in range(20)])
    b = tp.tensor([float(-i) for i in range(10, 30)])
    want = tp.add(a, b).tolist()
    target = tp.tensor(a.tolist())
    tp._C.add(target, b, out=target)
    assert target.tolist() == want

    # A transposed destination is dense in the other order, and the result has
    # to land in the parent storage at the transposed offsets.
    grid = tp.tensor([[float(r * 6 + c) for c in range(6)] for r in range(4)])
    shift = tp.tensor([[float(100 + r * 6 + c) for c in range(6)] for r in range(4)])
    buffer = tp.zeros(24)
    view = buffer.reshape(4, 6).t()
    tp._C.add(grid.t(), shift.t(), out=view)
    assert view.tolist() == tp.add(grid.t(), shift.t()).tolist()
    assert buffer.tolist() == tp.add(grid, shift).reshape(-1).tolist()
