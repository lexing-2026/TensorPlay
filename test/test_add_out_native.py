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
