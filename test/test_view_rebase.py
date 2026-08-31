import tensorplay as tp


def test_inplace_view_rebases_base_history():
    root = tp.arange(4.0, requires_grad=True)
    base = root.clone()
    view = base.narrow(0, 0, 2)

    view.mul_(2.0)
    base.sum().backward()

    expected = tp.tensor([2.0, 2.0, 1.0, 1.0])
    assert tp.allclose(root.grad, expected)


def test_inplace_view_of_view_reaches_root_base():
    root = tp.ones((2, 2), requires_grad=True)
    base = root.clone()
    first = base.narrow(0, 0, 1)
    second = first.narrow(1, 1, 1)

    second.mul_(2.0)
    base.sum().backward()

    expected = tp.tensor([[1.0, 2.0], [1.0, 1.0]])
    assert tp.allclose(root.grad, expected)
