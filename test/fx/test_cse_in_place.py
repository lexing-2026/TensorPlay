"""Equal calls are not merged in a region that writes into a value.

Merging two calls with the same arguments makes their results one value, so a
write into one of them would show through the other, and a write into an
argument between the two calls would be lost.
"""
import tensorplay as tp


def _compiled(fn, *args):
    return tp.compile(fn, backend="stax")(*args)


def test_a_write_into_one_result_leaves_the_equal_one_alone():
    x = tp.tensor([[0.5, -1.0, 2.0], [0.25, 4.0, -3.0]], requires_grad=True)

    def region(v):
        scaled = v * 3
        first = scaled.half()
        second = scaled.half()
        first.mul_(2)
        return first * 1.0, second * 1.0

    first, second = _compiled(region, x)
    assert first.tolist() == [[3.0, -6.0, 12.0], [1.5, 24.0, -18.0]]
    assert second.tolist() == [[1.5, -3.0, 6.0], [0.75, 12.0, -9.0]]
    (first.float().sum() + second.float().sum()).backward()
    # Read once doubled and once as it was: 3 * 2 + 3.
    assert x.grad.tolist() == [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]]


def test_a_write_into_an_argument_between_two_calls_is_seen():
    x = tp.tensor([1.0, -2.0, 0.5])

    def region(v):
        base = v + 1
        before = base * 2
        base.add_(1)
        after = base * 2
        return before, after

    before, after = _compiled(region, x)
    assert before.tolist() == [4.0, -2.0, 3.0]
    assert after.tolist() == [6.0, 0.0, 5.0]
