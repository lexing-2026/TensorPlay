import pytest

import tensorplay as tp


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_square_layout_changes_recompile(dynamic, requires_grad):
    value = tp.randn(4, 4, requires_grad=requires_grad)
    expanded = tp.randn(4, 1, requires_grad=requires_grad).expand(4, 4)
    sliced = tp.randn(4, 8, requires_grad=requires_grad)[:, ::2]
    compiled = tp.compile(lambda x: x.sin() * 2, dynamic=dynamic, fullgraph=True)

    for operand in (value, value.t(), expanded, sliced):
        result = compiled(operand)
        assert tp.allclose(result, operand.sin() * 2)
        if requires_grad:
            gradient, = tp.autograd.grad(result.sum(), operand)
            assert tp.allclose(gradient, operand.cos() * 2)
    assert len(compiled._tensorplay_cache) == 4
    for operand in (value.t(), value, sliced, expanded):
        assert tp.allclose(compiled(operand), operand.sin() * 2)
    assert len(compiled._tensorplay_cache) == 4


def test_dynamic_layouts_reuse_specializations_across_sizes():
    compiled = tp.compile(lambda x: x.sin() + 1, dynamic=True, fullgraph=True)
    for size in (4, 6, 8):
        value = tp.randn(size, size)
        for operand in (value, value.t()):
            assert tp.allclose(compiled(operand), operand.sin() + 1)
    assert len(compiled._tensorplay_cache) == 2


@pytest.mark.parametrize("dynamic", [False, True])
def test_nested_keyword_tensor_layouts_recompile(dynamic):
    compiled = tp.compile(lambda *, batch: batch["x"].cos(), dynamic=dynamic)
    value = tp.randn(4, 4)
    for operand in (value, value.t(), value):
        assert tp.allclose(compiled(batch={"x": operand}), operand.cos())
    assert len(compiled._tensorplay_cache) == 2
