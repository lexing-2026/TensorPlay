import tensorplay as tp
import pytest


def test_is_set_to_requires_same_storage_and_geometry():
    first = tp.empty((3, 4, 9, 10))
    second = tp.empty((3, 4, 9, 10))
    shared = tp.empty((0,)).set_(first)

    assert not first.is_set_to(second)
    assert first.is_set_to(shared)
    assert shared.is_set_to(first)

    reshaped = first.view((4, 3, 2, 45))
    assert not first.is_set_to(reshaped)
    assert not reshaped.is_set_to(first)


def test_is_set_to_rejects_undefined_storage_identity():
    first = tp.empty((0,))
    second = tp.empty((0,))

    assert not first.is_set_to(second)


def test_type_properties_use_native_metadata_entries():
    floating = tp.empty((2,), dtype=tp.float32)
    integral = tp.empty((2,), dtype=tp.int64)

    assert tp.is_floating_point(floating)
    assert not tp.is_floating_point(integral)
    assert not tp.is_distributed(floating)


def test_inference_mode_marks_allocations_and_views():
    source = tp.ones((2, 3), requires_grad=True)

    assert not source.is_inference()
    with tp.inference_mode():
        result = source + source
        view = result.view((3, 2))
        assert result.is_inference()
        assert view.is_inference()
        assert not result.requires_grad
        with pytest.raises(RuntimeError, match="do not track version counter"):
            _ = result._version

    with pytest.raises(RuntimeError, match="outside inference mode"):
        result.requires_grad = True
