from pathlib import Path

from tools.codegen.model import parse_native_yaml, parse_schema


ROOT = Path(__file__).resolve().parents[1]


def test_native_collection_retains_reference_records_and_indexes():
    funcs = parse_native_yaml(str(ROOT / "config" / "native_functions.yaml"))

    assert len(funcs) == len(funcs.reference_functions)
    assert len(funcs.reference_by_name) == len(funcs.reference_functions)
    # The native schema engine keeps no global backend index table; kernel
    # names are projected per-op from each record's yaml `dispatch:` section.
    assert isinstance(funcs.backend_indices, dict)
    assert not funcs.backend_indices
    assert all(function.reference is not None for function in funcs)
    assert all(function.location is not None for function in funcs)

    add = next(function for function in funcs if function.func_name == "add.Tensor")
    assert str(add.reference.func) == add.schema.replace("int64_t", "int")
    assert add.schema_kind == "functional"
    assert set(add.reference.variants) == set(add.variants)
    assert add.namespace == "tensorplay"
    assert add.backend("CPU").kernel == "add_cpu"
    assert add.backend("CUDA").kernel == "add_cuda"

    view = next(function for function in funcs if function.func_name == "view")
    assert view.returns_view_of_input
    assert view.view_input_name == "self"
    assert not view.view_metadata_changes

    reshape = next(function for function in funcs
                   if function.func_name == "reshape")
    assert reshape.returns_view_of_input
    assert reshape.view_input_name == "self"

    real = next(function for function in funcs if function.func_name == "real")
    assert real.returns_view_of_input
    assert real.view_input_name == "self"

    native_groups = funcs.grouped_native_functions()
    view_groups = funcs.grouped_view_functions()
    assert native_groups
    assert view_groups
    assert any(type(group).__name__ == "NativeFunctionsGroup"
               for group in native_groups)
    assert any(type(group).__name__ == "NativeFunctionsViewGroup"
               for group in view_groups)

    for name in ("retains_grad", "numel", "dim", "is_contiguous",
                 "select.int"):
        function = next(item for item in funcs if item.func_name == name)
        assert function.manual_cpp_binding
        assert function.manual_kernel_registration


def test_schema_projection_preserves_nested_type_shape():
    function = parse_schema(
        "sample(Tensor?[] values, int[2] shape, SymInt dim) -> Tensor?[]"
    )

    values, shape, dim = function.args
    assert str(values.type) == "Tensor?[]"
    assert values.type.is_opt is False
    assert values.type.is_list is True
    assert values.type.list_elem_opt is True
    assert shape.type.list_size == 2
    assert dim.type.symint is True
    assert str(function.returns[0].type) == "Tensor?[]"
