from pathlib import Path

import pytest

from tools.codegen.gen_structured import _render_header, validate_structured
from tools.codegen.gen_autograd import generate_autograd_nodes, load_derivatives
from tools.codegen.gen_tpx import generate_tpx_ops_cpp
from tools.codegen.gen_python_c import _is_variadic_shape_list
from tools.codegen.model import parse_native_yaml, parse_schema


ROOT = Path(__file__).resolve().parents[2]


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


def test_structured_groups_emit_dispatch_metadata():
    funcs = parse_native_yaml(str(ROOT / "config" / "native_functions.yaml"))

    assert validate_structured(funcs) == []
    groups = [group for group in funcs.grouped_native_functions()
              if group.structured]
    generated = _render_header(groups)

    assert len(groups) == 2
    assert '"_convert_indices_from_coo_to_csr_structured_cpu"' in generated
    assert '"_convert_indices_from_coo_to_csr_structured_cuda"' in generated
    assert '"_convert_indices_from_csr_to_coo_structured_cpu"' in generated
    assert '"_convert_indices_from_csr_to_coo_structured_cuda"' in generated
    assert "__line__" not in generated


def test_python_bridge_only_expands_shape_lists():
    funcs = parse_native_yaml(str(ROOT / "config" / "native_functions.yaml"))

    view = next(function for function in funcs if function.func_name == "view")
    reshape = next(function for function in funcs if function.func_name == "reshape")
    sparse_size = next(
        function for function in funcs
        if function.schema.startswith("sparse_csr_tensor.crow_col_value_size(")
    )
    as_strided = next(
        function for function in funcs
        if function.schema.startswith("as_strided(")
    )

    assert _is_variadic_shape_list(view, "method")
    assert _is_variadic_shape_list(reshape, "function")
    assert not _is_variadic_shape_list(sparse_size, "function")
    assert not _is_variadic_shape_list(as_strided, "function")


def test_derivatives_reject_a_second_entry_for_the_same_operator(tmp_path):
    # Two spellings of one schema (here differing only in a default) would
    # leave whichever entry is read last in charge.
    funcs = parse_native_yaml(str(ROOT / "config" / "native_functions.yaml"))
    path = tmp_path / "derivatives.yaml"
    path.write_text(
        "- name: flip(Tensor self, int[] dims) -> Tensor\n"
        "  self: grad.flip(dims)\n"
        "\n"
        "- name: flip(Tensor self, int[] dims=[]) -> Tensor\n"
        "  self: flip(grad, dims)\n"
    )
    with pytest.raises(ValueError, match="'flip' more than once"):
        load_derivatives(str(path), {function.func_name: function for function in funcs})


def test_non_differentiable_inputs_do_not_make_the_output_require_grad():
    # A backward kernel reads its forward input for the shape alone: the
    # declaration keeps that input out of the gradient slots and out of the
    # requires_grad test, so calling the kernel on a leaf yields a constant.
    funcs = parse_native_yaml(str(ROOT / "config" / "native_functions.yaml"))
    derivatives = load_derivatives(
        str(ROOT / "config" / "derivatives.yaml"),
        {function.func_name: function for function in funcs},
    )
    entry = derivatives["fft_fft2_backward"]
    assert entry.non_differentiable_args == {"self"}
    assert set(entry.formulas) == {"grad_output"}

    wrappers = generate_tpx_ops_cpp(
        funcs, autocast_ops=set(), derivatives=derivatives,
        native_op_names={function.cpp_name for function in funcs},
    )
    wrapper = wrappers.split("Tensor fft_fft2_backward(", 1)[1].split("\n}\n", 1)[0]
    assert "grad_output.requires_grad()" in wrapper
    assert "self.requires_grad()" not in wrapper


def test_group_norm_codegen_shares_backward_and_marks_saved_statistics():
    funcs = parse_native_yaml(str(ROOT / "config" / "native_functions.yaml"))
    derivatives = load_derivatives(
        str(ROOT / "config" / "derivatives.yaml"),
        {function.func_name: function for function in funcs},
    )
    node = generate_autograd_nodes(
        derivatives, native_op_names={function.cpp_name for function in funcs}
    )
    group_norm_node = node.split(
        "struct NativeGroupNormBackward : public Node {", 1
    )[1].split("struct InstanceNormBackward : public Node {", 1)[0]
    assert group_norm_node.count("ops::native_group_norm_backward(") == 1
    assert "grad_input_mask.push_back(should_compute_output(0))" in group_norm_node
    assert "grad_input_mask.push_back(should_compute_output(1))" in group_norm_node
    assert "grad_input_mask.push_back(should_compute_output(2))" in group_norm_node

    wrappers = generate_tpx_ops_cpp(
        funcs, autocast_ops=set(), derivatives=derivatives,
        native_op_names={function.cpp_name for function in funcs},
    )
    group_norm_wrapper = wrappers.split(
        "std::tuple<Tensor, Tensor, Tensor> native_group_norm(", 1
    )[1].split("Tensor index(", 1)[0].split(
        "std::tuple<Tensor, Tensor, Tensor> native_group_norm_backward(", 1
    )[0]
    assert "set_grad_fn(std::get<0>(__tp_wrapped_result)" in group_norm_wrapper
    assert "set_grad_fn(std::get<1>(__tp_wrapped_result)" not in group_norm_wrapper
    assert "set_grad_fn(std::get<2>(__tp_wrapped_result)" not in group_norm_wrapper
