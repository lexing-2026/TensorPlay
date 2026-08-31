import operator

import tensorplay as tp
from tensorplay.graph import Graph, GraphModule
from tensorplay.graph.passes.shape_prop import ShapeProp, TensorMetadata


def _module_with_tuple_output() -> GraphModule:
    graph = Graph()
    value = graph.placeholder("value")
    shifted = graph.call_function(operator.add, (value, 1))
    graph.output((shifted, value, "tag"))
    return GraphModule(None, graph)


def test_shape_prop_records_nested_tensor_metadata() -> None:
    module = _module_with_tuple_output()
    value = tp.tensor([1.0, 2.0])

    result = ShapeProp(module).propagate(value)

    assert result[0].tolist() == [2.0, 3.0]
    output_meta = module.graph.output_node.meta["tensor_meta"]
    assert isinstance(output_meta[0], TensorMetadata)
    assert isinstance(output_meta[1], TensorMetadata)
    assert output_meta[2] == "tag"
    assert output_meta[0].shape == value.shape
    assert output_meta[0].stride == value.stride()
    assert output_meta[0].memory_format == value.memory_format()


def test_shape_prop_supports_reference_propagate_call_style() -> None:
    module = _module_with_tuple_output()
    value = tp.tensor([3.0])

    result = ShapeProp(module)(value)

    assert result[0].tolist() == [4.0]
    assert module.graph.nodes[0].meta["type"] is tp.Tensor


def test_shape_prop_wraps_node_execution_errors() -> None:
    graph = Graph()
    value = graph.placeholder("value")
    failed = graph.call_function(operator.truediv, (value, 0))
    graph.output(failed)
    module = GraphModule(None, graph)

    try:
        ShapeProp(module).propagate(1)
    except RuntimeError as exc:
        assert "ShapeProp error for" in str(exc)
        assert "truediv" in str(exc)
    else:
        raise AssertionError("shape propagation must report the failing node")
