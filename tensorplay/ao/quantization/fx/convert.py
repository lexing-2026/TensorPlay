"""Conversion of observed graph modules to quantized modules."""

__all__ = ["convert"]


def convert(
    graph_module,
    conversion_plugin=None,
    is_standalone_module=False,
    backend_config=None,
    convert_custom_config=None,
    remove_qconfig=True,
):
    """Swap observed leaf modules in a traced graph for quantized ones."""
    from ..quantize import convert as eager_convert

    eager_convert(graph_module, inplace=True, remove_qconfig=remove_qconfig)
    return graph_module