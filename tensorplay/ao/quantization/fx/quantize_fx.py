"""Public entry points for graph-mode quantization."""

from tensorplay.graph import GraphModule

from .custom_config import ConvertCustomConfig, FuseCustomConfig, PrepareCustomConfig
from .tracer import QuantizationTracer

__all__ = [
    "prepare_fx",
    "convert_fx",
    "fuse_fx",
]


def _prepare_fx(
    model,
    qconfig_mapping,
    example_inputs,
    prepare_custom_config=None,
    backend_config=None,
):
    if prepare_custom_config is None:
        prepare_custom_config = PrepareCustomConfig()
    tracer = QuantizationTracer(
        skipped_module_names=prepare_custom_config.skipped_module_names,
        skipped_module_classes=prepare_custom_config.skipped_module_classes,
    )
    graph_module = tracer.trace(model)
    from .prepare import prepare

    return prepare(
        graph_module,
        qconfig_mapping,
        node_name_to_scope={},
        example_inputs=example_inputs,
        prepare_custom_config=prepare_custom_config,
        backend_config=backend_config,
    )


def prepare_fx(
    model,
    qconfig_mapping,
    example_inputs,
    prepare_custom_config_dict=None,
    backend_config=None,
):
    """Prepare a model for post-training quantization in graph mode."""
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = {}
    prepare_custom_config = PrepareCustomConfig()
    for key, value in prepare_custom_config_dict.items():
        setattr(prepare_custom_config, key, value)
    return _prepare_fx(
        model,
        qconfig_mapping,
        example_inputs,
        prepare_custom_config,
        backend_config,
    )


def convert_fx(
    graph_module,
    conversion_plugin=None,
    is_standalone_module=False,
    backend_config=None,
    convert_custom_config=None,
    remove_qconfig=True,
):
    """Convert an observed graph module to a quantized graph module."""
    from .convert import convert

    return convert(
        graph_module,
        conversion_plugin,
        is_standalone_module,
        backend_config,
        convert_custom_config,
        remove_qconfig,
    )


def fuse_fx(
    model,
    fuse_custom_config_dict=None,
    backend_config=None,
):
    """Fuse operators in a model before graph-mode quantization."""
    from .fuse import fuse

    if fuse_custom_config_dict is None:
        fuse_custom_config_dict = {}
    fuse_custom_config = FuseCustomConfig()
    for key, value in fuse_custom_config_dict.items():
        setattr(fuse_custom_config, key, value)
    tracer = QuantizationTracer()
    graph_module = tracer.trace(model)
    return fuse(graph_module, fuse_custom_config, backend_config)