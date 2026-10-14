"""Preparation (observer insertion) for graph-mode quantization.

The graph module produced by the quantization tracer keeps quantizable leaf
modules as ``call_module`` nodes.  Preparation propagates the qconfig mapping
over the module tree and reuses the eager observer-attachment workflow.
"""

__all__ = ["prepare"]


def _propagate_qconfig_mapping(model, qconfig_mapping):
    """Assign qconfig attributes from a QConfigMapping over the module tree."""

    def assign(module, prefix):
        config = qconfig_mapping.global_qconfig
        config = qconfig_mapping.module_name_qconfigs.get(prefix, config)
        config = qconfig_mapping.object_type_qconfigs.get(type(module), config)
        module.qconfig = getattr(module, "qconfig", config)
        for name, child in module.named_children():
            assign(child, prefix + "." + name if prefix else name)

    assign(model, "")


def prepare(
    model,
    qconfig_mapping,
    node_name_to_scope,
    example_inputs=None,
    prepare_custom_config=None,
    backend_config=None,
    is_standalone_module=False,
):
    """Attach observers to the quantizable leaves of a traced graph module."""
    from ..quantize import prepare as eager_prepare

    if qconfig_mapping is None:
        raise ValueError("qconfig_mapping must be provided")
    _propagate_qconfig_mapping(model, qconfig_mapping)
    eager_prepare(model, inplace=True)
    return model