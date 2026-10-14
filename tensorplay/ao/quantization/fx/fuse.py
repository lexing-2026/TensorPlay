"""Operator fusion for graph-mode quantization."""

from tensorplay import nn

__all__ = ["fuse"]


def _fusion_groups(model):
    """Yield lists of module names forming fusion patterns in the graph."""
    conv_rank = {nn.Conv1d: 1, nn.Conv2d: 2, nn.Conv3d: 3}
    groups = []
    current = []
    for node in model.graph.nodes:
        if node.op != "call_module":
            continue
        module = model.get_submodule(node.target)
        mtype = type(module)
        if mtype in conv_rank or mtype is nn.Linear:
            if current:
                groups.append(current)
                current = []
            current.append(node.target)
        elif mtype in (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.ReLU) and current:
            current.append(node.target)
        else:
            if current:
                groups.append(current)
                current = []
    if current:
        groups.append(current)
    # Keep only groups of length >= 2 and drop groups that are not
    # conv/linear followed by bn/relu.
    valid = []
    for group in groups:
        if len(group) < 2:
            continue
        first = model.get_submodule(group[0])
        first_is_weighted = isinstance(first, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear))
        if first_is_weighted:
            valid.append(group)
    return valid


def fuse(
    model,
    fuse_custom_config=None,
    backend_config=None,
    is_qat=False,
):
    """Fuse conv/linear-bn-relu patterns found in a traced graph module."""
    from ..fuse_modules import fuse_modules, fuse_modules_qat

    groups = _fusion_groups(model)
    if groups:
        fused = fuse_modules_qat(model, groups, inplace=True) if is_qat else fuse_modules(model, groups, inplace=True)
        return fused
    return model