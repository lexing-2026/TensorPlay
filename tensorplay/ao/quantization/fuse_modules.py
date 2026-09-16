"""Module fusion for inference and quantization-aware training.

Fuses supported patterns — convolution or linear followed by batch norm and
an optional relu — by folding the batch-norm affine transform into the
preceding operator's weights.  Folding is exact in inference (eval) mode:

    conv(x; W, b) followed by BN(gamma, beta, mean, var) with std =
    sqrt(var + eps) computes gamma * (conv_out - mean) / std + beta, which
    equals conv(x; W', b') with W' = W * gamma/std and
    b' = gamma * (b - mean)/std + beta.
"""

from __future__ import annotations

import tensorplay
from tensorplay import nn

__all__ = ["fuse_modules", "fuse_known_modules", "fuse_conv_bn_weights"]


def fuse_conv_bn_weights(conv_w, conv_b, bn_mean, bn_var, bn_gamma, bn_beta, eps):
    """Fold a batch norm affine transform into convolution/linear weights.

    Returns the folded weight and bias (as a plain tensor or ``None`` when
    the input bias is ``None`` and the folded bias is exactly the batch norm
    shift).
    """
    std = (bn_var + eps).sqrt()
    scale = bn_gamma / std
    # one broadcastable 1 after the channel axis, then one per spatial dim
    reshaped = scale.reshape([-1, 1] + [1] * (conv_w.dim() - 2))
    fused_w = conv_w * reshaped
    shift = bn_beta - bn_gamma * bn_mean / std
    if conv_b is not None:
        fused_b = conv_b * scale + shift
    else:
        fused_b = shift
    return fused_w, fused_b


_FUSED_MODULE_FOR = {}


def _register_fused_modules():
    from tensorplay.ao.nn.intrinsic import ConvReLU1d, ConvReLU2d, ConvReLU3d, LinearReLU

    _FUSED_MODULE_FOR.update({
        (nn.Conv1d, nn.ReLU): ConvReLU1d,
        (nn.Conv2d, nn.ReLU): ConvReLU2d,
        (nn.Conv3d, nn.ReLU): ConvReLU3d,
        (nn.Linear, nn.ReLU): LinearReLU,
    })


def _get_supported_fusion_patterns() -> dict:
    return {
        (nn.Conv1d, nn.BatchNorm1d): nn.Conv1d,
        (nn.Conv1d, nn.BatchNorm1d, nn.ReLU): ConvReLU1d_holder(),
        (nn.Conv2d, nn.BatchNorm2d): nn.Conv2d,
        (nn.Conv2d, nn.BatchNorm2d, nn.ReLU): ConvReLU2d_holder(),
        (nn.Conv3d, nn.BatchNorm3d): nn.Conv3d,
        (nn.Conv3d, nn.BatchNorm3d, nn.ReLU): ConvReLU3d_holder(),
        (nn.Linear, nn.BatchNorm1d): nn.Linear,
        (nn.Linear, nn.BatchNorm1d, nn.ReLU): LinearReLU_holder(),
        (nn.Linear, nn.ReLU): LinearReLU_holder(),
        (nn.Conv1d, nn.ReLU): ConvReLU1d_holder(),
        (nn.Conv2d, nn.ReLU): ConvReLU2d_holder(),
        (nn.Conv3d, nn.ReLU): ConvReLU3d_holder(),
    }


def ConvReLU1d_holder():
    from tensorplay.ao.nn.intrinsic import ConvReLU1d

    return ConvReLU1d


def ConvReLU2d_holder():
    from tensorplay.ao.nn.intrinsic import ConvReLU2d

    return ConvReLU2d


def ConvReLU3d_holder():
    from tensorplay.ao.nn.intrinsic import ConvReLU3d

    return ConvReLU3d


def LinearReLU_holder():
    from tensorplay.ao.nn.intrinsic import LinearReLU

    return LinearReLU


def _resolve_path(root, path):
    module = root
    for atom in path.split("."):
        module = module._modules[atom]
    return module


def _set_path(root, path, module):
    atoms = path.split(".")
    parent = root
    for atom in atoms[:-1]:
        parent = parent._modules[atom]
    parent._modules[atoms[-1]] = module


def fuse_known_modules(modules):
    """Fuse one pattern group given the resolved module instances.

    Returns the fused module, or ``None`` when the pattern is unsupported.
    """
    if len(modules) == 1:
        return modules[0]
    base = modules[0]
    act = modules[-1] if isinstance(modules[-1], nn.ReLU) else None
    body = modules[:-1] if act is not None else modules
    bn = body[1] if len(body) > 1 else None

    if bn is not None:
        if isinstance(base, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            expected_bn = {nn.Conv1d: nn.BatchNorm1d, nn.Conv2d: nn.BatchNorm2d,
                           nn.Conv3d: nn.BatchNorm3d}[type(base)]
        elif isinstance(base, nn.Linear):
            expected_bn = nn.BatchNorm1d
        else:
            return None
        if not isinstance(bn, expected_bn):
            return None
        fused_w, fused_b = fuse_conv_bn_weights(
            base.weight.detach(),
            base.bias.detach() if base.bias is not None else None,
            bn.running_mean.detach(),
            bn.running_var.detach(),
            bn.weight.detach() if bn.weight is not None else None,
            bn.bias.detach() if bn.bias is not None else None,
            bn.eps,
        )
        fused = _make_base(base, fused_w, fused_b)
    else:
        fused = _make_base(base, base.weight.detach(),
                           base.bias.detach() if base.bias is not None else None)

    if act is not None:
        _register_fused_modules()
        fused_cls = _FUSED_MODULE_FOR.get((type(base), nn.ReLU))
        if fused_cls is not None:
            fused = fused_cls.from_float(fused)
    return fused


def _make_base(base, weight, bias):
    if isinstance(base, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fused = type(base)(
            base.in_channels,
            base.out_channels,
            base.kernel_size,
            base.stride,
            base.padding,
            base.dilation,
            base.groups,
            bias is not None,
        )
    else:
        fused = nn.Linear(base.in_features, base.out_features, bias is not None)
    with tensorplay.no_grad():
        fused.weight.copy_(weight)
        if bias is not None:
            fused.bias.copy_(bias)
    fused.train(base.training)
    return fused


def _should_fuse(group, patterns) -> bool:
    types = tuple(type(m) for m in group)
    return types in patterns


def fuse_modules(model, modules_to_fuse, inplace=False, fuser_func=None):
    """Fuse supported modules in-place within a model.

    Args:
        model: the model to fuse.
        modules_to_fuse: an iterable of module-name groups, each a list of
            fully-qualified attribute paths, e.g.
            ``[["conv1", "bn1", "relu1"], ["fc"]]``.  Groups of length one
            are ignored (kept for call-site symmetry).
        inplace: mutate the model instead of returning a copy.
        fuser_func: replacement for :func:`fuse_known_modules`.

    Returns:
        The fused model (the same object when ``inplace``).
    """
    import copy

    if fuser_func is None:
        fuser_func = fuse_known_modules
    if not inplace:
        model = copy.deepcopy(model)

    patterns = _get_supported_fusion_patterns()
    for group_names in modules_to_fuse:
        if len(group_names) <= 1:
            continue
        group = [_resolve_path(model, name) for name in group_names]
        if not _should_fuse(group, patterns):
            raise ValueError(
                f"fuse_modules: unsupported fusion pattern "
                f"{[type(m).__name__ for m in group]} for "
                f"{list(group_names)}")
        fused = fuser_func(group)
        if fused is None:
            raise ValueError(
                f"fuse_modules: failed to fuse {list(group_names)}")
        _set_path(model, group_names[0], fused)
        for name in group_names[1:]:
            atoms = name.split(".")
            parent = model
            for atom in atoms[:-1]:
                parent = parent._modules[atom]
            del parent._modules[atoms[-1]]
    return model
