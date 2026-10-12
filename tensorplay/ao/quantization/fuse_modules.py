"""Module fusion for calibration, inference and quantization-aware training."""

import copy
import tensorplay
from tensorplay import nn
from tensorplay.ao.nn import intrinsic as nni
from tensorplay.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval

__all__ = ['fuse_modules', 'fuse_modules_qat', 'fuse_known_modules', 'fuse_conv_bn_weights']


def fuse_conv_bn_weights(conv_w, conv_b, bn_mean, bn_var, bn_gamma, bn_beta, eps):
    from tensorplay.nn.utils.fusion import fuse_conv_bn_weights as fold
    return fold(conv_w, conv_b, bn_mean, bn_var, eps, bn_gamma, bn_beta)


def fuse_known_modules(modules, is_qat=False):
    if len(modules) < 2:
        return modules[0]
    base = modules[0]
    conv_rank = {nn.Conv1d: 1, nn.Conv2d: 2, nn.Conv3d: 3}.get(type(base))
    linear = type(base) is nn.Linear
    relu = type(modules[-1]) is nn.ReLU
    bn = modules[1] if isinstance(modules[1], (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) else None
    if not (conv_rank or linear) or (len(modules) != (1 + int(bn is not None) + int(relu))):
        raise ValueError('Unsupported module fusion pattern')
    if len({m.training for m in modules}) != 1:
        raise ValueError('Fusion requires all modules to share training mode')
    if bn is not None:
        expected = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)[(conv_rank or 1) - 1]
        if type(bn) is not expected:
            raise ValueError('Normalization rank must match the preceding operator')
        if is_qat:
            if not bn.affine or not bn.track_running_stats:
                raise ValueError('Training fusion requires affine parameters and running statistics')
            if bn.num_features != (base.out_channels if conv_rank else base.out_features):
                raise ValueError('Normalization features must match the operator output')
            if linear and relu:
                raise ValueError('Linear normalization training fusion accepts two modules')
            fused = getattr(nni, f'ConvBn{"ReLU" if relu else ""}{conv_rank}d' if conv_rank else 'LinearBn1d')(*modules)
        else:
            fused = fuse_conv_bn_eval(base, bn) if conv_rank else fuse_linear_bn_eval(base, bn)
            if relu:
                fused = getattr(nni, f'ConvReLU{conv_rank}d' if conv_rank else 'LinearReLU')(fused, modules[-1])
    elif relu:
        fused = getattr(nni, f'ConvReLU{conv_rank}d' if conv_rank else 'LinearReLU')(base, modules[-1])
    else:
        raise ValueError('Unsupported module fusion pattern')
    if hasattr(base, 'qconfig'):
        fused.qconfig = base.qconfig
    fused.train(base.training)
    return fused


def _resolve_path(root, path):
    for atom in path.split('.'):
        root = getattr(root, atom)
    return root


def _set_path(root, path, value):
    atoms = path.split('.')
    for atom in atoms[:-1]:
        root = getattr(root, atom)
    setattr(root, atoms[-1], value)


def _fuse(model, groups, inplace, fuser_func, is_qat):
    if not inplace:
        model = copy.deepcopy(model)
    if all(isinstance(name, str) for name in groups):
        groups = [groups]
    for names in groups:
        if len(names) < 2:
            continue
        modules = [_resolve_path(model, name) for name in names]
        fused = fuser_func(modules) if fuser_func else fuse_known_modules(modules, is_qat)
        for hook in modules[0]._forward_pre_hooks.values():
            fused.register_forward_pre_hook(hook)
        modules[0]._forward_pre_hooks.clear()
        for hook in modules[-1]._forward_hooks.values():
            fused.register_forward_hook(hook)
        modules[-1]._forward_hooks.clear()
        _set_path(model, names[0], fused)
        for name in names[1:]:
            _set_path(model, name, nn.Identity().train(modules[0].training))
    return model


def fuse_modules(model, modules_to_fuse, inplace=False, fuser_func=None):
    return _fuse(model, modules_to_fuse, inplace, fuser_func, False)


def fuse_modules_qat(model, modules_to_fuse, inplace=False, fuser_func=None):
    return _fuse(model, modules_to_fuse, inplace, fuser_func, True)
