"""Module fusion for calibration, inference and quantization-aware training."""

import copy
import tensorplay
from tensorplay import nn
from tensorplay.ao.nn import intrinsic as nni
from tensorplay.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval
from tensorplay.nn.utils.parametrize import type_before_parametrizations

# for backward compatibility
from .fuser_method_mappings import (  # noqa: F401
    fuse_conv_bn,
    fuse_conv_bn_relu,
    fuse_linear_bn,
    fuse_convtranspose_bn,
    get_fuser_method,
    get_fuser_method_new,
)

__all__ = ['fuse_modules', 'fuse_modules_qat', 'fuse_known_modules', 'fuse_conv_bn_weights']


def fuse_conv_bn_weights(conv_w, conv_b, bn_mean, bn_var, bn_gamma, bn_beta, eps):
    from tensorplay.nn.utils.fusion import fuse_conv_bn_weights as fold
    return fold(conv_w, conv_b, bn_mean, bn_var, eps, bn_gamma, bn_beta)


def _get_module(model, submodule_key):
    tokens = submodule_key.split(".")
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split(".")
    cur_mod = model
    for s in tokens[:-1]:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def fuse_known_modules(mod_list, is_qat=False, additional_fuser_method_mapping=None):
    """Fuse a list of modules, returning a list with the fused module first
    followed by identity modules."""
    types = tuple(type_before_parametrizations(m) for m in mod_list)
    if len(mod_list) < 2:
        return mod_list
    try:
        fuser_method = get_fuser_method(types, additional_fuser_method_mapping)
    except AssertionError:
        fuser_method = None
    if fuser_method is not None:
        new_mod = [None] * len(mod_list)
        fused = fuser_method(is_qat, *mod_list)
        new_mod[0] = fused
        for i in range(1, len(mod_list)):
            identity = nn.Identity()
            identity.training = mod_list[0].training
            new_mod[i] = identity
        return new_mod

    # Fall back to the intrinsic pattern matcher below for patterns not in the
    # default registry (for example linear-bn or bn-relu).
    base = mod_list[0]
    conv_rank = {nn.Conv1d: 1, nn.Conv2d: 2, nn.Conv3d: 3}.get(type(base))
    linear = type(base) is nn.Linear
    relu = type(mod_list[-1]) is nn.ReLU
    bn = mod_list[1] if isinstance(mod_list[1], (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) else None
    if not (conv_rank or linear) or (len(mod_list) != (1 + int(bn is not None) + int(relu))):
        raise ValueError('Unsupported module fusion pattern')
    if len({m.training for m in mod_list}) != 1:
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
            fused = getattr(nni, f'ConvBn{"ReLU" if relu else ""}{conv_rank}d' if conv_rank else 'LinearBn1d')(*mod_list)
        else:
            fused = fuse_conv_bn_eval(base, bn) if conv_rank else fuse_linear_bn_eval(base, bn)
            if relu:
                fused = getattr(nni, f'ConvReLU{conv_rank}d' if conv_rank else 'LinearReLU')(fused, mod_list[-1])
    elif relu:
        fused = getattr(nni, f'ConvReLU{conv_rank}d' if conv_rank else 'LinearReLU')(base, mod_list[-1])
    else:
        raise ValueError('Unsupported module fusion pattern')
    if hasattr(base, 'qconfig'):
        fused.qconfig = base.qconfig
    fused.train(base.training)
    result = [None] * len(mod_list)
    result[0] = fused
    for i in range(1, len(mod_list)):
        identity = nn.Identity()
        identity.training = base.training
        result[i] = identity
    return result


def _fuse_modules_helper(
    model, modules_to_fuse, is_qat, fuser_func, fuse_custom_config_dict
):
    if fuse_custom_config_dict is None:
        fuse_custom_config_dict = {}
    additional_fuser_method_mapping = fuse_custom_config_dict.get(
        "additional_fuser_method_mapping", {}
    )
    mod_list = [_get_module(model, item) for item in modules_to_fuse]
    new_mod_list = fuser_func(mod_list, is_qat, additional_fuser_method_mapping)
    for i, item in enumerate(modules_to_fuse):
        _set_module(model, item, new_mod_list[i])


def _fuse_modules(
    model,
    modules_to_fuse,
    is_qat,
    inplace=False,
    fuser_func=fuse_known_modules,
    fuse_custom_config_dict=None,
):
    if not inplace:
        model = copy.deepcopy(model)
    if all(isinstance(module_element, str) for module_element in modules_to_fuse):
        _fuse_modules_helper(
            model, modules_to_fuse, is_qat, fuser_func, fuse_custom_config_dict
        )
    else:
        for module_list in modules_to_fuse:
            _fuse_modules_helper(
                model, module_list, is_qat, fuser_func, fuse_custom_config_dict
            )
    return model


def fuse_modules(
    model,
    modules_to_fuse,
    inplace=False,
    fuser_func=fuse_known_modules,
    fuse_custom_config_dict=None,
):
    """Fuse a list of modules into a single module.

    Supports conv-bn, conv-bn-relu, conv-relu, linear-relu and bn-relu
    sequences. The first listed module is replaced by the fused module and
    the remaining entries become identity modules.
    """
    return _fuse_modules(
        model,
        modules_to_fuse,
        is_qat=False,
        inplace=inplace,
        fuser_func=fuser_func,
        fuse_custom_config_dict=fuse_custom_config_dict,
    )


def fuse_modules_qat(
    model,
    modules_to_fuse,
    inplace=False,
    fuser_func=fuse_known_modules,
    fuse_custom_config_dict=None,
):
    """Quantization-aware training variant of :func:`fuse_modules`."""
    return _fuse_modules(
        model,
        modules_to_fuse,
        is_qat=True,
        inplace=inplace,
        fuser_func=fuser_func,
        fuse_custom_config_dict=fuse_custom_config_dict,
    )