"""Parameter folding for affine normalization layers."""
import copy
import tensorplay
def fuse_conv_bn_eval(conv, bn, transpose=False):
    if conv.training or bn.training:
        raise AssertionError('Fusion only for eval!')
    fused_conv = copy.deepcopy(conv)
    if bn.running_mean is None or bn.running_var is None:
        raise AssertionError('bn.running_mean and bn.running_var must not be None')
    fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias, transpose)
    return fused_conv

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=False):
    conv_weight_dtype = conv_w.dtype
    conv_bias_dtype = conv_b.dtype if conv_b is not None else conv_weight_dtype
    if conv_b is None:
        conv_b = tensorplay.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = tensorplay.ones_like(bn_rm)
    if bn_b is None:
        bn_b = tensorplay.zeros_like(bn_rm)
    bn_var_rsqrt = tensorplay.rsqrt(bn_rv + bn_eps)
    if transpose:
        shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
    else:
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)
    fused_conv_w = (conv_w * (bn_w * bn_var_rsqrt).reshape(shape)).to(dtype=conv_weight_dtype)
    fused_conv_b = ((conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b).to(dtype=conv_bias_dtype)
    return (tensorplay.nn.Parameter(fused_conv_w.detach(), conv_w.requires_grad), tensorplay.nn.Parameter(fused_conv_b.detach(), conv_b.requires_grad))

def fuse_linear_bn_eval(linear, bn):
    if linear.training or bn.training:
        raise AssertionError('Fusion only for eval!')
    fused_linear = copy.deepcopy(linear)
    if linear.out_features != bn.num_features and bn.num_features != 1:
        raise AssertionError(f'To fuse, linear.out_features == bn.num_features or bn.num_features == 1, got linear.out_features={linear.out_features} and bn.num_features={bn.num_features}')
    if bn.running_mean is None or bn.running_var is None:
        raise AssertionError('bn.running_mean and bn.running_var must not be None')
    fused_linear.weight, fused_linear.bias = fuse_linear_bn_weights(fused_linear.weight, fused_linear.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    return fused_linear

def fuse_linear_bn_weights(linear_w, linear_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    linear_weight_dtype = linear_w.dtype
    linear_bias_dtype = linear_b.dtype if linear_b is not None else linear_weight_dtype
    if linear_b is None:
        linear_b = tensorplay.zeros_like(bn_rm)
    bn_scale = bn_w * tensorplay.rsqrt(bn_rv + bn_eps)
    fused_w = linear_w * bn_scale.unsqueeze(-1).to(dtype=linear_weight_dtype)
    fused_b = ((linear_b - bn_rm) * bn_scale + bn_b).to(dtype=linear_bias_dtype)
    return (tensorplay.nn.Parameter(fused_w.detach(), linear_w.requires_grad), tensorplay.nn.Parameter(fused_b.detach(), linear_b.requires_grad))
