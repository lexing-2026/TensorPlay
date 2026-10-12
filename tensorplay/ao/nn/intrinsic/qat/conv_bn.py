"""Convolution and normalization training with simulated folded weights."""
import math
from tensorplay.ao.nn.qat.modules.conv import _conv_forward
import tensorplay
from tensorplay import nn
from tensorplay.nn import functional as F, init
from tensorplay.nn.parameter import Parameter
from tensorplay.nn.modules.utils import _pair, _single, _triple
from tensorplay.nn.utils.fusion import fuse_conv_bn_weights
from tensorplay.ao.nn import intrinsic as nni
_BN_CLASS_MAP = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
class _ConvBnNd(nn.modules.conv._ConvNd, nni._FusedModule):
    _version = 2
    _conv_forward = _conv_forward

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode, eps=1e-05, momentum=0.1, freeze_bn=False, qconfig=None, dim=2):
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, False, padding_mode)
        if not qconfig:
            raise AssertionError('qconfig must be provided for QAT module')
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = _BN_CLASS_MAP[dim](out_channels, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = Parameter(tensorplay.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_bn_parameters()
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()
        self._enable_slow_path_for_better_numerical_stability = False

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    def _forward(self, input):
        if self._enable_slow_path_for_better_numerical_stability:
            return self._forward_slow(input)
        return self._forward_approximate(input)

    def _forward_approximate(self, input):
        if self.bn.running_var is None:
            raise AssertionError('self.bn.running_var must not be None')
        running_std = tensorplay.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
        if self.bias is not None:
            zero_bias = tensorplay.zeros_like(self.bias, dtype=input.dtype)
        else:
            zero_bias = tensorplay.zeros(self.out_channels, device=scaled_weight.device, dtype=input.dtype)
        conv = self._conv_forward(input, scaled_weight, zero_bias)
        conv_orig = conv / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            conv_orig = conv_orig + self.bias.reshape(bias_shape)
        conv = self.bn(conv_orig)
        return conv

    def _forward_slow(self, input):
        if self.bn.running_var is None:
            raise AssertionError('self.bn.running_var must not be None')
        if self.bn.running_mean is None:
            raise AssertionError('self.bn.running_mean must not be None')
        zero_bias = tensorplay.zeros(self.out_channels, device=self.weight.device, dtype=input.dtype)
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        if self.bn.training:
            conv_out = self._conv_forward(input, self.weight, zero_bias)
            with tensorplay.no_grad():
                conv_out_bias = conv_out if self.bias is None else conv_out + self.bias.reshape(bias_shape)
                self.bn(conv_out_bias)
            running_std = tensorplay.sqrt(self.bn.running_var + self.bn.eps)
            scale_factor = self.bn.weight / running_std
            scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
            conv_bn = self._conv_forward(input, scaled_weight, zero_bias)
            avg_dims = [0] + list(range(2, len(self.weight.shape)))
            batch_mean = conv_out.mean(avg_dims)
            batch_var = tensorplay.square(conv_out - batch_mean.reshape(bias_shape)).mean(avg_dims)
            batch_std = tensorplay.sqrt(batch_var + self.bn.eps)
            unscale_factor = running_std / batch_std
            conv_bn *= unscale_factor.reshape(bias_shape)
            fused_mean = batch_mean
            fused_std = batch_std
        else:
            running_std = tensorplay.sqrt(self.bn.running_var + self.bn.eps)
            scale_factor = self.bn.weight / running_std
            scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
            conv_bn = self._conv_forward(input, scaled_weight, zero_bias)
            fused_mean = self.bn.running_mean - (self.bias if self.bias is not None else 0)
            fused_std = running_std
        fused_bias = self.bn.bias - self.bn.weight * fused_mean / fused_std
        conv_bn += fused_bias.reshape(bias_shape)
        if self.bias is not None:
            conv_bn += (self.bias - self.bias).reshape(bias_shape)
        return conv_bn

    def forward(self, input):
        return self._forward(input)

    def train(self, mode=True):
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version == 1:
            v2_to_v1_names = {'bn.weight': 'gamma', 'bn.bias': 'beta', 'bn.running_mean': 'running_mean', 'bn.running_var': 'running_var', 'bn.num_batches_tracked': 'num_batches_tracked'}
            for v2_name, v1_name in v2_to_v1_names.items():
                if prefix + v1_name in state_dict:
                    state_dict[prefix + v2_name] = state_dict[prefix + v1_name]
                    state_dict.pop(prefix + v1_name)
                elif prefix + v2_name in state_dict:
                    pass
                elif strict:
                    missing_keys.append(prefix + v2_name)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError('qat.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__)
        if not hasattr(mod, 'qconfig'):
            raise AssertionError('Input float module must have qconfig defined')
        if not mod.qconfig:
            raise AssertionError('Input float module must have a valid qconfig')
        qconfig = mod.qconfig
        conv, bn = (mod[0], mod[1])
        qat_convbn = cls(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode, bn.eps, bn.momentum, False, qconfig)
        qat_convbn.weight = conv.weight
        qat_convbn.bias = conv.bias
        qat_convbn.bn.weight = bn.weight
        qat_convbn.bn.bias = bn.bias
        qat_convbn.bn.running_mean = bn.running_mean
        qat_convbn.bn.running_var = bn.running_var
        qat_convbn.bn.num_batches_tracked = bn.num_batches_tracked
        return qat_convbn

    def to_float(self):
        cls = type(self)
        conv = cls._FLOAT_CONV_MODULE(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias is not None, self.padding_mode)
        conv.weight = tensorplay.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = tensorplay.nn.Parameter(self.bias.detach())
        if cls._FLOAT_BN_MODULE:
            if self.bn.running_var is None or self.bn.running_mean is None:
                raise AssertionError('self.bn.running_var and self.bn.running_mean must not be None')
            conv.weight, conv.bias = fuse_conv_bn_weights(conv.weight, conv.bias, self.bn.running_mean, self.bn.running_var, self.bn.eps, self.bn.weight, self.bn.bias)
        if cls._FLOAT_RELU_MODULE:
            modules = []
            modules.append(conv)
            relu = cls._FLOAT_RELU_MODULE()
            modules.append(relu)
            conv_relu = cls._FUSED_FLOAT_MODULE(*modules)
            conv_relu.train(self.training)
            return conv_relu
        else:
            conv.train(self.training)
            return conv

class ConvBn1d(_ConvBnNd, nn.Conv1d):
    _FLOAT_BN_MODULE = nn.BatchNorm1d
    _FLOAT_RELU_MODULE = None
    _FLOAT_MODULE = nni.ConvBn1d
    _FLOAT_CONV_MODULE = nn.Conv1d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None, padding_mode='zeros', eps=1e-05, momentum=0.1, freeze_bn=False, qconfig=None):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        _ConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, False, _single(0), groups, bias, padding_mode, eps, momentum, freeze_bn, qconfig, dim=1)

class ConvBnReLU1d(ConvBn1d):
    _FLOAT_MODULE = nni.ConvBnReLU1d
    _FLOAT_CONV_MODULE = nn.Conv1d
    _FLOAT_BN_MODULE = nn.BatchNorm1d
    _FLOAT_RELU_MODULE = nn.ReLU
    _FUSED_FLOAT_MODULE = nni.ConvReLU1d

    def forward(self, input):
        return F.relu(self._forward(input))

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant)

class ConvBn2d(_ConvBnNd, nn.Conv2d):
    _FLOAT_MODULE = nni.ConvBn2d
    _FLOAT_CONV_MODULE = nn.Conv2d
    _FLOAT_BN_MODULE = nn.BatchNorm2d
    _FLOAT_RELU_MODULE = None

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None, padding_mode='zeros', eps=1e-05, momentum=0.1, freeze_bn=False, qconfig=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _ConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode, eps, momentum, freeze_bn, qconfig, dim=2)

class ConvBnReLU2d(ConvBn2d):
    _FLOAT_MODULE = nni.ConvBnReLU2d
    _FLOAT_CONV_MODULE = nn.Conv2d
    _FLOAT_BN_MODULE = nn.BatchNorm2d
    _FLOAT_RELU_MODULE = nn.ReLU
    _FUSED_FLOAT_MODULE = nni.ConvReLU2d

    def forward(self, input):
        return F.relu(self._forward(input))

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant)

class ConvBn3d(_ConvBnNd, nn.Conv3d):
    _FLOAT_MODULE = nni.ConvBn3d
    _FLOAT_CONV_MODULE = nn.Conv3d
    _FLOAT_BN_MODULE = nn.BatchNorm3d
    _FLOAT_RELU_MODULE = None

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None, padding_mode='zeros', eps=1e-05, momentum=0.1, freeze_bn=False, qconfig=None):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        _ConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, False, _triple(0), groups, bias, padding_mode, eps, momentum, freeze_bn, qconfig, dim=3)

class ConvBnReLU3d(ConvBn3d):
    _FLOAT_MODULE = nni.ConvBnReLU3d
    _FLOAT_CONV_MODULE = nn.Conv3d
    _FLOAT_BN_MODULE = nn.BatchNorm3d
    _FLOAT_RELU_MODULE = nn.ReLU
    _FUSED_FLOAT_MODULE = nni.ConvReLU3d

    def forward(self, input):
        return F.relu(ConvBn3d._forward(self, input))

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)

def update_bn_stats(mod):
    if type(mod) in {ConvBnReLU1d, ConvBnReLU2d, ConvBnReLU3d, ConvBn1d, ConvBn2d, ConvBn3d}:
        mod.update_bn_stats()

def freeze_bn_stats(mod):
    if type(mod) in {ConvBnReLU1d, ConvBnReLU2d, ConvBnReLU3d, ConvBn1d, ConvBn2d, ConvBn3d}:
        mod.freeze_bn_stats()
