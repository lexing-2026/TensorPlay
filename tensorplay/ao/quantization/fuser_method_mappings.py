"""Default operator-pattern to fusion-method registry for eager quantization."""

import itertools
from collections.abc import Callable

import tensorplay.ao.nn.intrinsic as nni
from tensorplay import nn

from .utils import MatchAllNode, Pattern, get_combined_dict

__all__ = [
    "fuse_conv_bn",
    "fuse_conv_bn_relu",
    "fuse_linear_bn",
    "fuse_convtranspose_bn",
    "get_fuser_method",
    "get_fuser_method_new",
]


def fuse_conv_bn(is_qat, conv, bn):
    """Fuse a convolution with a spatial batch norm."""
    if conv.training != bn.training:
        raise AssertionError(
            "Conv and BN both must be in the same mode (train or eval)."
        )

    fused_module_class_map = {
        nn.Conv1d: nni.ConvBn1d,
        nn.Conv2d: nni.ConvBn2d,
        nn.Conv3d: nni.ConvBn3d,
    }

    if is_qat:
        if bn.num_features != conv.out_channels:
            raise AssertionError(
                "Output channel of Conv2d must match num_features of BatchNorm2d."
            )
        if not bn.affine:
            raise AssertionError(
                "Only support fusing BatchNorm2d with affine set to True"
            )
        if not bn.track_running_stats:
            raise AssertionError(
                "Only support fusing BatchNorm2d with tracking_running_stats set to True"
            )
        fused_module_class = fused_module_class_map.get(type(conv))
        if fused_module_class is not None:
            return fused_module_class(conv, bn)
        raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn)}")
    else:
        return nn.utils.fuse_conv_bn_eval(conv, bn)


def fuse_conv_bn_relu(is_qat, conv, bn, relu):
    """Fuse a convolution, batch norm and rectifier into one module."""
    if not (conv.training == bn.training == relu.training):
        raise AssertionError(
            "Conv and BN both must be in the same mode (train or eval)."
        )
    fused_module = None
    if is_qat:
        map_to_fused_module_train = {
            nn.Conv1d: nni.ConvBnReLU1d,
            nn.Conv2d: nni.ConvBnReLU2d,
            nn.Conv3d: nni.ConvBnReLU3d,
        }
        if bn.num_features != conv.out_channels:
            raise AssertionError(
                "Output channel of Conv2d must match num_features of BatchNorm2d"
            )
        if not bn.affine:
            raise AssertionError(
                "Only support fusing BatchNorm2d with affine set to True"
            )
        if not bn.track_running_stats:
            raise AssertionError(
                "Only support fusing BatchNorm2d with tracking_running_stats set to True"
            )
        fused_module = map_to_fused_module_train.get(type(conv))
        if fused_module is not None:
            return fused_module(conv, bn, relu)
        raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn, relu)}")
    else:
        map_to_fused_module_eval = {
            nn.Conv1d: nni.ConvReLU1d,
            nn.Conv2d: nni.ConvReLU2d,
            nn.Conv3d: nni.ConvReLU3d,
        }
        fused_module = map_to_fused_module_eval.get(type(conv))
        if fused_module is not None:
            fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
            return fused_module(fused_conv, relu)
        raise NotImplementedError(f"Cannot fuse eval modules: {(conv, bn, relu)}")


def fuse_linear_bn(is_qat, linear, bn):
    """Fuse a linear layer with a batch norm."""
    if linear.training != bn.training:
        raise AssertionError(
            "Linear and BN both must be in the same mode (train or eval)."
        )

    if is_qat:
        if bn.num_features != linear.out_features:
            raise AssertionError(
                "Output features of Linear must match num_features of BatchNorm1d"
            )
        if not bn.affine:
            raise AssertionError(
                "Only support fusing BatchNorm1d with affine set to True"
            )
        if not bn.track_running_stats:
            raise AssertionError(
                "Only support fusing BatchNorm1d with tracking_running_stats set to True"
            )
        return nni.LinearBn1d(linear, bn)
    else:
        return nn.utils.fusion.fuse_linear_bn_eval(linear, bn)


def fuse_convtranspose_bn(is_qat, convt, bn):
    """Fuse a transposed convolution with a batch norm."""
    if convt.training != bn.training:
        raise AssertionError(
            "ConvTranspose and BN both must be in the same mode (train or eval)."
        )

    if is_qat:
        raise Exception(  # noqa: TRY002
            "Fusing ConvTranspose+BatchNorm not yet supported in QAT."
        )
    else:
        return nn.utils.fusion.fuse_conv_bn_eval(convt, bn, transpose=True)


def _sequential_wrapper2(sequential):
    """Build a fuser method that always wraps two modules in ``sequential``."""

    def fuser_method(is_qat, m1, m2):
        return sequential(m1, m2)

    return fuser_method


_DEFAULT_OP_LIST_TO_FUSER_METHOD: dict[tuple, nn.Sequential | Callable] = {
    (nn.Conv1d, nn.BatchNorm1d): fuse_conv_bn,
    (nn.Conv1d, nn.BatchNorm1d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv2d, nn.BatchNorm2d): fuse_conv_bn,
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv3d, nn.BatchNorm3d): fuse_conv_bn,
    (nn.Conv3d, nn.BatchNorm3d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv1d, nn.ReLU): _sequential_wrapper2(nni.ConvReLU1d),
    (nn.Conv2d, nn.ReLU): _sequential_wrapper2(nni.ConvReLU2d),
    (nn.Conv3d, nn.ReLU): _sequential_wrapper2(nni.ConvReLU3d),
    (nn.Linear, nn.BatchNorm1d): fuse_linear_bn,
    (nn.Linear, nn.ReLU): _sequential_wrapper2(nni.LinearReLU),
    (nn.BatchNorm2d, nn.ReLU): _sequential_wrapper2(nni.BNReLU2d),
    (nn.BatchNorm3d, nn.ReLU): _sequential_wrapper2(nni.BNReLU3d),
    (nn.ConvTranspose1d, nn.BatchNorm1d): fuse_convtranspose_bn,
    (nn.ConvTranspose2d, nn.BatchNorm2d): fuse_convtranspose_bn,
    (nn.ConvTranspose3d, nn.BatchNorm3d): fuse_convtranspose_bn,
}


def get_fuser_method(op_list, additional_fuser_method_mapping=None):
    """Return the fuser method for a tuple of module types."""
    if additional_fuser_method_mapping is None:
        additional_fuser_method_mapping = {}
    all_mappings = get_combined_dict(
        _DEFAULT_OP_LIST_TO_FUSER_METHOD, additional_fuser_method_mapping
    )
    fuser_method = all_mappings.get(op_list, None)
    if fuser_method is None:
        raise AssertionError(f"did not find fuser method for: {op_list} ")
    return fuser_method


def _reverse2(f):
    def reversed(is_qat, x, y):
        return f(is_qat, y, x)

    return reversed


def _reverse3(f):
    def reversed(is_qat, x, w):
        y, z = w
        return f(is_qat, z, y, x)

    return reversed


def _get_valid_patterns(op_pattern):
    """Expand an op pattern into all patterns where ``MatchAllNode`` stands in
    for any concrete node type."""
    result = []
    if isinstance(op_pattern, (tuple, list)):
        sub_combs = [_get_valid_patterns(sub_pattern) for sub_pattern in op_pattern]
        result = list(itertools.product(*sub_combs))
    else:
        result = [op_pattern, MatchAllNode]
    return result


def get_fuser_method_new(op_pattern: Pattern, fuser_method_mapping: dict):
    """Look up a fuser method allowing ``MatchAllNode`` wildcards in patterns."""
    op_patterns = _get_valid_patterns(op_pattern)
    fuser_method = None
    for op_pattern in op_patterns:
        fuser_method = fuser_method_mapping.get(op_pattern)
        if fuser_method is not None:
            break
    if fuser_method is None:
        raise AssertionError(f"did not find fuser method for: {op_pattern} ")
    return fuser_method