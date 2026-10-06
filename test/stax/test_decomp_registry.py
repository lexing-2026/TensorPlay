import math

import pytest

import tensorplay as tp
from tensorplay._decomp import (
    core_decompositions,
    decomposition_table,
    get_decompositions,
    register_decomposition,
    remove_decompositions,
)
from tensorplay._ops import NATIVE_NAMESPACE

ops = getattr(tp.ops, NATIVE_NAMESPACE)
tp.manual_seed(0)


def _t(*shape, low=-2.0, high=2.0):
    return tp.rand(*shape) * (high - low) + low


def _unit(*shape):
    return tp.rand(*shape) * 0.8 + 0.1


def _lattice_bn_input():
    # Batch norm runs its statistics in a fused kernel on one side and a
    # composite var/mean walk on the other; unit-scale lattices keep the two
    # rounding paths inside the comparison tolerance.
    return tp.arange(24, dtype=tp.float32).reshape(2, 3, 2, 2) * 0.01


def _lat(*shape):
    # Same idea as the batch norm lattice, at any shape.
    return tp.arange(math.prod(shape), dtype=tp.float32).reshape(shape) * 0.01


def _pool2d_backward_sample():
    inp = tp.randn(2, 3, 6, 6)
    indices = ops.max_pool2d_with_indices.default(inp, [2, 2], [2, 2])[1]
    return (tp.randn(2, 3, 3, 3), inp, [2, 2], [2, 2]), {"indices": indices}


# Sample inputs per overload name: (args, kwargs).  Inputs stay away from
# points where an operator is discontinuous so both sides agree exactly
# up to rounding.
SAMPLES = {
    "addcmul.default": lambda: ((_t(3, 4), _t(3, 4), _t(3, 4)), {"value": 0.5}),
    "addcdiv.default": lambda: ((_t(3, 4), _t(3, 4), _unit(3, 4)), {"value": 2}),
    "rsub.Tensor": lambda: ((_t(3), _t(3)), {"alpha": 2}),
    "rsub.Scalar": lambda: ((_t(3), 1.5), {}),
    "clamp_min.default": lambda: ((_t(5), 0.3), {}),
    "clamp_min.Tensor": lambda: ((_t(5), _t(5)), {}),
    "clamp_max.default": lambda: ((_t(5), 0.3), {}),
    "clamp_max.Tensor": lambda: ((_t(5), _t(5)), {}),
    "deg2rad.default": lambda: ((_t(4),), {}),
    "rad2deg.default": lambda: ((_t(4),), {}),
    "frac.default": lambda: ((_t(6, low=-5, high=5),), {}),
    "sgn.default": lambda: ((tp.tensor([-2.0, 0.0, 3.0]),), {}),
    "sinc.default": lambda: ((tp.tensor([0.0, 0.5, -1.25]),), {}),
    "heaviside.default": lambda: ((tp.tensor([-1.0, 0.0, 2.0]), tp.tensor([0.5])), {}),
    "lerp.default": lambda: ((_t(4), _t(4), 0.3), {}),
    "lerp.Scalar": lambda: ((_t(4), _t(4), 0.8), {}),
    "lerp.Tensor": lambda: ((_t(4), _t(4), _unit(4)), {}),
    "logaddexp.default": lambda: ((_t(5), _t(5)), {}),
    "logaddexp2.default": lambda: ((_t(5), _t(5)), {}),
    "xlogy.default": lambda: ((tp.tensor([0.0, 1.0, 2.0]), _unit(3)), {}),
    "xlogy.Tensor": lambda: ((tp.tensor([0.0, 1.0, 2.0]), _unit(3)), {}),
    "xlogy.Scalar_Other": lambda: ((tp.tensor([0.0, 1.0, 2.0]), 0.5), {}),
    "xlogy.Scalar_Self": lambda: ((2.0, _unit(3)), {}),
    "nan_to_num.default": lambda: (
        (tp.tensor([float("nan"), float("inf"), -float("inf"), 1.0]),), {"nan": 2.0}
    ),
    "logit.default": lambda: ((_unit(4),), {}),
    "silu.default": lambda: ((_t(5),), {}),
    "silu_backward.default": lambda: ((_t(5), _t(5)), {}),
    "mish.default": lambda: ((_t(5),), {}),
    "mish_backward.default": lambda: ((_t(5), _t(5)), {}),
    "celu.default": lambda: ((_t(5),), {"alpha": 1.5}),
    "elu_backward.default": lambda: ((_t(5), 1.0, 1.0, 1.0, False, _t(5)), {}),
    "hardsigmoid.default": lambda: ((_t(6, low=-5, high=5),), {}),
    "hardsigmoid_backward.default": lambda: ((_t(6), _t(6, low=-5, high=5)), {}),
    "hardswish.default": lambda: ((_t(6, low=-5, high=5),), {}),
    "hardswish_backward.default": lambda: ((_t(6), _t(6, low=-5, high=5)), {}),
    "hardtanh_backward.default": lambda: ((_t(6), _t(6), -1.0, 1.0), {}),
    "hardshrink.default": lambda: ((_t(6),), {}),
    "softshrink.default": lambda: ((_t(6),), {}),
    "leaky_relu_backward.default": lambda: ((_t(6), _t(6), 0.1, False), {}),
    "gelu_backward.default": lambda: ((_t(6), _t(6)), {}),
    "softplus.default": lambda: ((_t(6),), {"beta": 2, "threshold": 3}),
    "softplus_backward.default": lambda: ((_t(6), _t(6), 2, 3), {}),
    "threshold.default": lambda: ((_t(6), 0.1, 20.0), {}),
    "threshold_backward.default": lambda: ((_t(6), _t(6), 0.1), {}),
    "sigmoid_backward.default": lambda: ((_t(6), _unit(6)), {}),
    "tanh_backward.default": lambda: ((_t(6), _t(6, low=-0.9, high=0.9)), {}),
    "logit_backward.default": lambda: ((_t(6), _unit(6)), {"eps": 0.2}),
    "glu.default": lambda: ((_t(4, 6),), {}),
    "mv.default": lambda: ((_t(3, 4), _t(4)), {}),
    "dot.default": lambda: ((_t(5), _t(5)), {}),
    "vdot.default": lambda: ((_t(5), _t(5)), {}),
    "trace.default": lambda: ((_t(4, 4),), {}),
    # creation
    "zeros.default": lambda: (([2, 3],), {}),
    "ones.default": lambda: (([2, 3],), {"dtype": tp.float64}),
    "zeros_like.default": lambda: ((_t(2, 3),), {}),
    "ones_like.default": lambda: ((_t(2, 3),), {"dtype": tp.int64}),
    "new_full.default": lambda: ((_t(2), [3, 2], 7.0), {}),
    "new_zeros.default": lambda: ((_t(2), [3]), {}),
    "new_ones.default": lambda: ((_t(2), [2, 2]), {"dtype": tp.float64}),
    "fill.Scalar": lambda: ((_t(3), 2.5), {}),
    "fill.Tensor": lambda: ((_t(3), tp.tensor(4.0)), {}),
    "eye.default": lambda: ((3,), {"m": 4}),
    "eye.m": lambda: ((3, 2), {}),
    "linspace.default": lambda: ((-1.0, 2.0, 7), {}),
    "linspace.Tensor_Tensor": lambda: ((tp.tensor(0.0), tp.tensor(1.0), 5), {}),
    "linspace.Tensor_Scalar": lambda: ((tp.tensor(0.0), 3.0, 4), {}),
    "linspace.Scalar_Tensor": lambda: ((1.0, tp.tensor(2.0), 3), {}),
    "logspace.default": lambda: ((0.0, 2.0, 5), {"base": 2.0}),
    "logspace.Tensor_Tensor": lambda: ((tp.tensor(0.0), tp.tensor(1.0), 4), {}),
    "logspace.Tensor_Scalar": lambda: ((tp.tensor(0.0), 1.0, 3), {}),
    "logspace.Scalar_Tensor": lambda: ((0.0, tp.tensor(1.0), 3), {}),
    "hann_window.default": lambda: ((8,), {}),
    "hann_window.periodic": lambda: ((8, False), {}),
    # views and copies
    "t.default": lambda: ((_t(2, 3),), {}),
    "transpose.default": lambda: ((_t(2, 3, 4), 0, 2), {}),
    "transpose.int": lambda: ((_t(2, 3, 4), -1, 1), {}),
    "_unsafe_view.default": lambda: ((_t(2, 6), [3, 4]), {}),
    "_reshape_alias.default": lambda: ((_t(2, 6), [4, 3], [3, 1]), {}),
    "expand_as.default": lambda: ((_t(1, 3), _t(4, 3)), {}),
    "detach.default": lambda: ((_t(3),), {}),
    "alias_copy.default": lambda: ((_t(3),), {}),
    "t_copy.default": lambda: ((_t(2, 3),), {}),
    "transpose_copy.int": lambda: ((_t(2, 3), 0, 1), {}),
    "squeeze_copy.default": lambda: ((_t(1, 3, 1),), {}),
    "squeeze_copy.dim": lambda: ((_t(1, 3, 1), 2), {}),
    "squeeze_copy.dims": lambda: ((_t(1, 3, 1), [0, 2]), {}),
    "unsqueeze_copy.default": lambda: ((_t(3), 0), {}),
    "diagonal_copy.default": lambda: ((_t(3, 4),), {"offset": 1}),
    "unfold_copy.default": lambda: ((_t(7), 0, 3, 2), {}),
    "expand_copy.default": lambda: ((_t(1, 3), [2, 3]), {}),
    "split.default": lambda: ((_t(7, 2), 3), {}),
    "split.Tensor": lambda: ((_t(2, 5), 2, 1), {}),
    "split.sizes": lambda: ((_t(6), [1, 2, 3]), {}),
    "unsafe_split.Tensor": lambda: ((_t(5), 2), {}),
    "unsafe_split_with_sizes.default": lambda: ((_t(5), [2, 3]), {}),
    "split_with_sizes_copy.default": lambda: ((_t(5), [4, 1]), {}),
    "lift.default": lambda: ((_t(3),), {}),
    "lift_fresh.default": lambda: ((_t(3),), {}),
    "as_strided_copy.default": lambda: ((tp.arange(6.0), [2, 3], [3, 1]), {"storage_offset": 0}),
    "as_strided_scatter.default": lambda: (
        (tp.arange(6.0), tp.arange(6.0).reshape(2, 3) + 100, [2, 3], [3, 1]),
        {"storage_offset": 0},
    ),
    "unbind.default": lambda: ((_t(3, 2),), {}),
    "unbind.int": lambda: ((_t(3, 2), 1), {}),
    "narrow.default": lambda: ((_t(5, 2), 0, 1, 3), {}),
    "narrow.Tensor": lambda: ((_t(5, 2), 0, tp.tensor(2), 2), {}),
    "stack.default": lambda: (([_t(2, 3), _t(2, 3)], 1), {}),
    "roll.default": lambda: ((_t(3, 4), [1, -2], [0, 1]), {}),
    "rot90.default": lambda: ((_t(2, 3), 3, [0, 1]), {}),
    "take.default": lambda: ((_t(3, 4), tp.tensor([[0, -1], [5, 2]])), {}),
    "tril.default": lambda: ((_t(4, 5), 1), {}),
    "triu.default": lambda: ((_t(4, 5), -1), {}),
    "diag_embed.default": lambda: ((_t(2, 3),), {"offset": 1}),
    "block_diag.default": lambda: (([_t(2, 2), _t(1, 3), _t(3)],), {}),
    "select_scatter.default": lambda: ((_t(3, 4), _t(4), 0, -1), {}),
    "select_backward.default": lambda: ((_t(4), _t(3, 4), 0, 1), {}),
    "slice_backward.default": lambda: ((_t(3, 2), _t(3, 5), 1, 1, 5, 2), {}),
    "diagonal_backward.default": lambda: ((_t(3), [3, 4], 0, 0, 1), {}),
    "unfold_backward.default": lambda: ((_t(3, 3), [7], 0, 3, 2), {}),
    "alias.default": lambda: ((_t(2, 3),), {}),
    "clone.default": lambda: ((_t(2, 3),), {}),
    "unsqueeze.default": lambda: ((_t(2, 3), 1), {}),
    "squeeze.default": lambda: ((_t(1, 3, 1),), {}),
    "squeeze.dim": lambda: ((_t(1, 3, 1), 0), {}),
    "squeeze.dims": lambda: ((_t(1, 3, 1), [0, 2]), {}),
    "permute.default": lambda: ((_t(2, 3, 4), [2, 0, 1]), {}),
    "expand.default": lambda: ((_t(1, 3), [4, 3]), {}),
    "flip.default": lambda: ((_t(2, 3), [0, 1]), {}),
    "slice.Tensor": lambda: ((_t(5, 4), 0, 1, 4, 2), {}),
    "slice_scatter.default": lambda: ((_t(5, 4), _t(2, 4), 0, 1, 4, 2), {}),
    "split_with_sizes.default": lambda: ((_t(6, 3), [2, 1, 3], 0), {}),
    "unfold.default": lambda: ((_t(6), 0, 3, 2), {}),
    "diagonal.default": lambda: ((_t(3, 4),), {"offset": 1}),
    "diagonal_scatter.default": lambda: ((_t(3, 4), _t(3), 1, 0, 1), {}),
    "view.default": lambda: ((_t(2, 6), [3, 4]), {}),
    "cat.default": lambda: (([_t(2, 3), _t(1, 3)], 0), {}),
    "meshgrid.default": lambda: (([_t(2), _t(3)], "ij"), {}),
    "meshgrid.indexing": lambda: (([_t(2), _t(3)],), {"indexing": "xy"}),
    "constant_pad_nd.default": lambda: ((_t(2, 3), [1, 2], 0.5), {}),
    "repeat.default": lambda: ((_t(2, 3), [2, 1, 2]), {}),
    "tril_indices.default": lambda: ((4, 5, -1), {}),
    "triu_indices.default": lambda: ((4, 5, 1), {}),
    "permute_copy.default": lambda: ((_t(2, 3, 4), [2, 0, 1]), {}),
    "narrow_copy.default": lambda: ((_t(4, 6), 1, 1, 3), {}),
    "view_copy.default": lambda: ((_t(2, 6), [4, 3]), {}),
    # fills and statistics
    "masked_fill.default": lambda: ((_t(4), tp.tensor([True, False, True, False]), 9.0), {}),
    "masked_fill.Scalar": lambda: ((_t(4), tp.tensor([True, False, True, False]), 9.0), {}),
    "masked_fill.Tensor": lambda: ((_t(4), tp.tensor([False, True, True, False]), tp.tensor(3.0)), {}),
    "index_add.default": lambda: ((_t(4, 3), 0, tp.tensor([0, 2, 0]), _t(3, 3)), {}),
    "index_copy.default": lambda: ((_t(4, 3), 0, tp.tensor([3, 1]), _t(2, 3)), {}),
    "index_fill.Scalar": lambda: ((_t(4, 3), 1, tp.tensor([0, 2]), 5.0), {}),
    "index_fill.Tensor": lambda: ((_t(4, 3), 0, tp.tensor([1]), tp.tensor(5.0)), {}),
    "index_fill.int_Scalar": lambda: ((_t(4, 3), 1, tp.tensor([-1]), 5.0), {}),
    "index_fill.int_Tensor": lambda: ((_t(4, 3), 0, tp.tensor([0, 3]), tp.tensor(-2.0)), {}),
    "count_nonzero.default": lambda: ((tp.tensor([[0.0, 1.0], [2.0, 0.0]]),), {}),
    "count_nonzero.dim_IntList": lambda: ((tp.tensor([[0.0, 1.0], [2.0, 0.0]]), [1]), {}),
    "nansum.default": lambda: ((tp.tensor([[1.0, float("nan")], [2.0, 3.0]]), [1]), {}),
    "isposinf.default": lambda: ((tp.tensor([1.0, float("inf"), -float("inf")]),), {}),
    "isneginf.default": lambda: ((tp.tensor([1.0, float("inf"), -float("inf")]),), {}),
    "isin.Tensor_Tensor": lambda: ((tp.tensor([1, 2, 3, 4]), tp.tensor([2, 4])), {}),
    "isin.Tensor_Scalar": lambda: ((tp.tensor([1, 2, 3]), 2), {"invert": True}),
    "isin.Scalar_Tensor": lambda: ((3, tp.tensor([1, 3])), {}),
    "std.default": lambda: ((_t(4, 5),), {}),
    "std.dim": lambda: ((_t(4, 5), [1]), {"keepdim": True}),
    "std.correction": lambda: ((_t(4, 5),), {"dim": [0], "correction": 0}),
    "std_mean.default": lambda: ((_t(4, 5), [1]), {}),
    "std_mean.dim": lambda: ((_t(4, 5), [0]), {"unbiased": False}),
    "std_mean.correction": lambda: ((_t(4, 5),), {"correction": 2}),
    # batch 3
    "leaky_relu.default": lambda: ((_t(6), 0.2), {}),
    "elu.default": lambda: ((_t(6), 1.5, 1.2, 0.8), {}),
    "gelu.default": lambda: ((_t(6),), {}),
    "hardtanh.default": lambda: ((_t(6), -0.5, 0.7), {}),
    "log_sigmoid_forward.default": lambda: ((_t(6, low=-5, high=5),), {}),
    "log_sigmoid_backward.default": lambda: ((_t(6), _t(6, low=-5, high=5)), {}),
    "glu_backward.default": lambda: ((_t(4, 3), _t(4, 6)), {}),
    "floor_divide.default": lambda: ((_t(6, low=-9, high=9), _unit(6) + 1), {}),
    "floor_divide.Scalar": lambda: ((_t(6, low=-9, high=9), 2.5), {}),
    "baddbmm.default": lambda: ((_t(2, 3, 4), _t(2, 3, 5), _t(2, 5, 4)), {"beta": 0.5, "alpha": 2}),
    "baddbmm.dtype": lambda: ((_t(2, 3, 4), _t(2, 3, 5), _t(2, 5, 4), tp.float32), {}),
    "addr.default": lambda: ((_t(3, 4), _t(3), _t(4)), {"beta": 2, "alpha": 0.5}),
    "all.default": lambda: ((tp.tensor([[1.0, 0.0], [2.0, 3.0]]),), {}),
    "all.dim": lambda: ((tp.tensor([[1.0, 0.0], [2.0, 3.0]]), 1), {"keepdim": True}),
    "all.dims": lambda: ((tp.tensor([[1.0, 0.0], [2.0, 3.0]]), [0]), {}),
    "aminmax.default": lambda: ((_t(3, 4), [1]), {}),
    "linalg_cross.default": lambda: ((_t(4, 3), _t(4, 3)), {}),
    "mvlgamma.default": lambda: ((_t(4, low=2, high=4), 3), {}),
    "renorm.default": lambda: ((_t(3, 4), 2, 0, 1.0), {}),
    "_lazy_clone.default": lambda: ((_t(3),), {}),
    "special_xlog1py.default": lambda: ((tp.tensor([0.0, 1.0, 2.0]), _unit(3)), {}),
    "special_xlog1py.other_scalar": lambda: ((tp.tensor([0.0, 1.0]), 0.5), {}),
    "special_xlog1py.self_scalar": lambda: ((2.0, _unit(3)), {}),
    "special_entr.default": lambda: ((tp.tensor([0.0, 0.5, -1.0, 2.0]),), {}),
    "special_log_ndtr.default": lambda: ((_t(6, low=-6, high=4),), {}),
    "_softmax_backward_data.default": lambda: ((_t(3, 4), tp.softmax(_t(3, 4), 1), 1, tp.float32), {}),
    "_log_softmax_backward_data.default": lambda: ((_t(3, 4), tp.log_softmax(_t(3, 4), 1), 1, tp.float32), {}),
    "_safe_softmax.default": lambda: ((tp.tensor([[1.0, 2.0], [-float("inf"), -float("inf")]]), 1), {}),
    "mse_loss.default": lambda: ((_t(3, 4), _t(3, 4)), {"reduction": 2}),
    "mse_loss_backward.default": lambda: ((tp.tensor(1.5), _t(3, 4), _t(3, 4)), {}),
    "l1_loss.default": lambda: ((_t(3, 4), _t(3, 4)), {}),
    "smooth_l1_loss.default": lambda: ((_t(8), _t(8)), {"beta": 0.5}),
    "smooth_l1_loss_backward.default": lambda: ((tp.tensor(1.0), _t(8), _t(8), 1, 0.5), {}),
    "huber_loss.default": lambda: ((_t(8), _t(8)), {"delta": 0.5, "reduction": 0}),
    "huber_loss_backward.default": lambda: ((_t(8), _t(8), _t(8), 0, 0.5), {}),
    "binary_cross_entropy.default": lambda: ((_unit(6), _unit(6)), {}),
    "binary_cross_entropy_backward.default": lambda: ((tp.tensor(1.0), _unit(6), _unit(6)), {}),
    "binary_cross_entropy_with_logits.default": lambda: ((_t(6), _unit(6)), {"pos_weight": _unit(6) + 0.5}),
    "soft_margin_loss.default": lambda: ((_t(6), tp.sign(_t(6)) + 0.0), {}),
    "soft_margin_loss_backward.default": lambda: ((tp.tensor(1.0), _t(6), tp.sign(_t(6)), 1), {}),
    "pixel_shuffle.default": lambda: ((_t(2, 8, 3, 3), 2), {}),
    "pixel_unshuffle.default": lambda: ((_t(2, 2, 4, 6), 2), {}),
    "channel_shuffle.default": lambda: ((_t(2, 6, 3), 3), {}),
    "_prelu_kernel.default": lambda: ((_t(2, 3, 4), _unit(3, 1)), {}),
    "_prelu_kernel_backward.default": lambda: ((_t(2, 3, 4), _t(2, 3, 4), _unit(3, 1)), {}),
    "_weight_norm_interface.default": lambda: ((_t(4, 3), _unit(4, 1)), {}),
    "embedding_dense_backward.default": lambda: (
        (_t(5, 3), tp.tensor([0, 2, 2, 4, 1]), 6, 2, True), {}
    ),
    # batch 4: normalization backward, dropout, negative log likelihood
    "native_batch_norm_backward.default": lambda: _bn_backward_sample(True),
    "native_layer_norm_backward.default": lambda: _ln_backward_sample(),
    "native_group_norm_backward.default": lambda: _gn_backward_sample(),
    "native_dropout_backward.default": lambda: ((_t(6), tp.tensor([True, False] * 3), 2.0), {}),
    "nll_loss_forward.default": lambda: (
        (tp.log_softmax(_t(4, 5), 1), tp.tensor([0, 2, -100, 4]), _unit(5), 1, -100), {}
    ),
    "nll_loss2d_forward.default": lambda: (
        (tp.log_softmax(_t(2, 3, 2, 2), 1), tp.tensor([[[0, 1], [2, 0]], [[1, 1], [0, 2]]]), None, 2, -100), {}
    ),
    "nll_loss_backward.default": lambda: _nll_backward_sample(),
    "nll_loss2d_backward.default": lambda: _nll2d_backward_sample(),
    # batch 4: padding, resampling, sliding blocks, unpooling
    "reflection_pad1d.default": lambda: ((_t(2, 3, 5), [2, 3]), {}),
    "reflection_pad2d.default": lambda: ((_t(2, 3, 4, 5), [1, 2, 3, 0]), {}),
    "reflection_pad3d.default": lambda: ((_t(1, 2, 3, 4, 3), [1, 1, 2, 0, 0, 2]), {}),
    "replication_pad1d.default": lambda: ((_t(2, 3, 5), [3, 1]), {}),
    "replication_pad2d.default": lambda: ((_t(3, 4, 5), [1, 4, 0, 2]), {}),
    "replication_pad3d.default": lambda: ((_t(1, 2, 3, 4, 3), [2, 0, 1, 1, 0, 3]), {}),
    "reflection_pad1d_backward.default": lambda: ((_t(2, 3, 10), _t(2, 3, 5), [2, 3]), {}),
    "reflection_pad2d_backward.default": lambda: ((_t(2, 3, 7, 8), _t(2, 3, 4, 5), [1, 2, 3, 0]), {}),
    "reflection_pad3d_backward.default": lambda: (
        (_t(1, 2, 5, 6, 5), _t(1, 2, 3, 4, 3), [1, 1, 2, 0, 0, 2]), {}
    ),
    "upsample_linear1d.default": lambda: ((_t(2, 3, 5), [9], False), {}),
    "upsample_bilinear2d.default": lambda: ((_t(1, 2, 4, 5), [7, 3], True), {}),
    "upsample_trilinear3d.default": lambda: ((_t(1, 2, 3, 4, 3), [5, 6, 4], False), {}),
    "upsample_linear1d.vec": lambda: ((_t(2, 3, 5), None, True, [1.7]), {}),
    "upsample_bilinear2d.vec": lambda: ((_t(1, 2, 4, 5), None, False, [1.5, 2.0]), {}),
    "upsample_trilinear3d.vec": lambda: ((_t(1, 1, 3, 4, 3), [4, 5, 6], False, None), {}),
    "upsample_nearest2d_backward.default": lambda: ((_t(2, 3, 7, 5), [7, 5], [2, 3, 3, 4]), {}),
    "im2col.default": lambda: ((_t(2, 3, 6, 7), [2, 3], [1, 2], [1, 0], [2, 1]), {}),
    "col2im.default": lambda: ((_t(2, 12, 8), [5, 6], [2, 2], [1, 1], [0, 1], [2, 2]), {}),
    "max_unpool2d.default": lambda: _max_unpool_sample(2),
    "max_unpool3d.default": lambda: _max_unpool_sample(3),
    # batch 4: unchecked indexing, margin losses, grids, ranges, norms
    "_unsafe_index.Tensor": lambda: ((_t(4, 5), [None, tp.tensor([4, 0, 2])]), {}),
    "_unsafe_index_put.default": lambda: (
        (_t(4, 5), [tp.tensor([3, 0, 3]), tp.tensor([1, 4, 1])], _t(3), True), {}
    ),
    "_unsafe_masked_index.default": lambda: (
        (_t(4, 5), tp.tensor([[True], [False], [True]]), [tp.tensor([3, 7, 0]), None], 1.5), {}
    ),
    "_unsafe_masked_index_put_accumulate.default": lambda: (
        (_t(4, 5), tp.tensor([[True], [False], [True]]), [tp.tensor([3, 1, 3])], _t(3, 5)), {}
    ),
    "multi_margin_loss.default": lambda: (
        (_t(4, 5), tp.tensor([0, 3, 4, 1]), 2, 0.7, _unit(5), 0), {}
    ),
    "multilabel_margin_loss_forward.default": lambda: (
        (_t(3, 4), tp.tensor([[3, 0, -1, 1], [0, 1, 2, 3], [-1, 2, 0, 0]]), 1), {}
    ),
    "affine_grid_generator.default": lambda: ((_t(2, 2, 3), [2, 3, 4, 5], False), {}),
    "rrelu_with_noise_backward.default": lambda: (
        (_t(6), _t(6), _unit(6), 0.1, 0.3, True, False), {}
    ),
    "bernoulli.default": lambda: ((tp.tensor([0.0, 1.0, 1.0, 0.0]),), {}),
    "arange.start": lambda: ((2, 9), {}),
    "arange.end": lambda: ((7,), {}),
    "linalg_vector_norm.default": lambda: ((_t(3, 4), 3), {"dim": [1], "keepdim": True}),
    "matmul.default": lambda: ((_t(3, 4), _t(4, 5)), {}),
    "native_group_norm.default": lambda: ((_t(2, 4, 5, 6), None, None, 2, 4, 30, 2, 1e-5), {}),
    "scaled_dot_product_attention.default": lambda: ((_t(2, 4, 8, 8), _t(2, 4, 8, 8), _t(2, 4, 8, 8)), {}),
    # Views and arrangements.
    "adjoint.default": lambda: ((_t(2, 3, 4),), {}),
    "mH.default": lambda: ((_t(2, 3, 4),), {}),
    "mT.default": lambda: ((_t(2, 3, 4),), {}),
    "swapaxes.default": lambda: ((_t(2, 3, 4), 0, 2), {}),
    "swapdims.default": lambda: ((_t(2, 3, 4), 2, 1), {}),
    "moveaxis.int": lambda: ((_t(2, 3, 4), 0, 2), {}),
    "moveaxis.intlist": lambda: ((_t(2, 3, 4), [0, 1], [2, 0]), {}),
    "movedim.default": lambda: ((_t(2, 3, 4), [2], [0]), {}),
    "movedim.int": lambda: ((_t(2, 3, 4), -1, 0), {}),
    "movedim.intlist": lambda: ((_t(2, 3, 4), [0, 2], [1, 0]), {}),
    "unflatten.int": lambda: ((_t(2, 12), 1, [3, -1]), {}),
    "atleast_1d.default": lambda: ((tp.tensor(1.5),), {}),
    "atleast_2d.default": lambda: ((_t(3),), {}),
    "atleast_3d.default": lambda: ((_t(2, 3),), {}),
    "atleast_1d.Sequence": lambda: (([tp.tensor(1.5), _t(3), _t(2, 3)],), {}),
    "atleast_2d.Sequence": lambda: (([tp.tensor(1.5), _t(3), _t(2, 3)],), {}),
    "atleast_3d.Sequence": lambda: (([tp.tensor(1.5), _t(3), _t(2, 3), _t(2, 3, 4)],), {}),
    "hstack.default": lambda: (([_t(2, 3), _t(2, 1)],), {}),
    "vstack.default": lambda: (([_t(3), _t(2, 3)],), {}),
    "tile.default": lambda: ((_t(2, 3), [2, 1, 2]), {}),
    "tensor_split.indices": lambda: ((_t(7, 2), [2, 5], 0), {}),
    "tensor_split.sections": lambda: ((_t(7, 2), 3, 0), {}),
    "tensor_split.tensor_indices_or_sections": lambda: ((_t(2, 7), tp.tensor([1, 4]), 1), {}),
    "diag.default": lambda: ((_t(4), 1), {}),
    # Gathers, sorts and products.
    "index_select.default": lambda: ((_t(4, 5), 1, tp.tensor([4, 0, 2])), {}),
    "take_along_dim.default": lambda: ((_t(3, 4), tp.tensor([[0, 3], [1, 1], [2, 0]]), 1), {}),
    "argsort.default": lambda: ((_t(3, 6), 1, True), {}),
    "argsort.stable": lambda: ((tp.tensor([[2.0, 1.0, 2.0, 0.0, 1.0]]),), {"stable": True, "dim": 1}),
    "msort.default": lambda: ((_t(5, 3),), {}),
    "kron.default": lambda: ((_t(2, 3), _t(3, 2)), {}),
    "outer.default": lambda: ((_t(3), _t(4)), {}),
    "diff.default": lambda: ((_t(3, 5), 2, -1, _t(3, 1), _t(3, 2)), {}),
    # Elementwise.
    "entr.default": lambda: ((tp.tensor([0.0, 0.5, 2.0, -1.0]),), {}),
    "xlog1py.default": lambda: ((tp.tensor([0.0, 1.0, 2.0]), _unit(3)), {}),
    "isclose.default": lambda: (
        (tp.tensor([1.0, 2.0, float("nan"), 3.0]), tp.tensor([1.0, 2.1, float("nan"), 3.0 + 1e-9])),
        {"equal_nan": True},
    ),
    "round.decimals": lambda: ((tp.tensor([1.2345, -2.5678, 0.1049]),), {"decimals": 2}),
    "log_sigmoid.default": lambda: ((_t(5, low=-30, high=30),), {}),
    # Reductions.
    "logsumexp.default": lambda: ((_t(3, 4), 1, True), {}),
    "nanmean.default": lambda: (
        (tp.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]]), 1), {}
    ),
    "norm.default": lambda: ((_t(3, 4), 3.0), {}),
    "norm.Scalar": lambda: ((_t(3, 4), 1), {}),
    "norm.dim": lambda: ((_t(3, 4), [1], 2.0, True), {}),
    "norm.ScalarOpt_dim": lambda: ((_t(3, 4), 3, [0], False), {}),
    "norm.ScalarOpt_dtype": lambda: ((_t(3, 4), 2), {"dtype": tp.float64}),
    "norm.ScalarOpt_dim_dtype": lambda: ((_t(3, 4), 2, [1], False), {"dtype": tp.float64}),
    "quantile.default": lambda: ((_t(3, 8), tp.tensor([0.25, 0.5, 0.9]), 1, False), {}),
    "nanquantile.default": lambda: (
        (tp.tensor([[1.0, float("nan"), 3.0, 0.5], [4.0, 5.0, 2.0, float("nan")]]),
         tp.tensor([0.3, 0.75]), 1, True),
        {},
    ),
    # Normalizations.
    "instance_norm.default": lambda: (
        (_t(2, 3, 4, 5), _unit(3), _t(3), None, None, True, 0.1, 1e-5), {}
    ),
    "rms_norm.default": lambda: ((_t(3, 4), [4], _unit(4), 1e-5), {}),
    # Losses (reduction 1 is the mean).
    "tp_l1_loss.default": lambda: ((_t(3, 4), _t(3, 4), 1), {}),
    "tp_kl_div.default": lambda: (
        (tp.log_softmax(_t(3, 4), 1), tp.softmax(_t(3, 4), 1), 1, False), {}
    ),
    "tp_poisson_nll_loss.default": lambda: ((_t(3, 4), _unit(3, 4) * 4, True, True, 1e-8, 1), {}),
    "tp_soft_margin_loss.default": lambda: (
        (_t(3, 4), tp.tensor([[1.0, -1.0, 1.0, -1.0]] * 3), 1), {}
    ),
    "tp_hinge_embedding_loss.default": lambda: (
        (_t(3, 4), tp.tensor([[1.0, -1.0, -1.0, 1.0]] * 3), 0.5, 1), {}
    ),
    "tp_margin_ranking_loss.default": lambda: (
        (_t(5), _t(5), tp.tensor([1.0, -1.0, 1.0, 1.0, -1.0]), 0.2, 1), {}
    ),
    # Activations, losses, and their backward plumbing.
    "sigmoid.default": lambda: ((_t(2, 3),), {}),
    "tanh.default": lambda: ((_t(2, 3),), {}),
    "atanh.default": lambda: ((tp.tanh(_t(2, 3)),), {}),
    "relu.default": lambda: ((_t(2, 3),), {}),
    "relu6.default": lambda: ((_t(2, 3, low=-8.0, high=8.0),), {}),
    "selu.default": lambda: ((_t(2, 3),), {}),
    "prelu.default": lambda: ((_t(2, 3, 4, 5), _unit(3)), {}),
    "margin_ranking_loss.default": lambda: (
        (_t(4), _t(4), tp.tensor([1.0, -1.0, 1.0, -1.0])), {"margin": 0.5},
    ),
    "hinge_embedding_loss.default": lambda: (
        (_t(4), tp.tensor([1.0, -1.0, 0.0, 1.0])), {"margin": 1.0},
    ),
    "nll_loss.default": lambda: ((_t(4, 8), tp.tensor([1, 0, 3, 2])), {}),
    "_softmax.default": lambda: ((_t(2, 3, 4), 1, False), {}),
    "_log_softmax.default": lambda: ((_t(2, 3, 4), 1, False), {}),
    "embedding.default": lambda: (
        (tp.randn(10, 4), tp.tensor([[0, 1, 2], [7, 8, 9]])), {},
    ),
    "batch_norm_backward.default": lambda: (
        (tp.randn(3, 4, 5, 5), tp.randn(3, 4, 5, 5), _unit(4), _t(4), _unit(4)),
        {"training": True, "eps": 1e-5},
    ),
    "max_pool2d_with_indices_backward.default": _pool2d_backward_sample,
    "rnn_tanh.input": lambda: (
        (_t(5, 3, 4), _t(1, 3, 2), [_t(2, 4), _t(2, 2), _unit(2), _unit(2)],
         True, 1, 0.0, False, False, False),
        {},
    ),
    "rnn_relu.input": lambda: (
        (_t(5, 3, 4), _t(1, 3, 2), [_t(2, 4), _t(2, 2), _unit(2), _unit(2)],
         True, 1, 0.0, False, False, False),
        {},
    ),
    # The gated RNNs fuse their gates into the input linear, so w_ih is
    # (3H or 4H, features) and hx carries one hidden per state per direction.
    "lstm.input": lambda: (
        (_t(5, 3, 4), [_t(1, 3, 2), _t(1, 3, 2)],
         [_t(8, 4), _t(8, 2), _unit(8), _unit(8)], True, 1, 0.0, False, False, False),
        {},
    ),
    "gru.input": lambda: (
        (_t(5, 3, 4), _t(1, 3, 2), [_t(6, 4), _t(6, 2), _unit(6), _unit(6)],
         True, 1, 0.0, False, False, False),
        {},
    ),
    # Special functions; the ranges keep inputs away from poles and the
    # boundaries of erfinv/ndtri, and zeta's first argument stays above one.
    "bessel_j0.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "bessel_j1.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "spherical_bessel_j0.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "digamma.default": lambda: ((_t(2, 3, low=0.5, high=3.0),), {}),
    "erf.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "erfc.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "erfinv.default": lambda: ((_t(2, 3, low=-0.9, high=0.9),), {}),
    "i0.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "i0e.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "i1.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "i1e.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "lgamma.default": lambda: ((_t(2, 3, low=0.5, high=3.0),), {}),
    "igamma.default": lambda: ((_t(2, 3, low=0.5, high=3.0), _t(2, 3, low=0.5, high=3.0)), {}),
    "igammac.default": lambda: ((_t(2, 3, low=0.5, high=3.0), _t(2, 3, low=0.5, high=3.0)), {}),
    "zeta.default": lambda: ((_t(2, 3, low=1.5, high=4.0), _t(2, 3, low=0.5, high=2.0)), {}),
    "special_bessel_j0.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "special_bessel_j1.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "special_spherical_bessel_j0.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "special_erfcx.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "special_i0e.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "special_i1.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "special_i1e.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "special_ndtr.default": lambda: ((_t(2, 3, low=-2.0, high=2.0),), {}),
    "special_ndtri.default": lambda: ((_t(2, 3, low=0.05, high=0.95),), {}),
    "special_zeta.default": lambda: ((_t(2, 3, low=1.5, high=4.0), _t(2, 3, low=0.5, high=2.0)), {}),
    # Transforms; the inverse forms round-trip through the prims and lose a
    # little precision, so their comparison relies on the allclose branch.
    "fft_fft.default": lambda: ((_t(8),), {}),
    "fft_ifft.default": lambda: ((tp.complex(_t(8), _t(8)),), {}),
    "fft_rfft.default": lambda: ((_t(8),), {}),
    "fft_irfft.default": lambda: ((tp.complex(_t(5), _t(5)),), {}),
    "fft_fft2.default": lambda: ((_t(4, 6),), {}),
    "fft_ifft2.default": lambda: ((tp.complex(_t(4, 6), _t(4, 6)),), {}),
    "fft_rfft2.default": lambda: ((_t(4, 6),), {}),
    "fft_irfft2.default": lambda: ((tp.complex(_t(4, 4), _t(4, 4)),), {}),
    # Resampling; the explicit-scale calls keep output sizes consistent with
    # floor(input * scale), which the gather indices assume.
    "upsample_nearest1d.default": lambda: ((_t(1, 2, 4), [8]), {}),
    "upsample_nearest2d.default": lambda: ((_t(1, 2, 3, 4), [7, 8]), {}),
    "upsample_nearest3d.default": lambda: ((_t(1, 2, 2, 3, 4), [5, 6, 7]), {}),
    "_upsample_nearest_exact1d.default": lambda: ((_t(1, 2, 4), [8], 2.0), {}),
    "_upsample_nearest_exact2d.default": lambda: ((_t(1, 2, 3, 4), [4, 8], 1.5, 2.0), {}),
    "_upsample_nearest_exact3d.default": lambda: ((_t(1, 2, 2, 3, 4), [5, 6, 7]), {}),
    "upsample_nearest1d.vec": lambda: ((_t(1, 2, 4), None, (2.0,)), {}),
    "upsample_nearest2d.vec": lambda: ((_t(1, 2, 3, 4), None, (1.5, 2.0)), {}),
    "upsample_nearest3d.vec": lambda: ((_t(1, 2, 2, 3, 4), [5, 6, 7], None), {}),
    "_upsample_nearest_exact1d.vec": lambda: ((_t(1, 2, 4), None, (2.0,)), {}),
    "_upsample_nearest_exact2d.vec": lambda: ((_t(1, 2, 3, 4), [7, 8], None), {}),
    "_upsample_nearest_exact3d.vec": lambda: ((_t(1, 2, 2, 3, 4), None, (1.0, 2.0, 1.5)), {}),
    "upsample_bicubic2d.default": lambda: ((_t(1, 2, 4, 5), [7, 8], False), {}),
    "upsample_bicubic2d.vec": lambda: ((_t(1, 2, 4, 5), None, False, (1.5, 2.0)), {}),
    "_upsample_bilinear2d_aa.vec": lambda: ((_t(1, 2, 4, 5), [7, 8], False, None), {}),
    "_upsample_bicubic2d_aa.vec": lambda: ((_t(1, 2, 4, 5), None, False, (1.5, 2.0)), {}),
    # Predicates, scalar reads, factories and BLAS glue.  The entries that
    # answer python values go through the scalar branch of the matcher.
    "is_same_size.default": lambda: ((tp.randn(2, 3), tp.randn(2, 3)), {}),
    "is_complex.default": lambda: ((tp.randn(2, 3).to(tp.complex64),), {}),
    "isreal.default": lambda: ((tp.randn(2, 3).to(tp.complex64),), {}),
    "item.default": lambda: ((tp.tensor([3.5]),), {}),
    "where.self": lambda: ((tp.tensor([[True, False], [True, True]]),
                            tp.randn(2, 2), tp.randn(2, 2)), {}),
    "where.ScalarSelf": lambda: ((tp.tensor([[True, False], [True, True]]), 1.5,
                                  tp.randn(2, 2)), {}),
    "where.ScalarOther": lambda: ((tp.tensor([[True, False], [True, True]]),
                                   tp.randn(2, 2), 2.5), {}),
    "where.Scalar": lambda: ((tp.tensor([[True, False], [True, True]]), 1.5, 2.5), {}),
    "where.default": lambda: ((tp.tensor([[True, False], [True, False]]),), {}),
    "complex.default": lambda: ((tp.randn(2, 3), tp.randn(2, 3)), {}),
    "polar.default": lambda: ((tp.rand(2, 3), tp.rand(2, 3) * 3.0), {}),
    "conj_physical.default": lambda: ((tp.randn(2, 3).to(tp.complex64),), {}),
    "full.default": lambda: (([2, 3], 2.5), {}),
    "bucketize.Tensor": lambda: ((tp.tensor([[0.05, 0.3, 0.7, 0.95, 1.2]]),
                                  tp.tensor([0.1, 0.5, 0.9])), {}),
    "bucketize.Scalar": lambda: ((0.35, tp.tensor([0.1, 0.5, 0.9])), {}),
    "addmm.default": lambda: ((tp.randn(2, 3), tp.randn(2, 4), tp.randn(4, 3)),
                              {"beta": 2.0, "alpha": 3.0}),
    "addmm.dtype": lambda: ((tp.randn(2, 3), tp.randn(2, 4), tp.randn(4, 3), tp.float64),
                            {"beta": 2, "alpha": 3}),
    "_addmm_activation.default": lambda: ((tp.randn(2, 3), tp.randn(2, 4), tp.randn(4, 3)),
                                          {"use_gelu": True}),
    "addmv.default": lambda: ((tp.randn(3), tp.randn(3, 4), tp.randn(4)),
                              {"beta": 2.0, "alpha": 0.5}),
    "dist.default": lambda: ((tp.randn(3, 5), tp.randn(3, 5)), {"p": 2.0}),
    "_euclidean_dist.default": lambda: ((tp.randn(3, 5), tp.randn(3, 5)), {}),
    "_to_copy.default": lambda: ((tp.randn(2, 3),), {"dtype": tp.float64}),
    "_adaptive_avg_pool2d.default": lambda: ((tp.randn(2, 3, 6, 5), [2, 2]), {}),
    # Distances and grid sampling; the grid stays inside roughly one pixel so
    # the gathered reads exercise in-bounds and zero-weight corners alike.
    "pairwise_distance.default": lambda: ((tp.randn(4, 3), tp.randn(4, 3)), {"p": 2.0}),
    "pdist.default": lambda: ((tp.randn(6, 3),), {"p": 2}),
    "grid_sampler_2d.default": lambda: ((tp.randn(2, 3, 4, 5),
                                         tp.rand(2, 6, 7, 2) * 1.6 - 0.8,
                                         0, 0, False), {}),
    # Reductions.  Corrections stay integral in the samples: the eager
    # var/std kernels truncate a fractional correction to an integer while
    # the walks apply it as a true divisor.
    "sum.default": lambda: ((tp.randn(2, 3, 4),), {}),
    "sum.dim_IntList": lambda: ((tp.randn(2, 3, 4), [0, 2], True), {}),
    "mean.default": lambda: ((tp.randn(2, 3, 4),), {}),
    "mean.dim": lambda: ((tp.randn(2, 3, 4), [1], False), {}),
    "prod.default": lambda: ((tp.rand(2, 3) + 0.5,), {}),
    "prod.dim_int": lambda: ((tp.rand(2, 3, 4) + 0.5, 1), {}),
    "prod.dim_IntList": lambda: ((tp.rand(2, 3, 4) + 0.5, [0, 1], True), {}),
    "var.default": lambda: ((_lat(2, 3, 4),), {"correction": 1}),
    "var.dim": lambda: ((_lat(2, 3, 4), [0, 2], 0, True), {}),
    "var.correction": lambda: ((_lat(2, 3, 4), [1]), {"correction": 2, "keepdim": True}),
    "var_mean.default": lambda: ((_lat(2, 3, 4), [1, 2], False), {}),
    "var_mean.dim": lambda: ((_lat(2, 3, 4), [0], True, False), {}),
    "var_mean.correction": lambda: ((_lat(2, 3, 4),), {"correction": 0}),
    "amax.default": lambda: ((tp.randn(2, 3, 4), [1], True), {}),
    "amin.default": lambda: ((tp.randn(2, 3, 4), [0, 1]), {}),
    "any.default": lambda: ((tp.tensor([[True, False], [False, False]]),), {}),
    "any.dim": lambda: ((tp.tensor([[True, False], [False, False]]), 0, True), {}),
    "any.dims": lambda: ((tp.rand(2, 3, 4) > 0.5, [0, 1], False), {}),
    "cumsum.default": lambda: ((tp.randn(2, 3, 4), 1), {}),
    "cumprod.default": lambda: ((tp.randn(2, 3, 4) * 0.5, 1), {}),
    # The CPU flash walk routes through the math attention and rebuilds the
    # log-sum-exp over the masked scores; small magnitudes keep the two
    # log-summation orders inside the tolerance.
    "_scaled_dot_product_flash_attention_for_cpu.default": lambda: (
        (tp.rand(1, 2, 4, 8) * 0.5, tp.rand(1, 2, 4, 8) * 0.5,
         tp.rand(1, 2, 4, 8) * 0.5, 0.0, False), {}),
    # Pooling; window maxima and indices are picked exactly, so the comparison
    # is bit-stable even on random inputs.
    "max_pool2d_with_indices.default": lambda: (
        (_t(2, 3, 6, 6), [3, 3], [2, 2], [1, 1], [1, 1], False), {}
    ),
    "max_pool3d_with_indices.default": lambda: (
        (_t(1, 2, 4, 6, 8), [3, 3, 3], [2, 2, 2], [1, 1, 1], [1, 1, 1], False), {}
    ),
    "adaptive_max_pool2d.default": lambda: ((_t(2, 3, 6, 6), (2, 2)), {}),
    "adaptive_max_pool3d.default": lambda: ((_t(1, 2, 4, 6, 8), (2, 2, 2)), {}),
    # Normalization; the statistics walk (var/mean, rsqrt) rounds differently
    # between the fused kernels and the composite walk at the atol boundary on
    # unit-scale inputs, so these samples run on a small deterministic lattice
    # where both sides agree to the last bit.
    "native_batch_norm.default": lambda: (
        (_lattice_bn_input(), tp.ones(3), tp.zeros(3), tp.zeros(3), tp.ones(3),
         True, 0.1, 1e-5), {}
    ),
    "_native_batch_norm_legit_no_training.default": lambda: (
        (_lattice_bn_input(), tp.ones(3), tp.zeros(3),
         tp.tensor([0.05, 0.10, 0.15]), tp.tensor([0.02, 0.04, 0.08]), 0.1, 1e-5), {}
    ),
    "_batch_norm_with_update.default": lambda: (
        (_lattice_bn_input(), tp.ones(3), tp.zeros(3), tp.zeros(3), tp.ones(3),
         0.1, 1e-5), {}
    ),
    "_batch_norm_no_update.default": lambda: (
        (_lattice_bn_input(), tp.ones(3), tp.zeros(3),
         tp.tensor([0.05, 0.10, 0.15]), tp.tensor([0.02, 0.04, 0.08]), 0.1, 1e-5), {}
    ),
    "native_layer_norm.default": lambda: (
        (tp.tensor([[-0.02, -0.01, 0.0, 0.01, 0.02], [0.02, 0.01, 0.0, -0.01, -0.02]]),
         [5], tp.ones(5), tp.zeros(5), 1e-5), {}
    ),
}

# Overloads whose kernels exist only on specific devices; exercised by the
# device tests instead of the CPU comparison above.
DEVICE_SPECIFIC = {
    "cudnn_batch_norm.default",
    "cudnn_batch_norm_backward.default",
    "miopen_batch_norm_backward.default",
}


def _max_unpool_sample(dims):
    x = _t(2, 3, *([4] * dims))
    if dims == 2:
        pooled, indices = tp.nn.functional.max_pool2d(x, 2, return_indices=True)
        return (pooled, indices, [4, 4]), {}
    pooled, indices = tp.nn.functional.max_pool3d(x, 2, return_indices=True)
    return (pooled, indices, [4, 4, 4], [2, 2, 2], [0, 0, 0]), {}


def _bn_backward_sample(train):
    x, w = _t(4, 3, 5), _unit(3)
    running_mean, running_var = tp.zeros(3), tp.ones(3)
    _, save_mean, save_invstd = ops.native_batch_norm.default(
        x, w, None, running_mean, running_var, train, 0.1, 1e-5
    )
    return ((_t(4, 3, 5), x, w, running_mean, running_var, save_mean, save_invstd, train, 1e-5,
             [True, True, True]), {})


def _ln_backward_sample():
    x, w, b = _t(3, 4, 5), _unit(5), _t(5)
    _, mean, rstd = ops.native_layer_norm.default(x, [5], w, b, 1e-5)
    return ((_t(3, 4, 5), x, [5], mean, rstd, w, b, [True, True, True]), {})


def _gn_backward_sample():
    x, w, b = _t(2, 6, 4), _unit(6), _t(6)
    _, mean, rstd = ops.native_group_norm.default(x, w, b, 2, 6, 4, 3, 1e-5)
    return ((_t(2, 6, 4), x, mean, rstd, w, 2, 6, 4, 3, [True, True, True]), {})


def _nll_backward_sample():
    scores, target, weight = tp.log_softmax(_t(4, 5), 1), tp.tensor([0, 2, -100, 4]), _unit(5)
    _, total_weight = ops.nll_loss_forward.default(scores, target, weight, 1, -100)
    return ((tp.tensor(1.0), scores, target, weight, 1, -100, total_weight), {})


def _nll2d_backward_sample():
    scores = tp.log_softmax(_t(2, 3, 2, 2), 1)
    target = tp.tensor([[[0, 1], [2, 0]], [[1, 1], [0, 2]]])
    _, total_weight = ops.nll_loss2d_forward.default(scores, target, None, 1, -100)
    return ((tp.tensor(1.0), scores, target, None, 1, -100, total_weight), {})


# Inputs for the in-place overloads that reuse the functional operator and
# write the answer into the value they were called on; each call builds
# fresh values so no case sees what a previous one wrote.
def _int_vals():
    return tp.tensor([5, 6, 9, 12])


def _shift_vals():
    return tp.tensor([1, 2, 0, 3])


def _bool_vals():
    return tp.tensor([True, False, True, True])


def _float_vals():
    return _t(4, low=-2.0, high=2.0)


def _positive_vals():
    return _t(4, low=0.5, high=3.0)


def _bounded_vals():
    return _t(4, low=-0.9, high=0.9)


WRITING_FORMS = {
    "bitwise_and_.Tensor": lambda: (_int_vals(), _int_vals()),
    "bitwise_and_.Scalar": lambda: (_int_vals(), 7),
    "bitwise_or_.Tensor": lambda: (_int_vals(), _int_vals()),
    "bitwise_or_.Scalar": lambda: (_int_vals(), 7),
    "bitwise_xor_.Tensor": lambda: (_int_vals(), _int_vals()),
    "bitwise_xor_.Scalar": lambda: (_int_vals(), 7),
    "bitwise_not_.default": lambda: (_int_vals(),),
    "bitwise_left_shift_.Tensor": lambda: (_int_vals(), _shift_vals()),
    "bitwise_left_shift_.Tensor_Scalar": lambda: (_int_vals(), 2),
    "bitwise_right_shift_.Tensor": lambda: (_int_vals(), _shift_vals()),
    "bitwise_right_shift_.Tensor_Scalar": lambda: (_int_vals(), 2),
    "logical_and_.default": lambda: (_bool_vals(), _bool_vals()),
    "logical_not_.default": lambda: (_bool_vals(),),
    "logical_or_.default": lambda: (_bool_vals(), _bool_vals()),
    "logical_xor_.default": lambda: (_bool_vals(), _bool_vals()),
    "relu_.default": lambda: (_t(4, low=-1.0, high=1.0),),
    "sigmoid_.default": lambda: (_t(4, low=-3.0, high=3.0),),
    "__iand__.Tensor": lambda: (_int_vals(), _int_vals()),
    "__iand__.Scalar": lambda: (_int_vals(), 7),
    "__ior__.Tensor": lambda: (_int_vals(), _int_vals()),
    "__ior__.Scalar": lambda: (_int_vals(), 7),
    "__ixor__.Tensor": lambda: (_int_vals(), _int_vals()),
    "__ixor__.Scalar": lambda: (_int_vals(), 7),
    "__ilshift__.Tensor": lambda: (_int_vals(), _shift_vals()),
    "__ilshift__.Scalar": lambda: (_int_vals(), 2),
    "__irshift__.Tensor": lambda: (_int_vals(), _shift_vals()),
    "__irshift__.Scalar": lambda: (_int_vals(), 2),
    # Arithmetic and rounding writing forms.  A comparison writes into a
    # boolean value: the answer is already boolean, so that is the only
    # target an eager write accepts.
    "add_.Tensor": lambda: ((_float_vals(), _float_vals()), {"alpha": 2}),
    "add_.Scalar": lambda: ((_float_vals(), 3.0), {"alpha": 2}),
    "sub_.Tensor": lambda: ((_float_vals(), _float_vals()), {"alpha": 2}),
    "sub_.Scalar": lambda: ((_float_vals(), 3.0), {"alpha": 2}),
    "mul_.Tensor": lambda: (_float_vals(), _float_vals()),
    "mul_.Scalar": lambda: (_float_vals(), 3.0),
    "div_.Tensor": lambda: (_float_vals(), _t(4, low=0.5, high=2.0)),
    "div_.Scalar": lambda: (_float_vals(), 2.0),
    "div_.Tensor_mode": lambda: (
        (_float_vals(), _t(4, low=0.5, high=2.0)), {"rounding_mode": "floor"},
    ),
    "div_.Scalar_mode": lambda: ((_float_vals(), 2.0), {"rounding_mode": "floor"}),
    "true_divide_.Tensor": lambda: (_float_vals(), _t(4, low=0.5, high=2.0)),
    "true_divide_.Scalar": lambda: (_float_vals(), 2.0),
    "remainder_.Tensor": lambda: (_float_vals(), _t(4, low=0.5, high=2.0)),
    "remainder_.Scalar": lambda: (_float_vals(), 2.0),
    "fmod_.Tensor": lambda: (_float_vals(), _t(4, low=1.0, high=2.0)),
    "fmod_.Scalar": lambda: (_float_vals(), 2.0),
    "pow_.Tensor": lambda: (_positive_vals(), _t(4, low=0.5, high=2.0)),
    "pow_.Scalar": lambda: (_float_vals(), 3),
    "float_power_.Tensor": lambda: (_positive_vals(), _t(4, low=0.5, high=2.0)),
    "float_power_.Scalar": lambda: (_float_vals(), 3),
    "copysign_.Tensor": lambda: (_float_vals(), _t(4, low=-1.0, high=1.0)),
    "copysign_.Scalar": lambda: (_float_vals(), -1.5),
    "atan2_.default": lambda: (_float_vals(), _float_vals()),
    "hypot_.default": lambda: (_float_vals(), _float_vals()),
    "ldexp_.default": lambda: (_int_vals(), _shift_vals()),
    "nextafter_.default": lambda: (_float_vals(), _float_vals()),
    "gcd_.default": lambda: (_int_vals(), _int_vals()),
    "lcm_.default": lambda: (_int_vals(), _int_vals()),
    "igamma_.default": lambda: (_positive_vals(), _positive_vals()),
    "igammac_.default": lambda: (_positive_vals(), _positive_vals()),
    "eq_.Tensor": lambda: (_bool_vals(), _bool_vals()),
    "eq_.Scalar": lambda: (_bool_vals(), True),
    "ne_.Tensor": lambda: (_bool_vals(), _bool_vals()),
    "ne_.Scalar": lambda: (_bool_vals(), True),
    "lt_.Tensor": lambda: (_bool_vals(), _bool_vals()),
    "lt_.Scalar": lambda: (_bool_vals(), False),
    "le_.Tensor": lambda: (_bool_vals(), _bool_vals()),
    "le_.Scalar": lambda: (_bool_vals(), False),
    "gt_.Tensor": lambda: (_bool_vals(), _bool_vals()),
    "gt_.Scalar": lambda: (_bool_vals(), False),
    "ge_.Tensor": lambda: (_bool_vals(), _bool_vals()),
    "ge_.Scalar": lambda: (_bool_vals(), False),
    "clamp_.default": lambda: (_float_vals(), 0.2, 0.8),
    "clamp_.Tensor": lambda: (_float_vals(), tp.tensor(0.2), tp.tensor(0.8)),
    # Unary mathematical writing forms; each value builder stays inside the
    # domain the operator is defined on.
    "abs_.default": lambda: (_float_vals(),),
    "neg_.default": lambda: (_float_vals(),),
    "reciprocal_.default": lambda: (_positive_vals(),),
    "sqrt_.default": lambda: (_positive_vals(),),
    "rsqrt_.default": lambda: (_positive_vals(),),
    "square_.default": lambda: (_float_vals(),),
    "sign_.default": lambda: (_float_vals(),),
    "exp_.default": lambda: (_float_vals(),),
    "exp2_.default": lambda: (_float_vals(),),
    "expm1_.default": lambda: (_float_vals(),),
    "log_.default": lambda: (_positive_vals(),),
    "log2_.default": lambda: (_positive_vals(),),
    "log10_.default": lambda: (_positive_vals(),),
    "log1p_.default": lambda: (_positive_vals(),),
    "floor_.default": lambda: (_float_vals(),),
    "ceil_.default": lambda: (_float_vals(),),
    "trunc_.default": lambda: (_float_vals(),),
    "round_.default": lambda: (_float_vals(),),
    "round_.decimals": lambda: ((_float_vals(),), {"decimals": 2}),
    "sin_.default": lambda: (_float_vals(),),
    "cos_.default": lambda: (_float_vals(),),
    "tan_.default": lambda: (_t(4, low=-1.2, high=1.2),),
    "asin_.default": lambda: (_bounded_vals(),),
    "acos_.default": lambda: (_bounded_vals(),),
    "atan_.default": lambda: (_float_vals(),),
    "sinh_.default": lambda: (_t(4, low=-1.5, high=1.5),),
    "cosh_.default": lambda: (_t(4, low=-1.5, high=1.5),),
    "tanh_.default": lambda: (_float_vals(),),
    "asinh_.default": lambda: (_float_vals(),),
    "acosh_.default": lambda: (_t(4, low=1.5, high=4.0),),
    "atanh_.default": lambda: (_bounded_vals(),),
    "erf_.default": lambda: (_float_vals(),),
    "erfc_.default": lambda: (_float_vals(),),
    "erfinv_.default": lambda: (_bounded_vals(),),
    "digamma_.default": lambda: (_positive_vals(),),
    "lgamma_.default": lambda: (_positive_vals(),),
    "i0_.default": lambda: (_float_vals(),),
    "conj_physical_.default": lambda: (_float_vals(),),
    # Running sums, small matrix chains, gated activation.
    "cumsum_.default": lambda: (_float_vals(), 0),
    "cumprod_.default": lambda: (_positive_vals(), 0),
    "addbmm_.default": lambda: (
        (_t(3, 5), _t(2, 3, 4), _t(2, 4, 5)), {"beta": 0.5, "alpha": 2},
    ),
    "addmm_.default": lambda: ((_t(3, 5), _t(3, 4), _t(4, 5)), {"beta": 0.5, "alpha": 2}),
    "addmv_.default": lambda: ((_t(5), _t(5, 4), _t(4)), {"beta": 0.5, "alpha": 2}),
    "selu_.default": lambda: (_float_vals(),),
    # Reads and writes at positions.
    "scatter_.src": lambda: (_t(3, 4), 0, tp.tensor([[0, 1, 2, 0], [2, 1, 0, 1]]), _t(2, 4)),
    "scatter_.value": lambda: (_t(3, 4), 0, tp.tensor([[0, 1, 2, 0], [2, 1, 0, 1]]), 7.0),
    "scatter_.reduce": lambda: (
        (_t(3, 4), 0, tp.tensor([[0, 1, 2, 0], [2, 1, 0, 1]]), _t(2, 4)), {"reduce": "add"},
    ),
    "scatter_.value_reduce": lambda: (
        (_t(3, 4), 0, tp.tensor([[0, 1, 2, 0], [2, 1, 0, 1]]), 7.0),
        {"reduce": "multiply"},
    ),
    "scatter_add_.default": lambda: (
        _t(3, 4), 0, tp.tensor([[0, 1, 2, 0], [2, 1, 0, 1]]), _t(2, 4),
    ),
    "scatter_reduce_.two": lambda: (
        (_t(3, 4), 0, tp.tensor([[0, 1, 2, 0], [2, 1, 0, 1]]), _t(2, 4)), {"reduce": "amax"},
    ),
    "index_put_.default": lambda: (_t(3, 4), [tp.tensor([0, 2])], _t(2, 4), False),
    "index_reduce_.default": lambda: (
        (_t(3, 4), 0, tp.tensor([0, 2]), _t(2, 4)), {"reduce": "mean"},
    ),
}


def _registered_functional():
    return {
        overload: fn
        for overload, fn in decomposition_table.items()
        if not overload._schema.is_mutable
    }


def test_batch_norm_legit_decompositions_match_the_kernels_they_replace():
    # The legit overloads share the batch norm statistics walk; no CPU kernel
    # serves them, so they are checked against the native_batch_norm kernel
    # (which accepts absent running statistics in training mode) and against
    # the composite contract for the eval-time refusal.
    get_decompositions([])
    x = _lattice_bn_input()
    w, b = tp.ones(3), tp.zeros(3)
    rm, rv = tp.zeros(3), tp.ones(3)

    legit = decomposition_table[ops._native_batch_norm_legit.default](
        x, w, b, rm.clone(), rv.clone(), True, 0.1, 1e-5
    )
    native = ops.native_batch_norm.default(x, w, b, rm.clone(), rv.clone(), True, 0.1, 1e-5)
    for l, n in zip(legit, native):
        assert tp.allclose(l, n, rtol=1e-5, atol=1e-6)

    no_stats = decomposition_table[ops._native_batch_norm_legit.no_stats](
        x, w, b, True, 0.1, 1e-5
    )
    native_no_stats = ops.native_batch_norm.default(x, w, b, None, None, True, 0.1, 1e-5)
    for l, n in zip(no_stats, native_no_stats):
        assert tp.allclose(l, n, rtol=1e-5, atol=1e-6)

    with pytest.raises(RuntimeError):
        decomposition_table[ops._native_batch_norm_legit.no_stats](x, w, b, False, 0.1, 1e-5)


def test_chunk_cat_decomposition():
    # No native kernel serves this overload on every backend; the expected
    # value is written out: each input is padded to a multiple of the chunk
    # count along dim, split into chunks, and the chunks are concatenated.
    get_decompositions([])
    a = tp.arange(5.0).reshape(5, 1)
    b = tp.arange(4.0).reshape(4, 1) + 10
    got = decomposition_table[ops._chunk_cat.default]([a, b], 0, 2)
    expected = tp.tensor([[0.0, 1.0, 2.0, 10.0, 11.0], [3.0, 4.0, 0.0, 12.0, 13.0]])
    assert tp.equal(got, expected)


def test_pad_sequence_decomposition_pads_to_the_longest_sequence():
    # No native kernel serves this overload; the expected values are written
    # out: sequences line up along their own axis, padded on one side.
    get_decompositions([])
    fn = decomposition_table[ops.pad_sequence.default]
    seqs = [tp.tensor([1.0, 2.0, 3.0]), tp.tensor([4.0]), tp.tensor([5.0, 6.0])]
    assert tp.equal(fn(seqs, False, 0.0, "right"), tp.tensor([[1.0, 4.0, 5.0], [2.0, 0.0, 6.0], [3.0, 0.0, 0.0]]))
    assert tp.equal(fn(seqs, True, -1.0, "left"), tp.tensor([[1.0, 2.0, 3.0], [-1.0, -1.0, 4.0], [-1.0, 5.0, 6.0]]))
    with_trailing = [tp.tensor([[1.0, 10.0], [2.0, 20.0]]), tp.tensor([[3.0, 30.0]])]
    assert tp.equal(
        fn(with_trailing, False, 9.0, "right"),
        tp.tensor([[[1.0, 10.0], [3.0, 30.0]], [[2.0, 20.0], [9.0, 9.0]]]),
    )


@pytest.mark.parametrize("kwargs", [{}, {"dtype": tp.float64}])
def test_empty_strided_metadata(kwargs):
    get_decompositions([])
    expected = ops.empty_strided.default([4, 1], [1, 4], **kwargs)
    got = decomposition_table[ops.empty_strided.default]([4, 1], [1, 4], **kwargs)
    assert got.dtype == expected.dtype
    assert tuple(got.shape) == tuple(expected.shape)
    assert tuple(got.stride()) == tuple(expected.stride())


def test_nll_loss_decomposition_reductions_and_ignore_index():
    get_decompositions([])
    fn = decomposition_table[ops.nll_loss.default]
    scores = _t(4, 6)
    target = tp.tensor([0, 3, 5, 2])
    weight = _unit(6)
    for reduction in (0, 1, 2):
        got = fn(scores, target, None, reduction, -100)
        expected = ops.nll_loss.default(scores, target, None, reduction, -100)
        assert tp.allclose(got[0], expected[0], rtol=1e-5, atol=1e-6)
    got = fn(scores, target, weight, 2, -100)
    expected = ops.nll_loss.default(scores, target, weight, 2, -100)
    assert tp.allclose(got[0], expected[0], rtol=1e-5, atol=1e-6)
    ignored = tp.tensor([0, -100, 5, 2])
    got = fn(scores, ignored, None, 0, -100)
    expected = ops.nll_loss.default(scores, ignored, None, 0, -100)
    assert tp.allclose(got[0], expected[0], rtol=1e-5, atol=1e-6)


def test_alpha_dropout_decomposition_edges():
    get_decompositions([])
    fn = decomposition_table[ops.alpha_dropout.default]
    x = _t(4, 5)
    assert tp.equal(fn(x, 0.0, True), x)
    assert tp.equal(fn(x, 0.5, False), x)
    assert tp.equal(fn(x, 1.0, True), x * 0)
    with pytest.raises(RuntimeError):
        fn(x, 1.5, True)
    with pytest.raises(RuntimeError):
        ops.alpha_dropout.default(x, 1.5, True)
    # Every element lands on one of the two support points of the
    # inverted-dropout rescaling.
    alpha = 1.7580993408473766
    p = 0.5
    a = 1.0 / math.sqrt((alpha * alpha * p + 1) * (1 - p))
    saturation = alpha * a
    out = fn(x, p, True)
    kept = (out - x * a - saturation * p).abs() < 1e-5
    dropped = (out - saturation * (p - 1)).abs() < 1e-5
    assert tp.all(kept | dropped)


def test_fused_dropout_decomposition_membership():
    get_decompositions([])
    fn = decomposition_table[ops._fused_dropout.default]
    x = _t(3, 7)
    res, mask = fn(x, 0.4, None)
    assert mask.dtype == tp.uint8 and res.dtype == x.dtype
    active = mask != 0
    kept = (res - x * 2.5).abs() < 1e-5
    assert tp.all(tp.where(active, kept, res == 0))
    with pytest.raises(AssertionError):
        fn(x, 0.4, "generator")


def test_batch_norm_backward_decomposition_training_and_eval():
    get_decompositions([])
    fn = decomposition_table[ops.batch_norm_backward.default]
    grad = _t(2, 4, 3, 3)
    x = _t(2, 4, 3, 3)
    weight = _unit(4)
    running_mean = _t(4)
    running_var = _unit(4)
    for training in (True, False):
        got = fn(grad, x, weight, running_mean, running_var, training, 1e-5)
        expected = ops.batch_norm_backward.default(
            grad, x, weight, running_mean, running_var, training, 1e-5
        )
        for g, e in zip(got, expected):
            assert g.dtype == e.dtype
            assert tp.allclose(g, e, rtol=1e-4, atol=1e-5)


def test_softmax_decomposition_reduces_an_empty_dimension():
    get_decompositions([])
    x = tp.randn(2, 0, 3)
    for op in (ops._softmax.default, ops._log_softmax.default):
        got = decomposition_table[op](x, 1, False)
        expected = op(x, 1, False)
        assert tuple(got.shape) == tuple(expected.shape) == (2, 0, 3)
    half = tp.randn(2, 3).to(tp.float16)
    assert decomposition_table[ops._softmax.default](half, 1, False).dtype == tp.float16


def test_rnn_decomposition_matches_the_eager_loop():
    get_decompositions([])
    # Bidirectional, batch-first, biased: the input is (batch, time, features).
    hx = _t(2, 3, 2)
    params = [_t(2, 4), _t(2, 2), _unit(2), _unit(2),
              _t(2, 4), _t(2, 2), _unit(2), _unit(2)]
    for op in (ops.rnn_tanh.input, ops.rnn_relu.input):
        fn = decomposition_table[op]
        xb = _t(3, 5, 4)
        got = fn(xb, hx, params, True, 1, 0.0, False, True, True)
        expected = op(xb, hx, params, True, 1, 0.0, False, True, True)
        for g, e in zip(got, expected):
            assert tp.allclose(g, e, rtol=1e-4, atol=1e-5)
    # Two layers without biases: the input is (time, batch, features) and
    # hx stacks one hidden per layer per direction.  The second layer sees
    # the two directions' outputs concatenated, so its input weights are
    # twice as wide.
    x = _t(5, 3, 4)
    hx = _t(4, 3, 2)
    params = [_t(2, 4), _t(2, 2), _t(2, 4), _t(2, 2),
              _t(2, 4), _t(2, 2), _t(2, 4), _t(2, 2)]
    for op in (ops.rnn_tanh.input, ops.rnn_relu.input):
        fn = decomposition_table[op]
        got = fn(x, hx, params, False, 2, 0.0, False, True, False)
        expected = op(x, hx, params, False, 2, 0.0, False, True, False)
        for g, e in zip(got, expected):
            assert tp.allclose(g, e, rtol=1e-4, atol=1e-5)


def test_gated_rnn_decomposition_matches_the_eager_loop():
    get_decompositions([])
    # The gates are fused into the input linear, so a group is
    # [w_ih (4H, F), w_hh (4H, H), b_ih, b_hh] for lstm and (3H, ...) for gru;
    # hx carries one hidden per direction (lstm: per state too).
    xb = _t(3, 5, 4)
    hx = _t(2, 3, 2)
    hc = _t(2, 3, 2)
    lstm_params = [_t(8, 4), _t(8, 2), _unit(8), _unit(8)] * 2
    got = decomposition_table[ops.lstm.input](
        xb, [hx, hc], lstm_params, True, 1, 0.0, False, True, True)
    expected = ops.lstm.input(
        xb, [hx, hc], lstm_params, True, 1, 0.0, False, True, True)
    for g, e in zip(got, expected):
        assert tp.allclose(g, e, rtol=1e-4, atol=1e-5)

    gru_params = [_t(6, 4), _t(6, 2), _unit(6), _unit(6)] * 2
    got = decomposition_table[ops.gru.input](
        xb, hx, gru_params, True, 1, 0.0, False, True, True)
    expected = ops.gru.input(
        xb, hx, gru_params, True, 1, 0.0, False, True, True)
    for g, e in zip(got, expected):
        assert tp.allclose(g, e, rtol=1e-4, atol=1e-5)

    # Two layers: the second layer's input width is the first layer's hidden
    # width, not the raw feature count.
    x = _t(5, 3, 4)
    lstm_params = [_t(8, 4), _t(8, 2), _unit(8), _unit(8),
                   _t(8, 2), _t(8, 2), _unit(8), _unit(8)]
    got = decomposition_table[ops.lstm.input](
        x, [hx, hc], lstm_params, True, 2, 0.0, False, False, False)
    expected = ops.lstm.input(
        x, [hx, hc], lstm_params, True, 2, 0.0, False, False, False)
    for g, e in zip(got, expected):
        assert tp.allclose(g, e, rtol=1e-4, atol=1e-5)


def test_predicate_factory_and_blas_decompositions_follow_their_contracts():
    get_decompositions([])
    with pytest.raises(RuntimeError):
        decomposition_table[ops.item.default](tp.tensor([1.0, 2.0]))
    with pytest.raises(RuntimeError):
        decomposition_table[ops.complex.default](
            tp.randn(2), tp.randn(2, dtype=tp.float64))
    with pytest.raises(RuntimeError):
        decomposition_table[ops.complex.default](
            tp.randn(2, dtype=tp.int64), tp.randn(2, dtype=tp.int64))
    with pytest.raises(RuntimeError):
        decomposition_table[ops.addmv.default](
            tp.randn(3), tp.randn(3, 4, dtype=tp.float64), tp.randn(4, dtype=tp.float64))
    with pytest.raises(RuntimeError):
        decomposition_table[ops.bucketize.Tensor](tp.randn(2), tp.randn(2, 2))
    with pytest.raises(RuntimeError):
        decomposition_table[ops._to_copy.default](tp.randn(2), pin_memory=True)
    with pytest.raises(RuntimeError):
        decomposition_table[ops.where.self](tp.randn(2, 2), tp.randn(2, 2), tp.randn(2, 2))

    # An empty boundary list answers zero everywhere, in the result dtype.
    got = decomposition_table[ops.bucketize.Tensor](tp.randn(2, 2), tp.zeros(0))
    assert got.dtype == tp.int64
    assert tp.equal(got, tp.zeros(2, 2, dtype=tp.int64))

    # A uniform window split is the strided average.
    x = tp.randn(2, 3, 6, 6)
    got = decomposition_table[ops._adaptive_avg_pool2d.default](x, [2, 3])
    assert tp.equal(got, ops._adaptive_avg_pool2d.default(x, [2, 3]))

    # _to_copy with no conversion options is a clone.
    x = tp.randn(2, 3)
    got = decomposition_table[ops._to_copy.default](x)
    assert tp.equal(got, x)

    # Integer addmm casts the scalars so the product stays integral: the
    # matmul sums 2*3 twice over the reduction dim, then 5*12 + 2*self.
    got = decomposition_table[ops.addmm.default](
        tp.ones(2, 2, dtype=tp.int64), 2 * tp.ones(2, 2, dtype=tp.int64),
        3 * tp.ones(2, 2, dtype=tp.int64), beta=2, alpha=5)
    assert tp.equal(got, 62 * tp.ones(2, 2, dtype=tp.int64))


def test_normal_decompositions_read_a_standard_normal_and_affine_it():
    get_decompositions([])
    # A large draw recovers the affine parameters; the walk scales and shifts
    # a standard normal read.
    samples = decomposition_table[ops.normal.float_float](
        2.0, 0.5, [200000], dtype=tp.float32)
    assert abs(samples.mean().item() - 2.0) < 0.02
    assert abs(samples.std().item() - 0.5) < 0.02

    # Tensor parameters shape the draw by broadcasting and promote the dtype.
    got = decomposition_table[ops.normal.Tensor_Tensor](
        tp.full([2, 1], 3.0), tp.full([1, 2], 0.5))
    assert tuple(got.shape) == (2, 2)
    assert got.dtype == tp.float32

    res = decomposition_table[ops.normal_functional.default](
        tp.empty(3, 4, dtype=tp.float32), 0.0, 1.0)
    assert tuple(res.shape) == (3, 4)
    assert res.dtype == tp.float32

    with pytest.raises(RuntimeError):
        decomposition_table[ops.normal.float_float](0.0, -1.0, [3])
    with pytest.raises(AssertionError):
        decomposition_table[ops.normal.float_float](0.0, 1.0, [3], generator=True)
    with pytest.raises(RuntimeError):
        decomposition_table[ops.pdist.default](tp.randn(3), 2)
    with pytest.raises(RuntimeError):
        decomposition_table[ops.grid_sampler_2d.default](
            tp.randn(2, 3, 4, 5), tp.randn(2, 6, 7, 2), 3, 0, False)


def test_reduction_decompositions_follow_their_contracts():
    get_decompositions([])
    # Bool sums and products accumulate in int64.
    got = decomposition_table[ops.sum.default](
        tp.tensor([[True, False], [True, True]]))
    assert got.dtype == tp.int64 and got.item() == 3
    got = decomposition_table[ops.prod.dim_IntList](
        tp.tensor([[True, False], [True, True]]), [1])
    assert got.dtype == tp.int64 and tp.equal(got, tp.tensor([0, 1], dtype=tp.int64))

    # any keeps the legacy uint8 mask spelling.
    got = decomposition_table[ops.any.default](tp.tensor([2, 0, 5], dtype=tp.uint8))
    assert got.dtype == tp.uint8 and got.item() == 1

    # mean refuses integer inputs.
    with pytest.raises(RuntimeError):
        decomposition_table[ops.mean.default](tp.ones(3, dtype=tp.int64))

    # An empty dim list reduces every dimension.
    x = tp.randn(2, 3, 4)
    got = decomposition_table[ops.sum.dim_IntList](x, [])
    assert tp.allclose(got, ops.sum.dim_IntList(x, [0, 1, 2]), rtol=1e-5, atol=1e-6)

    # Scans over a 0-d tensor answer the value itself.
    got = decomposition_table[ops.cumsum.default](tp.tensor(2.5), 0)
    assert got.dim() == 0 and got.item() == 2.5
    got = decomposition_table[ops.cumprod.default](tp.tensor(3.0), 0)
    assert got.dim() == 0 and got.item() == 3.0

    # The cumsum mask reproduces a sequential scan exactly on a lattice.
    x = _lat(2, 4)
    got = decomposition_table[ops.cumsum.default](x, 1)
    assert tp.equal(got, ops.cumsum.default(x, 1))


def test_fused_rms_norm_against_formula_and_autograd():
    # No CPU kernel serves these overloads; the oracle is the formula itself
    # and its autograd gradient.
    get_decompositions([])
    x = _t(3, 4)
    w = _unit(4)
    out, rstd = decomposition_table[ops._fused_rms_norm.default](x, [4], w, 1e-5)
    expected = x * tp.rsqrt((x * x).mean(dim=[1], keepdim=True) + 1e-5) * w
    assert tp.allclose(out, expected, rtol=1e-5, atol=1e-6)

    grad = _t(3, 4)
    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    (xr * tp.rsqrt((xr * xr).mean(dim=[1], keepdim=True) + 1e-5) * wr).backward(grad)
    d_input, d_weight = decomposition_table[ops._fused_rms_norm_backward.default](
        grad, x, [4], rstd, w, [True, True]
    )
    assert tp.allclose(d_input, xr.grad, rtol=1e-4, atol=1e-5)
    assert tp.allclose(d_weight, wr.grad, rtol=1e-4, atol=1e-5)


def test_registry_loads():
    get_decompositions([])
    assert len(decomposition_table) > 0


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_decomposition_matches_operator(name):
    get_decompositions([])
    packet, overload_name = name.split(".")
    overload = getattr(getattr(ops, packet), overload_name)
    assert overload in decomposition_table
    args, kwargs = SAMPLES[name]()
    expected = overload(*args, **kwargs)
    got = decomposition_table[overload](*args, **kwargs)
    expected = list(expected) if isinstance(expected, (list, tuple)) else [expected]
    got = list(got) if isinstance(got, (list, tuple)) else [got]
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        if e is None:
            assert g is None
            continue
        if isinstance(e, (bool, int, float, complex)):
            assert g == e, (g, e)
            continue
        assert g.dtype == e.dtype, (g.dtype, e.dtype)
        assert tuple(g.shape) == tuple(e.shape), (tuple(g.shape), tuple(e.shape))
        if e.dtype == tp.bool or (not e.is_floating_point() and not e.is_complex()):
            assert tp.equal(g, e)
        else:
            assert tp.allclose(g, e, rtol=1e-5, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("name", ["empty_like.default", "new_empty.default"])
def test_uninitialized_creation_metadata(name):
    get_decompositions([])
    packet, overload_name = name.split(".")
    overload = getattr(getattr(ops, packet), overload_name)
    args = (_t(2, 3),) if packet == "empty_like" else (_t(2), [4, 1])
    expected = overload(*args)
    got = decomposition_table[overload](*args)
    assert got.dtype == expected.dtype and tuple(got.shape) == tuple(expected.shape)


def test_every_functional_decomposition_has_a_sample():
    get_decompositions([])
    missing = sorted(
        str(o).split(".", 1)[1] for o in _registered_functional()
        if str(o).split(".", 1)[1] not in SAMPLES
        and str(o).split(".", 1)[1] not in {
            "empty_like.default", "new_empty.default", "_chunk_cat.default",
            "_fused_rms_norm.default", "_fused_rms_norm_backward.default",
            "dropout.default", "native_dropout.default",
            "alpha_dropout.default", "_fused_dropout.default",
            "new_empty_strided.default", "randn.default", "sym_numel.default",
            "empty_strided.default", "pad_sequence.default",
            # Batch norm legit forms have no CPU eager kernels; the walk is
            # shared with native_batch_norm and exercised against it below.
            "_native_batch_norm_legit.default", "_native_batch_norm_legit.no_stats",
            # Transforms without CPU eager kernels; validated through the
            # round-trip identities in the fft test module instead.
            "fft_hfft.default", "fft_ihfft.default",
            "fft_fftn.default", "fft_ifftn.default", "fft_rfftn.default",
            "fft_irfftn.default", "fft_hfftn.default", "fft_ihfftn.default",
            "fft_hfft2.default", "fft_ihfft2.default",
            "fft_fftshift.default", "fft_ifftshift.default",
            # Anti-aliased lanczos has no CPU kernel at any level; its vec
            # decomposition only unpacks the scale factors onto the default op.
            "_upsample_lanczos2d_aa.vec",
            # The normal family draws random values; an eager comparison would
            # compare two different draws, so the walk is checked against the
            # distribution instead (below).
            "normal.Tensor_Tensor", "normal.Tensor_float",
            "normal.float_Tensor", "normal.float_float",
            "normal_functional.default",
        }
        and str(o).split(".", 1)[1] not in DEVICE_SPECIFIC
        and not any(a.is_out for a in o._schema.arguments)
    )
    assert missing == []


def test_inplace_decomposition_writes_self():
    get_decompositions([])
    x = _t(5)
    expected = x.clone().clamp_min_(0.2)
    fn = decomposition_table[ops.clamp_min_.default]
    y = x.clone()
    result = fn(y, 0.2)
    assert result is y
    assert tp.allclose(y, expected)


@pytest.mark.parametrize("name", sorted(WRITING_FORMS))
def test_inplace_writing_form_matches_operator(name):
    get_decompositions([])
    packet, overload_name = name.split(".")
    overload = getattr(getattr(ops, packet), overload_name)
    assert overload in decomposition_table
    built = WRITING_FORMS[name]()
    if len(built) == 2 and isinstance(built[1], dict):
        args, kwargs = built
    else:
        args, kwargs = built, {}
    expected = overload(args[0].clone(), *args[1:], **kwargs)
    inputs = (args[0].clone(),) + tuple(
        a.clone() if isinstance(a, tp.Tensor) else a for a in args[1:]
    )
    result = decomposition_table[overload](*inputs, **kwargs)
    assert result is inputs[0]
    if expected.dtype == tp.bool or not expected.is_floating_point():
        assert tp.equal(result, expected)
    else:
        assert tp.allclose(result, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kwargs", [{}, {"dtype": tp.float64}, {"dtype": tp.int32}])
def test_new_empty_strided_metadata(kwargs):
    get_decompositions([])
    x = _t(2)
    expected = ops.new_empty_strided.default(x, [4, 1], [1, 4], **kwargs)
    got = decomposition_table[ops.new_empty_strided.default](x, [4, 1], [1, 4], **kwargs)
    assert got.dtype == expected.dtype
    assert tuple(got.shape) == tuple(expected.shape)
    assert tuple(got.stride()) == tuple(expected.stride())


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"dtype": tp.float64}, {"requires_grad": True}, {"device": tp.get_default_device()}],
)
def test_randn_decomposition_draws_at_shape(kwargs):
    get_decompositions([])
    got = decomposition_table[ops.randn.default]([3, 4], **kwargs)
    assert tuple(got.shape) == (3, 4)
    assert got.dtype == kwargs.get("dtype", tp.get_default_dtype())
    assert got.requires_grad == kwargs.get("requires_grad", False)
    assert bool(tp.isfinite(got).all())


def test_sym_numel_counts_positions():
    get_decompositions([])
    x = _t(2, 3, 4)
    got = decomposition_table[ops.sym_numel.default](x)
    assert isinstance(got, int)
    assert got == 24
    assert got == ops.sym_numel.default(x)


def test_out_overload_writes_destination():
    get_decompositions([])
    grad, out = _t(4), _t(4)
    destination = tp.empty(0)
    fn = decomposition_table[ops.tanh_backward.grad_input]
    result = fn(grad, out, grad_input=destination)
    assert result is destination
    assert tp.allclose(destination, ops.tanh_backward.default(grad, out))


def test_get_and_remove_by_packet():
    table = get_decompositions([ops.lerp, ops.silu.default])
    assert ops.lerp.Tensor in table and ops.silu.default in table
    remove_decompositions(table, [ops.lerp])
    assert ops.lerp.Tensor not in table and ops.silu.default in table


def test_duplicate_registration_is_rejected():
    registry = {}

    @register_decomposition(ops.silu.default, registry)
    def first(x):
        return x

    with pytest.raises(RuntimeError, match="duplicate"):

        @register_decomposition(ops.silu.default, registry)
        def second(x):
            return x


def test_core_table_keeps_core_operators():
    table = core_decompositions()
    assert all("core" not in overload.tags for overload in table)


@pytest.mark.parametrize("ord", [0, 1, 2, 3.5, float("inf"), float("-inf")])
@pytest.mark.parametrize("dim", [None, [0], [0, 1]])
def test_vector_norm_orders(ord, dim):
    get_decompositions([])
    overload = ops.linalg_vector_norm.default
    x = _t(3, 4)
    x[1, 2] = 0.0
    expected = overload(x, ord, dim)
    got = decomposition_table[overload](x, ord, dim)
    assert tuple(got.shape) == tuple(expected.shape)
    assert tp.allclose(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("uniform_.default", {}),
        ("normal_.default", {}),
        ("cauchy_.default", {"median": 0.0, "sigma": 1.0}),
        ("exponential_.default", {"lambd": 1.5}),
        ("geometric_.default", {"p": 0.5}),
        ("log_normal_.default", {"mean": 0.0, "std": 1.0}),
    ],
)
def test_rng_fill_decomposition_leaves_the_draw_in_the_value(name, kwargs):
    get_decompositions([])
    packet, overload_name = name.split(".")
    fn = decomposition_table[getattr(getattr(ops, packet), overload_name)]
    dest = tp.zeros(3, 4)
    assert fn(dest, **kwargs) is dest
    assert tuple(dest.shape) == (3, 4)
    assert dest.dtype == tp.get_default_dtype()
    assert bool(tp.isfinite(dest).all())
    assert float(dest.abs().sum()) > 0.0


@pytest.mark.parametrize(
    "name, call",
    [
        ("uniform_.default", lambda fn, t: fn(t, 0, 1, generator=object())),
        ("cauchy_.default", lambda fn, t: fn(t, 0, 1, generator=object())),
        ("exponential_.default", lambda fn, t: fn(t, 1, generator=object())),
        ("geometric_.default", lambda fn, t: fn(t, 0.5, generator=object())),
        ("log_normal_.default", lambda fn, t: fn(t, 0, 1, generator=object())),
    ],
)
def test_rng_fill_decomposition_refuses_a_generator(name, call):
    get_decompositions([])
    packet, overload_name = name.split(".")
    fn = decomposition_table[getattr(getattr(ops, packet), overload_name)]
    with pytest.raises(AssertionError):
        call(fn, tp.zeros(3))
