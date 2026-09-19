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


def _registered_functional():
    return {
        overload: fn
        for overload, fn in decomposition_table.items()
        if not overload._schema.is_mutable
    }


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
        assert g.dtype == e.dtype, (g.dtype, e.dtype)
        assert tuple(g.shape) == tuple(e.shape), (tuple(g.shape), tuple(e.shape))
        if e.dtype == tp.bool or not e.is_floating_point():
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
