// Shape-only kernels for the convolutions on the meta device.  The
// result carries the weight's output channels and, per spatial axis, the
// extents the stride, padding and dilation leave behind; the transposed
// forms grow with the stride instead.  The weight's leading channel
// extent must agree with the input's (divided by groups), and the bias
// plane follows the output channels.
#include "MetaCommon.h"

#include "Tensor.h"
#include "TensorImpl.h"
#include "Convolution.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Utils.h"
#include "Context.h"
#include "Generator.h"
#include "Scalar.h"
#include "DType.h"
#include "DTypeNames.h"
#include "DistributionDispatch.h"
#include "TypePromotion.h"
#include "TypeProperties.h"
#include "../BlasComposite.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
namespace tensorplay {
namespace meta {
namespace {
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Convolution. The result carries the weight's output channels and, per
// spatial dimension, the extent the direct-convolution formula gives for the
// padded input and the dilated kernel; the geometry is validated by the same
// shared rules the computing backends use, so a call they would reject is
// rejected here too.
// ---------------------------------------------------------------------------
Tensor conv2d_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     const std::vector<int64_t>& stride,
                     const std::vector<int64_t>& padding,
                     const std::vector<int64_t>& dilation, int64_t groups) {
    if (input.dim() != 4 || weight.dim() != 4) {
        TP_THROW(RuntimeError, "conv2d: Expected 4D input and weight");
    }
    convolution::check_conv_shapes(input, weight, bias, groups, /*transposed=*/false);
    convolution::check_conv_geometry(input, weight, stride, padding, dilation, "conv2d");
    const std::vector<int64_t> spatial =
        convolution::conv_output_spatial_sizes(input, weight, stride, padding, dilation);
    std::vector<int64_t> out{input.size(0), weight.size(0)};
    out.insert(out.end(), spatial.begin(), spatial.end());
    return empty_meta(out, input.dtype(), input.device());
}

Tensor conv1d_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     const std::vector<int64_t>& stride,
                     const std::vector<int64_t>& padding,
                     const std::vector<int64_t>& dilation, int64_t groups) {
    if (input.dim() != 3 || weight.dim() != 3) {
        TP_THROW(RuntimeError, "conv1d: Expected 3D input and weight");
    }
    convolution::check_conv_shapes(input, weight, bias, groups, /*transposed=*/false);
    convolution::check_conv_geometry(input, weight, stride, padding, dilation, "conv1d");
    const std::vector<int64_t> spatial =
        convolution::conv_output_spatial_sizes(input, weight, stride, padding, dilation);
    std::vector<int64_t> out{input.size(0), weight.size(0)};
    out.insert(out.end(), spatial.begin(), spatial.end());
    return empty_meta(out, input.dtype(), input.device());
}

Tensor conv3d_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     const std::vector<int64_t>& stride,
                     const std::vector<int64_t>& padding,
                     const std::vector<int64_t>& dilation, int64_t groups) {
    if (input.dim() != 5 || weight.dim() != 5) {
        TP_THROW(RuntimeError, "conv3d: Expected 5D input and weight");
    }
    convolution::check_conv_shapes(input, weight, bias, groups, /*transposed=*/false);
    convolution::check_conv_geometry(input, weight, stride, padding, dilation, "conv3d");
    const std::vector<int64_t> spatial =
        convolution::conv_output_spatial_sizes(input, weight, stride, padding, dilation);
    std::vector<int64_t> out{input.size(0), weight.size(0)};
    out.insert(out.end(), spatial.begin(), spatial.end());
    return empty_meta(out, input.dtype(), input.device());
}

// A transposed convolution scatters each input element through the kernel, so
// its output grows with the stride and the dilation and shrinks by the
// cropping padding: (in - 1) * stride - 2 * padding + dilation * (kernel - 1)
// + output_padding + 1.  The weight's first channel dimension is the input's.
// Parameter lists accept one shared value, one per spatial axis, or one per
// side, and only the axis entries reach the formula; no extent is policed
// beyond that, because the formula is applied exactly as the computing
// backend applies it -- a zero result extent names a valid empty tensor,
// a negative one cannot name a tensor at all.
Tensor conv_transpose2d_kernel(const Tensor& input, const Tensor& weight,
                               const Tensor& bias,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& output_padding,
                               int64_t groups,
                               const std::vector<int64_t>& dilation) {
    if (input.dim() != 4 || weight.dim() != 4) {
        TP_THROW(RuntimeError, "conv_transpose2d: Expected 4D input and weight");
    }
    convolution::check_conv_shapes(input, weight, bias, groups, /*transposed=*/true);
    const auto expand = [](const std::vector<int64_t>& list, const char* name) {
        switch (list.size()) {
            case 1:
                return std::vector<int64_t>{list[0], list[0]};
            case 2:
                return list;
            case 4:
                return std::vector<int64_t>{list[0], list[1]};
            default:
                TP_THROW(RuntimeError, "conv_transpose2d: ", name,
                         " must have size 1 or 2 or 4");
        }
    };
    const std::vector<int64_t> st = expand(stride, "stride");
    const std::vector<int64_t> pd = expand(padding, "padding");
    const std::vector<int64_t> op = expand(output_padding, "output_padding");
    const std::vector<int64_t> dl = expand(dilation, "dilation");
    const int64_t out_channels = weight.size(1) * groups;
    const int64_t h_out =
        (input.size(2) - 1) * st[0] - 2 * pd[0] + dl[0] * (weight.size(2) - 1) +
        op[0] + 1;
    const int64_t w_out =
        (input.size(3) - 1) * st[1] - 2 * pd[1] + dl[1] * (weight.size(3) - 1) +
        op[1] + 1;
    if (h_out < 0 || w_out < 0) {
        TP_THROW(RuntimeError,
                 "conv_transpose2d: Calculated output size is too small");
    }
    return empty_meta({input.size(0), out_channels, h_out, w_out},
                      input.dtype(), input.device());
}

// The one-dimensional transposed convolution runs on the two-dimensional one
// with a unit-height axis: both operands gain a middle extent of one, the
// parameter lists keep their leading entry per axis (padding, output padding
// and dilation ride the added axis at zero, one and zero), and the extra
// output extent of one is squeezed away again.
Tensor conv_transpose1d_kernel(const Tensor& input, const Tensor& weight,
                               const Tensor& bias,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& output_padding,
                               int64_t groups,
                               const std::vector<int64_t>& dilation) {
    if (input.dim() != 3) {
        TP_THROW(RuntimeError, "conv_transpose1d: Expected 3D input (N, C, L)");
    }
    if (weight.dim() != 3) {
        TP_THROW(RuntimeError, "conv_transpose1d: Expected 3D weight");
    }
    convolution::check_conv_shapes(input, weight, bias, groups, /*transposed=*/true);
    Tensor in_2d = input.unsqueeze(2);
    Tensor w_2d = weight.unsqueeze(2);
    std::vector<int64_t> s_2d = {1, stride.empty() ? 1 : stride[0]};
    std::vector<int64_t> p_2d = {0, padding.empty() ? 0 : padding[0]};
    std::vector<int64_t> op_2d = {0, output_padding.empty() ? 0 : output_padding[0]};
    std::vector<int64_t> d_2d = {1, dilation.empty() ? 1 : dilation[0]};
    Tensor out_2d = conv_transpose2d_kernel(in_2d, w_2d, bias, s_2d, p_2d, op_2d,
                                            groups, d_2d);
    return out_2d.squeeze(2);
}

Tensor conv_transpose3d_kernel(const Tensor& input, const Tensor& weight,
                               const Tensor& bias,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& output_padding,
                               int64_t groups,
                               const std::vector<int64_t>& dilation) {
    if (input.dim() != 5 || weight.dim() != 5) {
        TP_THROW(RuntimeError, "conv_transpose3d: Expected 5D input and weight");
    }
    convolution::check_conv_shapes(input, weight, bias, groups, /*transposed=*/true);
    const auto expand = [](const std::vector<int64_t>& list, const char* name) {
        switch (list.size()) {
            case 1:
                return std::vector<int64_t>{list[0], list[0], list[0]};
            case 3:
                return list;
            case 6:
                return std::vector<int64_t>{list[0], list[1], list[2]};
            default:
                TP_THROW(RuntimeError, "conv_transpose3d: ", name,
                         " must have size 1 or 3 or 6");
        }
    };
    const std::vector<int64_t> st = expand(stride, "stride");
    const std::vector<int64_t> pd = expand(padding, "padding");
    const std::vector<int64_t> op = expand(output_padding, "output_padding");
    const std::vector<int64_t> dl = expand(dilation, "dilation");
    const int64_t out_channels = weight.size(1) * groups;
    const int64_t d_out =
        (input.size(2) - 1) * st[0] - 2 * pd[0] + dl[0] * (weight.size(2) - 1) +
        op[0] + 1;
    const int64_t h_out =
        (input.size(3) - 1) * st[1] - 2 * pd[1] + dl[1] * (weight.size(3) - 1) +
        op[1] + 1;
    const int64_t w_out =
        (input.size(4) - 1) * st[2] - 2 * pd[2] + dl[2] * (weight.size(4) - 1) +
        op[2] + 1;
    if (d_out < 0 || h_out < 0 || w_out < 0) {
        TP_THROW(RuntimeError,
                 "conv_transpose3d: Calculated output size is too small");
    }
    return empty_meta({input.size(0), out_channels, d_out, h_out, w_out},
                      input.dtype(), input.device());
}

// ---------------------------------------------------------------------------
// The generic `convolution` spelling owns no geometry of its own: it pins the
// spatial rank of the operand pair and forwards to the rank-specialized
// kernel above, the same routing the computing backends apply, so both
// spellings of the operator answer identically.
// ---------------------------------------------------------------------------
Tensor convolution_meta(const Tensor& input, const Tensor& weight,
                        const std::optional<Tensor>& bias,
                        const std::vector<int64_t>& stride,
                        const std::vector<int64_t>& padding,
                        const std::vector<int64_t>& dilation, bool transposed,
                        const std::vector<int64_t>& output_padding,
                        int64_t groups) {
    const Tensor bias_t = bias.has_value() ? *bias : Tensor();
    const int64_t k = convolution::spatial_dims(input, weight, "convolution");
    if (!transposed) {
        switch (k) {
            case 1: return conv1d_kernel(input, weight, bias_t, stride, padding,
                                         dilation, groups);
            case 2: return conv2d_kernel(input, weight, bias_t, stride, padding,
                                         dilation, groups);
            default: return conv3d_kernel(input, weight, bias_t, stride, padding,
                                          dilation, groups);
        }
    }
    switch (k) {
        case 1: return conv_transpose1d_kernel(input, weight, bias_t, stride,
                                               padding, output_padding, groups,
                                               dilation);
        case 2: return conv_transpose2d_kernel(input, weight, bias_t, stride,
                                               padding, output_padding, groups,
                                               dilation);
        default: return conv_transpose3d_kernel(input, weight, bias_t, stride,
                                                padding, output_padding, groups,
                                                dilation);
    }
}

// The backward answers one gradient per requested slot: the input gradient
// repeats the input geometry, the weight gradient the weight geometry, and
// the bias gradient is the per-channel reduction of grad_output.  Every
// gradient carries grad_output's dtype, the width the scatter kernels
// accumulate in (low-precision inputs route through a widening pass and cast
// back).  Slots the mask leaves unset stay undefined; that is the signal the
// autograd engine reads for "no gradient flows to this input".
std::tuple<Tensor, Tensor, Tensor> convolution_backward_overrideable_meta(
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation, bool transposed,
        const std::vector<int64_t>& output_padding, int64_t groups,
        const std::vector<bool>& output_mask) {
    convolution::spatial_dims(input, weight, "convolution_backward");
    const convolution::GradRequest want = convolution::decode_mask(output_mask);
    if (want.bias) {
        convolution::check_bias_sizes(std::nullopt, grad_output.size(1));
    }
    if (!transposed) {
        convolution::check_conv_geometry(input, weight, stride, padding,
                                         dilation, "convolution_backward");
    }
    Tensor grad_input;
    Tensor grad_weight;
    Tensor grad_bias;
    if (want.input) {
        grad_input = empty_meta(shape_of(input), grad_output.dtype(),
                                grad_output.device());
    }
    if (want.weight) {
        grad_weight = empty_meta(shape_of(weight), grad_output.dtype(),
                                 grad_output.device());
    }
    if (want.bias) {
        grad_bias = empty_meta({grad_output.size(1)}, grad_output.dtype(),
                               grad_output.device());
    }
    return {grad_input, grad_weight, grad_bias};
}
} // anonymous namespace
TENSORPLAY_LIBRARY_IMPL(Meta, MetaConv) {
    m.impl("conv2d", conv2d_kernel);
    m.impl("conv1d", conv1d_kernel);
    m.impl("conv3d", conv3d_kernel);
    m.impl("conv_transpose1d", conv_transpose1d_kernel);
    m.impl("conv_transpose2d", conv_transpose2d_kernel);
    m.impl("conv_transpose3d", conv_transpose3d_kernel);
    m.impl("convolution", convolution_meta);
    m.impl("convolution_overrideable", convolution_meta);
    m.impl("convolution_backward_overrideable",
           convolution_backward_overrideable_meta);
}

} // namespace meta
} // namespace tensorplay
