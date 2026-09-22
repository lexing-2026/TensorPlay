#include "Convolution.h"

#include "Dispatcher.h"
#include "Exception.h"

#include <string>

namespace tensorplay {

namespace convolution {

int64_t spatial_dims(const Tensor& input, const Tensor& weight, const char* name) {
    if (input.dim() != weight.dim()) {
        TP_THROW(RuntimeError, std::string(name) +
                 ": input and weight must have the same rank, got " +
                 std::to_string(input.dim()) + " and " + std::to_string(weight.dim()));
    }
    const int64_t k = input.dim() - 2;
    if (k < 1 || k > 3) {
        TP_THROW(RuntimeError, std::string(name) +
                 ": only 1-D, 2-D and 3-D convolutions are supported, got a " +
                 std::to_string(input.dim()) + "-D input");
    }
    return k;
}

void check_bias_sizes(const std::optional<std::vector<int64_t>>& bias_sizes,
                      int64_t out_channels) {
    if (bias_sizes.has_value() && bias_sizes->size() == 1 &&
        (*bias_sizes)[0] != out_channels) {
        TP_THROW(RuntimeError,
                 "convolution_backward: bias_sizes does not match the output channel count");
    }
}

namespace {

// One value per spatial dimension: an empty vector means the operator
// defaults, a single value replicates, and a two-values-per-dimension vector
// (accepted for the per-side spellings) contributes its first half.
std::vector<int64_t> per_dim_values(const std::vector<int64_t>& v, int64_t dims,
                                    int64_t default_value, const char* name,
                                    const char* what) {
    if (v.empty()) return std::vector<int64_t>(dims, default_value);
    if (v.size() == 1) return std::vector<int64_t>(dims, v[0]);
    if (v.size() == static_cast<size_t>(dims)) return v;
    if (v.size() == static_cast<size_t>(2 * dims)) {
        return std::vector<int64_t>(v.begin(), v.begin() + dims);
    }
    TP_THROW(RuntimeError, std::string(name) + ": " + what +
               " must hold one value per spatial dimension");
}

// Per-dimension padding totals: a single value or one value per dimension
// pads both sides of that dimension, so the total doubles; the per-side
// spelling stores the two sides of each dimension adjacently (e.g. top,
// bottom, left, right for 2-D) and their sum is the total.
std::vector<int64_t> per_dim_padding(const std::vector<int64_t>& v, int64_t dims,
                                     const char* name) {
    if (v.empty()) return std::vector<int64_t>(dims, 0);
    if (v.size() == 1) return std::vector<int64_t>(dims, 2 * v[0]);
    if (v.size() == static_cast<size_t>(dims)) {
        std::vector<int64_t> totals(dims);
        for (int64_t i = 0; i < dims; ++i) {
            totals[i] = 2 * v[i];
        }
        return totals;
    }
    if (v.size() == static_cast<size_t>(2 * dims)) {
        std::vector<int64_t> totals(dims);
        for (int64_t i = 0; i < dims; ++i) {
            totals[i] = v[2 * i] + v[2 * i + 1];
        }
        return totals;
    }
    TP_THROW(RuntimeError, std::string(name) +
               ": padding must hold one value per spatial dimension or one per side");
}

} // namespace

void check_conv_geometry(const Tensor& input, const Tensor& weight,
                         const std::vector<int64_t>& stride,
                         const std::vector<int64_t>& padding,
                         const std::vector<int64_t>& dilation,
                         const char* name) {
    const int64_t dims = spatial_dims(input, weight, name);
    const std::vector<int64_t> s = per_dim_values(stride, dims, 1, name, "stride");
    const std::vector<int64_t> d = per_dim_values(dilation, dims, 1, name, "dilation");
    const std::vector<int64_t> p = per_dim_padding(padding, dims, name);

    for (int64_t i = 0; i < dims; ++i) {
        if (p[i] < 0) {
            TP_THROW(RuntimeError, "negative padding is not supported");
        }
        if (s[i] <= 0) {
            TP_THROW(RuntimeError, "non-positive stride is not supported");
        }
        if (d[i] <= 0) {
            TP_THROW(RuntimeError, "dilation should be greater than zero");
        }
    }

    std::vector<int64_t> padded(dims);
    std::vector<int64_t> kernel(dims);
    bool fits = true;
    for (int64_t i = 0; i < dims; ++i) {
        padded[i] = input.size(2 + i) + p[i];
        kernel[i] = d[i] * (weight.size(2 + i) - 1) + 1;
        fits = fits && padded[i] >= kernel[i];
    }
    if (fits) {
        return;
    }

    std::string padded_list;
    std::string kernel_list;
    const char* separator = "";
    for (int64_t i = 0; i < dims; ++i) {
        padded_list += separator + std::to_string(padded[i]);
        kernel_list += separator + std::to_string(kernel[i]);
        separator = " x ";
    }
    TP_THROW(RuntimeError,
             "Calculated padded input size per channel: (" + padded_list + "). "
             "Kernel size: (" + kernel_list +
             "). Kernel size can't be greater than actual input size");
}

} // namespace convolution

namespace cpu {

Tensor convolution_cpu(const Tensor& input, const Tensor& weight,
                       const std::optional<Tensor>& bias,
                       const std::vector<int64_t>& stride,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& dilation,
                       bool transposed,
                       const std::vector<int64_t>& output_padding,
                       int64_t groups) {
    const Tensor bias_t = bias.has_value() ? *bias : Tensor();
    const int64_t k = convolution::spatial_dims(input, weight, "convolution");
    if (!transposed) {
        switch (k) {
            case 1: return conv1d_cpu(input, weight, bias_t, stride, padding, dilation, groups);
            case 2: return conv2d_cpu(input, weight, bias_t, stride, padding, dilation, groups);
            default: return conv3d_cpu(input, weight, bias_t, stride, padding, dilation, groups);
        }
    }
    switch (k) {
        case 1: return conv_transpose1d_cpu(input, weight, bias_t, stride, padding,
                                            output_padding, groups, dilation);
        case 2: return conv_transpose2d_cpu(input, weight, bias_t, stride, padding,
                                            output_padding, groups, dilation);
        default: return conv_transpose3d_cpu(input, weight, bias_t, stride, padding,
                                             output_padding, groups, dilation);
    }
}

std::tuple<Tensor, Tensor, Tensor> convolution_backward_cpu(
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,
        const std::optional<std::vector<int64_t>>& bias_sizes,
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation, bool transposed,
        const std::vector<int64_t>& output_padding, int64_t groups,
        const std::vector<bool>& output_mask) {
    const int64_t k = convolution::spatial_dims(input, weight, "convolution_backward");
    const convolution::GradRequest want = convolution::decode_mask(output_mask);
    if (want.bias) convolution::check_bias_sizes(bias_sizes, grad_output.size(1));

    if (!transposed) {
        convolution::check_conv_geometry(input, weight, stride, padding,
                                         dilation, "convolution_backward");
    }

    // The autograd engine may hand down a broadcast view of the output
    // gradient (e.g. the expand() a sum-backward produces).  The rank-
    // specialized backward kernels index grad_output with contiguous
    // NCHW strides, so normalize the layout once here.
    const Tensor grad_out = grad_output.is_contiguous()
                                ? grad_output : grad_output.contiguous();

    // Slots the caller did not ask for stay undefined; that is the signal the
    // autograd engine reads for "no gradient flows to this input".
    Tensor grad_input;
    Tensor grad_weight;
    Tensor grad_bias;

#define TP_CONV_BACKWARD_CASE(fn_prefix, ...)                                              \
    do {                                                                                   \
        if (want.input) grad_input = fn_prefix##_grad_input_cpu(__VA_ARGS__);               \
        if (want.weight) grad_weight = fn_prefix##_grad_weight_cpu(__VA_ARGS__);            \
        if (want.bias) grad_bias = fn_prefix##_grad_bias_cpu(__VA_ARGS__);                  \
    } while (0)

    if (!transposed) {
        switch (k) {
            case 1: TP_CONV_BACKWARD_CASE(conv1d, grad_out, input, weight, stride,
                                          padding, dilation, groups); break;
            case 2: TP_CONV_BACKWARD_CASE(conv2d, grad_out, input, weight, stride,
                                          padding, dilation, groups); break;
            default: TP_CONV_BACKWARD_CASE(conv3d, grad_out, input, weight, stride,
                                           padding, dilation, groups); break;
        }
    } else {
        switch (k) {
            case 1: TP_CONV_BACKWARD_CASE(conv_transpose1d, grad_out, input, weight,
                                          stride, padding, output_padding, groups,
                                          dilation); break;
            case 2: TP_CONV_BACKWARD_CASE(conv_transpose2d, grad_out, input, weight,
                                          stride, padding, output_padding, groups,
                                          dilation); break;
            default: TP_CONV_BACKWARD_CASE(conv_transpose3d, grad_out, input, weight,
                                           stride, padding, output_padding, groups,
                                           dilation); break;
        }
    }
#undef TP_CONV_BACKWARD_CASE

    return {grad_input, grad_weight, grad_bias};
}

// Extension point for backends registered outside this tree.  CPU and CUDA
// answer it with the same routing as `convolution` so the two spellings never
// disagree on a device this build already supports.
Tensor convolution_overrideable_cpu(const Tensor& input, const Tensor& weight,
                                    const std::optional<Tensor>& bias,
                                    const std::vector<int64_t>& stride,
                                    const std::vector<int64_t>& padding,
                                    const std::vector<int64_t>& dilation,
                                    bool transposed,
                                    const std::vector<int64_t>& output_padding,
                                    int64_t groups) {
    return convolution_cpu(input, weight, bias, stride, padding, dilation,
                           transposed, output_padding, groups);
}

std::tuple<Tensor, Tensor, Tensor> convolution_backward_overrideable_cpu(
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation, bool transposed,
        const std::vector<int64_t>& output_padding, int64_t groups,
        const std::vector<bool>& output_mask) {
    return convolution_backward_cpu(grad_output, input, weight, std::nullopt, stride,
                                    padding, dilation, transposed, output_padding,
                                    groups, output_mask);
}

TENSORPLAY_LIBRARY_IMPL(CPU, Convolution) {
    m.impl("convolution", convolution_cpu);
    m.impl("convolution_backward", convolution_backward_cpu);
    m.impl("convolution_overrideable", convolution_overrideable_cpu);
    m.impl("convolution_backward_overrideable", convolution_backward_overrideable_cpu);
}

} // namespace cpu
} // namespace tensorplay
