#pragma once

// Declarations for the native quantized kernels on the CUDA backend.  The
// implementations live in per-family translation units (activation, cat,
// pooling, convolution, batch norm) mirroring the reference layout of the
// quantized CPU backend, so each family can be compiled and tuned on its own.

#include "Tensor.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace cuda {

// Activation family.  relu/relu6 are order-preserving and run entirely on
// the Int8 code grid (the input quantizer is inherited); the rest compute in
// the dequantized domain inside the kernel and requantize under the output
// qparams.
Tensor quantized_relu_cuda(const Tensor& self);
Tensor quantized_relu6_cuda(const Tensor& self);
Tensor quantized_leaky_relu_cuda(const Tensor& self, double negative_slope,
                                 double output_scale,
                                 int64_t output_zero_point);
Tensor quantized_elu_cuda(const Tensor& self, double output_scale,
                          int64_t output_zero_point, double alpha,
                          double scale, double input_scale);
Tensor quantized_hardswish_cuda(const Tensor& self, double output_scale,
                                int64_t output_zero_point);
Tensor quantized_hardsigmoid_cuda(const Tensor& self, double output_scale,
                                  int64_t output_zero_point);
Tensor quantized_sigmoid_cuda(const Tensor& self, double output_scale,
                              int64_t output_zero_point);
Tensor quantized_tanh_cuda(const Tensor& self, double output_scale,
                           int64_t output_zero_point);

// Concatenation family.
Tensor quantized_cat_cuda(const std::vector<Tensor>& tensors, int64_t dim,
                          std::optional<double> scale,
                          std::optional<int64_t> zero_point);
Tensor quantized_cat_relu_cuda(const std::vector<Tensor>& tensors,
                               int64_t dim, std::optional<double> scale,
                               std::optional<int64_t> zero_point);

// Pooling family: window maximum on the Int8 code storage, the input
// quantizer is inherited.
Tensor quantized_max_pool1d_cuda(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode);
Tensor quantized_max_pool3d_cuda(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode);

// Convolution family: native integer-domain accumulation with a single
// requantization at the output under (output_scale, output_zero_point).
Tensor quantized_conv1d_cuda(
    const Tensor& input, const Tensor& weight,
    const std::optional<Tensor>& bias, double input_scale,
    int64_t input_zero_point, double weight_scale, int64_t weight_zero_point,
    double output_scale, int64_t output_zero_point,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int64_t groups);
Tensor quantized_conv2d_cuda(
    const Tensor& input, const Tensor& weight,
    const std::optional<Tensor>& bias, double input_scale,
    int64_t input_zero_point, double weight_scale, int64_t weight_zero_point,
    double output_scale, int64_t output_zero_point,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int64_t groups);
Tensor quantized_conv3d_cuda(
    const Tensor& input, const Tensor& weight,
    const std::optional<Tensor>& bias, double input_scale,
    int64_t input_zero_point, double weight_scale, int64_t weight_zero_point,
    double output_scale, int64_t output_zero_point,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int64_t groups);
std::tuple<Tensor, Tensor> quantized_conv2d_prepack_cuda(
    const Tensor& weight, const Tensor& weight_scales,
    const Tensor& weight_zero_points, const std::optional<Tensor>& bias,
    bool transposed);
std::tuple<Tensor, Tensor> quantized_conv2d_unpack_cuda(
    const Tensor& weight_packed, const Tensor& bias_packed,
    const std::vector<int64_t>& weight_sizes, bool transposed,
    bool depthwise);

// Batch norm family: fused per-channel alpha/beta computed on the host, then a
// single pass over the code storage in the integer domain.
Tensor quantized_batch_norm_cuda(
    const Tensor& input, const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias, const Tensor& mean, const Tensor& var,
    double eps, double output_scale, int64_t output_zero_point);

} // namespace cuda
} // namespace tensorplay