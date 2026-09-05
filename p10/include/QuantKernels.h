#pragma once

#include "Tensor.h"

#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace cpu {

// Affine quantization dispatches on the requested quantized dtype.  Quantized
// tensors retain their input shape and dequantize to Float32.
Tensor quantize_per_tensor_cpu(const Tensor& self, double scale,
                                int64_t zero_point, DType dtype);
Tensor quantize_per_tensor_dtype_cpu(const Tensor& self, double scale,
                                      int64_t zero_point, DType dtype);
Tensor dequantize_per_tensor_dtype_cpu(const Tensor& self, double scale,
                                        int64_t zero_point, DType dtype);
Tensor quantize_per_channel_cpu(const Tensor& self, const Tensor& scales,
                                const Tensor& zero_points, int64_t axis,
                                DType dtype);
Tensor quantize_per_channel_dtype_cpu(const Tensor& self, const Tensor& scales,
                                      const Tensor& zero_points, int64_t axis,
                                      DType dtype);
Tensor dequantize_per_channel_dtype_cpu(
    const Tensor& self, const Tensor& scales, const Tensor& zero_points,
    int64_t axis, DType dtype);
// Fused Int8 GEMM: out[m,n] = x_scale * w_scale[n] * Σ_k (x_q[m,k]-x_zp) *
// (w_q[n,k]-w_zp[n]) + bias[n] -> Float32 [M,N].
Tensor quantized_linear_cpu(const Tensor& input, const Tensor& weight,
                             double input_scale, int64_t input_zero_point,
                             const Tensor& weight_scales,
                             const Tensor& weight_zero_points,
                             std::optional<Tensor> bias);

// Quantized elementwise arithmetic: dequantize both operands with their
// affine qparams, apply the float operation, requantize into
// [-128, 127] under the output qparams.
Tensor quantized_add_cpu(const Tensor& a, const Tensor& b,
                          double a_scale, int64_t a_zero_point,
                          double b_scale, int64_t b_zero_point,
                          double out_scale, int64_t out_zero_point);
Tensor quantized_sub_cpu(const Tensor& a, const Tensor& b,
                          double a_scale, int64_t a_zero_point,
                          double b_scale, int64_t b_zero_point,
                          double out_scale, int64_t out_zero_point);
Tensor quantized_mul_cpu(const Tensor& a, const Tensor& b,
                          double a_scale, int64_t a_zero_point,
                          double b_scale, int64_t b_zero_point,
                          double out_scale, int64_t out_zero_point);
Tensor quantized_div_cpu(const Tensor& a, const Tensor& b,
                          double a_scale, int64_t a_zero_point,
                          double b_scale, int64_t b_zero_point,
                          double out_scale, int64_t out_zero_point);
Tensor quantized_clamp_cpu(const Tensor& self, double self_scale,
                            int64_t self_zero_point, double out_scale,
                            int64_t out_zero_point,
                            std::optional<Scalar> min,
                            std::optional<Scalar> max);
// Window maximum on Int8 storage; the result inherits the input qparams.
Tensor quantized_max_pool2d_cpu(const Tensor& self,
                                 const std::vector<int64_t>& kernel_size,
                                 const std::vector<int64_t>& stride,
                                 const std::vector<int64_t>& padding,
                                 const std::vector<int64_t>& dilation,
                                 bool ceil_mode);

// Quantized 2D convolution: dequantize input and weight bytes, run the
// float convolution (float-domain bias added after the accumulation), and
// requantize into the output qparams.
Tensor quantized_conv2d_cpu(
    const Tensor& input, const Tensor& weight, std::optional<Tensor> bias,
    double input_scale, int64_t input_zero_point, double weight_scale,
    int64_t weight_zero_point, double out_scale, int64_t out_zero_point,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int64_t groups);

// Fake quantization family.  The forward maps values through the affine
// grid (round-half-even on x * inv_scale) and back; the cachemask variants
// also emit a Bool mask marking elements whose raw grid position stayed
// inside [quant_min, quant_max].  The learnable variants accept tensor
// qparams; their backward ops return (dX, dScale, dZeroPoint).
std::tuple<Tensor, Tensor> fake_quantize_per_tensor_affine_cachemask_cpu(
    const Tensor& self, double scale, int64_t zero_point, int64_t quant_min,
    int64_t quant_max);
Tensor fake_quantize_per_tensor_affine_cpu(const Tensor& self, double scale,
                                            int64_t zero_point,
                                            int64_t quant_min,
                                            int64_t quant_max);
Tensor fake_quantize_per_tensor_affine_tensor_qparams_cpu(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t quant_min, int64_t quant_max);
std::tuple<Tensor, Tensor>
_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cpu(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    const Tensor& fake_quant_enabled, int64_t quant_min, int64_t quant_max);
Tensor fake_quantize_per_tensor_affine_cachemask_backward_cpu(
    const Tensor& grad, const Tensor& mask);
Tensor _fake_quantize_learnable_per_tensor_affine_cpu(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t quant_min, int64_t quant_max, double grad_factor);
std::tuple<Tensor, Tensor, Tensor>
_fake_quantize_learnable_per_tensor_affine_backward_cpu(
    const Tensor& grad, const Tensor& self, const Tensor& scale,
    const Tensor& zero_point, int64_t quant_min, int64_t quant_max,
    double grad_factor);
std::tuple<Tensor, Tensor> fake_quantize_per_channel_affine_cachemask_cpu(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max);
Tensor fake_quantize_per_channel_affine_cpu(const Tensor& self,
                                             const Tensor& scale,
                                             const Tensor& zero_point,
                                             int64_t axis, int64_t quant_min,
                                             int64_t quant_max);
Tensor fake_quantize_per_channel_affine_cachemask_backward_cpu(
    const Tensor& grad, const Tensor& mask);
Tensor _fake_quantize_learnable_per_channel_affine_cpu(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor);
std::tuple<Tensor, Tensor, Tensor>
_fake_quantize_learnable_per_channel_affine_backward_cpu(
    const Tensor& grad, const Tensor& self, const Tensor& scale,
    const Tensor& zero_point, int64_t axis, int64_t quant_min,
    int64_t quant_max, double grad_factor);
// Quantizes with qparams derived from the tensor's own min/max; supports
// Int8/UInt8 storage outputs and a Float16 passthrough.
Tensor quantize_per_tensor_dynamic_cpu(const Tensor& self, DType dtype,
                                       bool reduce_range);
// Returns (scale, zero_point) for the [0, 255] grid over the tensor range.
std::tuple<double, int64_t> _choose_qparams_per_tensor_cpu(
    const Tensor& self, bool reduce_range);
// Updates the running min/max under the observer flag, derives qparams from
// the running range, and fake-quantizes; the running-state, scale and
// zero-point tensors are mutated in place.
std::tuple<Tensor, Tensor> _fused_moving_avg_obs_fq_helper_cpu(
    const Tensor& self, const Tensor& observer_on,
    const Tensor& fake_quant_on, Tensor& running_min, Tensor& running_max,
    Tensor& scale, Tensor& zero_point, double averaging_const,
    int64_t quant_min, int64_t quant_max, int64_t ch_axis,
    bool per_row_fake_quant, bool symmetric_quant);
Tensor fused_moving_avg_obs_fake_quant_cpu(
    const Tensor& self, const Tensor& observer_on,
    const Tensor& fake_quant_on, Tensor& running_min, Tensor& running_max,
    Tensor& scale, Tensor& zero_point, double averaging_const,
    int64_t quant_min, int64_t quant_max, int64_t ch_axis,
    bool per_row_fake_quant, bool symmetric_quant);

// Tensor-level quantization metadata: quantized tensors carry an immutable
// quantizer on their impl; these read it (q_scale and friends), strip it
// (int_repr returns the integer codes as a plain tensor), reinterpret raw
// integer code storage as a quantized tensor (_make_per_*), or dispatch a
// quantized tensor to the float domain (dequantize.self; non-quantized
// tensors pass through).
bool is_quantized_cpu(const Tensor& self);
int64_t qscheme_cpu(const Tensor& self);
double q_scale_cpu(const Tensor& self);
int64_t q_zero_point_cpu(const Tensor& self);
Tensor q_per_channel_scales_cpu(const Tensor& self);
Tensor q_per_channel_zero_points_cpu(const Tensor& self);
int64_t q_per_channel_axis_cpu(const Tensor& self);
Tensor int_repr_cpu(const Tensor& self);
Tensor dequantize_self_cpu(const Tensor& self);
Tensor _make_per_tensor_quantized_tensor_cpu(const Tensor& self, double scale,
                                             int64_t zero_point);
Tensor _make_per_channel_quantized_tensor_cpu(const Tensor& self,
                                              const Tensor& scale,
                                              const Tensor& zero_point,
                                              int64_t axis);

} // namespace cpu

namespace vulkan {
namespace ops {

Tensor quantize_per_tensor_kernel(const Tensor& self, double scale,
                                  int64_t zero_point, DType dtype);
Tensor dequantize_per_tensor_kernel(const Tensor& self, double scale,
                                    int64_t zero_point, DType dtype);

// Runs a pre-packed quantized convolution on the Vulkan compute path; the
// packing layout and shader selection follow the quantized convolution
// family (sliding window / depthwise / 2x2 pointwise / transposed).
Tensor quantized_conv2d_run_kernel(
    const Tensor& input,
    const Tensor& weight_packed,
    const Tensor& bias_packed,
    double input_scale,
    int64_t input_zero_point,
    double out_scale,
    int64_t out_zero_point,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    const std::vector<int64_t>& output_padding,
    int64_t groups,
    bool transposed,
    std::optional<Scalar> output_min,
    std::optional<Scalar> output_max);

} // namespace ops
} // namespace vulkan

#ifdef USE_CUDA
namespace cuda {

Tensor quantize_per_tensor_cuda(const Tensor& self, double scale,
                                int64_t zero_point, DType dtype);
Tensor quantize_per_tensor_dtype_cuda(const Tensor& self, double scale,
                                      int64_t zero_point, DType dtype);
Tensor dequantize_per_tensor_dtype_cuda(const Tensor& self, double scale,
                                        int64_t zero_point, DType dtype);
Tensor quantize_per_channel_cuda(const Tensor& self, const Tensor& scales,
                                 const Tensor& zero_points, int64_t axis,
                                 DType dtype);
Tensor quantize_per_channel_dtype_cuda(
    const Tensor& self, const Tensor& scales, const Tensor& zero_points,
    int64_t axis, DType dtype);
Tensor dequantize_per_channel_dtype_cuda(
    const Tensor& self, const Tensor& scales, const Tensor& zero_points,
    int64_t axis, DType dtype);
Tensor quantized_linear_cuda(const Tensor& input, const Tensor& weight,
                              double input_scale, int64_t input_zero_point,
                              const Tensor& weight_scales,
                              const Tensor& weight_zero_points,
                              std::optional<Tensor> bias);
Tensor quantized_add_cuda(const Tensor& a, const Tensor& b,
                           double a_scale, int64_t a_zero_point,
                           double b_scale, int64_t b_zero_point,
                           double out_scale, int64_t out_zero_point);
Tensor quantized_sub_cuda(const Tensor& a, const Tensor& b,
                           double a_scale, int64_t a_zero_point,
                           double b_scale, int64_t b_zero_point,
                           double out_scale, int64_t out_zero_point);
Tensor quantized_mul_cuda(const Tensor& a, const Tensor& b,
                           double a_scale, int64_t a_zero_point,
                           double b_scale, int64_t b_zero_point,
                           double out_scale, int64_t out_zero_point);
Tensor quantized_div_cuda(const Tensor& a, const Tensor& b,
                           double a_scale, int64_t a_zero_point,
                           double b_scale, int64_t b_zero_point,
                           double out_scale, int64_t out_zero_point);
Tensor quantized_clamp_cuda(const Tensor& self, double self_scale,
                             int64_t self_zero_point, double out_scale,
                             int64_t out_zero_point,
                             std::optional<Scalar> min,
                             std::optional<Scalar> max);
Tensor quantized_max_pool2d_cuda(const Tensor& self,
                                  const std::vector<int64_t>& kernel_size,
                                  const std::vector<int64_t>& stride,
                                  const std::vector<int64_t>& padding,
                                  const std::vector<int64_t>& dilation,
                                  bool ceil_mode);
Tensor quantized_conv2d_cuda(
    const Tensor& input, const Tensor& weight, std::optional<Tensor> bias,
    double input_scale, int64_t input_zero_point, double weight_scale,
    int64_t weight_zero_point, double out_scale, int64_t out_zero_point,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int64_t groups);

std::tuple<Tensor, Tensor> fake_quantize_per_tensor_affine_cachemask_cuda(
    const Tensor& self, double scale, int64_t zero_point, int64_t quant_min,
    int64_t quant_max);
Tensor fake_quantize_per_tensor_affine_cuda(const Tensor& self, double scale,
                                             int64_t zero_point,
                                             int64_t quant_min,
                                             int64_t quant_max);
Tensor fake_quantize_per_tensor_affine_tensor_qparams_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t quant_min, int64_t quant_max);
std::tuple<Tensor, Tensor>
_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    const Tensor& fake_quant_enabled, int64_t quant_min, int64_t quant_max);
Tensor fake_quantize_per_tensor_affine_cachemask_backward_cuda(
    const Tensor& grad, const Tensor& mask);
Tensor _fake_quantize_learnable_per_tensor_affine_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t quant_min, int64_t quant_max, double grad_factor);
std::tuple<Tensor, Tensor, Tensor>
_fake_quantize_learnable_per_tensor_affine_backward_cuda(
    const Tensor& grad, const Tensor& self, const Tensor& scale,
    const Tensor& zero_point, int64_t quant_min, int64_t quant_max,
    double grad_factor);
std::tuple<Tensor, Tensor> fake_quantize_per_channel_affine_cachemask_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max);
Tensor fake_quantize_per_channel_affine_cuda(const Tensor& self,
                                              const Tensor& scale,
                                              const Tensor& zero_point,
                                              int64_t axis,
                                              int64_t quant_min,
                                              int64_t quant_max);
Tensor fake_quantize_per_channel_affine_cachemask_backward_cuda(
    const Tensor& grad, const Tensor& mask);
Tensor _fake_quantize_learnable_per_channel_affine_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor);
std::tuple<Tensor, Tensor, Tensor>
_fake_quantize_learnable_per_channel_affine_backward_cuda(
    const Tensor& grad, const Tensor& self, const Tensor& scale,
    const Tensor& zero_point, int64_t axis, int64_t quant_min,
    int64_t quant_max, double grad_factor);
Tensor quantize_per_tensor_dynamic_cuda(const Tensor& self, DType dtype,
                                        bool reduce_range);
std::tuple<double, int64_t> _choose_qparams_per_tensor_cuda(
    const Tensor& self, bool reduce_range);
std::tuple<Tensor, Tensor> _fused_moving_avg_obs_fq_helper_cuda(
    const Tensor& self, const Tensor& observer_on,
    const Tensor& fake_quant_on, Tensor& running_min, Tensor& running_max,
    Tensor& scale, Tensor& zero_point, double averaging_const,
    int64_t quant_min, int64_t quant_max, int64_t ch_axis,
    bool per_row_fake_quant, bool symmetric_quant);
Tensor fused_moving_avg_obs_fake_quant_cuda(
    const Tensor& self, const Tensor& observer_on,
    const Tensor& fake_quant_on, Tensor& running_min, Tensor& running_max,
    Tensor& scale, Tensor& zero_point, double averaging_const,
    int64_t quant_min, int64_t quant_max, int64_t ch_axis,
    bool per_row_fake_quant, bool symmetric_quant);

bool is_quantized_cuda(const Tensor& self);
int64_t qscheme_cuda(const Tensor& self);
double q_scale_cuda(const Tensor& self);
int64_t q_zero_point_cuda(const Tensor& self);
Tensor q_per_channel_scales_cuda(const Tensor& self);
Tensor q_per_channel_zero_points_cuda(const Tensor& self);
int64_t q_per_channel_axis_cuda(const Tensor& self);
Tensor int_repr_cuda(const Tensor& self);
Tensor dequantize_self_cuda(const Tensor& self);
Tensor _make_per_tensor_quantized_tensor_cuda(const Tensor& self, double scale,
                                              int64_t zero_point);
Tensor _make_per_channel_quantized_tensor_cuda(const Tensor& self,
                                               const Tensor& scale,
                                               const Tensor& zero_point,
                                               int64_t axis);

} // namespace cuda
#endif
} // namespace tensorplay
