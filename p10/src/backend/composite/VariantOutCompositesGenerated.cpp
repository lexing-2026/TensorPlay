#include "Tensor.h"
#include "Dispatcher.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "OutWrite.h"

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

Tensor& out_wrap__new_zeros_with_same_feature_meta_out(const Tensor& self, const Tensor& other, int64_t self_num_batch_dims, Tensor& out) {
    write_out(out, ops::_new_zeros_with_same_feature_meta(self, other, self_num_batch_dims));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap__cudnn_ctc_loss_out(const Tensor& log_probs, const Tensor& targets, const std::vector<int64_t>& input_lengths, const std::vector<int64_t>& target_lengths, int64_t blank, bool deterministic, bool zero_infinity, Tensor& out0, Tensor& out1) {
    const auto answers = ops::_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap__cudnn_rnn_flatten_weight_out(const std::vector<Tensor>& weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, bool bidirectional, Tensor& out) {
    write_out(out, ops::_cudnn_rnn_flatten_weight(weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional));
    return out;
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> out_wrap__cudnn_rnn_out(const Tensor& input, const std::vector<Tensor>& weight, int64_t weight_stride0, const std::optional<Tensor>& weight_buf, const Tensor& hx, const std::optional<Tensor>& cx, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, const std::vector<int64_t>& batch_sizes, const std::optional<Tensor>& dropout_state, Tensor& out0, Tensor& out1, Tensor& out2, Tensor& out3, Tensor& out4) {
    const auto answers = ops::_cudnn_rnn(input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    write_out(out3, std::get<3>(answers));
    write_out(out4, std::get<4>(answers));
    return {out0, out1, out2, out3, out4};
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> out_wrap__cudnn_rnn_backward_out(const Tensor& input, const std::vector<Tensor>& weight, int64_t weight_stride0, const Tensor& weight_buf, const Tensor& hx, const std::optional<Tensor>& cx, const Tensor& output, const std::optional<Tensor>& grad_output, const std::optional<Tensor>& grad_hy, const std::optional<Tensor>& grad_cy, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, const std::vector<int64_t>& batch_sizes, const std::optional<Tensor>& dropout_state, const Tensor& reserve, const std::vector<bool>& output_mask, Tensor& out0, Tensor& out1, Tensor& out2, std::vector<Tensor> out3) {
    const auto answers = ops::_cudnn_rnn_backward(input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    for (size_t item = 0;
         item < std::get<3>(answers).size() && item < out3.size();
         ++item) {
        write_out(out3[item], std::get<3>(answers)[item]);
    }
    return {out0, out1, out2, out3};
}

Tensor& out_wrap__cudnn_init_dropout_state_out(double dropout, bool train, int64_t dropout_seed, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::_cudnn_init_dropout_state(dropout, train, dropout_seed, dtype, layout, device, pin_memory));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap__fused_dropout_out(const Tensor& self, double p, std::optional<Generator> generator, Tensor& out0, Tensor& out1) {
    const auto answers = ops::_fused_dropout(self, p, generator);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap__masked_scale_out(const Tensor& self, const Tensor& mask, double scale, Tensor& out) {
    write_out(out, ops::_masked_scale(self, mask, scale));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap_native_dropout_out(const Tensor& input, double p, std::optional<bool> train, Tensor& out0, Tensor& out1) {
    const auto answers = ops::native_dropout(input, p, train);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap_native_dropout_backward_out(const Tensor& grad_output, const Tensor& mask, double scale, Tensor& out) {
    write_out(out, ops::native_dropout_backward(grad_output, mask, scale));
    return out;
}

Tensor& out_wrap__conj_physical_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::_conj_physical(self));
    return out;
}

Tensor& out_wrap_avg_pool1d_out(const Tensor& self, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, bool ceil_mode, bool count_include_pad, Tensor& out) {
    write_out(out, ops::avg_pool1d(self, kernel_size, stride, padding, ceil_mode, count_include_pad));
    return out;
}

Tensor& out_wrap_adaptive_avg_pool1d_out(const Tensor& self, const std::vector<int64_t>& output_size, Tensor& out) {
    write_out(out, ops::adaptive_avg_pool1d(self, output_size));
    return out;
}

Tensor& out_wrap__add_relu_Scalar_out(const Tensor& self, const Scalar& other, const Scalar& alpha, Tensor& out) {
    write_out(out, ops::_add_relu(self, other, alpha));
    return out;
}

Tensor& out_wrap_add_Scalar_out(const Tensor& self, const Scalar& other, const Scalar& alpha, Tensor& out) {
    write_out(out, ops::add(self, other, alpha));
    return out;
}

Tensor& out_wrap_affine_grid_generator_out(const Tensor& theta, const std::vector<int64_t>& size, bool align_corners, Tensor& out) {
    write_out(out, ops::affine_grid_generator(theta, size, align_corners));
    return out;
}

Tensor& out_wrap__test_functorch_fallback_out(const Tensor& self, const Tensor& other, Tensor& out) {
    write_out(out, ops::_test_functorch_fallback(self, other));
    return out;
}

Tensor& out_wrap_bartlett_window_out(int64_t window_length, bool periodic, std::optional<DType> dtype, Tensor& out) {
    write_out(out, ops::bartlett_window(window_length, periodic, dtype));
    return out;
}

Tensor& out_wrap_bartlett_window_periodic_out(int64_t window_length, bool periodic, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::bartlett_window(window_length, periodic, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap_quantized_batch_norm_out(const Tensor& input, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, const Tensor& mean, const Tensor& var, double eps, double output_scale, int64_t output_zero_point, Tensor& out) {
    write_out(out, ops::quantized_batch_norm(input, weight, bias, mean, var, eps, output_scale, output_zero_point));
    return out;
}

Tensor& out_wrap_bernoulli_Tensor_out(const Tensor& self, const Tensor& p, std::optional<Generator> generator, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::bernoulli_(answer, p, generator);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_bernoulli_float_out(const Tensor& self, double p, std::optional<Generator> generator, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::bernoulli_(answer, p, generator);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_binary_cross_entropy_with_logits_out(const Tensor& self, const Tensor& target, const std::optional<Tensor>& weight, const std::optional<Tensor>& pos_weight, Tensor& out) {
    write_out(out, ops::binary_cross_entropy_with_logits(self, target, weight, pos_weight));
    return out;
}

Tensor& out_wrap_bincount_out(const Tensor& self, const std::optional<Tensor>& weights, int64_t minlength, Tensor& out) {
    write_out(out, ops::bincount(self, weights, minlength));
    return out;
}

Tensor& out_wrap_blackman_window_out(int64_t window_length, bool periodic, std::optional<DType> dtype, Tensor& out) {
    write_out(out, ops::blackman_window(window_length, periodic, dtype));
    return out;
}

Tensor& out_wrap_blackman_window_periodic_out(int64_t window_length, bool periodic, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::blackman_window(window_length, periodic, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap_block_diag_out(const std::vector<Tensor>& tensors, Tensor& out) {
    write_out(out, ops::block_diag(tensors));
    return out;
}

Tensor& out_wrap_constant_pad_nd_out(const Tensor& self, const std::vector<int64_t>& pad, const Scalar& value, Tensor& out) {
    write_out(out, ops::constant_pad_nd(self, pad, value));
    return out;
}

Tensor& out_wrap_convolution_out(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool transposed, const std::vector<int64_t>& output_padding, int64_t groups, Tensor& out) {
    write_out(out, ops::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups));
    return out;
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_convolution_backward_out(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::optional<std::vector<int64_t>>& bias_sizes, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool transposed, const std::vector<int64_t>& output_padding, int64_t groups, const std::vector<bool>& output_mask, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

Tensor& out_wrap_convolution_overrideable_out(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool transposed, const std::vector<int64_t>& output_padding, int64_t groups, Tensor& out) {
    write_out(out, ops::convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups));
    return out;
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_convolution_backward_overrideable_out(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool transposed, const std::vector<int64_t>& output_padding, int64_t groups, const std::vector<bool>& output_mask, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::convolution_backward_overrideable(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

Tensor& out_wrap__convolution_out(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool transposed, const std::vector<int64_t>& output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32, Tensor& out) {
    write_out(out, ops::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32));
    return out;
}

Tensor& out_wrap_conv_tbc_out(const Tensor& self, const Tensor& weight, const Tensor& bias, int64_t pad, Tensor& out) {
    write_out(out, ops::conv_tbc(self, weight, bias, pad));
    return out;
}

Tensor& out_wrap_copy_out(const Tensor& self, const Tensor& src, bool non_blocking, Tensor& out) {
    write_out(out, ops::copy(self, src, non_blocking));
    return out;
}

Tensor& out_wrap__copy_from_out(const Tensor& self, const Tensor& dst, bool non_blocking, Tensor& out) {
    write_out(out, ops::_copy_from(self, dst, non_blocking));
    return out;
}

Tensor& out_wrap__copy_from_and_resize_out(const Tensor& self, const Tensor& dst, Tensor& out) {
    write_out(out, ops::_copy_from_and_resize(self, dst));
    return out;
}

Tensor& out_wrap_count_nonzero_dim_IntList_out(const Tensor& self, const std::vector<int64_t>& dim, Tensor& out) {
    write_out(out, ops::count_nonzero(self, dim));
    return out;
}

Tensor& out_wrap_count_nonzero_out(const Tensor& self, const std::vector<int64_t>& dim, Tensor& out) {
    write_out(out, ops::count_nonzero(self, dim));
    return out;
}

Tensor& out_wrap_cudnn_affine_grid_generator_out(const Tensor& theta, int64_t N, int64_t C, int64_t H, int64_t W, Tensor& out) {
    write_out(out, ops::cudnn_affine_grid_generator(theta, N, C, H, W));
    return out;
}

Tensor& out_wrap_cudnn_affine_grid_generator_backward_out(const Tensor& grad, int64_t N, int64_t C, int64_t H, int64_t W, Tensor& out) {
    write_out(out, ops::cudnn_affine_grid_generator_backward(grad, N, C, H, W));
    return out;
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_cudnn_batch_norm_backward_out(const Tensor& input, const Tensor& grad_output, const Tensor& weight, const std::optional<Tensor>& running_mean, const std::optional<Tensor>& running_var, const std::optional<Tensor>& save_mean, const std::optional<Tensor>& save_var, double epsilon, const Tensor& reserveSpace, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::cudnn_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

Tensor& out_wrap_cudnn_convolution_transpose_out(const Tensor& self, const Tensor& weight, const std::vector<int64_t>& padding, const std::vector<int64_t>& output_padding, const std::vector<int64_t>& stride, const std::vector<int64_t>& dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32, Tensor& out) {
    write_out(out, ops::cudnn_convolution_transpose(self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32));
    return out;
}

Tensor& out_wrap__mps_convolution_transpose_out(const Tensor& self, const Tensor& weight, const std::vector<int64_t>& padding, const std::vector<int64_t>& output_padding, const std::vector<int64_t>& stride, const std::vector<int64_t>& dilation, int64_t groups, Tensor& out) {
    write_out(out, ops::_mps_convolution_transpose(self, weight, padding, output_padding, stride, dilation, groups));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap_mps_convolution_transpose_backward_out(const Tensor& self, const Tensor& grad_output, const Tensor& weight, const std::vector<int64_t>& padding, const std::vector<int64_t>& output_padding, const std::vector<int64_t>& stride, const std::vector<int64_t>& dilation, int64_t groups, const std::vector<bool>& output_mask, Tensor& out0, Tensor& out1) {
    const auto answers = ops::mps_convolution_transpose_backward(self, grad_output, weight, padding, output_padding, stride, dilation, groups, output_mask);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap_cudnn_convolution_relu_out(const Tensor& self, const Tensor& weight, const std::optional<Tensor>& bias, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, int64_t groups, Tensor& out) {
    write_out(out, ops::cudnn_convolution_relu(self, weight, bias, stride, padding, dilation, groups));
    return out;
}

Tensor& out_wrap_cudnn_convolution_add_relu_out(const Tensor& self, const Tensor& weight, const Tensor& z, const std::optional<Scalar>& alpha, const std::optional<Tensor>& bias, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, int64_t groups, Tensor& out) {
    write_out(out, ops::cudnn_convolution_add_relu(self, weight, z, alpha, bias, stride, padding, dilation, groups));
    return out;
}

Tensor& out_wrap_cudnn_grid_sampler_out(const Tensor& self, const Tensor& grid, Tensor& out) {
    write_out(out, ops::cudnn_grid_sampler(self, grid));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap_cudnn_grid_sampler_backward_out(const Tensor& self, const Tensor& grid, const Tensor& grad_output, Tensor& out0, Tensor& out1) {
    const auto answers = ops::cudnn_grid_sampler_backward(self, grid, grad_output);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

std::tuple<Tensor, Tensor> out_wrap__ctc_loss_out(const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths, int64_t blank, bool zero_infinity, Tensor& out0, Tensor& out1) {
    const auto answers = ops::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

std::tuple<Tensor, Tensor> out_wrap__ctc_loss_Tensor_out(const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths, int64_t blank, bool zero_infinity, Tensor& out0, Tensor& out1) {
    const auto answers = ops::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap__ctc_loss_backward_out(const Tensor& grad_output, const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths, const Tensor& neg_log_likelihood, const Tensor& log_alpha, int64_t blank, bool zero_infinity, Tensor& out) {
    write_out(out, ops::_ctc_loss_backward(grad_output, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity));
    return out;
}

Tensor& out_wrap_diag_embed_out(const Tensor& self, int64_t offset, int64_t dim1, int64_t dim2, Tensor& out) {
    write_out(out, ops::diag_embed(self, offset, dim1, dim2));
    return out;
}

Tensor& out_wrap_diagonal_backward_out(const Tensor& grad_output, const std::vector<int64_t>& input_sizes, int64_t offset, int64_t dim1, int64_t dim2, Tensor& out) {
    write_out(out, ops::diagonal_backward(grad_output, input_sizes, offset, dim1, dim2));
    return out;
}

Tensor& out_wrap_div_Scalar_out(const Tensor& self, const Scalar& other, Tensor& out) {
    write_out(out, ops::div(self, other));
    return out;
}

Tensor& out_wrap_div_Scalar_mode_out(const Tensor& self, const Scalar& other, const std::optional<std::string>& rounding_mode, Tensor& out) {
    write_out(out, ops::div(self, other, rounding_mode));
    return out;
}

Tensor& out_wrap_embedding_out(const Tensor& weight, const Tensor& indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse, Tensor& out) {
    write_out(out, ops::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse));
    return out;
}

Tensor& out_wrap_embedding_dense_backward_out(const Tensor& grad_output, const Tensor& indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, Tensor& out) {
    write_out(out, ops::embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq));
    return out;
}

Tensor& out_wrap_embedding_renorm_out(const Tensor& self, const Tensor& indices, double max_norm, double norm_type, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::embedding_renorm_(answer, indices, max_norm, norm_type);
    write_out(out, answer);
    return out;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> out_wrap__embedding_bag_forward_only_out(const Tensor& weight, const Tensor& indices, const Tensor& offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const std::optional<Tensor>& per_sample_weights, bool include_last_offset, int64_t padding_idx, Tensor& out0, Tensor& out1, Tensor& out2, Tensor& out3) {
    const auto answers = ops::_embedding_bag_forward_only(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    write_out(out3, std::get<3>(answers));
    return {out0, out1, out2, out3};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> out_wrap__embedding_bag_out(const Tensor& weight, const Tensor& indices, const Tensor& offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const std::optional<Tensor>& per_sample_weights, bool include_last_offset, int64_t padding_idx, Tensor& out0, Tensor& out1, Tensor& out2, Tensor& out3) {
    const auto answers = ops::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    write_out(out3, std::get<3>(answers));
    return {out0, out1, out2, out3};
}

Tensor& out_wrap__embedding_bag_dense_backward_out(const Tensor& grad_output, const Tensor& indices, const Tensor& offset2bag, const Tensor& bag_size, const Tensor& maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const std::optional<Tensor>& per_sample_weights, int64_t padding_idx, Tensor& out) {
    write_out(out, ops::_embedding_bag_dense_backward(grad_output, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx));
    return out;
}

Tensor& out_wrap__embedding_bag_per_sample_weights_backward_out(const Tensor& grad_output, const Tensor& weight, const Tensor& indices, const Tensor& offsets, const Tensor& offset2bag, int64_t mode, int64_t padding_idx, Tensor& out) {
    write_out(out, ops::_embedding_bag_per_sample_weights_backward(grad_output, weight, indices, offsets, offset2bag, mode, padding_idx));
    return out;
}

Tensor& out_wrap_empty_permuted_out(const std::vector<int64_t>& size, const std::vector<int64_t>& physical_layout, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::empty_permuted(size, physical_layout, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap_new_empty_out(const Tensor& self, const std::vector<int64_t>& size, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::new_empty(self, size, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap_new_empty_strided_out(const Tensor& self, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::new_empty_strided(self, size, stride, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap_new_full_out(const Tensor& self, const std::vector<int64_t>& size, const Scalar& fill_value, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::new_full(self, size, fill_value, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap_new_zeros_out(const Tensor& self, const std::vector<int64_t>& size, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::new_zeros(self, size, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap_new_ones_out(const Tensor& self, const std::vector<int64_t>& size, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::new_ones(self, size, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap__empty_affine_quantized_out(const std::vector<int64_t>& size, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, double scale, int64_t zero_point, std::optional<int64_t> memory_format, Tensor& out) {
    write_out(out, ops::_empty_affine_quantized(size, dtype, layout, device, pin_memory, scale, zero_point, memory_format));
    return out;
}

Tensor& out_wrap__empty_per_channel_affine_quantized_out(const std::vector<int64_t>& size, const Tensor& scales, const Tensor& zero_points, int64_t axis, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<int64_t> memory_format, Tensor& out) {
    write_out(out, ops::_empty_per_channel_affine_quantized(size, scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format));
    return out;
}

Tensor& out_wrap_resize_out(const Tensor& self, const std::vector<int64_t>& size, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::resize_(answer, size);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap__resize_output_out(const Tensor& self, const std::vector<int64_t>& size, Device device, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::_resize_output_(answer, size, device);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_empty_quantized_out(const std::vector<int64_t>& size, const Tensor& qtensor, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<int64_t> memory_format, Tensor& out) {
    write_out(out, ops::empty_quantized(size, qtensor, dtype, layout, device, pin_memory, memory_format));
    return out;
}

Tensor& out_wrap_empty_like_out(const Tensor& self, DType dtype, std::optional<Device> device, Tensor& out) {
    write_out(out, ops::empty_like(self, dtype, device, false));
    return out;
}

Tensor& out_wrap_empty_strided_out(const std::vector<int64_t>& size, const std::vector<int64_t>& stride, std::optional<DType> dtype, std::optional<Device> device, bool pin_memory, Tensor& out) {
    write_out(out, ops::empty_strided(size, stride, dtype, device, pin_memory));
    return out;
}

Tensor& out_wrap_fill_Scalar_out(const Tensor& self, const Scalar& value, Tensor& out) {
    write_out(out, ops::fill(self, value));
    return out;
}

Tensor& out_wrap_fill_Tensor_out(const Tensor& self, const Tensor& value, Tensor& out) {
    write_out(out, ops::fill(self, value));
    return out;
}

Tensor& out_wrap_floor_divide_Scalar_out(const Tensor& self, const Scalar& other, Tensor& out) {
    write_out(out, ops::floor_divide(self, other));
    return out;
}

Tensor& out_wrap_full_like_out(const Tensor& self, const Scalar& fill_value, DType dtype, std::optional<Device> device, Tensor& out) {
    write_out(out, ops::full_like(self, fill_value, dtype, device, false));
    return out;
}

Tensor& out_wrap_from_file_out(const std::string& filename, std::optional<bool> shared, std::optional<int64_t> size, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::from_file(filename, shared, size, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap_grid_sampler_2d_out(const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, Tensor& out) {
    write_out(out, ops::grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap_grid_sampler_2d_backward_out(const Tensor& grad_output, const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, const std::vector<bool>& output_mask, Tensor& out0, Tensor& out1) {
    const auto answers = ops::grid_sampler_2d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap__grid_sampler_2d_cpu_fallback_out(const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, Tensor& out) {
    write_out(out, ops::_grid_sampler_2d_cpu_fallback(input, grid, interpolation_mode, padding_mode, align_corners));
    return out;
}

Tensor& out_wrap_grid_sampler_3d_out(const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, Tensor& out) {
    write_out(out, ops::grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap_grid_sampler_3d_backward_out(const Tensor& grad_output, const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, const std::vector<bool>& output_mask, Tensor& out0, Tensor& out1) {
    const auto answers = ops::grid_sampler_3d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap_hann_window_out(int64_t window_length, bool periodic, std::optional<DType> dtype, Tensor& out) {
    write_out(out, ops::hann_window(window_length, periodic, dtype));
    return out;
}

Tensor& out_wrap_hann_window_periodic_out(int64_t window_length, bool periodic, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::hann_window(window_length, periodic, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap_hamming_window_out(int64_t window_length, bool periodic, double alpha, double beta, std::optional<DType> dtype, Tensor& out) {
    write_out(out, ops::hamming_window(window_length, periodic, alpha, beta, dtype));
    return out;
}

Tensor& out_wrap_hamming_window_periodic_out(int64_t window_length, bool periodic, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::hamming_window(window_length, periodic, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap_hamming_window_periodic_alpha_out(int64_t window_length, bool periodic, double alpha, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::hamming_window(window_length, periodic, alpha, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap_hamming_window_periodic_alpha_beta_out(int64_t window_length, bool periodic, double alpha, double beta, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::hamming_window(window_length, periodic, alpha, beta, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap_kaiser_window_out(int64_t window_length, std::optional<DType> dtype, std::optional<Device> device, bool pin_memory, Tensor& out) {
    write_out(out, ops::kaiser_window(window_length, dtype, device, pin_memory));
    return out;
}

Tensor& out_wrap_kaiser_window_periodic_out(int64_t window_length, bool periodic, std::optional<DType> dtype, std::optional<Device> device, bool pin_memory, Tensor& out) {
    write_out(out, ops::kaiser_window(window_length, periodic, dtype, device, pin_memory));
    return out;
}

Tensor& out_wrap_kaiser_window_beta_out(int64_t window_length, bool periodic, double beta, std::optional<DType> dtype, std::optional<Device> device, bool pin_memory, Tensor& out) {
    write_out(out, ops::kaiser_window(window_length, periodic, beta, dtype, device, pin_memory));
    return out;
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_native_group_norm_out(const Tensor& input, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::native_group_norm(input, weight, bias, N, C, HxW, group, eps);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_native_group_norm_backward_out(const Tensor& grad_out, const Tensor& input, const Tensor& mean, const Tensor& rstd, const std::optional<Tensor>& weight, int64_t N, int64_t C, int64_t HxW, int64_t group, const std::vector<bool>& output_mask, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::native_group_norm_backward(grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

Tensor& out_wrap_index_put_out(const Tensor& self, const std::vector<std::optional<Tensor>>& indices, const Tensor& values, bool accumulate, Tensor& out) {
    write_out(out, ops::index_put(self, indices, values, accumulate));
    return out;
}

Tensor& out_wrap__index_put_impl_out(const Tensor& self, const std::vector<std::optional<Tensor>>& indices, const Tensor& values, bool accumulate, bool unsafe, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::_index_put_impl_(answer, indices, values, accumulate, unsafe);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_isnan_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::isnan(self));
    return out;
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_native_layer_norm_out(const Tensor& input, const std::vector<int64_t>& normalized_shape, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, double eps, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::native_layer_norm(input, normalized_shape, weight, bias, eps);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_native_layer_norm_backward_out(const Tensor& grad_out, const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& mean, const Tensor& rstd, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, const std::vector<bool>& output_mask, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_linear_backward_out(const Tensor& self, const Tensor& grad_output, const Tensor& weight, const std::vector<bool>& output_mask, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::linear_backward(self, grad_output, weight, output_mask);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

Tensor& out_wrap_mkldnn_linear_out(const Tensor& self, const Tensor& weight, const std::optional<Tensor>& bias, Tensor& out) {
    write_out(out, ops::mkldnn_linear(self, weight, bias));
    return out;
}

Tensor& out_wrap_mkldnn_linear_backward_input_out(const std::vector<int64_t>& input_size, const Tensor& grad_output, const Tensor& weight, Tensor& out) {
    write_out(out, ops::mkldnn_linear_backward_input(input_size, grad_output, weight));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap_mkldnn_linear_backward_weights_out(const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined, Tensor& out0, Tensor& out1) {
    const auto answers = ops::mkldnn_linear_backward_weights(grad_output, input, weight, bias_defined);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_mkldnn_linear_backward_out(const Tensor& self, const Tensor& grad_output, const Tensor& weight, const std::vector<bool>& output_mask, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::mkldnn_linear_backward(self, grad_output, weight, output_mask);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

std::tuple<Tensor, Tensor> out_wrap_matmul_backward_out(const Tensor& grad, const Tensor& self, const Tensor& other, const std::vector<bool>& mask, Tensor& out0, Tensor& out1) {
    const auto answers = ops::matmul_backward(grad, self, other, mask);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

std::tuple<Tensor, Tensor> out_wrap__aminmax_out(const Tensor& self, Tensor& out0, Tensor& out1) {
    const auto answers = ops::_aminmax(self);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

std::tuple<Tensor, Tensor> out_wrap__aminmax_dim_out(const Tensor& self, int64_t dim, bool keepdim, Tensor& out0, Tensor& out1) {
    const auto answers = ops::_aminmax(self, dim, keepdim);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap_max_pool2d_backward_out(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceil_mode, Tensor& out) {
    write_out(out, ops::max_pool2d_backward(grad_output, input, kernel_size, stride, padding, dilation, ceil_mode));
    return out;
}

Tensor& out_wrap_mkldnn_max_pool2d_out(const Tensor& self, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceil_mode, Tensor& out) {
    write_out(out, ops::mkldnn_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode));
    return out;
}

Tensor& out_wrap_mkldnn_max_pool2d_backward_out(const Tensor& grad_output, const Tensor& output, const Tensor& input, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceil_mode, Tensor& out) {
    write_out(out, ops::mkldnn_max_pool2d_backward(grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode));
    return out;
}

Tensor& out_wrap_mkldnn_max_pool3d_out(const Tensor& self, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceil_mode, Tensor& out) {
    write_out(out, ops::mkldnn_max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode));
    return out;
}

Tensor& out_wrap_mkldnn_max_pool3d_backward_out(const Tensor& grad_output, const Tensor& output, const Tensor& input, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceil_mode, Tensor& out) {
    write_out(out, ops::mkldnn_max_pool3d_backward(grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode));
    return out;
}

Tensor& out_wrap_quantized_max_pool1d_out(const Tensor& self, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceil_mode, Tensor& out) {
    write_out(out, ops::quantized_max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode));
    return out;
}

Tensor& out_wrap_quantized_max_pool2d_out(const Tensor& self, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceil_mode, Tensor& out) {
    write_out(out, ops::quantized_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode));
    return out;
}

Tensor& out_wrap_quantized_max_pool3d_out(const Tensor& self, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceil_mode, Tensor& out) {
    write_out(out, ops::quantized_max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode));
    return out;
}

Tensor& out_wrap_median_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::median(self));
    return out;
}

Tensor& out_wrap_nanmedian_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::nanmedian(self));
    return out;
}

Tensor& out_wrap__mps_convolution_out(const Tensor& self, const Tensor& weight, const std::optional<Tensor>& bias, const std::vector<int64_t>& padding, const std::vector<int64_t>& stride, const std::vector<int64_t>& dilation, int64_t groups, Tensor& out) {
    write_out(out, ops::_mps_convolution(self, weight, bias, padding, stride, dilation, groups));
    return out;
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_mps_convolution_backward_out(const Tensor& self, const Tensor& grad_output, const Tensor& weight, const std::vector<int64_t>& padding, const std::vector<int64_t>& stride, const std::vector<int64_t>& dilation, int64_t groups, const std::vector<bool>& output_mask, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::mps_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, output_mask);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

Tensor& out_wrap_mkldnn_convolution_out(const Tensor& self, const Tensor& weight, const std::optional<Tensor>& bias, const std::vector<int64_t>& padding, const std::vector<int64_t>& stride, const std::vector<int64_t>& dilation, int64_t groups, Tensor& out) {
    write_out(out, ops::mkldnn_convolution(self, weight, bias, padding, stride, dilation, groups));
    return out;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> out_wrap_mkldnn_rnn_layer_out(const Tensor& input, const Tensor& weight0, const Tensor& weight1, const Tensor& weight2, const Tensor& weight3, const Tensor& hx_, const Tensor& cx_, bool reverse, const std::vector<int64_t>& batch_sizes, int64_t mode, int64_t hidden_size, int64_t num_layers, bool has_biases, bool bidirectional, bool batch_first, bool train, Tensor& out0, Tensor& out1, Tensor& out2, Tensor& out3) {
    const auto answers = ops::mkldnn_rnn_layer(input, weight0, weight1, weight2, weight3, hx_, cx_, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    write_out(out3, std::get<3>(answers));
    return {out0, out1, out2, out3};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> out_wrap_mkldnn_rnn_layer_backward_out(const Tensor& input, const Tensor& weight1, const Tensor& weight2, const Tensor& weight3, const Tensor& weight4, const Tensor& hx_, const Tensor& cx_tmp, const Tensor& output, const Tensor& hy_, const Tensor& cy_, const std::optional<Tensor>& grad_output, const std::optional<Tensor>& grad_hy, const std::optional<Tensor>& grad_cy, bool reverse, int64_t mode, int64_t hidden_size, int64_t num_layers, bool has_biases, bool train, bool bidirectional, const std::vector<int64_t>& batch_sizes, bool batch_first, const Tensor& workspace, Tensor& out0, Tensor& out1, Tensor& out2, Tensor& out3, Tensor& out4, Tensor& out5, Tensor& out6) {
    const auto answers = ops::mkldnn_rnn_layer_backward(input, weight1, weight2, weight3, weight4, hx_, cx_tmp, output, hy_, cy_, grad_output, grad_hy, grad_cy, reverse, mode, hidden_size, num_layers, has_biases, train, bidirectional, batch_sizes, batch_first, workspace);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    write_out(out3, std::get<3>(answers));
    write_out(out4, std::get<4>(answers));
    write_out(out5, std::get<5>(answers));
    write_out(out6, std::get<6>(answers));
    return {out0, out1, out2, out3, out4, out5, out6};
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_miopen_batch_norm_out(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias, const std::optional<Tensor>& running_mean, const std::optional<Tensor>& running_var, bool training, double exponential_average_factor, double epsilon, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::miopen_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_miopen_batch_norm_backward_out(const Tensor& input, const Tensor& grad_output, const Tensor& weight, const std::optional<Tensor>& running_mean, const std::optional<Tensor>& running_var, const std::optional<Tensor>& save_mean, const std::optional<Tensor>& save_var, double epsilon, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::miopen_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

Tensor& out_wrap_miopen_convolution_out(const Tensor& self, const Tensor& weight, const std::optional<Tensor>& bias, const std::vector<int64_t>& padding, const std::vector<int64_t>& stride, const std::vector<int64_t>& dilation, int64_t groups, bool benchmark, bool deterministic, Tensor& out) {
    write_out(out, ops::miopen_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic));
    return out;
}

Tensor& out_wrap_miopen_convolution_transpose_out(const Tensor& self, const Tensor& weight, const std::optional<Tensor>& bias, const std::vector<int64_t>& padding, const std::vector<int64_t>& output_padding, const std::vector<int64_t>& stride, const std::vector<int64_t>& dilation, int64_t groups, bool benchmark, bool deterministic, Tensor& out) {
    write_out(out, ops::miopen_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic));
    return out;
}

Tensor& out_wrap_miopen_depthwise_convolution_out(const Tensor& self, const Tensor& weight, const std::optional<Tensor>& bias, const std::vector<int64_t>& padding, const std::vector<int64_t>& stride, const std::vector<int64_t>& dilation, int64_t groups, bool benchmark, bool deterministic, Tensor& out) {
    write_out(out, ops::miopen_depthwise_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic));
    return out;
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> out_wrap_miopen_rnn_out(const Tensor& input, const std::vector<Tensor>& weight, int64_t weight_stride0, const Tensor& hx, const std::optional<Tensor>& cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, const std::vector<int64_t>& batch_sizes, const std::optional<Tensor>& dropout_state, Tensor& out0, Tensor& out1, Tensor& out2, Tensor& out3, Tensor& out4) {
    const auto answers = ops::miopen_rnn(input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    write_out(out3, std::get<3>(answers));
    write_out(out4, std::get<4>(answers));
    return {out0, out1, out2, out3, out4};
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> out_wrap_miopen_rnn_backward_out(const Tensor& input, const std::vector<Tensor>& weight, int64_t weight_stride0, const Tensor& weight_buf, const Tensor& hx, const std::optional<Tensor>& cx, const Tensor& output, const std::optional<Tensor>& grad_output, const std::optional<Tensor>& grad_hy, const std::optional<Tensor>& grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, const std::vector<int64_t>& batch_sizes, const std::optional<Tensor>& dropout_state, const Tensor& reserve, const std::vector<bool>& output_mask, Tensor& out0, Tensor& out1, Tensor& out2, std::vector<Tensor> out3) {
    const auto answers = ops::miopen_rnn_backward(input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    for (size_t item = 0;
         item < std::get<3>(answers).size() && item < out3.size();
         ++item) {
        write_out(out3[item], std::get<3>(answers)[item]);
    }
    return {out0, out1, out2, out3};
}

std::tuple<Tensor, Tensor> out_wrap_miopen_ctc_loss_out(const Tensor& log_probs, const Tensor& targets, const std::vector<int64_t>& input_lengths, const std::vector<int64_t>& target_lengths, int64_t blank, bool deterministic, bool zero_infinity, Tensor& out0, Tensor& out1) {
    const auto answers = ops::miopen_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap__sparse_sparse_matmul_out(const Tensor& self, const Tensor& other, Tensor& out) {
    write_out(out, ops::_sparse_sparse_matmul(self, other));
    return out;
}

Tensor& out_wrap_mul_Scalar_out(const Tensor& self, const Scalar& other, Tensor& out) {
    write_out(out, ops::mul(self, other));
    return out;
}

std::tuple<Tensor, Tensor, Tensor> out_wrap__native_batch_norm_legit_no_training_out(const Tensor& input, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, const Tensor& running_mean, const Tensor& running_var, double momentum, double eps, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::_native_batch_norm_legit_no_training(input, weight, bias, running_mean, running_var, momentum, eps);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

std::tuple<Tensor, Tensor> out_wrap_batch_norm_stats_out(const Tensor& input, double eps, Tensor& out0, Tensor& out1) {
    const auto answers = ops::batch_norm_stats(input, eps);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

std::tuple<Tensor, Tensor> out_wrap_batch_norm_gather_stats_out(const Tensor& input, const Tensor& mean, const Tensor& invstd, const std::optional<Tensor>& running_mean, const std::optional<Tensor>& running_var, double momentum, double eps, int64_t count, Tensor& out0, Tensor& out1) {
    const auto answers = ops::batch_norm_gather_stats(input, mean, invstd, running_mean, running_var, momentum, eps, count);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

std::tuple<Tensor, Tensor> out_wrap_batch_norm_gather_stats_with_counts_out(const Tensor& input, const Tensor& mean, const Tensor& invstd, const std::optional<Tensor>& running_mean, const std::optional<Tensor>& running_var, double momentum, double eps, const Tensor& counts, Tensor& out0, Tensor& out1) {
    const auto answers = ops::batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_native_batch_norm_backward_out(const Tensor& grad_out, const Tensor& input, const std::optional<Tensor>& weight, const std::optional<Tensor>& running_mean, const std::optional<Tensor>& running_var, const std::optional<Tensor>& save_mean, const std::optional<Tensor>& save_invstd, bool train, double eps, const std::vector<bool>& output_mask, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::native_batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> out_wrap_batch_norm_backward_reduce_out(const Tensor& grad_out, const Tensor& input, const Tensor& mean, const Tensor& invstd, const std::optional<Tensor>& weight, bool input_g, bool weight_g, bool bias_g, Tensor& out0, Tensor& out1, Tensor& out2, Tensor& out3) {
    const auto answers = ops::batch_norm_backward_reduce(grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    write_out(out3, std::get<3>(answers));
    return {out0, out1, out2, out3};
}

Tensor& out_wrap_batch_norm_backward_elemt_out(const Tensor& grad_out, const Tensor& input, const Tensor& mean, const Tensor& invstd, const std::optional<Tensor>& weight, const Tensor& sum_dy, const Tensor& sum_dy_xmu, const Tensor& count, Tensor& out) {
    write_out(out, ops::batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap_batch_norm_update_stats_out(const Tensor& input, const std::optional<Tensor>& running_mean, const std::optional<Tensor>& running_var, double momentum, Tensor& out0, Tensor& out1) {
    const auto answers = ops::batch_norm_update_stats(input, running_mean, running_var, momentum);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap__nnpack_spatial_convolution_out(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias, const std::vector<int64_t>& padding, const std::vector<int64_t>& stride, Tensor& out) {
    write_out(out, ops::_nnpack_spatial_convolution(input, weight, bias, padding, stride));
    return out;
}

Tensor& out_wrap_ones_like_out(const Tensor& self, DType dtype, std::optional<Device> device, Tensor& out) {
    write_out(out, ops::ones_like(self, dtype, device, false));
    return out;
}

Tensor& out_wrap__euclidean_dist_out(const Tensor& x1, const Tensor& x2, Tensor& out) {
    write_out(out, ops::_euclidean_dist(x1, x2));
    return out;
}

Tensor& out_wrap__cdist_forward_out(const Tensor& x1, const Tensor& x2, double p, std::optional<int64_t> compute_mode, Tensor& out) {
    write_out(out, ops::_cdist_forward(x1, x2, p, compute_mode));
    return out;
}

Tensor& out_wrap__cdist_backward_out(const Tensor& grad_output, const Tensor& x1, const Tensor& x2, double p, const Tensor& cdist, Tensor& out) {
    write_out(out, ops::_cdist_backward(grad_output, x1, x2, p, cdist));
    return out;
}

Tensor& out_wrap__pdist_forward_out(const Tensor& self, double p, Tensor& out) {
    write_out(out, ops::_pdist_forward(self, p));
    return out;
}

Tensor& out_wrap__pdist_backward_out(const Tensor& grad_output, const Tensor& self, double p, const Tensor& pdist, Tensor& out) {
    write_out(out, ops::_pdist_backward(grad_output, self, p, pdist));
    return out;
}

Tensor& out_wrap_pixel_shuffle_out(const Tensor& self, int64_t upscale_factor, Tensor& out) {
    write_out(out, ops::pixel_shuffle(self, upscale_factor));
    return out;
}

Tensor& out_wrap_pixel_unshuffle_out(const Tensor& self, int64_t downscale_factor, Tensor& out) {
    write_out(out, ops::pixel_unshuffle(self, downscale_factor));
    return out;
}

Tensor& out_wrap_channel_shuffle_out(const Tensor& self, int64_t groups, Tensor& out) {
    write_out(out, ops::channel_shuffle(self, groups));
    return out;
}

Tensor& out_wrap__pin_memory_out(const Tensor& self, std::optional<Device> device, Tensor& out) {
    write_out(out, ops::_pin_memory(self, device));
    return out;
}

Tensor& out_wrap_scalar_tensor_out(const Scalar& s, std::optional<DType> dtype, std::optional<Device> device, bool pin_memory, Tensor& out) {
    write_out(out, ops::scalar_tensor(s, dtype, device, pin_memory));
    return out;
}

Tensor& out_wrap_rand_like_out(const Tensor& self, DType dtype, std::optional<Device> device, Tensor& out) {
    write_out(out, ops::rand_like(self, dtype, device, false));
    return out;
}

Tensor& out_wrap_rand_like_generator_out(const Tensor& self, std::optional<Generator> generator, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<int64_t> memory_format, Tensor& out) {
    write_out(out, ops::rand_like(self, generator, dtype, layout, device, pin_memory, memory_format));
    return out;
}

Tensor& out_wrap_randint_like_out(const Tensor& self, int64_t low, int64_t high, DType dtype, std::optional<Device> device, Tensor& out) {
    write_out(out, ops::randint_like(self, low, high, dtype, device, false));
    return out;
}

Tensor& out_wrap_randint_like_generator_out(const Tensor& self, int64_t high, std::optional<Generator> generator, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<int64_t> memory_format, Tensor& out) {
    write_out(out, ops::randint_like(self, high, generator, dtype, layout, device, pin_memory, memory_format));
    return out;
}

Tensor& out_wrap_randint_like_Tensor_out(const Tensor& self, const Tensor& high, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<int64_t> memory_format, Tensor& out) {
    write_out(out, ops::randint_like(self, high, dtype, layout, device, pin_memory, memory_format));
    return out;
}

Tensor& out_wrap_randint_like_Tensor_generator_out(const Tensor& self, const Tensor& high, std::optional<Generator> generator, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<int64_t> memory_format, Tensor& out) {
    write_out(out, ops::randint_like(self, high, generator, dtype, layout, device, pin_memory, memory_format));
    return out;
}

Tensor& out_wrap_randint_like_low_dtype_out(const Tensor& self, int64_t low, int64_t high, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<int64_t> memory_format, Tensor& out) {
    write_out(out, ops::randint_like(self, low, high, dtype, layout, device, pin_memory, memory_format));
    return out;
}

Tensor& out_wrap_randint_like_low_generator_dtype_out(const Tensor& self, int64_t low, int64_t high, std::optional<Generator> generator, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<int64_t> memory_format, Tensor& out) {
    write_out(out, ops::randint_like(self, low, high, generator, dtype, layout, device, pin_memory, memory_format));
    return out;
}

Tensor& out_wrap_randn_like_out(const Tensor& self, DType dtype, std::optional<Device> device, Tensor& out) {
    write_out(out, ops::randn_like(self, dtype, device, false));
    return out;
}

Tensor& out_wrap_randn_like_generator_out(const Tensor& self, std::optional<Generator> generator, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<int64_t> memory_format, Tensor& out) {
    write_out(out, ops::randn_like(self, generator, dtype, layout, device, pin_memory, memory_format));
    return out;
}

Tensor& out_wrap_repeat_out(const Tensor& self, const std::vector<int64_t>& repeats, Tensor& out) {
    write_out(out, ops::repeat(self, repeats));
    return out;
}

Tensor& out_wrap_repeat_interleave_Tensor_out(const Tensor& repeats, std::optional<int64_t> output_size, Tensor& out) {
    write_out(out, ops::repeat_interleave(repeats, output_size));
    return out;
}

Tensor& out_wrap__mkldnn_reshape_out(const Tensor& self, const std::vector<int64_t>& shape, Tensor& out) {
    write_out(out, ops::_mkldnn_reshape(self, shape));
    return out;
}

Tensor& out_wrap_prelu_out(const Tensor& self, const Tensor& weight, Tensor& out) {
    write_out(out, ops::prelu(self, weight));
    return out;
}

Tensor& out_wrap_select_backward_out(const Tensor& grad_output, const Tensor& self, int64_t dim, int64_t index, Tensor& out) {
    write_out(out, ops::select_backward(grad_output, self, dim, index));
    return out;
}

Tensor& out_wrap_slice_backward_out(const Tensor& grad_output, const Tensor& self, int64_t dim, std::optional<int64_t> start, std::optional<int64_t> end, int64_t step, Tensor& out) {
    write_out(out, ops::slice_backward(grad_output, self, dim, start, end, step));
    return out;
}

Tensor& out_wrap_slice_scatter_out(const Tensor& self, const Tensor& src, int64_t dim, std::optional<int64_t> start, std::optional<int64_t> end, int64_t step, Tensor& out) {
    write_out(out, ops::slice_scatter(self, src, dim, start, end, step));
    return out;
}

Tensor& out_wrap_select_scatter_out(const Tensor& self, const Tensor& src, int64_t dim, int64_t index, Tensor& out) {
    write_out(out, ops::select_scatter(self, src, dim, index));
    return out;
}

Tensor& out_wrap_diagonal_scatter_out(const Tensor& self, const Tensor& src, int64_t offset, int64_t dim1, int64_t dim2, Tensor& out) {
    write_out(out, ops::diagonal_scatter(self, src, offset, dim1, dim2));
    return out;
}

Tensor& out_wrap_as_strided_scatter_out(const Tensor& self, const Tensor& src, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, std::optional<int64_t> storage_offset, Tensor& out) {
    write_out(out, ops::as_strided_scatter(self, src, size, stride, storage_offset));
    return out;
}

void out_wrap_unsafe_split_Tensor_out(const Tensor& self, int64_t split_size, int64_t dim, std::vector<Tensor> out) {
    const auto answers = ops::unsafe_split(self, split_size, dim);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap_unsafe_split_with_sizes_out(const Tensor& self, const std::vector<int64_t>& split_sizes, int64_t dim, std::vector<Tensor> out) {
    const auto answers = ops::unsafe_split_with_sizes(self, split_sizes, dim);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

Tensor& out_wrap_sum_out(const Tensor& self, DType dtype, Tensor& out) {
    write_out(out, ops::sum(self, dtype));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap_std_mean_correction_out(const Tensor& self, const std::optional<std::vector<int64_t>>& dim, const std::optional<Scalar>& correction, bool keepdim, Tensor& out0, Tensor& out1) {
    const auto answers = ops::std_mean(self, dim, correction, keepdim);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap_prod_out(const Tensor& self, DType dtype, Tensor& out) {
    write_out(out, ops::prod(self, dtype));
    return out;
}

Tensor& out_wrap__mkldnn_transpose_out(const Tensor& self, int64_t dim0, int64_t dim1, Tensor& out) {
    write_out(out, ops::_mkldnn_transpose(self, dim0, dim1));
    return out;
}

Tensor& out_wrap_flip_out(const Tensor& self, const std::vector<int64_t>& dims, Tensor& out) {
    write_out(out, ops::flip(self, dims));
    return out;
}

Tensor& out_wrap_roll_out(const Tensor& self, const std::vector<int64_t>& shifts, const std::vector<int64_t>& dims, Tensor& out) {
    write_out(out, ops::roll(self, shifts, dims));
    return out;
}

Tensor& out_wrap_rot90_out(const Tensor& self, int64_t k, const std::vector<int64_t>& dims, Tensor& out) {
    write_out(out, ops::rot90(self, k, dims));
    return out;
}

std::tuple<Tensor, Tensor, Tensor> out_wrap__transform_bias_rescale_qkv_out(const Tensor& qkv, const Tensor& qkv_bias, int64_t num_heads, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::_transform_bias_rescale_qkv(qkv, qkv_bias, num_heads);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

Tensor& out_wrap__nested_tensor_from_mask_out(const Tensor& t, const Tensor& mask, bool mask_check, Tensor& out) {
    write_out(out, ops::_nested_tensor_from_mask(t, mask, mask_check));
    return out;
}

Tensor& out_wrap__nested_from_padded_out(const Tensor& padded, const Tensor& cpu_nested_shape_example, bool fuse_transform_0213, Tensor& out) {
    write_out(out, ops::_nested_from_padded(padded, cpu_nested_shape_example, fuse_transform_0213));
    return out;
}

Tensor& out_wrap__nested_tensor_size_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::_nested_tensor_size(self));
    return out;
}

Tensor& out_wrap__nested_tensor_strides_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::_nested_tensor_strides(self));
    return out;
}

Tensor& out_wrap__nested_tensor_storage_offsets_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::_nested_tensor_storage_offsets(self));
    return out;
}

Tensor& out_wrap__nested_from_padded_and_nested_example_out(const Tensor& padded, const Tensor& nt_example, Tensor& out) {
    write_out(out, ops::_nested_from_padded_and_nested_example(padded, nt_example));
    return out;
}

Tensor& out_wrap__nested_view_from_buffer_copy_out(const Tensor& self, const Tensor& nested_size, const Tensor& nested_strides, const Tensor& offsets, Tensor& out) {
    write_out(out, ops::_nested_view_from_buffer_copy(self, nested_size, nested_strides, offsets));
    return out;
}

Tensor& out_wrap__nested_view_from_jagged_copy_out(const Tensor& self, const Tensor& offsets, const Tensor& dummy, const std::optional<Tensor>& lengths, int64_t ragged_idx, const std::optional<Tensor>& min_seqlen, const std::optional<Tensor>& max_seqlen, Tensor& out) {
    write_out(out, ops::_nested_view_from_jagged_copy(self, offsets, dummy, lengths, ragged_idx, min_seqlen, max_seqlen));
    return out;
}

Tensor& out_wrap__nested_get_values_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::_nested_get_values_copy(self));
    return out;
}

Tensor& out_wrap__trilinear_out(const Tensor& i1, const Tensor& i2, const Tensor& i3, const std::vector<int64_t>& expand1, const std::vector<int64_t>& expand2, const std::vector<int64_t>& expand3, const std::vector<int64_t>& sumdim, int64_t unroll_dim, Tensor& out) {
    write_out(out, ops::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap__unique_out(const Tensor& self, bool sorted, bool return_inverse, Tensor& out0, Tensor& out1) {
    const auto answers = ops::_unique(self, sorted, return_inverse);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_unique_dim_out(const Tensor& self, int64_t dim, bool sorted, bool return_inverse, bool return_counts, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::unique_dim(self, dim, sorted, return_inverse, return_counts);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_unique_consecutive_out(const Tensor& self, bool return_inverse, bool return_counts, std::optional<int64_t> dim, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::unique_consecutive(self, return_inverse, return_counts, dim);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

std::tuple<Tensor, Tensor, Tensor> out_wrap_unique_dim_consecutive_out(const Tensor& self, int64_t dim, bool return_inverse, bool return_counts, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::unique_dim_consecutive(self, dim, return_inverse, return_counts);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

std::tuple<Tensor, Tensor, Tensor> out_wrap__unique2_out(const Tensor& self, bool sorted, bool return_inverse, bool return_counts, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::_unique2(self, sorted, return_inverse, return_counts);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

Tensor& out_wrap__unsafe_view_out(const Tensor& self, const std::vector<int64_t>& size, Tensor& out) {
    write_out(out, ops::_unsafe_view(self, size));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap_var_mean_correction_out(const Tensor& self, const std::optional<std::vector<int64_t>>& dim, const std::optional<Scalar>& correction, bool keepdim, Tensor& out0, Tensor& out1) {
    const auto answers = ops::var_mean(self, dim, correction, keepdim);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

std::tuple<Tensor, Tensor> out_wrap__weight_norm_interface_out(const Tensor& v, const Tensor& g, int64_t dim, Tensor& out0, Tensor& out1) {
    const auto answers = ops::_weight_norm_interface(v, g, dim);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

std::tuple<Tensor, Tensor> out_wrap__weight_norm_interface_backward_out(const Tensor& grad_w, const Tensor& saved_v, const Tensor& saved_g, const Tensor& saved_norms, int64_t dim, Tensor& out0, Tensor& out1) {
    const auto answers = ops::_weight_norm_interface_backward(grad_w, saved_v, saved_g, saved_norms, dim);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap__efficientzerotensor_out(const std::vector<int64_t>& size, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::_efficientzerotensor(size, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap_zeros_like_out(const Tensor& self, DType dtype, std::optional<Device> device, Tensor& out) {
    write_out(out, ops::zeros_like(self, dtype, device, false));
    return out;
}

Tensor& out_wrap__standard_gamma_grad_out(const Tensor& self, const Tensor& output, Tensor& out) {
    write_out(out, ops::_standard_gamma_grad(self, output));
    return out;
}

Tensor& out_wrap__standard_gamma_out(const Tensor& self, std::optional<Generator> generator, Tensor& out) {
    write_out(out, ops::_standard_gamma(self, generator));
    return out;
}

Tensor& out_wrap__philox_normal_out(const Tensor& self, const Tensor& key, double mean, double std, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::_philox_normal_(answer, key, mean, std);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap__philox_uniform_out(const Tensor& self, const Tensor& key, double low, double high, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::_philox_uniform_(answer, key, low, high);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap__dirichlet_grad_out(const Tensor& x, const Tensor& alpha, const Tensor& total, Tensor& out) {
    write_out(out, ops::_dirichlet_grad(x, alpha, total));
    return out;
}

Tensor& out_wrap__sample_dirichlet_out(const Tensor& self, std::optional<Generator> generator, Tensor& out) {
    write_out(out, ops::_sample_dirichlet(self, generator));
    return out;
}

Tensor& out_wrap_poisson_out(const Tensor& self, std::optional<Generator> generator, Tensor& out) {
    write_out(out, ops::poisson(self, generator));
    return out;
}

Tensor& out_wrap_binomial_out(const Tensor& count, const Tensor& prob, std::optional<Generator> generator, Tensor& out) {
    write_out(out, ops::binomial(count, prob, generator));
    return out;
}

Tensor& out_wrap_native_norm_out(const Tensor& self, const Scalar& p, Tensor& out) {
    write_out(out, ops::native_norm(self, p));
    return out;
}

Tensor& out_wrap_native_norm_ScalarOpt_dim_dtype_out(const Tensor& self, const std::optional<Scalar>& p, const std::vector<int64_t>& dim, bool keepdim, std::optional<DType> dtype, Tensor& out) {
    write_out(out, ops::native_norm(self, p, dim, keepdim, dtype));
    return out;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> out_wrap__batch_norm_no_update_out(const Tensor& input, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, const std::optional<Tensor>& running_mean, const std::optional<Tensor>& running_var, double momentum, double eps, Tensor& out0, Tensor& out1, Tensor& out2, Tensor& out3) {
    const auto answers = ops::_batch_norm_no_update(input, weight, bias, running_mean, running_var, momentum, eps);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    write_out(out3, std::get<3>(answers));
    return {out0, out1, out2, out3};
}

Tensor& out_wrap__sparse_sum_dim_out(const Tensor& self, const std::vector<int64_t>& dim, Tensor& out) {
    write_out(out, ops::_sparse_sum(self, dim));
    return out;
}

Tensor& out_wrap__sparse_sum_backward_out(const Tensor& grad_output, const Tensor& self, const std::vector<int64_t>& dim, Tensor& out) {
    write_out(out, ops::_sparse_sum_backward(grad_output, self, dim));
    return out;
}

Tensor& out_wrap__sparse_csr_sum_dim_dtype_out(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, std::optional<DType> dtype, Tensor& out) {
    write_out(out, ops::_sparse_csr_sum(self, dim, keepdim, dtype));
    return out;
}

Tensor& out_wrap__sparse_csr_prod_dim_dtype_out(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, std::optional<DType> dtype, Tensor& out) {
    write_out(out, ops::_sparse_csr_prod(self, dim, keepdim, dtype));
    return out;
}

Tensor& out_wrap__sparse_softmax_out(const Tensor& self, int64_t dim, bool half_to_float, Tensor& out) {
    write_out(out, ops::_sparse_softmax(self, dim, half_to_float));
    return out;
}

Tensor& out_wrap__sparse_softmax_backward_data_out(const Tensor& grad_output, const Tensor& output, int64_t dim, const Tensor& self, Tensor& out) {
    write_out(out, ops::_sparse_softmax_backward_data(grad_output, output, dim, self));
    return out;
}

Tensor& out_wrap__sparse_log_softmax_out(const Tensor& self, int64_t dim, bool half_to_float, Tensor& out) {
    write_out(out, ops::_sparse_log_softmax(self, dim, half_to_float));
    return out;
}

Tensor& out_wrap__sparse_log_softmax_backward_data_out(const Tensor& grad_output, const Tensor& output, int64_t dim, const Tensor& self, Tensor& out) {
    write_out(out, ops::_sparse_log_softmax_backward_data(grad_output, output, dim, self));
    return out;
}

Tensor& out_wrap__spdiags_out(const Tensor& diagonals, const Tensor& offsets, const std::vector<int64_t>& shape, std::optional<int64_t> layout, Tensor& out) {
    write_out(out, ops::_spdiags(diagonals, offsets, shape, layout));
    return out;
}

Tensor& out_wrap_norm_ScalarOpt_dtype_out(const Tensor& self, const std::optional<Scalar>& p, DType dtype, Tensor& out) {
    write_out(out, ops::norm(self, p, dtype));
    return out;
}

Tensor& out_wrap_norm_Scalar_out(const Tensor& self, const Scalar& p, Tensor& out) {
    write_out(out, ops::norm(self, p));
    return out;
}

Tensor& out_wrap_clone_out(const Tensor& self, std::optional<int64_t> memory_format, Tensor& out) {
    write_out(out, ops::clone(self, memory_format));
    return out;
}

Tensor& out_wrap_resize_as_out(const Tensor& self, const Tensor& other, std::optional<int64_t> memory_format, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::resize_as_(answer, other, memory_format);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_resize_as_sparse_out(const Tensor& self, const Tensor& the_template, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::resize_as_sparse_(answer, the_template);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_zero_out(const Tensor& self, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::zero_(answer);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_sub_Scalar_out(const Tensor& self, const Scalar& other, const Scalar& alpha, Tensor& out) {
    write_out(out, ops::sub(self, other, alpha));
    return out;
}

Tensor& out_wrap_rsub_Tensor_out(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& out) {
    write_out(out, ops::rsub(self, other, alpha));
    return out;
}

Tensor& out_wrap_rsub_Scalar_out(const Tensor& self, const Scalar& other, const Scalar& alpha, Tensor& out) {
    write_out(out, ops::rsub(self, other, alpha));
    return out;
}

Tensor& out_wrap__sparse_addmm_out(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Tensor& out) {
    write_out(out, ops::_sparse_addmm(self, mat1, mat2, beta, alpha));
    return out;
}

Tensor& out_wrap_sparse_coo_tensor_size_out(const std::vector<int64_t>& size, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::sparse_coo_tensor(size, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap__sparse_coo_tensor_with_dims_out(int64_t sparse_dim, int64_t dense_dim, const std::vector<int64_t>& size, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::_sparse_coo_tensor_with_dims(sparse_dim, dense_dim, size, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap__sparse_coo_tensor_with_dims_and_tensors_out(int64_t sparse_dim, int64_t dense_dim, const std::vector<int64_t>& size, const Tensor& indices, const Tensor& values, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<bool> is_coalesced, Tensor& out) {
    write_out(out, ops::_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, indices, values, dtype, layout, device, pin_memory, is_coalesced));
    return out;
}

Tensor& out_wrap_sparse_resize_out(const Tensor& self, const std::vector<int64_t>& size, int64_t sparse_dim, int64_t dense_dim, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::sparse_resize_(answer, size, sparse_dim, dense_dim);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_sparse_resize_and_clear_out(const Tensor& self, const std::vector<int64_t>& size, int64_t sparse_dim, int64_t dense_dim, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::sparse_resize_and_clear_(answer, size, sparse_dim, dense_dim);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_sparse_mask_out(const Tensor& self, const Tensor& mask, Tensor& out) {
    write_out(out, ops::sparse_mask(self, mask));
    return out;
}

Tensor& out_wrap__sparse_mask_projection_out(const Tensor& self, const Tensor& mask, bool accumulate_matches, Tensor& out) {
    write_out(out, ops::_sparse_mask_projection(self, mask, accumulate_matches));
    return out;
}

Tensor& out_wrap__to_dense_out(const Tensor& self, std::optional<DType> dtype, std::optional<bool> masked_grad, Tensor& out) {
    write_out(out, ops::_to_dense(self, dtype, masked_grad));
    return out;
}

Tensor& out_wrap__coalesce_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::_coalesce(self));
    return out;
}

Tensor& out_wrap__coalesced_out(const Tensor& self, bool coalesced, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::_coalesced_(answer, coalesced);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_copy_sparse_to_sparse_out(const Tensor& self, const Tensor& src, bool non_blocking, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::copy_sparse_to_sparse_(answer, src, non_blocking);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap__to_sparse_sparse_dim_out(const Tensor& self, int64_t sparse_dim, Tensor& out) {
    write_out(out, ops::_to_sparse(self, sparse_dim));
    return out;
}

Tensor& out_wrap__to_sparse_out(const Tensor& self, std::optional<int64_t> layout, const std::optional<std::vector<int64_t>>& blocksize, std::optional<int64_t> dense_dim, Tensor& out) {
    write_out(out, ops::_to_sparse(self, layout, blocksize, dense_dim));
    return out;
}

Tensor& out_wrap__to_sparse_csr_out(const Tensor& self, std::optional<int64_t> dense_dim, Tensor& out) {
    write_out(out, ops::_to_sparse_csr(self, dense_dim));
    return out;
}

Tensor& out_wrap__to_sparse_csc_out(const Tensor& self, std::optional<int64_t> dense_dim, Tensor& out) {
    write_out(out, ops::_to_sparse_csc(self, dense_dim));
    return out;
}

Tensor& out_wrap__to_sparse_bsr_out(const Tensor& self, const std::vector<int64_t>& blocksize, std::optional<int64_t> dense_dim, Tensor& out) {
    write_out(out, ops::_to_sparse_bsr(self, blocksize, dense_dim));
    return out;
}

Tensor& out_wrap__to_sparse_bsc_out(const Tensor& self, const std::vector<int64_t>& blocksize, std::optional<int64_t> dense_dim, Tensor& out) {
    write_out(out, ops::_to_sparse_bsc(self, blocksize, dense_dim));
    return out;
}

Tensor& out_wrap_to_mkldnn_out(const Tensor& self, std::optional<DType> dtype, Tensor& out) {
    write_out(out, ops::to_mkldnn(self, dtype));
    return out;
}

Tensor& out_wrap_mkldnn_reorder_conv2d_weight_out(const Tensor& self, const std::vector<int64_t>& padding, const std::vector<int64_t>& stride, const std::vector<int64_t>& dilation, int64_t groups, const std::optional<std::vector<int64_t>>& input_size, Tensor& out) {
    write_out(out, ops::mkldnn_reorder_conv2d_weight(self, padding, stride, dilation, groups, input_size));
    return out;
}

Tensor& out_wrap_mkldnn_reorder_conv3d_weight_out(const Tensor& self, const std::vector<int64_t>& padding, const std::vector<int64_t>& stride, const std::vector<int64_t>& dilation, int64_t groups, const std::optional<std::vector<int64_t>>& input_size, Tensor& out) {
    write_out(out, ops::mkldnn_reorder_conv3d_weight(self, padding, stride, dilation, groups, input_size));
    return out;
}

Tensor& out_wrap_quantize_per_tensor_dynamic_out(const Tensor& self, DType dtype, bool reduce_range, Tensor& out) {
    write_out(out, ops::quantize_per_tensor_dynamic(self, dtype, reduce_range));
    return out;
}

Tensor& out_wrap_quantize_per_tensor_out(const Tensor& self, double scale, int64_t zero_point, DType dtype, Tensor& out) {
    write_out(out, ops::quantize_per_tensor(self, scale, zero_point, dtype));
    return out;
}

Tensor& out_wrap_quantize_per_tensor_tensor_qparams_out(const Tensor& self, const Tensor& scale, const Tensor& zero_point, DType dtype, Tensor& out) {
    write_out(out, ops::quantize_per_tensor(self, scale, zero_point, dtype));
    return out;
}

void out_wrap_quantize_per_tensor_tensors_out(const std::vector<Tensor>& tensors, const Tensor& scales, const Tensor& zero_points, DType dtype, std::vector<Tensor> out) {
    const auto answers = ops::quantize_per_tensor(tensors, scales, zero_points, dtype);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

Tensor& out_wrap_quantize_per_channel_out(const Tensor& self, const Tensor& scales, const Tensor& zero_points, int64_t axis, DType dtype, Tensor& out) {
    write_out(out, ops::quantize_per_channel(self, scales, zero_points, axis, dtype));
    return out;
}

Tensor& out_wrap_dequantize_self_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::dequantize(self));
    return out;
}

void out_wrap_dequantize_tensors_out(const std::vector<Tensor>& tensors, std::vector<Tensor> out) {
    const auto answers = ops::dequantize(tensors);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

Tensor& out_wrap_q_per_channel_scales_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::q_per_channel_scales(self));
    return out;
}

Tensor& out_wrap_q_per_channel_zero_points_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::q_per_channel_zero_points(self));
    return out;
}

Tensor& out_wrap_int_repr_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::int_repr(self));
    return out;
}

Tensor& out_wrap__make_per_tensor_quantized_tensor_out(const Tensor& self, double scale, int64_t zero_point, Tensor& out) {
    write_out(out, ops::_make_per_tensor_quantized_tensor(self, scale, zero_point));
    return out;
}

Tensor& out_wrap__make_per_channel_quantized_tensor_out(const Tensor& self, const Tensor& scale, const Tensor& zero_point, int64_t axis, Tensor& out) {
    write_out(out, ops::_make_per_channel_quantized_tensor(self, scale, zero_point, axis));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap_fake_quantize_per_tensor_affine_cachemask_out(const Tensor& self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max, Tensor& out0, Tensor& out1) {
    const auto answers = ops::fake_quantize_per_tensor_affine_cachemask(self, scale, zero_point, quant_min, quant_max);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

std::tuple<Tensor, Tensor> out_wrap__fake_quantize_per_tensor_affine_cachemask_tensor_qparams_out(const Tensor& self, const Tensor& scale, const Tensor& zero_point, const Tensor& fake_quant_enabled, int64_t quant_min, int64_t quant_max, Tensor& out0, Tensor& out1) {
    const auto answers = ops::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(self, scale, zero_point, fake_quant_enabled, quant_min, quant_max);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap__fake_quantize_learnable_per_tensor_affine_out(const Tensor& self, const Tensor& scale, const Tensor& zero_point, int64_t quant_min, int64_t quant_max, double grad_factor, Tensor& out) {
    write_out(out, ops::_fake_quantize_learnable_per_tensor_affine(self, scale, zero_point, quant_min, quant_max, grad_factor));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap_fake_quantize_per_channel_affine_cachemask_out(const Tensor& self, const Tensor& scale, const Tensor& zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, Tensor& out0, Tensor& out1) {
    const auto answers = ops::fake_quantize_per_channel_affine_cachemask(self, scale, zero_point, axis, quant_min, quant_max);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap__fake_quantize_learnable_per_channel_affine_out(const Tensor& self, const Tensor& scale, const Tensor& zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor, Tensor& out) {
    write_out(out, ops::_fake_quantize_learnable_per_channel_affine(self, scale, zero_point, axis, quant_min, quant_max, grad_factor));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap__fused_moving_avg_obs_fq_helper_out(const Tensor& self, const Tensor& observer_on, const Tensor& fake_quant_on, Tensor& running_min, Tensor& running_max, Tensor& scale, Tensor& zero_point, double averaging_const, int64_t quant_min, int64_t quant_max, int64_t ch_axis, bool per_row_fake_quant, bool symmetric_quant, Tensor& out0, Tensor& out1) {
    const auto answers = ops::_fused_moving_avg_obs_fq_helper(self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap__to_copy_out(const Tensor& self, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, bool non_blocking, std::optional<int64_t> memory_format, Tensor& out) {
    write_out(out, ops::_to_copy(self, dtype, layout, device, pin_memory, non_blocking, memory_format));
    return out;
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> out_wrap__lstm_mps_out(const Tensor& input, const std::vector<Tensor>& hx, const std::vector<Tensor>& params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, Tensor& out0, Tensor& out1, Tensor& out2, Tensor& out3, Tensor& out4, Tensor& out5) {
    const auto answers = ops::_lstm_mps(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    write_out(out3, std::get<3>(answers));
    write_out(out4, std::get<4>(answers));
    write_out(out5, std::get<5>(answers));
    return {out0, out1, out2, out3, out4, out5};
}

std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>> out_wrap_lstm_mps_backward_out(const std::optional<Tensor>& grad_y, const std::optional<Tensor>& grad_hy, const std::optional<Tensor>& grad_cy, const Tensor& z_state, const Tensor& cell_state_fwd, const Tensor& input, const Tensor& layersOutputs, const std::vector<Tensor>& hx, const std::vector<Tensor>& params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, Tensor& out0, std::vector<Tensor> out1, std::vector<Tensor> out2) {
    const auto answers = ops::lstm_mps_backward(grad_y, grad_hy, grad_cy, z_state, cell_state_fwd, input, layersOutputs, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
    write_out(out0, std::get<0>(answers));
    for (size_t item = 0;
         item < std::get<1>(answers).size() && item < out1.size();
         ++item) {
        write_out(out1[item], std::get<1>(answers)[item]);
    }
    for (size_t item = 0;
         item < std::get<2>(answers).size() && item < out2.size();
         ++item) {
        write_out(out2[item], std::get<2>(answers)[item]);
    }
    return {out0, out1, out2};
}

std::tuple<Tensor, Tensor, Tensor> out_wrap__thnn_fused_lstm_cell_out(const Tensor& input_gates, const Tensor& hidden_gates, const Tensor& cx, const std::optional<Tensor>& input_bias, const std::optional<Tensor>& hidden_bias, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::_thnn_fused_lstm_cell(input_gates, hidden_gates, cx, input_bias, hidden_bias);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

std::tuple<Tensor, Tensor, Tensor> out_wrap__thnn_fused_lstm_cell_backward_impl_out(const std::optional<Tensor>& grad_hy, const std::optional<Tensor>& grad_cy, const Tensor& cx, const Tensor& cy, const Tensor& workspace, bool has_bias, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::_thnn_fused_lstm_cell_backward_impl(grad_hy, grad_cy, cx, cy, workspace, has_bias);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

std::tuple<Tensor, Tensor> out_wrap__thnn_fused_gru_cell_out(const Tensor& input_gates, const Tensor& hidden_gates, const Tensor& hx, const std::optional<Tensor>& input_bias, const std::optional<Tensor>& hidden_bias, Tensor& out0, Tensor& out1) {
    const auto answers = ops::_thnn_fused_gru_cell(input_gates, hidden_gates, hx, input_bias, hidden_bias);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> out_wrap__thnn_fused_gru_cell_backward_out(const Tensor& grad_hy, const Tensor& workspace, bool has_bias, Tensor& out0, Tensor& out1, Tensor& out2, Tensor& out3, Tensor& out4) {
    const auto answers = ops::_thnn_fused_gru_cell_backward(grad_hy, workspace, has_bias);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    write_out(out3, std::get<3>(answers));
    write_out(out4, std::get<4>(answers));
    return {out0, out1, out2, out3, out4};
}

std::tuple<Tensor, Tensor> out_wrap__pack_padded_sequence_out(const Tensor& input, const Tensor& lengths, bool batch_first, Tensor& out0, Tensor& out1) {
    const auto answers = ops::_pack_padded_sequence(input, lengths, batch_first);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap_set_source_Storage_out(const Tensor& self, Storage source, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::set_(answer, source);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_set_source_Storage_storage_offset_out(const Tensor& self, Storage source, int64_t storage_offset, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::set_(answer, source, storage_offset, size, stride);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_set_source_Tensor_out(const Tensor& self, const Tensor& source, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::set_(answer, source);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_set_out(const Tensor& self, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::set_(answer);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_lift_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::lift(self));
    return out;
}

Tensor& out_wrap_lift_fresh_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::lift_fresh_copy(self));
    return out;
}

Tensor& out_wrap_masked_fill_Scalar_out(const Tensor& self, const Tensor& mask, const Scalar& value, Tensor& out) {
    write_out(out, ops::masked_fill(self, mask, value));
    return out;
}

Tensor& out_wrap_masked_fill_Tensor_out(const Tensor& self, const Tensor& mask, const Tensor& value, Tensor& out) {
    write_out(out, ops::masked_fill(self, mask, value));
    return out;
}

Tensor& out_wrap_masked_scatter_out(const Tensor& self, const Tensor& mask, const Tensor& source, Tensor& out) {
    write_out(out, ops::masked_scatter(self, mask, source));
    return out;
}

Tensor& out_wrap__masked_softmax_out(const Tensor& self, const Tensor& mask, std::optional<int64_t> dim, std::optional<int64_t> mask_type, Tensor& out) {
    write_out(out, ops::_masked_softmax(self, mask, dim, mask_type));
    return out;
}

Tensor& out_wrap__masked_softmax_backward_out(const Tensor& grad_output, const Tensor& output, const Tensor& mask, std::optional<int64_t> dim, Tensor& out) {
    write_out(out, ops::_masked_softmax_backward(grad_output, output, mask, dim));
    return out;
}

Tensor& out_wrap_put_out(const Tensor& self, const Tensor& index, const Tensor& source, bool accumulate, Tensor& out) {
    write_out(out, ops::put(self, index, source, accumulate));
    return out;
}

Tensor& out_wrap_index_fill_int_Scalar_out(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value, Tensor& out) {
    write_out(out, ops::index_fill(self, dim, index, value));
    return out;
}

Tensor& out_wrap_index_fill_int_Tensor_out(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& value, Tensor& out) {
    write_out(out, ops::index_fill(self, dim, index, value));
    return out;
}

Tensor& out_wrap_bitwise_and_Scalar_Tensor_out(const Scalar& self, const Tensor& other, Tensor& out) {
    write_out(out, ops::bitwise_and(self, other));
    return out;
}

Tensor& out_wrap_bitwise_or_Scalar_Tensor_out(const Scalar& self, const Tensor& other, Tensor& out) {
    write_out(out, ops::bitwise_or(self, other));
    return out;
}

Tensor& out_wrap_bitwise_xor_Scalar_Tensor_out(const Scalar& self, const Tensor& other, Tensor& out) {
    write_out(out, ops::bitwise_xor(self, other));
    return out;
}

Tensor& out_wrap___lshift___Scalar_out(const Tensor& self, const Scalar& other, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::__ilshift__(answer, other);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap___lshift___Tensor_out(const Tensor& self, const Tensor& other, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::__ilshift__(answer, other);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_bitwise_left_shift_Scalar_Tensor_out(const Scalar& self, const Tensor& other, Tensor& out) {
    write_out(out, ops::bitwise_left_shift(self, other));
    return out;
}

Tensor& out_wrap___rshift___Scalar_out(const Tensor& self, const Scalar& other, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::__irshift__(answer, other);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap___rshift___Tensor_out(const Tensor& self, const Tensor& other, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::__irshift__(answer, other);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_bitwise_right_shift_Scalar_Tensor_out(const Scalar& self, const Tensor& other, Tensor& out) {
    write_out(out, ops::bitwise_right_shift(self, other));
    return out;
}

Tensor& out_wrap_random_from_out(const Tensor& self, int64_t from, std::optional<int64_t> to, std::optional<Generator> generator, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::random_(answer, from, to, generator);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_random_to_out(const Tensor& self, int64_t to, std::optional<Generator> generator, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::random_(answer, to, generator);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_random_out(const Tensor& self, std::optional<Generator> generator, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::random_(answer, generator);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_uniform_out(const Tensor& self, double from, double to, std::optional<Generator> generator, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::uniform_(answer, from, to, generator);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_cauchy_out(const Tensor& self, double median, double sigma, std::optional<Generator> generator, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::cauchy_(answer, median, sigma, generator);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_log_normal_out(const Tensor& self, double mean, double std, std::optional<Generator> generator, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::log_normal_(answer, mean, std, generator);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_exponential_out(const Tensor& self, double lambd, std::optional<Generator> generator, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::exponential_(answer, lambd, generator);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_geometric_out(const Tensor& self, double p, std::optional<Generator> generator, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::geometric_(answer, p, generator);
    write_out(out, answer);
    return out;
}

Tensor& out_wrap_tril_indices_out(int64_t row, int64_t col, int64_t offset, DType dtype, std::optional<Device> device, bool pin_memory, Tensor& out) {
    write_out(out, ops::tril_indices(row, col, offset, dtype, device, pin_memory));
    return out;
}

Tensor& out_wrap_triu_indices_out(int64_t row, int64_t col, int64_t offset, DType dtype, std::optional<Device> device, bool pin_memory, Tensor& out) {
    write_out(out, ops::triu_indices(row, col, offset, dtype, device, pin_memory));
    return out;
}

Tensor& out_wrap_trace_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::trace(self));
    return out;
}

Tensor& out_wrap__cholesky_solve_helper_out(const Tensor& self, const Tensor& A, bool upper, Tensor& out) {
    write_out(out, ops::_cholesky_solve_helper(self, A, upper));
    return out;
}

Tensor& out_wrap_dist_out(const Tensor& self, const Tensor& other, const Scalar& p, Tensor& out) {
    write_out(out, ops::dist(self, other, p));
    return out;
}

void out_wrap__histogramdd_bin_edges_out(const Tensor& self, const std::vector<int64_t>& bins, const std::optional<std::vector<double>>& range, const std::optional<Tensor>& weight, bool density, std::vector<Tensor> out) {
    const auto answers = ops::_histogramdd_bin_edges(self, bins, range, weight, density);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

Tensor& out_wrap__histogramdd_from_bin_cts_out(const Tensor& self, const std::vector<int64_t>& bins, const std::optional<std::vector<double>>& range, const std::optional<Tensor>& weight, bool density, Tensor& out) {
    write_out(out, ops::_histogramdd_from_bin_cts(self, bins, range, weight, density));
    return out;
}

Tensor& out_wrap__histogramdd_from_bin_tensors_out(const Tensor& self, const std::vector<Tensor>& bins, const std::optional<Tensor>& weight, bool density, Tensor& out) {
    write_out(out, ops::_histogramdd_from_bin_tensors(self, bins, weight, density));
    return out;
}

Tensor& out_wrap_remainder_Scalar_Tensor_out(const Scalar& self, const Tensor& other, Tensor& out) {
    write_out(out, ops::remainder(self, other));
    return out;
}

Tensor& out_wrap_unfold_backward_out(const Tensor& grad_output, const std::vector<int64_t>& input_sizes, int64_t dim, int64_t size, int64_t step, Tensor& out) {
    write_out(out, ops::unfold_backward(grad_output, input_sizes, dim, size, step));
    return out;
}

Tensor& out_wrap_normal_out(const Tensor& self, double mean, double std, std::optional<Generator> generator, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::normal_(answer, mean, std, generator);
    write_out(out, answer);
    return out;
}

void out_wrap__amp_foreach_non_finite_check_and_unscale_out(const std::vector<Tensor>& self, Tensor& found_inf, const Tensor& inv_scale, std::vector<Tensor> out) {
    std::vector<Tensor> answer;
    for (const Tensor& item : self) {
        answer.push_back(ops::clone(item));
    }
    ops::_amp_foreach_non_finite_check_and_unscale_(answer, found_inf, inv_scale);
    for (size_t index = 0;
         index < answer.size() && index < out.size();
         ++index) {
        write_out(out[index], answer[index]);
    }
}

Tensor& out_wrap__amp_update_scale_out(const Tensor& self, Tensor& growth_tracker, const Tensor& found_inf, double growth_factor, double backoff_factor, int64_t growth_interval, Tensor& out) {
    Tensor answer = ops::clone(self);
    ops::_amp_update_scale_(answer, growth_tracker, found_inf, growth_factor, backoff_factor, growth_interval);
    write_out(out, answer);
    return out;
}

void out_wrap__foreach_add_Scalar_out(const std::vector<Tensor>& self, const Scalar& scalar, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_add(self, scalar);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_add_List_out(const std::vector<Tensor>& self, const std::vector<Tensor>& other, const Scalar& alpha, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_add(self, other, alpha);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_add_ScalarList_out(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_add(self, scalars);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_add_Tensor_out(const std::vector<Tensor>& self, const Tensor& other, const Scalar& alpha, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_add(self, other, alpha);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_sub_Scalar_out(const std::vector<Tensor>& self, const Scalar& scalar, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_sub(self, scalar);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_sub_List_out(const std::vector<Tensor>& self, const std::vector<Tensor>& other, const Scalar& alpha, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_sub(self, other, alpha);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_sub_ScalarList_out(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_sub(self, scalars);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_mul_Scalar_out(const std::vector<Tensor>& self, const Scalar& scalar, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_mul(self, scalar);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_mul_List_out(const std::vector<Tensor>& self, const std::vector<Tensor>& other, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_mul(self, other);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_mul_ScalarList_out(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_mul(self, scalars);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_mul_Tensor_out(const std::vector<Tensor>& self, const Tensor& other, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_mul(self, other);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_div_Scalar_out(const std::vector<Tensor>& self, const Scalar& scalar, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_div(self, scalar);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_div_List_out(const std::vector<Tensor>& self, const std::vector<Tensor>& other, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_div(self, other);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_div_ScalarList_out(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_div(self, scalars);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_div_Tensor_out(const std::vector<Tensor>& self, const Tensor& other, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_div(self, other);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_clamp_max_Scalar_out(const std::vector<Tensor>& self, const Scalar& scalar, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_clamp_max(self, scalar);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_clamp_max_List_out(const std::vector<Tensor>& self, const std::vector<Tensor>& other, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_clamp_max(self, other);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_clamp_max_ScalarList_out(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_clamp_max(self, scalars);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_clamp_min_Scalar_out(const std::vector<Tensor>& self, const Scalar& scalar, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_clamp_min(self, scalar);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_clamp_min_List_out(const std::vector<Tensor>& self, const std::vector<Tensor>& other, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_clamp_min(self, other);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_clamp_min_ScalarList_out(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_clamp_min(self, scalars);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_maximum_Scalar_out(const std::vector<Tensor>& self, const Scalar& scalar, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_maximum(self, scalar);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_maximum_List_out(const std::vector<Tensor>& self, const std::vector<Tensor>& other, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_maximum(self, other);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_maximum_ScalarList_out(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_maximum(self, scalars);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_minimum_Scalar_out(const std::vector<Tensor>& self, const Scalar& scalar, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_minimum(self, scalar);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_minimum_List_out(const std::vector<Tensor>& self, const std::vector<Tensor>& other, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_minimum(self, other);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_minimum_ScalarList_out(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_minimum(self, scalars);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_addcdiv_Scalar_out(const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1, const std::vector<Tensor>& tensor2, const Scalar& value, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_addcdiv(self, tensor1, tensor2, value);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_addcdiv_ScalarList_out(const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1, const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_addcdiv(self, tensor1, tensor2, scalars);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_addcdiv_Tensor_out(const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1, const std::vector<Tensor>& tensor2, const Tensor& scalars, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_addcdiv(self, tensor1, tensor2, scalars);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_addcmul_Scalar_out(const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1, const std::vector<Tensor>& tensor2, const Scalar& value, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_addcmul(self, tensor1, tensor2, value);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_addcmul_ScalarList_out(const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1, const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_addcmul(self, tensor1, tensor2, scalars);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_addcmul_Tensor_out(const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1, const std::vector<Tensor>& tensor2, const Tensor& scalars, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_addcmul(self, tensor1, tensor2, scalars);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_abs_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_abs(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_acos_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_acos(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_asin_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_asin(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_atan_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_atan(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_ceil_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_ceil(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_cos_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_cos(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_cosh_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_cosh(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_erf_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_erf(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_erfc_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_erfc(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_exp_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_exp(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_expm1_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_expm1(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_floor_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_floor(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_frac_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_frac(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_lerp_List_out(const std::vector<Tensor>& self, const std::vector<Tensor>& tensors1, const std::vector<Tensor>& weights, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_lerp(self, tensors1, weights);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_lerp_Scalar_out(const std::vector<Tensor>& self, const std::vector<Tensor>& tensors1, const Scalar& weight, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_lerp(self, tensors1, weight);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_lerp_ScalarList_out(const std::vector<Tensor>& self, const std::vector<Tensor>& tensors1, const std::vector<Scalar>& weight, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_lerp(self, tensors1, weight);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_lgamma_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_lgamma(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_log_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_log(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_log10_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_log10(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_log1p_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_log1p(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_log2_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_log2(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_max_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_max(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_neg_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_neg(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_norm_Scalar_out(const std::vector<Tensor>& self, const Scalar& ord, std::optional<DType> dtype, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_norm(self, ord, dtype);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_powsum_Scalar_out(const std::vector<Tensor>& self, const Scalar& ord, std::optional<DType> dtype, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_powsum(self, ord, dtype);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_pow_List_out(const std::vector<Tensor>& self, const std::vector<Tensor>& exponent, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_pow(self, exponent);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_pow_Scalar_out(const std::vector<Tensor>& self, const Scalar& exponent, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_pow(self, exponent);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_pow_ScalarList_out(const std::vector<Tensor>& self, const std::vector<Scalar>& exponent, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_pow(self, exponent);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_reciprocal_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_reciprocal(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_round_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_round(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_rsqrt_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_rsqrt(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_sigmoid_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_sigmoid(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_sign_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_sign(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_sin_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_sin(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_sinh_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_sinh(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_sqrt_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_sqrt(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_tan_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_tan(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_tanh_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_tanh(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_trunc_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_trunc(self);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_zero_out(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    std::vector<Tensor> answer;
    for (const Tensor& item : self) {
        answer.push_back(ops::clone(item));
    }
    ops::_foreach_zero_(answer);
    for (size_t index = 0;
         index < answer.size() && index < out.size();
         ++index) {
        write_out(out[index], answer[index]);
    }
}

void out_wrap__foreach_clone_out(const std::vector<Tensor>& self, std::optional<int64_t> memory_format, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_clone(self, memory_format);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

void out_wrap__foreach_copy_out(const std::vector<Tensor>& self, const std::vector<Tensor>& src, bool non_blocking, std::vector<Tensor> out) {
    const auto answers = ops::_foreach_copy(self, src, non_blocking);
    for (size_t index = 0;
         index < answers.size() && index < out.size();
         ++index) {
        write_out(out[index], answers[index]);
    }
}

Tensor& out_wrap_bucketize_Scalar_out(const Scalar& self, const Tensor& boundaries, bool out_int32, bool right, Tensor& out) {
    write_out(out, ops::bucketize(self, boundaries, out_int32, right));
    return out;
}

Tensor& out_wrap_glu_jvp_out(const Tensor& glu, const Tensor& x, const Tensor& dx, int64_t dim, Tensor& out) {
    write_out(out, ops::glu_jvp(glu, x, dx, dim));
    return out;
}

Tensor& out_wrap_glu_backward_jvp_out(const Tensor& grad_x, const Tensor& grad_glu, const Tensor& x, const Tensor& dgrad_glu, const Tensor& dx, int64_t dim, Tensor& out) {
    write_out(out, ops::glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, dim));
    return out;
}

Tensor& out_wrap_rrelu_with_noise_backward_out(const Tensor& grad_output, const Tensor& self, const Tensor& noise, const Scalar& lower, const Scalar& upper, bool training, bool self_is_result, Tensor& out) {
    write_out(out, ops::rrelu_with_noise_backward(grad_output, self, noise, lower, upper, training, self_is_result));
    return out;
}

Tensor& out_wrap_mkldnn_adaptive_avg_pool2d_backward_out(const Tensor& grad_output, const Tensor& self, Tensor& out) {
    write_out(out, ops::mkldnn_adaptive_avg_pool2d_backward(grad_output, self));
    return out;
}

Tensor& out_wrap__adaptive_avg_pool2d_out(const Tensor& self, const std::vector<int64_t>& output_size, Tensor& out) {
    write_out(out, ops::_adaptive_avg_pool2d(self, output_size));
    return out;
}

Tensor& out_wrap__adaptive_avg_pool2d_backward_out(const Tensor& grad_output, const Tensor& self, Tensor& out) {
    write_out(out, ops::_adaptive_avg_pool2d_backward(grad_output, self));
    return out;
}

Tensor& out_wrap__adaptive_avg_pool3d_out(const Tensor& self, const std::vector<int64_t>& output_size, Tensor& out) {
    write_out(out, ops::_adaptive_avg_pool3d(self, output_size));
    return out;
}

Tensor& out_wrap__adaptive_avg_pool3d_backward_out(const Tensor& grad_output, const Tensor& self, Tensor& out) {
    write_out(out, ops::_adaptive_avg_pool3d_backward(grad_output, self));
    return out;
}

Tensor& out_wrap_upsample_linear1d_vec_out(const Tensor& input, const std::optional<std::vector<int64_t>>& output_size, bool align_corners, const std::optional<std::vector<double>>& scale_factors, Tensor& out) {
    write_out(out, ops::upsample_linear1d(input, output_size, align_corners, scale_factors));
    return out;
}

Tensor& out_wrap_upsample_bilinear2d_vec_out(const Tensor& input, const std::optional<std::vector<int64_t>>& output_size, bool align_corners, const std::optional<std::vector<double>>& scale_factors, Tensor& out) {
    write_out(out, ops::upsample_bilinear2d(input, output_size, align_corners, scale_factors));
    return out;
}

Tensor& out_wrap__upsample_bilinear2d_aa_vec_out(const Tensor& input, const std::optional<std::vector<int64_t>>& output_size, bool align_corners, const std::optional<std::vector<double>>& scale_factors, Tensor& out) {
    write_out(out, ops::_upsample_bilinear2d_aa(input, output_size, align_corners, scale_factors));
    return out;
}

Tensor& out_wrap_upsample_trilinear3d_vec_out(const Tensor& input, const std::optional<std::vector<int64_t>>& output_size, bool align_corners, const std::optional<std::vector<double>>& scale_factors, Tensor& out) {
    write_out(out, ops::upsample_trilinear3d(input, output_size, align_corners, scale_factors));
    return out;
}

Tensor& out_wrap_upsample_bicubic2d_vec_out(const Tensor& input, const std::optional<std::vector<int64_t>>& output_size, bool align_corners, const std::optional<std::vector<double>>& scale_factors, Tensor& out) {
    write_out(out, ops::upsample_bicubic2d(input, output_size, align_corners, scale_factors));
    return out;
}

Tensor& out_wrap__upsample_bicubic2d_aa_vec_out(const Tensor& input, const std::optional<std::vector<int64_t>>& output_size, bool align_corners, const std::optional<std::vector<double>>& scale_factors, Tensor& out) {
    write_out(out, ops::_upsample_bicubic2d_aa(input, output_size, align_corners, scale_factors));
    return out;
}

Tensor& out_wrap__upsample_lanczos2d_aa_vec_out(const Tensor& input, const std::optional<std::vector<int64_t>>& output_size, bool align_corners, const std::optional<std::vector<double>>& scale_factors, Tensor& out) {
    write_out(out, ops::_upsample_lanczos2d_aa(input, output_size, align_corners, scale_factors));
    return out;
}

Tensor& out_wrap_upsample_nearest1d_vec_out(const Tensor& input, const std::optional<std::vector<int64_t>>& output_size, const std::optional<std::vector<double>>& scale_factors, Tensor& out) {
    write_out(out, ops::upsample_nearest1d(input, output_size, scale_factors));
    return out;
}

Tensor& out_wrap__upsample_nearest_exact1d_vec_out(const Tensor& input, const std::optional<std::vector<int64_t>>& output_size, const std::optional<std::vector<double>>& scale_factors, Tensor& out) {
    write_out(out, ops::_upsample_nearest_exact1d(input, output_size, scale_factors));
    return out;
}

Tensor& out_wrap_upsample_nearest2d_vec_out(const Tensor& input, const std::optional<std::vector<int64_t>>& output_size, const std::optional<std::vector<double>>& scale_factors, Tensor& out) {
    write_out(out, ops::upsample_nearest2d(input, output_size, scale_factors));
    return out;
}

Tensor& out_wrap__upsample_nearest_exact2d_vec_out(const Tensor& input, const std::optional<std::vector<int64_t>>& output_size, const std::optional<std::vector<double>>& scale_factors, Tensor& out) {
    write_out(out, ops::_upsample_nearest_exact2d(input, output_size, scale_factors));
    return out;
}

Tensor& out_wrap_upsample_nearest3d_vec_out(const Tensor& input, const std::optional<std::vector<int64_t>>& output_size, const std::optional<std::vector<double>>& scale_factors, Tensor& out) {
    write_out(out, ops::upsample_nearest3d(input, output_size, scale_factors));
    return out;
}

Tensor& out_wrap__upsample_nearest_exact3d_vec_out(const Tensor& input, const std::optional<std::vector<int64_t>>& output_size, const std::optional<std::vector<double>>& scale_factors, Tensor& out) {
    write_out(out, ops::_upsample_nearest_exact3d(input, output_size, scale_factors));
    return out;
}

std::tuple<Tensor, Tensor, Tensor> out_wrap__slow_conv2d_backward_output_mask_out(const Tensor& grad_output, const Tensor& self, const Tensor& weight, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<bool>& output_mask, Tensor& out0, Tensor& out1, Tensor& out2) {
    const auto answers = ops::_slow_conv2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_mask);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    write_out(out2, std::get<2>(answers));
    return {out0, out1, out2};
}

Tensor& out_wrap_conv_depthwise3d_out(const Tensor& self, const Tensor& weight, const std::vector<int64_t>& kernel_size, const std::optional<Tensor>& bias, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, Tensor& out) {
    write_out(out, ops::conv_depthwise3d(self, weight, kernel_size, bias, stride, padding, dilation));
    return out;
}

Tensor& out_wrap_slow_conv_dilated2d_out(const Tensor& self, const Tensor& weight, const std::vector<int64_t>& kernel_size, const std::optional<Tensor>& bias, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, Tensor& out) {
    write_out(out, ops::slow_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation));
    return out;
}

Tensor& out_wrap_slow_conv_dilated3d_out(const Tensor& self, const Tensor& weight, const std::vector<int64_t>& kernel_size, const std::optional<Tensor>& bias, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, Tensor& out) {
    write_out(out, ops::slow_conv_dilated3d(self, weight, kernel_size, bias, stride, padding, dilation));
    return out;
}

Tensor& out_wrap_isinf_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::isinf(self));
    return out;
}

Tensor& out_wrap_linalg_matrix_exp_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::linalg_matrix_exp(self));
    return out;
}

Tensor& out_wrap_linalg_matrix_sqrth_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::linalg_matrix_sqrth(self));
    return out;
}

Tensor& out_wrap__test_optional_intlist_out(const Tensor& values, const std::optional<std::vector<int64_t>>& addends, Tensor& out) {
    write_out(out, ops::_test_optional_intlist(values, addends));
    return out;
}

Tensor& out_wrap__test_optional_filled_intlist_out(const Tensor& values, const std::optional<std::vector<int64_t>>& addends, Tensor& out) {
    write_out(out, ops::_test_optional_filled_intlist(values, addends));
    return out;
}

Tensor& out_wrap__test_optional_floatlist_out(const Tensor& values, const std::optional<std::vector<double>>& addends, Tensor& out) {
    write_out(out, ops::_test_optional_floatlist(values, addends));
    return out;
}

Tensor& out_wrap__test_warn_in_autograd_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::_test_warn_in_autograd(self));
    return out;
}

Tensor& out_wrap__test_autograd_multiple_dispatch_fullcoverage_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::_test_autograd_multiple_dispatch(self));
    return out;
}

Tensor& out_wrap__test_autograd_multiple_dispatch_view_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::_test_autograd_multiple_dispatch_view_copy(self));
    return out;
}

Tensor& out_wrap_segment_reduce_out(const Tensor& data, const std::string& reduce, const std::optional<Tensor>& lengths, const std::optional<Tensor>& indices, const std::optional<Tensor>& offsets, int64_t axis, bool unsafe, const std::optional<Scalar>& initial, Tensor& out) {
    write_out(out, ops::segment_reduce(data, reduce, lengths, indices, offsets, axis, unsafe, initial));
    return out;
}

Tensor& out_wrap__segment_reduce_backward_out(const Tensor& grad_output, const Tensor& output, const Tensor& data, const std::string& reduce, const std::optional<Tensor>& lengths, const std::optional<Tensor>& offsets, int64_t axis, const std::optional<Scalar>& initial, Tensor& out) {
    write_out(out, ops::_segment_reduce_backward(grad_output, output, data, reduce, lengths, offsets, axis, initial));
    return out;
}

Tensor& out_wrap__nested_tensor_from_tensor_list_out(const std::vector<Tensor>& list, std::optional<DType> dtype, std::optional<int64_t> layout, std::optional<Device> device, std::optional<bool> pin_memory, Tensor& out) {
    write_out(out, ops::_nested_tensor_from_tensor_list(list, dtype, layout, device, pin_memory));
    return out;
}

Tensor& out_wrap__fw_primal_copy_out(const Tensor& self, int64_t level, Tensor& out) {
    write_out(out, ops::_fw_primal_copy(self, level));
    return out;
}

Tensor& out_wrap__make_dual_copy_out(const Tensor& primal, const Tensor& tangent, int64_t level, Tensor& out) {
    write_out(out, ops::_make_dual_copy(primal, tangent, level));
    return out;
}

Tensor& out_wrap_view_as_real_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::view_as_real_copy(self));
    return out;
}

Tensor& out_wrap_view_as_complex_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::view_as_complex_copy(self));
    return out;
}

Tensor& out_wrap__conj_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::_conj_copy(self));
    return out;
}

Tensor& out_wrap__neg_view_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::_neg_view_copy(self));
    return out;
}

Tensor& out_wrap_as_strided_copy_out(const Tensor& self, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, std::optional<int64_t> storage_offset, Tensor& out) {
    write_out(out, ops::as_strided_copy(self, size, stride, storage_offset));
    return out;
}

Tensor& out_wrap__sparse_broadcast_to_copy_out(const Tensor& self, const std::vector<int64_t>& size, Tensor& out) {
    write_out(out, ops::_sparse_broadcast_to_copy(self, size));
    return out;
}

Tensor& out_wrap_diagonal_copy_out(const Tensor& self, int64_t offset, int64_t dim1, int64_t dim2, Tensor& out) {
    write_out(out, ops::diagonal_copy(self, offset, dim1, dim2));
    return out;
}

Tensor& out_wrap_expand_copy_out(const Tensor& self, const std::vector<int64_t>& size, bool implicit, Tensor& out) {
    write_out(out, ops::expand_copy(self, size, implicit));
    return out;
}

Tensor& out_wrap_permute_copy_out(const Tensor& self, const std::vector<int64_t>& dims, Tensor& out) {
    write_out(out, ops::permute_copy(self, dims));
    return out;
}

Tensor& out_wrap__reshape_alias_copy_out(const Tensor& self, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, Tensor& out) {
    write_out(out, ops::_reshape_alias_copy(self, size, stride));
    return out;
}

Tensor& out_wrap_select_copy_int_out(const Tensor& self, int64_t dim, int64_t index, Tensor& out) {
    write_out(out, ops::select_copy(self, dim, index));
    return out;
}

Tensor& out_wrap_detach_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::detach_copy(self));
    return out;
}

Tensor& out_wrap_slice_copy_Tensor_out(const Tensor& self, int64_t dim, std::optional<int64_t> start, std::optional<int64_t> end, int64_t step, Tensor& out) {
    write_out(out, ops::slice_copy(self, dim, start, end, step));
    return out;
}

Tensor& out_wrap_squeeze_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::squeeze_copy(self));
    return out;
}

Tensor& out_wrap_squeeze_copy_dim_out(const Tensor& self, int64_t dim, Tensor& out) {
    write_out(out, ops::squeeze_copy(self, dim));
    return out;
}

Tensor& out_wrap_squeeze_copy_dims_out(const Tensor& self, const std::vector<int64_t>& dim, Tensor& out) {
    write_out(out, ops::squeeze_copy(self, dim));
    return out;
}

Tensor& out_wrap_t_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::t_copy(self));
    return out;
}

Tensor& out_wrap_transpose_copy_int_out(const Tensor& self, int64_t dim0, int64_t dim1, Tensor& out) {
    write_out(out, ops::transpose_copy(self, dim0, dim1));
    return out;
}

Tensor& out_wrap_unsqueeze_copy_out(const Tensor& self, int64_t dim, Tensor& out) {
    write_out(out, ops::unsqueeze_copy(self, dim));
    return out;
}

Tensor& out_wrap__indices_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::_indices_copy(self));
    return out;
}

Tensor& out_wrap__values_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::_values_copy(self));
    return out;
}

Tensor& out_wrap_indices_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::indices_copy(self));
    return out;
}

Tensor& out_wrap_values_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::values_copy(self));
    return out;
}

Tensor& out_wrap_crow_indices_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::crow_indices_copy(self));
    return out;
}

Tensor& out_wrap_col_indices_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::col_indices_copy(self));
    return out;
}

Tensor& out_wrap_ccol_indices_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::ccol_indices_copy(self));
    return out;
}

Tensor& out_wrap_row_indices_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::row_indices_copy(self));
    return out;
}

Tensor& out_wrap_view_copy_out(const Tensor& self, const std::vector<int64_t>& size, Tensor& out) {
    write_out(out, ops::view_copy(self, size));
    return out;
}

Tensor& out_wrap_view_copy_dtype_out(const Tensor& self, DType dtype, Tensor& out) {
    write_out(out, ops::view_copy(self, dtype));
    return out;
}

Tensor& out_wrap_unfold_copy_out(const Tensor& self, int64_t dimension, int64_t size, int64_t step, Tensor& out) {
    write_out(out, ops::unfold_copy(self, dimension, size, step));
    return out;
}

Tensor& out_wrap_alias_copy_out(const Tensor& self, Tensor& out) {
    write_out(out, ops::alias_copy(self));
    return out;
}

Tensor& out_wrap_to_padded_tensor_out(const Tensor& self, double padding, const std::optional<std::vector<int64_t>>& output_size, Tensor& out) {
    write_out(out, ops::to_padded_tensor(self, padding, output_size));
    return out;
}

Tensor& out_wrap__transformer_encoder_layer_fwd_out(const Tensor& src, int64_t embed_dim, int64_t num_heads, const Tensor& qkv_weight, const Tensor& qkv_bias, const Tensor& proj_weight, const Tensor& proj_bias, bool use_gelu, bool norm_first, double eps, const Tensor& norm_weight_1, const Tensor& norm_bias_1, const Tensor& norm_weight_2, const Tensor& norm_bias_2, const Tensor& ffn_weight_1, const Tensor& ffn_bias_1, const Tensor& ffn_weight_2, const Tensor& ffn_bias_2, const std::optional<Tensor>& mask, std::optional<int64_t> mask_type, Tensor& out) {
    write_out(out, ops::_transformer_encoder_layer_fwd(src, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias, use_gelu, norm_first, eps, norm_weight_1, norm_bias_1, norm_weight_2, norm_bias_2, ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2, mask, mask_type));
    return out;
}

std::tuple<Tensor, Tensor> out_wrap__native_multi_head_attention_out(const Tensor& query, const Tensor& key, const Tensor& value, int64_t embed_dim, int64_t num_head, const Tensor& qkv_weight, const Tensor& qkv_bias, const Tensor& proj_weight, const Tensor& proj_bias, const std::optional<Tensor>& mask, bool need_weights, bool average_attn_weights, std::optional<int64_t> mask_type, Tensor& out0, Tensor& out1) {
    const auto answers = ops::_native_multi_head_attention(query, key, value, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask, need_weights, average_attn_weights, mask_type);
    write_out(out0, std::get<0>(answers));
    write_out(out1, std::get<1>(answers));
    return {out0, out1};
}

Tensor& out_wrap_scaled_dot_product_attention_out(const Tensor& query, const Tensor& key, const Tensor& value, const std::optional<Tensor>& attn_mask, double dropout_p, bool is_causal, std::optional<double> scale, bool enable_gqa, Tensor& out) {
    write_out(out, ops::scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa));
    return out;
}

Tensor& out_wrap__triton_scaled_dot_attention_out(const Tensor& q, const Tensor& k, const Tensor& v, double dropout_p, Tensor& out) {
    write_out(out, ops::_triton_scaled_dot_attention(q, k, v, dropout_p));
    return out;
}

Tensor& out_wrap__triton_multi_head_attention_out(const Tensor& query, const Tensor& key, const Tensor& value, int64_t embed_dim, int64_t num_head, const Tensor& qkv_weight, const Tensor& qkv_bias, const Tensor& proj_weight, const Tensor& proj_bias, const std::optional<Tensor>& mask, Tensor& out) {
    write_out(out, ops::_triton_multi_head_attention(query, key, value, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask));
    return out;
}

Tensor& out_wrap__foobar_out(const Tensor& self, bool arg1, bool arg2, bool arg3, Tensor& out) {
    write_out(out, ops::_foobar(self, arg1, arg2, arg3));
    return out;
}

void out_wrap__fused_adam_out(const std::vector<Tensor>& self, std::vector<Tensor> grads, std::vector<Tensor> exp_avgs, std::vector<Tensor> exp_avg_sqs, std::vector<Tensor> max_exp_avg_sqs, const std::vector<Tensor>& state_steps, double lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const std::optional<Tensor>& grad_scale, const std::optional<Tensor>& found_inf, bool exact, std::vector<Tensor> out) {
    std::vector<Tensor> answer;
    for (const Tensor& item : self) {
        answer.push_back(ops::clone(item));
    }
    ops::_fused_adam_(answer, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf, exact);
    for (size_t index = 0;
         index < answer.size() && index < out.size();
         ++index) {
        write_out(out[index], answer[index]);
    }
}

void out_wrap__fused_adam_tensor_lr_out(const std::vector<Tensor>& self, std::vector<Tensor> grads, std::vector<Tensor> exp_avgs, std::vector<Tensor> exp_avg_sqs, std::vector<Tensor> max_exp_avg_sqs, const std::vector<Tensor>& state_steps, const Tensor& lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const std::optional<Tensor>& grad_scale, const std::optional<Tensor>& found_inf, bool exact, std::vector<Tensor> out) {
    std::vector<Tensor> answer;
    for (const Tensor& item : self) {
        answer.push_back(ops::clone(item));
    }
    ops::_fused_adam_(answer, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf, exact);
    for (size_t index = 0;
         index < answer.size() && index < out.size();
         ++index) {
        write_out(out[index], answer[index]);
    }
}

void out_wrap__fused_adamw_out(const std::vector<Tensor>& self, std::vector<Tensor> grads, std::vector<Tensor> exp_avgs, std::vector<Tensor> exp_avg_sqs, std::vector<Tensor> max_exp_avg_sqs, const std::vector<Tensor>& state_steps, double lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const std::optional<Tensor>& grad_scale, const std::optional<Tensor>& found_inf, bool exact, std::vector<Tensor> out) {
    std::vector<Tensor> answer;
    for (const Tensor& item : self) {
        answer.push_back(ops::clone(item));
    }
    ops::_fused_adamw_(answer, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf, exact);
    for (size_t index = 0;
         index < answer.size() && index < out.size();
         ++index) {
        write_out(out[index], answer[index]);
    }
}

void out_wrap__fused_adamw_tensor_lr_out(const std::vector<Tensor>& self, std::vector<Tensor> grads, std::vector<Tensor> exp_avgs, std::vector<Tensor> exp_avg_sqs, std::vector<Tensor> max_exp_avg_sqs, const std::vector<Tensor>& state_steps, const Tensor& lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const std::optional<Tensor>& grad_scale, const std::optional<Tensor>& found_inf, bool exact, std::vector<Tensor> out) {
    std::vector<Tensor> answer;
    for (const Tensor& item : self) {
        answer.push_back(ops::clone(item));
    }
    ops::_fused_adamw_(answer, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf, exact);
    for (size_t index = 0;
         index < answer.size() && index < out.size();
         ++index) {
        write_out(out[index], answer[index]);
    }
}

void out_wrap__fused_sgd_out(const std::vector<Tensor>& self, std::vector<Tensor> grads, std::vector<Tensor> momentum_buffer_list, double weight_decay, double momentum, double lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const std::optional<Tensor>& grad_scale, const std::optional<Tensor>& found_inf, std::vector<Tensor> out) {
    std::vector<Tensor> answer;
    for (const Tensor& item : self) {
        answer.push_back(ops::clone(item));
    }
    ops::_fused_sgd_(answer, grads, momentum_buffer_list, weight_decay, momentum, lr, dampening, nesterov, maximize, is_first_step, grad_scale, found_inf);
    for (size_t index = 0;
         index < answer.size() && index < out.size();
         ++index) {
        write_out(out[index], answer[index]);
    }
}

void out_wrap__fused_sgd_tensor_lr_out(const std::vector<Tensor>& self, std::vector<Tensor> grads, std::vector<Tensor> momentum_buffer_list, double weight_decay, double momentum, const Tensor& lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const std::optional<Tensor>& grad_scale, const std::optional<Tensor>& found_inf, std::vector<Tensor> out) {
    std::vector<Tensor> answer;
    for (const Tensor& item : self) {
        answer.push_back(ops::clone(item));
    }
    ops::_fused_sgd_(answer, grads, momentum_buffer_list, weight_decay, momentum, lr, dampening, nesterov, maximize, is_first_step, grad_scale, found_inf);
    for (size_t index = 0;
         index < answer.size() && index < out.size();
         ++index) {
        write_out(out[index], answer[index]);
    }
}

void out_wrap__fused_adagrad_out(const std::vector<Tensor>& self, std::vector<Tensor> grads, std::vector<Tensor> state_sums, std::vector<Tensor> state_steps, double lr, double lr_decay, double weight_decay, double eps, bool maximize, const std::optional<Tensor>& grad_scale, const std::optional<Tensor>& found_inf, std::vector<Tensor> out) {
    std::vector<Tensor> answer;
    for (const Tensor& item : self) {
        answer.push_back(ops::clone(item));
    }
    ops::_fused_adagrad_(answer, grads, state_sums, state_steps, lr, lr_decay, weight_decay, eps, maximize, grad_scale, found_inf);
    for (size_t index = 0;
         index < answer.size() && index < out.size();
         ++index) {
        write_out(out[index], answer[index]);
    }
}

void out_wrap__fused_adagrad_tensor_lr_out(const std::vector<Tensor>& self, std::vector<Tensor> grads, std::vector<Tensor> state_sums, std::vector<Tensor> state_steps, const Tensor& lr, double lr_decay, double weight_decay, double eps, bool maximize, const std::optional<Tensor>& grad_scale, const std::optional<Tensor>& found_inf, std::vector<Tensor> out) {
    std::vector<Tensor> answer;
    for (const Tensor& item : self) {
        answer.push_back(ops::clone(item));
    }
    ops::_fused_adagrad_(answer, grads, state_sums, state_steps, lr, lr_decay, weight_decay, eps, maximize, grad_scale, found_inf);
    for (size_t index = 0;
         index < answer.size() && index < out.size();
         ++index) {
        write_out(out[index], answer[index]);
    }
}

} // namespace composite
} // namespace tensorplay

namespace tensorplay {
TENSORPLAY_LIBRARY_IMPL(Composite, VariantOutGenerated) {
    m.impl("_new_zeros_with_same_feature_meta.out", composite::out_wrap__new_zeros_with_same_feature_meta_out);
    m.impl("_cudnn_ctc_loss.out", composite::out_wrap__cudnn_ctc_loss_out);
    m.impl("_cudnn_rnn_flatten_weight.out", composite::out_wrap__cudnn_rnn_flatten_weight_out);
    m.impl("_cudnn_rnn.out", composite::out_wrap__cudnn_rnn_out);
    m.impl("_cudnn_rnn_backward.out", composite::out_wrap__cudnn_rnn_backward_out);
    m.impl("_cudnn_init_dropout_state.out", composite::out_wrap__cudnn_init_dropout_state_out);
    m.impl("_fused_dropout.out", composite::out_wrap__fused_dropout_out);
    m.impl("_masked_scale.out", composite::out_wrap__masked_scale_out);
    m.impl("native_dropout.out", composite::out_wrap_native_dropout_out);
    m.impl("native_dropout_backward.out", composite::out_wrap_native_dropout_backward_out);
    m.impl("_conj_physical.out", composite::out_wrap__conj_physical_out);
    m.impl("avg_pool1d.out", composite::out_wrap_avg_pool1d_out);
    m.impl("adaptive_avg_pool1d.out", composite::out_wrap_adaptive_avg_pool1d_out);
    m.impl("_add_relu.Scalar_out", composite::out_wrap__add_relu_Scalar_out);
    m.impl("add.Scalar_out", composite::out_wrap_add_Scalar_out);
    m.impl("affine_grid_generator.out", composite::out_wrap_affine_grid_generator_out);
    m.impl("_test_functorch_fallback.out", composite::out_wrap__test_functorch_fallback_out);
    m.impl("bartlett_window.out", composite::out_wrap_bartlett_window_out);
    m.impl("bartlett_window.periodic_out", composite::out_wrap_bartlett_window_periodic_out);
    m.impl("quantized_batch_norm.out", composite::out_wrap_quantized_batch_norm_out);
    m.impl("bernoulli.Tensor_out", composite::out_wrap_bernoulli_Tensor_out);
    m.impl("bernoulli.float_out", composite::out_wrap_bernoulli_float_out);
    m.impl("binary_cross_entropy_with_logits.out", composite::out_wrap_binary_cross_entropy_with_logits_out);
    m.impl("bincount.out", composite::out_wrap_bincount_out);
    m.impl("blackman_window.out", composite::out_wrap_blackman_window_out);
    m.impl("blackman_window.periodic_out", composite::out_wrap_blackman_window_periodic_out);
    m.impl("block_diag.out", composite::out_wrap_block_diag_out);
    m.impl("constant_pad_nd.out", composite::out_wrap_constant_pad_nd_out);
    m.impl("convolution.out", composite::out_wrap_convolution_out);
    m.impl("convolution_backward.out", composite::out_wrap_convolution_backward_out);
    m.impl("convolution_overrideable.out", composite::out_wrap_convolution_overrideable_out);
    m.impl("convolution_backward_overrideable.out", composite::out_wrap_convolution_backward_overrideable_out);
    m.impl("_convolution.out", composite::out_wrap__convolution_out);
    m.impl("conv_tbc.out", composite::out_wrap_conv_tbc_out);
    m.impl("copy.out", composite::out_wrap_copy_out);
    m.impl("_copy_from.out", composite::out_wrap__copy_from_out);
    m.impl("_copy_from_and_resize.out", composite::out_wrap__copy_from_and_resize_out);
    m.impl("count_nonzero.dim_IntList_out", composite::out_wrap_count_nonzero_dim_IntList_out);
    m.impl("count_nonzero.out", composite::out_wrap_count_nonzero_out);
    m.impl("cudnn_affine_grid_generator.out", composite::out_wrap_cudnn_affine_grid_generator_out);
    m.impl("cudnn_affine_grid_generator_backward.out", composite::out_wrap_cudnn_affine_grid_generator_backward_out);
    m.impl("cudnn_batch_norm_backward.out", composite::out_wrap_cudnn_batch_norm_backward_out);
    m.impl("cudnn_convolution_transpose.out", composite::out_wrap_cudnn_convolution_transpose_out);
    m.impl("_mps_convolution_transpose.out", composite::out_wrap__mps_convolution_transpose_out);
    m.impl("mps_convolution_transpose_backward.out", composite::out_wrap_mps_convolution_transpose_backward_out);
    m.impl("cudnn_convolution_relu.out", composite::out_wrap_cudnn_convolution_relu_out);
    m.impl("cudnn_convolution_add_relu.out", composite::out_wrap_cudnn_convolution_add_relu_out);
    m.impl("cudnn_grid_sampler.out", composite::out_wrap_cudnn_grid_sampler_out);
    m.impl("cudnn_grid_sampler_backward.out", composite::out_wrap_cudnn_grid_sampler_backward_out);
    m.impl("_ctc_loss.out", composite::out_wrap__ctc_loss_out);
    m.impl("_ctc_loss.Tensor_out", composite::out_wrap__ctc_loss_Tensor_out);
    m.impl("_ctc_loss_backward.out", composite::out_wrap__ctc_loss_backward_out);
    m.impl("diag_embed.out", composite::out_wrap_diag_embed_out);
    m.impl("diagonal_backward.out", composite::out_wrap_diagonal_backward_out);
    m.impl("div.Scalar_out", composite::out_wrap_div_Scalar_out);
    m.impl("div.Scalar_mode_out", composite::out_wrap_div_Scalar_mode_out);
    m.impl("embedding.out", composite::out_wrap_embedding_out);
    m.impl("embedding_dense_backward.out", composite::out_wrap_embedding_dense_backward_out);
    m.impl("embedding_renorm.out", composite::out_wrap_embedding_renorm_out);
    m.impl("_embedding_bag_forward_only.out", composite::out_wrap__embedding_bag_forward_only_out);
    m.impl("_embedding_bag.out", composite::out_wrap__embedding_bag_out);
    m.impl("_embedding_bag_dense_backward.out", composite::out_wrap__embedding_bag_dense_backward_out);
    m.impl("_embedding_bag_per_sample_weights_backward.out", composite::out_wrap__embedding_bag_per_sample_weights_backward_out);
    m.impl("empty_permuted.out", composite::out_wrap_empty_permuted_out);
    m.impl("new_empty.out", composite::out_wrap_new_empty_out);
    m.impl("new_empty_strided.out", composite::out_wrap_new_empty_strided_out);
    m.impl("new_full.out", composite::out_wrap_new_full_out);
    m.impl("new_zeros.out", composite::out_wrap_new_zeros_out);
    m.impl("new_ones.out", composite::out_wrap_new_ones_out);
    m.impl("_empty_affine_quantized.out", composite::out_wrap__empty_affine_quantized_out);
    m.impl("_empty_per_channel_affine_quantized.out", composite::out_wrap__empty_per_channel_affine_quantized_out);
    m.impl("resize.out", composite::out_wrap_resize_out);
    m.impl("_resize_output.out", composite::out_wrap__resize_output_out);
    m.impl("empty_quantized.out", composite::out_wrap_empty_quantized_out);
    m.impl("empty_like.out", composite::out_wrap_empty_like_out);
    m.impl("empty_strided.out", composite::out_wrap_empty_strided_out);
    m.impl("fill.Scalar_out", composite::out_wrap_fill_Scalar_out);
    m.impl("fill.Tensor_out", composite::out_wrap_fill_Tensor_out);
    m.impl("floor_divide.Scalar_out", composite::out_wrap_floor_divide_Scalar_out);
    m.impl("full_like.out", composite::out_wrap_full_like_out);
    m.impl("from_file.out", composite::out_wrap_from_file_out);
    m.impl("grid_sampler_2d.out", composite::out_wrap_grid_sampler_2d_out);
    m.impl("grid_sampler_2d_backward.out", composite::out_wrap_grid_sampler_2d_backward_out);
    m.impl("_grid_sampler_2d_cpu_fallback.out", composite::out_wrap__grid_sampler_2d_cpu_fallback_out);
    m.impl("grid_sampler_3d.out", composite::out_wrap_grid_sampler_3d_out);
    m.impl("grid_sampler_3d_backward.out", composite::out_wrap_grid_sampler_3d_backward_out);
    m.impl("hann_window.out", composite::out_wrap_hann_window_out);
    m.impl("hann_window.periodic_out", composite::out_wrap_hann_window_periodic_out);
    m.impl("hamming_window.out", composite::out_wrap_hamming_window_out);
    m.impl("hamming_window.periodic_out", composite::out_wrap_hamming_window_periodic_out);
    m.impl("hamming_window.periodic_alpha_out", composite::out_wrap_hamming_window_periodic_alpha_out);
    m.impl("hamming_window.periodic_alpha_beta_out", composite::out_wrap_hamming_window_periodic_alpha_beta_out);
    m.impl("kaiser_window.out", composite::out_wrap_kaiser_window_out);
    m.impl("kaiser_window.periodic_out", composite::out_wrap_kaiser_window_periodic_out);
    m.impl("kaiser_window.beta_out", composite::out_wrap_kaiser_window_beta_out);
    m.impl("native_group_norm.out", composite::out_wrap_native_group_norm_out);
    m.impl("native_group_norm_backward.out", composite::out_wrap_native_group_norm_backward_out);
    m.impl("index_put.out", composite::out_wrap_index_put_out);
    m.impl("_index_put_impl.out", composite::out_wrap__index_put_impl_out);
    m.impl("isnan.out", composite::out_wrap_isnan_out);
    m.impl("native_layer_norm.out", composite::out_wrap_native_layer_norm_out);
    m.impl("native_layer_norm_backward.out", composite::out_wrap_native_layer_norm_backward_out);
    m.impl("linear_backward.out", composite::out_wrap_linear_backward_out);
    m.impl("mkldnn_linear.out", composite::out_wrap_mkldnn_linear_out);
    m.impl("mkldnn_linear_backward_input.out", composite::out_wrap_mkldnn_linear_backward_input_out);
    m.impl("mkldnn_linear_backward_weights.out", composite::out_wrap_mkldnn_linear_backward_weights_out);
    m.impl("mkldnn_linear_backward.out", composite::out_wrap_mkldnn_linear_backward_out);
    m.impl("matmul_backward.out", composite::out_wrap_matmul_backward_out);
    m.impl("_aminmax.out", composite::out_wrap__aminmax_out);
    m.impl("_aminmax.dim_out", composite::out_wrap__aminmax_dim_out);
    m.impl("max_pool2d_backward.out", composite::out_wrap_max_pool2d_backward_out);
    m.impl("mkldnn_max_pool2d.out", composite::out_wrap_mkldnn_max_pool2d_out);
    m.impl("mkldnn_max_pool2d_backward.out", composite::out_wrap_mkldnn_max_pool2d_backward_out);
    m.impl("mkldnn_max_pool3d.out", composite::out_wrap_mkldnn_max_pool3d_out);
    m.impl("mkldnn_max_pool3d_backward.out", composite::out_wrap_mkldnn_max_pool3d_backward_out);
    m.impl("quantized_max_pool1d.out", composite::out_wrap_quantized_max_pool1d_out);
    m.impl("quantized_max_pool2d.out", composite::out_wrap_quantized_max_pool2d_out);
    m.impl("quantized_max_pool3d.out", composite::out_wrap_quantized_max_pool3d_out);
    m.impl("median.out", composite::out_wrap_median_out);
    m.impl("nanmedian.out", composite::out_wrap_nanmedian_out);
    m.impl("_mps_convolution.out", composite::out_wrap__mps_convolution_out);
    m.impl("mps_convolution_backward.out", composite::out_wrap_mps_convolution_backward_out);
    m.impl("mkldnn_convolution.out", composite::out_wrap_mkldnn_convolution_out);
    m.impl("mkldnn_rnn_layer.out", composite::out_wrap_mkldnn_rnn_layer_out);
    m.impl("mkldnn_rnn_layer_backward.out", composite::out_wrap_mkldnn_rnn_layer_backward_out);
    m.impl("miopen_batch_norm.out", composite::out_wrap_miopen_batch_norm_out);
    m.impl("miopen_batch_norm_backward.out", composite::out_wrap_miopen_batch_norm_backward_out);
    m.impl("miopen_convolution.out", composite::out_wrap_miopen_convolution_out);
    m.impl("miopen_convolution_transpose.out", composite::out_wrap_miopen_convolution_transpose_out);
    m.impl("miopen_depthwise_convolution.out", composite::out_wrap_miopen_depthwise_convolution_out);
    m.impl("miopen_rnn.out", composite::out_wrap_miopen_rnn_out);
    m.impl("miopen_rnn_backward.out", composite::out_wrap_miopen_rnn_backward_out);
    m.impl("miopen_ctc_loss.out", composite::out_wrap_miopen_ctc_loss_out);
    m.impl("_sparse_sparse_matmul.out", composite::out_wrap__sparse_sparse_matmul_out);
    m.impl("mul.Scalar_out", composite::out_wrap_mul_Scalar_out);
    m.impl("_native_batch_norm_legit_no_training.out", composite::out_wrap__native_batch_norm_legit_no_training_out);
    m.impl("batch_norm_stats.out", composite::out_wrap_batch_norm_stats_out);
    m.impl("batch_norm_gather_stats.out", composite::out_wrap_batch_norm_gather_stats_out);
    m.impl("batch_norm_gather_stats_with_counts.out", composite::out_wrap_batch_norm_gather_stats_with_counts_out);
    m.impl("native_batch_norm_backward.out", composite::out_wrap_native_batch_norm_backward_out);
    m.impl("batch_norm_backward_reduce.out", composite::out_wrap_batch_norm_backward_reduce_out);
    m.impl("batch_norm_backward_elemt.out", composite::out_wrap_batch_norm_backward_elemt_out);
    m.impl("batch_norm_update_stats.out", composite::out_wrap_batch_norm_update_stats_out);
    m.impl("_nnpack_spatial_convolution.out", composite::out_wrap__nnpack_spatial_convolution_out);
    m.impl("ones_like.out", composite::out_wrap_ones_like_out);
    m.impl("_euclidean_dist.out", composite::out_wrap__euclidean_dist_out);
    m.impl("_cdist_forward.out", composite::out_wrap__cdist_forward_out);
    m.impl("_cdist_backward.out", composite::out_wrap__cdist_backward_out);
    m.impl("_pdist_forward.out", composite::out_wrap__pdist_forward_out);
    m.impl("_pdist_backward.out", composite::out_wrap__pdist_backward_out);
    m.impl("pixel_shuffle.out", composite::out_wrap_pixel_shuffle_out);
    m.impl("pixel_unshuffle.out", composite::out_wrap_pixel_unshuffle_out);
    m.impl("channel_shuffle.out", composite::out_wrap_channel_shuffle_out);
    m.impl("_pin_memory.out", composite::out_wrap__pin_memory_out);
    m.impl("scalar_tensor.out", composite::out_wrap_scalar_tensor_out);
    m.impl("rand_like.out", composite::out_wrap_rand_like_out);
    m.impl("rand_like.generator_out", composite::out_wrap_rand_like_generator_out);
    m.impl("randint_like.out", composite::out_wrap_randint_like_out);
    m.impl("randint_like.generator_out", composite::out_wrap_randint_like_generator_out);
    m.impl("randint_like.Tensor_out", composite::out_wrap_randint_like_Tensor_out);
    m.impl("randint_like.Tensor_generator_out", composite::out_wrap_randint_like_Tensor_generator_out);
    m.impl("randint_like.low_dtype_out", composite::out_wrap_randint_like_low_dtype_out);
    m.impl("randint_like.low_generator_dtype_out", composite::out_wrap_randint_like_low_generator_dtype_out);
    m.impl("randn_like.out", composite::out_wrap_randn_like_out);
    m.impl("randn_like.generator_out", composite::out_wrap_randn_like_generator_out);
    m.impl("repeat.out", composite::out_wrap_repeat_out);
    m.impl("repeat_interleave.Tensor_out", composite::out_wrap_repeat_interleave_Tensor_out);
    m.impl("_mkldnn_reshape.out", composite::out_wrap__mkldnn_reshape_out);
    m.impl("prelu.out", composite::out_wrap_prelu_out);
    m.impl("select_backward.out", composite::out_wrap_select_backward_out);
    m.impl("slice_backward.out", composite::out_wrap_slice_backward_out);
    m.impl("slice_scatter.out", composite::out_wrap_slice_scatter_out);
    m.impl("select_scatter.out", composite::out_wrap_select_scatter_out);
    m.impl("diagonal_scatter.out", composite::out_wrap_diagonal_scatter_out);
    m.impl("as_strided_scatter.out", composite::out_wrap_as_strided_scatter_out);
    m.impl("unsafe_split.Tensor_out", composite::out_wrap_unsafe_split_Tensor_out);
    m.impl("unsafe_split_with_sizes.out", composite::out_wrap_unsafe_split_with_sizes_out);
    m.impl("sum.out", composite::out_wrap_sum_out);
    m.impl("std_mean.correction_out", composite::out_wrap_std_mean_correction_out);
    m.impl("prod.out", composite::out_wrap_prod_out);
    m.impl("_mkldnn_transpose.out", composite::out_wrap__mkldnn_transpose_out);
    m.impl("flip.out", composite::out_wrap_flip_out);
    m.impl("roll.out", composite::out_wrap_roll_out);
    m.impl("rot90.out", composite::out_wrap_rot90_out);
    m.impl("_transform_bias_rescale_qkv.out", composite::out_wrap__transform_bias_rescale_qkv_out);
    m.impl("_nested_tensor_from_mask.out", composite::out_wrap__nested_tensor_from_mask_out);
    m.impl("_nested_from_padded.out", composite::out_wrap__nested_from_padded_out);
    m.impl("_nested_tensor_size.out", composite::out_wrap__nested_tensor_size_out);
    m.impl("_nested_tensor_strides.out", composite::out_wrap__nested_tensor_strides_out);
    m.impl("_nested_tensor_storage_offsets.out", composite::out_wrap__nested_tensor_storage_offsets_out);
    m.impl("_nested_from_padded_and_nested_example.out", composite::out_wrap__nested_from_padded_and_nested_example_out);
    m.impl("_nested_view_from_buffer_copy.out", composite::out_wrap__nested_view_from_buffer_copy_out);
    m.impl("_nested_view_from_jagged_copy.out", composite::out_wrap__nested_view_from_jagged_copy_out);
    m.impl("_nested_get_values_copy.out", composite::out_wrap__nested_get_values_copy_out);
    m.impl("_trilinear.out", composite::out_wrap__trilinear_out);
    m.impl("_unique.out", composite::out_wrap__unique_out);
    m.impl("unique_dim.out", composite::out_wrap_unique_dim_out);
    m.impl("unique_consecutive.out", composite::out_wrap_unique_consecutive_out);
    m.impl("unique_dim_consecutive.out", composite::out_wrap_unique_dim_consecutive_out);
    m.impl("_unique2.out", composite::out_wrap__unique2_out);
    m.impl("_unsafe_view.out", composite::out_wrap__unsafe_view_out);
    m.impl("var_mean.correction_out", composite::out_wrap_var_mean_correction_out);
    m.impl("_weight_norm_interface.out", composite::out_wrap__weight_norm_interface_out);
    m.impl("_weight_norm_interface_backward.out", composite::out_wrap__weight_norm_interface_backward_out);
    m.impl("_efficientzerotensor.out", composite::out_wrap__efficientzerotensor_out);
    m.impl("zeros_like.out", composite::out_wrap_zeros_like_out);
    m.impl("_standard_gamma_grad.out", composite::out_wrap__standard_gamma_grad_out);
    m.impl("_standard_gamma.out", composite::out_wrap__standard_gamma_out);
    m.impl("_philox_normal.out", composite::out_wrap__philox_normal_out);
    m.impl("_philox_uniform.out", composite::out_wrap__philox_uniform_out);
    m.impl("_dirichlet_grad.out", composite::out_wrap__dirichlet_grad_out);
    m.impl("_sample_dirichlet.out", composite::out_wrap__sample_dirichlet_out);
    m.impl("poisson.out", composite::out_wrap_poisson_out);
    m.impl("binomial.out", composite::out_wrap_binomial_out);
    m.impl("native_norm.out", composite::out_wrap_native_norm_out);
    m.impl("native_norm.ScalarOpt_dim_dtype_out", composite::out_wrap_native_norm_ScalarOpt_dim_dtype_out);
    m.impl("_batch_norm_no_update.out", composite::out_wrap__batch_norm_no_update_out);
    m.impl("_sparse_sum.dim_out", composite::out_wrap__sparse_sum_dim_out);
    m.impl("_sparse_sum_backward.out", composite::out_wrap__sparse_sum_backward_out);
    m.impl("_sparse_csr_sum.dim_dtype_out", composite::out_wrap__sparse_csr_sum_dim_dtype_out);
    m.impl("_sparse_csr_prod.dim_dtype_out", composite::out_wrap__sparse_csr_prod_dim_dtype_out);
    m.impl("_sparse_softmax.out", composite::out_wrap__sparse_softmax_out);
    m.impl("_sparse_softmax_backward_data.out", composite::out_wrap__sparse_softmax_backward_data_out);
    m.impl("_sparse_log_softmax.out", composite::out_wrap__sparse_log_softmax_out);
    m.impl("_sparse_log_softmax_backward_data.out", composite::out_wrap__sparse_log_softmax_backward_data_out);
    m.impl("_spdiags.out", composite::out_wrap__spdiags_out);
    m.impl("norm.ScalarOpt_dtype_out", composite::out_wrap_norm_ScalarOpt_dtype_out);
    m.impl("norm.Scalar_out", composite::out_wrap_norm_Scalar_out);
    m.impl("clone.out", composite::out_wrap_clone_out);
    m.impl("resize_as.out", composite::out_wrap_resize_as_out);
    m.impl("resize_as_sparse.out", composite::out_wrap_resize_as_sparse_out);
    m.impl("zero.out", composite::out_wrap_zero_out);
    m.impl("sub.Scalar_out", composite::out_wrap_sub_Scalar_out);
    m.impl("rsub.Tensor_out", composite::out_wrap_rsub_Tensor_out);
    m.impl("rsub.Scalar_out", composite::out_wrap_rsub_Scalar_out);
    m.impl("_sparse_addmm.out", composite::out_wrap__sparse_addmm_out);
    m.impl("sparse_coo_tensor.size_out", composite::out_wrap_sparse_coo_tensor_size_out);
    m.impl("_sparse_coo_tensor_with_dims.out", composite::out_wrap__sparse_coo_tensor_with_dims_out);
    m.impl("_sparse_coo_tensor_with_dims_and_tensors.out", composite::out_wrap__sparse_coo_tensor_with_dims_and_tensors_out);
    m.impl("sparse_resize.out", composite::out_wrap_sparse_resize_out);
    m.impl("sparse_resize_and_clear.out", composite::out_wrap_sparse_resize_and_clear_out);
    m.impl("sparse_mask.out", composite::out_wrap_sparse_mask_out);
    m.impl("_sparse_mask_projection.out", composite::out_wrap__sparse_mask_projection_out);
    m.impl("_to_dense.out", composite::out_wrap__to_dense_out);
    m.impl("_coalesce.out", composite::out_wrap__coalesce_out);
    m.impl("_coalesced.out", composite::out_wrap__coalesced_out);
    m.impl("copy_sparse_to_sparse.out", composite::out_wrap_copy_sparse_to_sparse_out);
    m.impl("_to_sparse.sparse_dim_out", composite::out_wrap__to_sparse_sparse_dim_out);
    m.impl("_to_sparse.out", composite::out_wrap__to_sparse_out);
    m.impl("_to_sparse_csr.out", composite::out_wrap__to_sparse_csr_out);
    m.impl("_to_sparse_csc.out", composite::out_wrap__to_sparse_csc_out);
    m.impl("_to_sparse_bsr.out", composite::out_wrap__to_sparse_bsr_out);
    m.impl("_to_sparse_bsc.out", composite::out_wrap__to_sparse_bsc_out);
    m.impl("to_mkldnn.out", composite::out_wrap_to_mkldnn_out);
    m.impl("mkldnn_reorder_conv2d_weight.out", composite::out_wrap_mkldnn_reorder_conv2d_weight_out);
    m.impl("mkldnn_reorder_conv3d_weight.out", composite::out_wrap_mkldnn_reorder_conv3d_weight_out);
    m.impl("quantize_per_tensor_dynamic.out", composite::out_wrap_quantize_per_tensor_dynamic_out);
    m.impl("quantize_per_tensor.out", composite::out_wrap_quantize_per_tensor_out);
    m.impl("quantize_per_tensor.tensor_qparams_out", composite::out_wrap_quantize_per_tensor_tensor_qparams_out);
    m.impl("quantize_per_tensor.tensors_out", composite::out_wrap_quantize_per_tensor_tensors_out);
    m.impl("quantize_per_channel.out", composite::out_wrap_quantize_per_channel_out);
    m.impl("dequantize.self_out", composite::out_wrap_dequantize_self_out);
    m.impl("dequantize.tensors_out", composite::out_wrap_dequantize_tensors_out);
    m.impl("q_per_channel_scales.out", composite::out_wrap_q_per_channel_scales_out);
    m.impl("q_per_channel_zero_points.out", composite::out_wrap_q_per_channel_zero_points_out);
    m.impl("int_repr.out", composite::out_wrap_int_repr_out);
    m.impl("_make_per_tensor_quantized_tensor.out", composite::out_wrap__make_per_tensor_quantized_tensor_out);
    m.impl("_make_per_channel_quantized_tensor.out", composite::out_wrap__make_per_channel_quantized_tensor_out);
    m.impl("fake_quantize_per_tensor_affine_cachemask.out", composite::out_wrap_fake_quantize_per_tensor_affine_cachemask_out);
    m.impl("_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.out", composite::out_wrap__fake_quantize_per_tensor_affine_cachemask_tensor_qparams_out);
    m.impl("_fake_quantize_learnable_per_tensor_affine.out", composite::out_wrap__fake_quantize_learnable_per_tensor_affine_out);
    m.impl("fake_quantize_per_channel_affine_cachemask.out", composite::out_wrap_fake_quantize_per_channel_affine_cachemask_out);
    m.impl("_fake_quantize_learnable_per_channel_affine.out", composite::out_wrap__fake_quantize_learnable_per_channel_affine_out);
    m.impl("_fused_moving_avg_obs_fq_helper.out", composite::out_wrap__fused_moving_avg_obs_fq_helper_out);
    m.impl("_to_copy.out", composite::out_wrap__to_copy_out);
    m.impl("_lstm_mps.out", composite::out_wrap__lstm_mps_out);
    m.impl("lstm_mps_backward.out", composite::out_wrap_lstm_mps_backward_out);
    m.impl("_thnn_fused_lstm_cell.out", composite::out_wrap__thnn_fused_lstm_cell_out);
    m.impl("_thnn_fused_lstm_cell_backward_impl.out", composite::out_wrap__thnn_fused_lstm_cell_backward_impl_out);
    m.impl("_thnn_fused_gru_cell.out", composite::out_wrap__thnn_fused_gru_cell_out);
    m.impl("_thnn_fused_gru_cell_backward.out", composite::out_wrap__thnn_fused_gru_cell_backward_out);
    m.impl("_pack_padded_sequence.out", composite::out_wrap__pack_padded_sequence_out);
    m.impl("set.source_Storage_out", composite::out_wrap_set_source_Storage_out);
    m.impl("set.source_Storage_storage_offset_out", composite::out_wrap_set_source_Storage_storage_offset_out);
    m.impl("set.source_Tensor_out", composite::out_wrap_set_source_Tensor_out);
    m.impl("set.out", composite::out_wrap_set_out);
    m.impl("lift.out", composite::out_wrap_lift_out);
    m.impl("lift_fresh_copy.out", composite::out_wrap_lift_fresh_copy_out);
    m.impl("masked_fill.Scalar_out", composite::out_wrap_masked_fill_Scalar_out);
    m.impl("masked_fill.Tensor_out", composite::out_wrap_masked_fill_Tensor_out);
    m.impl("masked_scatter.out", composite::out_wrap_masked_scatter_out);
    m.impl("_masked_softmax.out", composite::out_wrap__masked_softmax_out);
    m.impl("_masked_softmax_backward.out", composite::out_wrap__masked_softmax_backward_out);
    m.impl("put.out", composite::out_wrap_put_out);
    m.impl("index_fill.int_Scalar_out", composite::out_wrap_index_fill_int_Scalar_out);
    m.impl("index_fill.int_Tensor_out", composite::out_wrap_index_fill_int_Tensor_out);
    m.impl("bitwise_and.Scalar_Tensor_out", composite::out_wrap_bitwise_and_Scalar_Tensor_out);
    m.impl("bitwise_or.Scalar_Tensor_out", composite::out_wrap_bitwise_or_Scalar_Tensor_out);
    m.impl("bitwise_xor.Scalar_Tensor_out", composite::out_wrap_bitwise_xor_Scalar_Tensor_out);
    m.impl("__lshift__.Scalar_out", composite::out_wrap___lshift___Scalar_out);
    m.impl("__lshift__.Tensor_out", composite::out_wrap___lshift___Tensor_out);
    m.impl("bitwise_left_shift.Scalar_Tensor_out", composite::out_wrap_bitwise_left_shift_Scalar_Tensor_out);
    m.impl("__rshift__.Scalar_out", composite::out_wrap___rshift___Scalar_out);
    m.impl("__rshift__.Tensor_out", composite::out_wrap___rshift___Tensor_out);
    m.impl("bitwise_right_shift.Scalar_Tensor_out", composite::out_wrap_bitwise_right_shift_Scalar_Tensor_out);
    m.impl("random.from_out", composite::out_wrap_random_from_out);
    m.impl("random.to_out", composite::out_wrap_random_to_out);
    m.impl("random.out", composite::out_wrap_random_out);
    m.impl("uniform.out", composite::out_wrap_uniform_out);
    m.impl("cauchy.out", composite::out_wrap_cauchy_out);
    m.impl("log_normal.out", composite::out_wrap_log_normal_out);
    m.impl("exponential.out", composite::out_wrap_exponential_out);
    m.impl("geometric.out", composite::out_wrap_geometric_out);
    m.impl("tril_indices.out", composite::out_wrap_tril_indices_out);
    m.impl("triu_indices.out", composite::out_wrap_triu_indices_out);
    m.impl("trace.out", composite::out_wrap_trace_out);
    m.impl("_cholesky_solve_helper.out", composite::out_wrap__cholesky_solve_helper_out);
    m.impl("dist.out", composite::out_wrap_dist_out);
    m.impl("_histogramdd_bin_edges.out", composite::out_wrap__histogramdd_bin_edges_out);
    m.impl("_histogramdd_from_bin_cts.out", composite::out_wrap__histogramdd_from_bin_cts_out);
    m.impl("_histogramdd_from_bin_tensors.out", composite::out_wrap__histogramdd_from_bin_tensors_out);
    m.impl("remainder.Scalar_Tensor_out", composite::out_wrap_remainder_Scalar_Tensor_out);
    m.impl("unfold_backward.out", composite::out_wrap_unfold_backward_out);
    m.impl("normal.out", composite::out_wrap_normal_out);
    m.impl("_amp_foreach_non_finite_check_and_unscale.out", composite::out_wrap__amp_foreach_non_finite_check_and_unscale_out);
    m.impl("_amp_update_scale.out", composite::out_wrap__amp_update_scale_out);
    m.impl("_foreach_add.Scalar_out", composite::out_wrap__foreach_add_Scalar_out);
    m.impl("_foreach_add.List_out", composite::out_wrap__foreach_add_List_out);
    m.impl("_foreach_add.ScalarList_out", composite::out_wrap__foreach_add_ScalarList_out);
    m.impl("_foreach_add.Tensor_out", composite::out_wrap__foreach_add_Tensor_out);
    m.impl("_foreach_sub.Scalar_out", composite::out_wrap__foreach_sub_Scalar_out);
    m.impl("_foreach_sub.List_out", composite::out_wrap__foreach_sub_List_out);
    m.impl("_foreach_sub.ScalarList_out", composite::out_wrap__foreach_sub_ScalarList_out);
    m.impl("_foreach_mul.Scalar_out", composite::out_wrap__foreach_mul_Scalar_out);
    m.impl("_foreach_mul.List_out", composite::out_wrap__foreach_mul_List_out);
    m.impl("_foreach_mul.ScalarList_out", composite::out_wrap__foreach_mul_ScalarList_out);
    m.impl("_foreach_mul.Tensor_out", composite::out_wrap__foreach_mul_Tensor_out);
    m.impl("_foreach_div.Scalar_out", composite::out_wrap__foreach_div_Scalar_out);
    m.impl("_foreach_div.List_out", composite::out_wrap__foreach_div_List_out);
    m.impl("_foreach_div.ScalarList_out", composite::out_wrap__foreach_div_ScalarList_out);
    m.impl("_foreach_div.Tensor_out", composite::out_wrap__foreach_div_Tensor_out);
    m.impl("_foreach_clamp_max.Scalar_out", composite::out_wrap__foreach_clamp_max_Scalar_out);
    m.impl("_foreach_clamp_max.List_out", composite::out_wrap__foreach_clamp_max_List_out);
    m.impl("_foreach_clamp_max.ScalarList_out", composite::out_wrap__foreach_clamp_max_ScalarList_out);
    m.impl("_foreach_clamp_min.Scalar_out", composite::out_wrap__foreach_clamp_min_Scalar_out);
    m.impl("_foreach_clamp_min.List_out", composite::out_wrap__foreach_clamp_min_List_out);
    m.impl("_foreach_clamp_min.ScalarList_out", composite::out_wrap__foreach_clamp_min_ScalarList_out);
    m.impl("_foreach_maximum.Scalar_out", composite::out_wrap__foreach_maximum_Scalar_out);
    m.impl("_foreach_maximum.List_out", composite::out_wrap__foreach_maximum_List_out);
    m.impl("_foreach_maximum.ScalarList_out", composite::out_wrap__foreach_maximum_ScalarList_out);
    m.impl("_foreach_minimum.Scalar_out", composite::out_wrap__foreach_minimum_Scalar_out);
    m.impl("_foreach_minimum.List_out", composite::out_wrap__foreach_minimum_List_out);
    m.impl("_foreach_minimum.ScalarList_out", composite::out_wrap__foreach_minimum_ScalarList_out);
    m.impl("_foreach_addcdiv.Scalar_out", composite::out_wrap__foreach_addcdiv_Scalar_out);
    m.impl("_foreach_addcdiv.ScalarList_out", composite::out_wrap__foreach_addcdiv_ScalarList_out);
    m.impl("_foreach_addcdiv.Tensor_out", composite::out_wrap__foreach_addcdiv_Tensor_out);
    m.impl("_foreach_addcmul.Scalar_out", composite::out_wrap__foreach_addcmul_Scalar_out);
    m.impl("_foreach_addcmul.ScalarList_out", composite::out_wrap__foreach_addcmul_ScalarList_out);
    m.impl("_foreach_addcmul.Tensor_out", composite::out_wrap__foreach_addcmul_Tensor_out);
    m.impl("_foreach_abs.out", composite::out_wrap__foreach_abs_out);
    m.impl("_foreach_acos.out", composite::out_wrap__foreach_acos_out);
    m.impl("_foreach_asin.out", composite::out_wrap__foreach_asin_out);
    m.impl("_foreach_atan.out", composite::out_wrap__foreach_atan_out);
    m.impl("_foreach_ceil.out", composite::out_wrap__foreach_ceil_out);
    m.impl("_foreach_cos.out", composite::out_wrap__foreach_cos_out);
    m.impl("_foreach_cosh.out", composite::out_wrap__foreach_cosh_out);
    m.impl("_foreach_erf.out", composite::out_wrap__foreach_erf_out);
    m.impl("_foreach_erfc.out", composite::out_wrap__foreach_erfc_out);
    m.impl("_foreach_exp.out", composite::out_wrap__foreach_exp_out);
    m.impl("_foreach_expm1.out", composite::out_wrap__foreach_expm1_out);
    m.impl("_foreach_floor.out", composite::out_wrap__foreach_floor_out);
    m.impl("_foreach_frac.out", composite::out_wrap__foreach_frac_out);
    m.impl("_foreach_lerp.List_out", composite::out_wrap__foreach_lerp_List_out);
    m.impl("_foreach_lerp.Scalar_out", composite::out_wrap__foreach_lerp_Scalar_out);
    m.impl("_foreach_lerp.ScalarList_out", composite::out_wrap__foreach_lerp_ScalarList_out);
    m.impl("_foreach_lgamma.out", composite::out_wrap__foreach_lgamma_out);
    m.impl("_foreach_log.out", composite::out_wrap__foreach_log_out);
    m.impl("_foreach_log10.out", composite::out_wrap__foreach_log10_out);
    m.impl("_foreach_log1p.out", composite::out_wrap__foreach_log1p_out);
    m.impl("_foreach_log2.out", composite::out_wrap__foreach_log2_out);
    m.impl("_foreach_max.out", composite::out_wrap__foreach_max_out);
    m.impl("_foreach_neg.out", composite::out_wrap__foreach_neg_out);
    m.impl("_foreach_norm.Scalar_out", composite::out_wrap__foreach_norm_Scalar_out);
    m.impl("_foreach_powsum.Scalar_out", composite::out_wrap__foreach_powsum_Scalar_out);
    m.impl("_foreach_pow.List_out", composite::out_wrap__foreach_pow_List_out);
    m.impl("_foreach_pow.Scalar_out", composite::out_wrap__foreach_pow_Scalar_out);
    m.impl("_foreach_pow.ScalarList_out", composite::out_wrap__foreach_pow_ScalarList_out);
    m.impl("_foreach_reciprocal.out", composite::out_wrap__foreach_reciprocal_out);
    m.impl("_foreach_round.out", composite::out_wrap__foreach_round_out);
    m.impl("_foreach_rsqrt.out", composite::out_wrap__foreach_rsqrt_out);
    m.impl("_foreach_sigmoid.out", composite::out_wrap__foreach_sigmoid_out);
    m.impl("_foreach_sign.out", composite::out_wrap__foreach_sign_out);
    m.impl("_foreach_sin.out", composite::out_wrap__foreach_sin_out);
    m.impl("_foreach_sinh.out", composite::out_wrap__foreach_sinh_out);
    m.impl("_foreach_sqrt.out", composite::out_wrap__foreach_sqrt_out);
    m.impl("_foreach_tan.out", composite::out_wrap__foreach_tan_out);
    m.impl("_foreach_tanh.out", composite::out_wrap__foreach_tanh_out);
    m.impl("_foreach_trunc.out", composite::out_wrap__foreach_trunc_out);
    m.impl("_foreach_zero.out", composite::out_wrap__foreach_zero_out);
    m.impl("_foreach_clone.out", composite::out_wrap__foreach_clone_out);
    m.impl("_foreach_copy.out", composite::out_wrap__foreach_copy_out);
    m.impl("bucketize.Scalar_out", composite::out_wrap_bucketize_Scalar_out);
    m.impl("glu_jvp.out", composite::out_wrap_glu_jvp_out);
    m.impl("glu_backward_jvp.out", composite::out_wrap_glu_backward_jvp_out);
    m.impl("rrelu_with_noise_backward.out", composite::out_wrap_rrelu_with_noise_backward_out);
    m.impl("mkldnn_adaptive_avg_pool2d_backward.out", composite::out_wrap_mkldnn_adaptive_avg_pool2d_backward_out);
    m.impl("_adaptive_avg_pool2d.out", composite::out_wrap__adaptive_avg_pool2d_out);
    m.impl("_adaptive_avg_pool2d_backward.out", composite::out_wrap__adaptive_avg_pool2d_backward_out);
    m.impl("_adaptive_avg_pool3d.out", composite::out_wrap__adaptive_avg_pool3d_out);
    m.impl("_adaptive_avg_pool3d_backward.out", composite::out_wrap__adaptive_avg_pool3d_backward_out);
    m.impl("upsample_linear1d.vec_out", composite::out_wrap_upsample_linear1d_vec_out);
    m.impl("upsample_bilinear2d.vec_out", composite::out_wrap_upsample_bilinear2d_vec_out);
    m.impl("_upsample_bilinear2d_aa.vec_out", composite::out_wrap__upsample_bilinear2d_aa_vec_out);
    m.impl("upsample_trilinear3d.vec_out", composite::out_wrap_upsample_trilinear3d_vec_out);
    m.impl("upsample_bicubic2d.vec_out", composite::out_wrap_upsample_bicubic2d_vec_out);
    m.impl("_upsample_bicubic2d_aa.vec_out", composite::out_wrap__upsample_bicubic2d_aa_vec_out);
    m.impl("_upsample_lanczos2d_aa.vec_out", composite::out_wrap__upsample_lanczos2d_aa_vec_out);
    m.impl("upsample_nearest1d.vec_out", composite::out_wrap_upsample_nearest1d_vec_out);
    m.impl("_upsample_nearest_exact1d.vec_out", composite::out_wrap__upsample_nearest_exact1d_vec_out);
    m.impl("upsample_nearest2d.vec_out", composite::out_wrap_upsample_nearest2d_vec_out);
    m.impl("_upsample_nearest_exact2d.vec_out", composite::out_wrap__upsample_nearest_exact2d_vec_out);
    m.impl("upsample_nearest3d.vec_out", composite::out_wrap_upsample_nearest3d_vec_out);
    m.impl("_upsample_nearest_exact3d.vec_out", composite::out_wrap__upsample_nearest_exact3d_vec_out);
    m.impl("_slow_conv2d_backward.output_mask_out", composite::out_wrap__slow_conv2d_backward_output_mask_out);
    m.impl("conv_depthwise3d.out", composite::out_wrap_conv_depthwise3d_out);
    m.impl("slow_conv_dilated2d.out", composite::out_wrap_slow_conv_dilated2d_out);
    m.impl("slow_conv_dilated3d.out", composite::out_wrap_slow_conv_dilated3d_out);
    m.impl("isinf.out", composite::out_wrap_isinf_out);
    m.impl("linalg_matrix_exp.out", composite::out_wrap_linalg_matrix_exp_out);
    m.impl("linalg_matrix_sqrth.out", composite::out_wrap_linalg_matrix_sqrth_out);
    m.impl("_test_optional_intlist.out", composite::out_wrap__test_optional_intlist_out);
    m.impl("_test_optional_filled_intlist.out", composite::out_wrap__test_optional_filled_intlist_out);
    m.impl("_test_optional_floatlist.out", composite::out_wrap__test_optional_floatlist_out);
    m.impl("_test_warn_in_autograd.out", composite::out_wrap__test_warn_in_autograd_out);
    m.impl("_test_autograd_multiple_dispatch.fullcoverage_out", composite::out_wrap__test_autograd_multiple_dispatch_fullcoverage_out);
    m.impl("_test_autograd_multiple_dispatch_view_copy.out", composite::out_wrap__test_autograd_multiple_dispatch_view_copy_out);
    m.impl("segment_reduce.out", composite::out_wrap_segment_reduce_out);
    m.impl("_segment_reduce_backward.out", composite::out_wrap__segment_reduce_backward_out);
    m.impl("_nested_tensor_from_tensor_list.out", composite::out_wrap__nested_tensor_from_tensor_list_out);
    m.impl("_fw_primal_copy.out", composite::out_wrap__fw_primal_copy_out);
    m.impl("_make_dual_copy.out", composite::out_wrap__make_dual_copy_out);
    m.impl("view_as_real_copy.out", composite::out_wrap_view_as_real_copy_out);
    m.impl("view_as_complex_copy.out", composite::out_wrap_view_as_complex_copy_out);
    m.impl("_conj_copy.out", composite::out_wrap__conj_copy_out);
    m.impl("_neg_view_copy.out", composite::out_wrap__neg_view_copy_out);
    m.impl("as_strided_copy.out", composite::out_wrap_as_strided_copy_out);
    m.impl("_sparse_broadcast_to_copy.out", composite::out_wrap__sparse_broadcast_to_copy_out);
    m.impl("diagonal_copy.out", composite::out_wrap_diagonal_copy_out);
    m.impl("expand_copy.out", composite::out_wrap_expand_copy_out);
    m.impl("permute_copy.out", composite::out_wrap_permute_copy_out);
    m.impl("_reshape_alias_copy.out", composite::out_wrap__reshape_alias_copy_out);
    m.impl("select_copy.int_out", composite::out_wrap_select_copy_int_out);
    m.impl("detach_copy.out", composite::out_wrap_detach_copy_out);
    m.impl("slice_copy.Tensor_out", composite::out_wrap_slice_copy_Tensor_out);
    m.impl("squeeze_copy.out", composite::out_wrap_squeeze_copy_out);
    m.impl("squeeze_copy.dim_out", composite::out_wrap_squeeze_copy_dim_out);
    m.impl("squeeze_copy.dims_out", composite::out_wrap_squeeze_copy_dims_out);
    m.impl("t_copy.out", composite::out_wrap_t_copy_out);
    m.impl("transpose_copy.int_out", composite::out_wrap_transpose_copy_int_out);
    m.impl("unsqueeze_copy.out", composite::out_wrap_unsqueeze_copy_out);
    m.impl("_indices_copy.out", composite::out_wrap__indices_copy_out);
    m.impl("_values_copy.out", composite::out_wrap__values_copy_out);
    m.impl("indices_copy.out", composite::out_wrap_indices_copy_out);
    m.impl("values_copy.out", composite::out_wrap_values_copy_out);
    m.impl("crow_indices_copy.out", composite::out_wrap_crow_indices_copy_out);
    m.impl("col_indices_copy.out", composite::out_wrap_col_indices_copy_out);
    m.impl("ccol_indices_copy.out", composite::out_wrap_ccol_indices_copy_out);
    m.impl("row_indices_copy.out", composite::out_wrap_row_indices_copy_out);
    m.impl("view_copy.out", composite::out_wrap_view_copy_out);
    m.impl("view_copy.dtype_out", composite::out_wrap_view_copy_dtype_out);
    m.impl("unfold_copy.out", composite::out_wrap_unfold_copy_out);
    m.impl("alias_copy.out", composite::out_wrap_alias_copy_out);
    m.impl("to_padded_tensor.out", composite::out_wrap_to_padded_tensor_out);
    m.impl("_transformer_encoder_layer_fwd.out", composite::out_wrap__transformer_encoder_layer_fwd_out);
    m.impl("_native_multi_head_attention.out", composite::out_wrap__native_multi_head_attention_out);
    m.impl("scaled_dot_product_attention.out", composite::out_wrap_scaled_dot_product_attention_out);
    m.impl("_triton_scaled_dot_attention.out", composite::out_wrap__triton_scaled_dot_attention_out);
    m.impl("_triton_multi_head_attention.out", composite::out_wrap__triton_multi_head_attention_out);
    m.impl("_foobar.out", composite::out_wrap__foobar_out);
    m.impl("_fused_adam.out", composite::out_wrap__fused_adam_out);
    m.impl("_fused_adam.tensor_lr_out", composite::out_wrap__fused_adam_tensor_lr_out);
    m.impl("_fused_adamw.out", composite::out_wrap__fused_adamw_out);
    m.impl("_fused_adamw.tensor_lr_out", composite::out_wrap__fused_adamw_tensor_lr_out);
    m.impl("_fused_sgd.out", composite::out_wrap__fused_sgd_out);
    m.impl("_fused_sgd.tensor_lr_out", composite::out_wrap__fused_sgd_tensor_lr_out);
    m.impl("_fused_adagrad.out", composite::out_wrap__fused_adagrad_out);
    m.impl("_fused_adagrad.tensor_lr_out", composite::out_wrap__fused_adagrad_tensor_lr_out);
}
} // namespace tensorplay
