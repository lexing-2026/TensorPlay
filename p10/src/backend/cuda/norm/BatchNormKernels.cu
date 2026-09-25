#include "BatchNormNativeKernels.cuh"
#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "CUDAContext.h"
#include "Exception.h"
#include "CUDNNUtils.h"
#include "Half.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "BFloat16.h"
#include "Scalar.h"

#include <cuda_runtime.h>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace ops = tensorplay::tpx::ops;

#ifdef USE_CUDNN

namespace {

bool defined(const std::optional<Tensor>& value) {
    return value.has_value() && value->defined();
}

cudnnTensorDescriptor_t derive_bn_descriptor(cudnnTensorDescriptor_t x_desc) {
    cudnnTensorDescriptor_t bn_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
        bn_desc, x_desc, CUDNN_BATCHNORM_SPATIAL));
    return bn_desc;
}

Tensor batch_norm_view(const Tensor& input) {
    std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(input.shape());
    while (shape.size() < 4) shape.push_back(1);
    if (shape.size() > 5) {
        TP_THROW(RuntimeError, "batch_norm CUDA supports 2D through 5D inputs");
    }
    if (shape.size() == static_cast<size_t>(input.dim())) return input;
    return input.reshape(shape);
}

void check_batch_norm_input(const Tensor& input) {
    if (input.dim() < 2 || input.dim() > 5) {
        TP_THROW(RuntimeError, "batch_norm CUDA supports 2D through 5D inputs");
    }
    if (input.dtype() != DType::Float32) {
        TP_THROW(NotImplementedError, "batch_norm CUDA currently supports Float32 only");
    }
}

__global__ void inverse_variance_kernel(
    int64_t channels, const float* variance, float* inverse_variance, float eps) {
    int64_t channel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel < channels) {
        inverse_variance[channel] = 1.0f / sqrtf(variance[channel] + eps);
    }
}

void check_cuda_launch(const char* name) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string(name) + ": " + cudaGetErrorString(error));
    }
}

}

Tensor batch_norm_cuda(
    const Tensor& input,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    bool training,
    double momentum,
    double eps) {
    check_batch_norm_input(input);

    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor input_bn = batch_norm_view(input_contig);
    Tensor output = Tensor::empty_like(input_contig, DType::Undefined, input_contig.device());
    Tensor output_bn = batch_norm_view(output);

    int64_t channels = input.size(1);
    Tensor scale = defined(weight_opt)
        ? *weight_opt
        : Tensor::ones({channels}, DType::Float32, input.device());
    Tensor bias = defined(bias_opt)
        ? *bias_opt
        : Tensor::zeros({channels}, DType::Float32, input.device());
    Tensor running_mean = defined(running_mean_opt)
        ? *running_mean_opt
        : Tensor::zeros({channels}, DType::Float32, input.device());
    Tensor running_var = defined(running_var_opt)
        ? *running_var_opt
        : Tensor::ones({channels}, DType::Float32, input.device());

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    cudnnTensorDescriptor_t x_desc = createTensorDescriptor(input_bn);
    cudnnTensorDescriptor_t y_desc = createTensorDescriptor(output_bn);
    cudnnTensorDescriptor_t bn_desc = derive_bn_descriptor(x_desc);

    float alpha = 1.0f;
    float beta = 0.0f;
    Tensor saved_mean;
    Tensor saved_inverse_variance;

    if (training) {
        saved_mean = Tensor::empty({channels}, DType::Float32, input.device());
        saved_inverse_variance = Tensor::empty({channels}, DType::Float32, input.device());
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &beta,
            x_desc,
            input_bn.data_ptr(),
            y_desc,
            output_bn.data_ptr(),
            bn_desc,
            scale.data_ptr(),
            bias.data_ptr(),
            momentum,
            running_mean.data_ptr(),
            running_var.data_ptr(),
            eps,
            saved_mean.data_ptr(),
            saved_inverse_variance.data_ptr()));
    } else {
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &beta,
            x_desc,
            input_bn.data_ptr(),
            y_desc,
            output_bn.data_ptr(),
            bn_desc,
            scale.data_ptr(),
            bias.data_ptr(),
            running_mean.data_ptr(),
            running_var.data_ptr(),
            eps));
    }

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    return output;
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    bool training,
    double eps) {
    check_batch_norm_input(input);
    if (grad_output.dtype() != DType::Float32 || grad_output.numel() != input.numel()) {
        TP_THROW(RuntimeError, "batch_norm_backward CUDA expects a Float32 gradient matching input");
    }

    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor grad_output_contig = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor input_bn = batch_norm_view(input_contig);
    Tensor grad_output_bn = batch_norm_view(grad_output_contig);
    Tensor grad_input = Tensor::empty_like(input_contig, DType::Undefined, input_contig.device());
    Tensor grad_input_bn = batch_norm_view(grad_input);

    int64_t channels = input.size(1);
    bool has_weight = defined(weight_opt);
    Tensor scale = has_weight
        ? *weight_opt
        : Tensor::ones({channels}, DType::Float32, input.device());
    Tensor grad_scale = Tensor::zeros({channels}, DType::Float32, input.device());
    Tensor grad_bias = Tensor::zeros({channels}, DType::Float32, input.device());

    Tensor saved_mean = Tensor::empty({channels}, DType::Float32, input.device());
    Tensor saved_inverse_variance = Tensor::empty({channels}, DType::Float32, input.device());
    if (training) {
        Tensor forward_output = Tensor::empty_like(input_contig, DType::Undefined, input_contig.device());
        Tensor forward_output_bn = batch_norm_view(forward_output);
        Tensor scratch_running_mean = Tensor::zeros({channels}, DType::Float32, input.device());
        Tensor scratch_running_var = Tensor::ones({channels}, DType::Float32, input.device());
        Tensor zero_bias = Tensor::zeros({channels}, DType::Float32, input.device());

        cudnnHandle_t handle = CUDAContext::getCudnnHandle();
        cudnnTensorDescriptor_t x_desc = createTensorDescriptor(input_bn);
        cudnnTensorDescriptor_t y_desc = createTensorDescriptor(forward_output_bn);
        cudnnTensorDescriptor_t bn_desc = derive_bn_descriptor(x_desc);
        float alpha = 1.0f;
        float beta = 0.0f;
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &beta,
            x_desc,
            input_bn.data_ptr(),
            y_desc,
            forward_output_bn.data_ptr(),
            bn_desc,
            scale.data_ptr(),
            zero_bias.data_ptr(),
            1.0,
            scratch_running_mean.data_ptr(),
            scratch_running_var.data_ptr(),
            eps,
            saved_mean.data_ptr(),
            saved_inverse_variance.data_ptr()));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    } else {
        if (!defined(running_mean_opt) || !defined(running_var_opt)) {
            TP_THROW(RuntimeError, "batch_norm_backward CUDA eval mode requires running statistics");
        }
        Tensor running_mean = *running_mean_opt;
        Tensor running_var = *running_var_opt;
        int threads = 256;
        int blocks = static_cast<int>((channels + threads - 1) / threads);
        inverse_variance_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            channels,
            running_var.data_ptr<float>(),
            saved_inverse_variance.data_ptr<float>(),
            static_cast<float>(eps));
        cudaMemcpyAsync(
            saved_mean.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            static_cast<size_t>(channels) * sizeof(float),
            cudaMemcpyDeviceToDevice,
            getCurrentCUDAStream().stream());
        check_cuda_launch("batch_norm_backward eval statistics");
    }

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    cudnnTensorDescriptor_t x_desc = createTensorDescriptor(input_bn);
    cudnnTensorDescriptor_t dy_desc = createTensorDescriptor(grad_output_bn);
    cudnnTensorDescriptor_t dx_desc = createTensorDescriptor(grad_input_bn);
    cudnnTensorDescriptor_t bn_desc = derive_bn_descriptor(x_desc);
    float alpha_data = 1.0f;
    float beta_data = 0.0f;
    float alpha_param = 1.0f;
    float beta_param = 0.0f;
    CUDNN_CHECK(cudnnBatchNormalizationBackward(
        handle,
        CUDNN_BATCHNORM_SPATIAL,
        &alpha_data,
        &beta_data,
        &alpha_param,
        &beta_param,
        x_desc,
        input_bn.data_ptr(),
        dy_desc,
        grad_output_bn.data_ptr(),
        dx_desc,
        grad_input_bn.data_ptr(),
        bn_desc,
        scale.data_ptr(),
        grad_scale.data_ptr(),
        grad_bias.data_ptr(),
        eps,
        saved_mean.data_ptr(),
        saved_inverse_variance.data_ptr()));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dx_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dy_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));

    if (!has_weight) {
        grad_scale = Tensor();
        grad_bias = Tensor();
    }
    return std::make_tuple(grad_input, grad_scale, grad_bias);
}

#else

Tensor batch_norm_cuda(
    const Tensor&, const std::optional<Tensor>&, const std::optional<Tensor>&,
    const std::optional<Tensor>&, const std::optional<Tensor>&, bool, double, double) {
    TP_THROW(NotImplementedError, "batch_norm CUDA requires cuDNN");
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda(
    const Tensor&, const Tensor&, const std::optional<Tensor>&,
    const std::optional<Tensor>&, const std::optional<Tensor>&, bool, double) {
    TP_THROW(NotImplementedError, "batch_norm_backward CUDA requires cuDNN");
}

#endif

namespace batch_norm {
namespace {

void check_bn_status(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError,
                 std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

void check_bn_launch(const char* operation) {
    check_bn_status(cudaGetLastError(), operation);
}

// Parameters wider than the data select the accumulate type for the whole
// kernel; otherwise the statistics share the operand type.
bool is_mixed_type(const Tensor& input, const std::optional<Tensor>& param) {
    if (!param.has_value() || !param->defined()) return false;
    return param->dtype() != input.dtype();
}

DType accumulate_dtype(DType dtype) {
    return dtype == DType::Float64 ? DType::Float64 : DType::Float32;
}

#define TP_BN_DISPATCH_FLOATING(DTYPE, NAME, ...)                           \
    switch (DTYPE) {                                                       \
        case DType::Float32: {                                             \
            using scalar_t = float;                                        \
            using accscalar_t = float;                                     \
            __VA_ARGS__;                                                   \
            break;                                                         \
        }                                                                  \
        case DType::Float64: {                                             \
            using scalar_t = double;                                       \
            using accscalar_t = double;                                    \
            __VA_ARGS__;                                                   \
            break;                                                         \
        }                                                                  \
        case DType::Float16: {                                             \
            using scalar_t = tensorplay::Half;                             \
            using accscalar_t = float;                                     \
            __VA_ARGS__;                                                   \
            break;                                                         \
        }                                                                  \
        case DType::BFloat16: {                                            \
            using scalar_t = tensorplay::BFloat16;                         \
            using accscalar_t = float;                                     \
            __VA_ARGS__;                                                   \
            break;                                                         \
        }                                                                  \
        default:                                                           \
            TP_THROW(NotImplementedError,                                  \
                     std::string(NAME) +                                   \
                         " supports floating-point dtypes only");           \
    }

// Layout classes mirror the operand layout: the dense kernels treat the
// channel axis as the plane, the channels-last kernels walk the reduction
// axis with C contiguous lanes, and the strided kernel addresses every
// operand through explicit strides.
enum class Layout { Contiguous, ChannelsLast, General };

bool is_channels_last(const Tensor& t) {
    return t.is_contiguous(MemoryFormat::ChannelsLast) ||
           t.is_contiguous(MemoryFormat::ChannelsLast3d) ||
           (t.is_contiguous() && t.dim() > 1 && t.strides()[1] == 1);
}

Layout choose_layout(const Tensor& t) {
    if (t.numel() > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return Layout::General;
    }
    if (t.is_contiguous()) {
        return t.strides()[1] == 1 ? Layout::ChannelsLast : Layout::Contiguous;
    }
    if (t.is_contiguous(MemoryFormat::ChannelsLast) ||
        t.is_contiguous(MemoryFormat::ChannelsLast3d)) {
        return Layout::ChannelsLast;
    }
    return Layout::General;
}

Layout choose_layout(const Tensor& a, const Tensor& b) {
    const Layout la = choose_layout(a);
    const Layout lb = choose_layout(b);
    if (la == Layout::General || lb == Layout::General) return Layout::General;
    return la == lb ? la : Layout::General;
}

// (batch, channel, feature) view: the trailing dimensions are merged.
Tensor merge_features(const Tensor& t) {
    if (t.dim() == 3) return t;
    int64_t tail = 1;
    for (int64_t d = 2; d < static_cast<int64_t>(t.dim()); ++d) {
        tail *= t.size(d);
    }
    return t.contiguous().view({t.size(0), t.size(1), tail});
}

BnStridedLayout make_strided_layout(const Tensor& input, const Tensor& out) {
    BnStridedLayout layout;
    const int64_t ndim = static_cast<int64_t>(input.dim());
    layout.ndim = static_cast<int>(ndim);
    layout.numel = input.numel();
    const std::vector<int64_t> in_strides = input.strides();
    const std::vector<int64_t> out_strides = out.strides();
    for (int64_t d = 0; d < ndim; ++d) {
        layout.sizes[d] = input.size(d);
        layout.in_stride[d] = in_strides[d];
        layout.out_stride[d] = out_strides[d];
    }
    layout.channel_size = ndim > 1 ? input.size(1) : 1;
    int64_t tail = 1;
    for (int64_t d = 2; d < ndim; ++d) tail *= input.size(d);
    layout.channel_div = tail;
    return layout;
}

int pointwise_threads_x(int64_t feature_size) {
    return std::max(bn_get_num_threads(feature_size / 4),
                    std::min(bn_get_num_threads(feature_size), 64));
}

int pointwise_threads_y(int threads_x) { return std::max(64 / threads_x, 1); }

// Tile geometry for the channels-last kernels: the channel axis wants a
// narrow tile, the reduction axis a deep one, and the reduction axis is
// capped so a single block can finish the merge in shared memory.
void flexible_launch_configs(int64_t reduction, int64_t stride, dim3* block,
                             dim3* grid, bool coop_flag = false) {
    int block_x = std::min(bn_last_pow2(stride), kBnOptimalTileW);
    int block_y = std::min(
        bn_last_pow2(bn_ceil_div(reduction, kBnElementsPerIter)),
        kBnMaxBlockSize / block_x);
    if (block_x * block_y != kBnMaxBlockSize) {
        block_x = std::min(bn_last_pow2(stride), kBnMaxBlockSize / block_y);
    }
    int grid_x = static_cast<int>(bn_ceil_div(stride, block_x));
    int grid_y = std::min(
        static_cast<int>(bn_ceil_div(reduction, block_y * kBnElementsPerIter)),
        kBnMaxHBlock);
    if (coop_flag) {
        grid_y = grid_y < 8 ? 1 : grid_y;
    }
    block->x = static_cast<unsigned int>(block_x);
    block->y = static_cast<unsigned int>(block_y);
    block->z = 1;
    grid->x = static_cast<unsigned int>(grid_x);
    grid->y = static_cast<unsigned int>(grid_y);
    grid->z = 1;
}

dim3 pointwise_blocks(int64_t channels, int64_t batch, int threads_y) {
    const int64_t by_batch = bn_ceil_div(batch, threads_y);
    const int64_t per_channel = (256 * 1024) / std::max<int64_t>(channels, 1);
    dim3 blocks(static_cast<unsigned int>(channels),
                static_cast<unsigned int>(
                    std::max<int64_t>(1, std::min(per_channel, by_batch))));
    blocks.y = std::min<unsigned int>(blocks.y, kBnMaxGridSize);
    return blocks;
}

}  // namespace

// ---------------------------------------------------------------------------
// batch_norm_stats: per-channel mean and reciprocal standard deviation
// ---------------------------------------------------------------------------

template <typename scalar_t, typename accscalar_t, typename VarTransform,
          typename index_t>
void launch_collect_statistics(Tensor& save_mean, Tensor& save_transformed_var,
                               const Tensor& input, double epsilon) {
    const Tensor input_reshaped = merge_features(input);
    auto in_acc = bn_acc3<const scalar_t, index_t>(input_reshaped);
    auto mean_acc = bn_acc1<accscalar_t, index_t>(save_mean);
    auto var_acc = bn_acc1<accscalar_t, index_t>(save_transformed_var);
    cudaStream_t stream = getCurrentCUDAStream().stream();
    const dim3 blocks(static_cast<unsigned int>(in_acc.size(1)));
    const int tf = bn_get_num_threads(in_acc.size(2));
    const dim3 threads(static_cast<unsigned int>(tf),
                       static_cast<unsigned int>(
                           std::max(1, kBnMaxBlockSize / tf)));
    bn_collect_statistics_kernel<VarTransform, scalar_t, scalar_t, accscalar_t,
                                 index_t>
        <<<blocks, threads, 0, stream>>>(in_acc, epsilon,
                                         static_cast<accscalar_t>(0), mean_acc,
                                         var_acc);
    check_bn_launch("batch_norm_stats");
}

template <typename scalar_t, typename accscalar_t, typename VarTransform>
void launch_collect_statistics_channels_last(Tensor& save_mean,
                                             Tensor& save_transformed_var,
                                             const Tensor& input,
                                             double epsilon) {
    const int64_t stride = input.size(1);
    const int64_t reduction_size = input.numel() / std::max<int64_t>(stride, 1);
    dim3 block;
    dim3 grid;
    flexible_launch_configs(reduction_size, stride, &block, &grid, true);

    Tensor staging_data;
    Tensor semaphores;
    if (grid.y > 1) {
        staging_data = Tensor::empty({4 * stride * grid.y}, save_mean.dtype(),
                                     save_mean.device());
        semaphores = Tensor::zeros({grid.x}, DType::Int32, input.device());
    }
    accscalar_t* staging_ptr =
        grid.y > 1 ? staging_data.data_ptr<accscalar_t>() : nullptr;
    int* semaphores_ptr = grid.y > 1 ? semaphores.data_ptr<int32_t>() : nullptr;
    cudaStream_t stream = getCurrentCUDAStream().stream();
    bn_collect_statistics_channels_last_kernel<VarTransform, scalar_t,
                                               accscalar_t,
                                               kBnElementsPerIter>
        <<<grid, block, 0, stream>>>(
            input.data_ptr<scalar_t>(), save_mean.data_ptr<accscalar_t>(),
            save_transformed_var.data_ptr<accscalar_t>(), staging_ptr,
            semaphores_ptr, static_cast<int>(reduction_size),
            static_cast<int>(stride), static_cast<accscalar_t>(epsilon));
    check_bn_launch("batch_norm_stats channels-last");
}

template <typename scalar_t, typename accscalar_t, typename VarTransform,
          typename index_t>
void collect_statistics_layout(const Tensor& input, Tensor& save_mean,
                               Tensor& save_transformed_var, double epsilon) {
    if (is_channels_last(input)) {
        // A layout with a unit channel stride already stores the reduction
        // axis contiguously, so the channels-last kernels read it as is.
        launch_collect_statistics_channels_last<scalar_t, accscalar_t,
                                                VarTransform>(
            save_mean, save_transformed_var, input, epsilon);
        return;
    }
    launch_collect_statistics<scalar_t, accscalar_t, VarTransform, index_t>(
        save_mean, save_transformed_var, input, epsilon);
}

template <typename scalar_t, typename accscalar_t, typename VarTransform>
void collect_statistics(const Tensor& input, Tensor& save_mean,
                        Tensor& save_transformed_var, double epsilon) {
    if (input.numel() <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        collect_statistics_layout<scalar_t, accscalar_t, VarTransform, int32_t>(
            input, save_mean, save_transformed_var, epsilon);
    } else {
        collect_statistics_layout<scalar_t, accscalar_t, VarTransform, int64_t>(
            input, save_mean, save_transformed_var, epsilon);
    }
}

std::tuple<Tensor, Tensor> batch_norm_stats_cuda(const Tensor& input,
                                                 double epsilon) {
    const int64_t channels = input.size(1);
    const DType acc_dtype = accumulate_dtype(input.dtype());
    Tensor save_mean =
        Tensor::empty({channels}, acc_dtype, input.device());
    Tensor save_invstd =
        Tensor::empty({channels}, acc_dtype, input.device());
    TP_BN_DISPATCH_FLOATING(input.dtype(), "batch_norm_stats", {
        collect_statistics<scalar_t, accscalar_t, BnInvStd>(
            input, save_mean, save_invstd, epsilon);
    });
    return std::make_tuple(save_mean, save_invstd);
}

// ---------------------------------------------------------------------------
// batch_norm_elemt: affine transform with precomputed statistics
// ---------------------------------------------------------------------------

template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t, typename index_t>
void launch_transform_input(const Tensor& input, Tensor& output,
                            const std::optional<Tensor>& weight,
                            const std::optional<Tensor>& bias,
                            const Tensor& mean, const Tensor& invstd) {
    const Tensor input_reshaped = merge_features(input);
    const std::vector<int64_t> out_shape = {input_reshaped.size(0),
                                            input_reshaped.size(1),
                                            input_reshaped.size(2)};
    Tensor output_reshaped = static_cast<std::vector<int64_t>>(output.shape()) ==
                                     out_shape
                                 ? output
                                 : output.view(out_shape);
    auto in_acc = bn_acc3<const input_scalar_t, index_t>(input_reshaped);
    auto out_acc = bn_acc3<input_scalar_t, index_t>(output_reshaped);
    auto mean_acc =
        bn_acc_or_dummy<stat_accscalar_t, index_t>(std::make_optional(mean));
    auto invstd_acc =
        bn_acc_or_dummy<stat_accscalar_t, index_t>(std::make_optional(invstd));
    auto weight_acc = bn_acc_or_dummy<stat_scalar_t, index_t>(weight);
    auto bias_acc = bn_acc_or_dummy<stat_scalar_t, index_t>(bias);
    cudaStream_t stream = getCurrentCUDAStream().stream();
    const int tf = pointwise_threads_x(in_acc.size(2));
    const int tb = pointwise_threads_y(tf);
    const dim3 blocks =
        pointwise_blocks(in_acc.size(1), in_acc.size(0), tb);
    const dim3 threads(static_cast<unsigned int>(tf),
                       static_cast<unsigned int>(tb));
    // The transform reads a precomputed inverse deviation, so epsilon is
    // unused on this path.
    bn_transform_input_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t,
                              true, index_t>
        <<<blocks, threads, 0, stream>>>(
            in_acc, out_acc, mean_acc, invstd_acc, weight_acc, bias_acc,
            static_cast<stat_accscalar_t>(1e-5));
    check_bn_launch("batch_norm_elemt");
}

template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t, typename index_t>
void launch_elementwise_strided(const Tensor& input, Tensor& output,
                                const std::optional<Tensor>& weight,
                                const std::optional<Tensor>& bias,
                                const Tensor& mean, const Tensor& invstd) {
    const BnStridedLayout layout = make_strided_layout(input, output);
    const int threads = 256;
    const int blocks = static_cast<int>(
        bn_ceil_div(layout.numel, static_cast<int64_t>(threads)));
    cudaStream_t stream = getCurrentCUDAStream().stream();
    const stat_scalar_t* weight_p =
        (weight.has_value() && weight->defined()) ? weight->data_ptr<stat_scalar_t>()
                                                  : nullptr;
    const stat_scalar_t* bias_p =
        (bias.has_value() && bias->defined()) ? bias->data_ptr<stat_scalar_t>()
                                              : nullptr;
    const stat_accscalar_t* mean_p = mean.data_ptr<stat_accscalar_t>();
    const stat_accscalar_t* invstd_p = invstd.data_ptr<stat_accscalar_t>();
    bn_elementwise_strided_kernel<input_scalar_t, stat_scalar_t,
                                  stat_accscalar_t>
        <<<blocks, threads, 0, stream>>>(layout, input.data_ptr<input_scalar_t>(),
                                         output.data_ptr<input_scalar_t>(),
                                         weight_p, bias_p, mean_p, invstd_p);
    check_bn_launch("batch_norm_elemt strided");
}

template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t>
void launch_transform_input_channels_last(const Tensor& input, Tensor& output,
                                          const std::optional<Tensor>& weight,
                                          const std::optional<Tensor>& bias,
                                          const Tensor& mean,
                                          const Tensor& invstd) {
    const int64_t stride = input.size(1);
    const int64_t reduction_size = input.numel() / std::max<int64_t>(stride, 1);
    dim3 block;
    dim3 grid;
    flexible_launch_configs(reduction_size, stride, &block, &grid);
    const bool weight_defined = weight.has_value() && weight->defined();
    const bool bias_defined = bias.has_value() && bias->defined();
    const stat_scalar_t* weight_ptr =
        weight_defined ? weight->data_ptr<stat_scalar_t>() : nullptr;
    const stat_scalar_t* bias_ptr =
        bias_defined ? bias->data_ptr<stat_scalar_t>() : nullptr;
    cudaStream_t stream = getCurrentCUDAStream().stream();
    bn_transform_input_channels_last_kernel<input_scalar_t, stat_accscalar_t,
                                            stat_scalar_t, kBnElementsPerIter>
        <<<grid, block, 0, stream>>>(
            input.data_ptr<input_scalar_t>(),
            mean.data_ptr<stat_accscalar_t>(),
            invstd.data_ptr<stat_accscalar_t>(), weight_ptr, bias_ptr,
            output.data_ptr<input_scalar_t>(), static_cast<int>(reduction_size),
            static_cast<int>(stride));
    check_bn_launch("batch_norm_elemt channels-last");
}

template <typename input_scalar_t, typename stat_accscalar_t>
void elementwise_dispatch(const Tensor& input, Tensor& output,
                          const std::optional<Tensor>& weight,
                          const std::optional<Tensor>& bias,
                          const Tensor& mean, const Tensor& invstd) {
    if (choose_layout(input) == Layout::General) {
        if (input.numel() <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            launch_elementwise_strided<input_scalar_t, input_scalar_t,
                                       stat_accscalar_t, int32_t>(
                input, output, weight, bias, mean, invstd);
        } else {
            launch_elementwise_strided<input_scalar_t, input_scalar_t,
                                       stat_accscalar_t, int64_t>(
                input, output, weight, bias, mean, invstd);
        }
        return;
    }
    const bool mixed = is_mixed_type(input, weight) || is_mixed_type(input, bias);
    if (is_channels_last(input)) {
        const DType second_dtype =
            (weight.has_value() && weight->defined())
                ? weight->dtype()
                : ((bias.has_value() && bias->defined()) ? bias->dtype()
                                                          : input.dtype());
        if (mixed || second_dtype != input.dtype()) {
            launch_transform_input_channels_last<input_scalar_t, stat_accscalar_t,
                                                stat_accscalar_t>(
                input, output, weight, bias, mean, invstd);
        } else {
            launch_transform_input_channels_last<input_scalar_t, input_scalar_t,
                                                stat_accscalar_t>(
                input, output, weight, bias, mean, invstd);
        }
        return;
    }
    const Tensor input_dense = merge_features(input);
    if (mixed) {
        if (input.numel() <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            launch_transform_input<input_scalar_t, stat_accscalar_t,
                                   stat_accscalar_t, int32_t>(
                input_dense, output, weight, bias, mean, invstd);
        } else {
            launch_transform_input<input_scalar_t, stat_accscalar_t,
                                   stat_accscalar_t, int64_t>(
                input_dense, output, weight, bias, mean, invstd);
        }
    } else {
        if (input.numel() <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            launch_transform_input<input_scalar_t, input_scalar_t,
                                   stat_accscalar_t, int32_t>(
                input_dense, output, weight, bias, mean, invstd);
        } else {
            launch_transform_input<input_scalar_t, input_scalar_t,
                                   stat_accscalar_t, int64_t>(
                input_dense, output, weight, bias, mean, invstd);
        }
    }
}

void batch_norm_elementwise(Tensor& output, const Tensor& input,
                            const std::optional<Tensor>& weight,
                            const std::optional<Tensor>& bias,
                            const Tensor& mean, const Tensor& invstd) {
    TP_BN_DISPATCH_FLOATING(input.dtype(), "batch_norm_elemt", {
        elementwise_dispatch<scalar_t, accscalar_t>(input, output, weight, bias,
                                                    mean, invstd);
    });
}

Tensor batch_norm_elemt_cuda(const Tensor& input,
                             const std::optional<Tensor>& weight,
                             const std::optional<Tensor>& bias,
                             const Tensor& mean, const Tensor& invstd,
                             double epsilon) {
    (void)epsilon;
    Tensor output = Tensor::empty_like(input);
    batch_norm_elementwise(output, input, weight, bias, mean, invstd);
    return output;
}

Tensor& batch_norm_elemt_cuda_out(const Tensor& input,
                                  const std::optional<Tensor>& weight,
                                  const std::optional<Tensor>& bias,
                                  const Tensor& mean, const Tensor& invstd,
                                  double epsilon, Tensor& output) {
    (void)epsilon;
    if (static_cast<std::vector<int64_t>>(output.shape()) !=
        static_cast<std::vector<int64_t>>(input.shape())) {
        output = Tensor::empty(input.shape(), input.dtype(), input.device());
    }
    batch_norm_elementwise(output, input, weight, bias, mean, invstd);
    return output;
}

// ---------------------------------------------------------------------------
// batch_norm_gather_stats: fold per-rank statistics into the running buffers
// ---------------------------------------------------------------------------

template <typename scalar_t, typename accscalar_t, typename index_t>
void launch_reduce_statistics(const Tensor& mean, const Tensor& invstd,
                              const std::optional<Tensor>& running_mean,
                              const std::optional<Tensor>& running_var,
                              double momentum, double epsilon,
                              const Tensor& counts, Tensor& save_mean,
                              Tensor& save_invstd) {
    const int64_t features = mean.size(1);
    auto mean_acc = bn_acc2<accscalar_t, index_t>(mean);
    auto invstd_acc = bn_acc2<accscalar_t, index_t>(invstd);
    auto save_mean_acc = bn_acc1<accscalar_t, index_t>(save_mean);
    auto save_invstd_acc = bn_acc1<accscalar_t, index_t>(save_invstd);
    auto running_mean_acc = bn_acc_or_dummy<scalar_t, index_t>(running_mean);
    auto running_var_acc = bn_acc_or_dummy<scalar_t, index_t>(running_var);
    auto counts_acc = bn_acc_or_dummy<scalar_t, index_t>(
        std::make_optional(counts));
    cudaStream_t stream = getCurrentCUDAStream().stream();
    const int block = bn_get_num_threads(features);
    const int grid =
        std::max<int>(1, static_cast<int>(bn_ceil_div(features, block)));
    bn_reduce_statistics_kernel<scalar_t, accscalar_t, index_t>
        <<<grid, block, 0, stream>>>(
            mean_acc, invstd_acc, save_mean_acc, save_invstd_acc,
            running_mean_acc, running_var_acc,
            static_cast<accscalar_t>(epsilon),
            static_cast<accscalar_t>(momentum), counts_acc);
    check_bn_launch("batch_norm_gather_stats");
}

std::tuple<Tensor, Tensor> batch_norm_gather_stats_with_counts_cuda(
        const Tensor& input, const Tensor& mean, const Tensor& invstd,
        const std::optional<Tensor>& running_mean,
        const std::optional<Tensor>& running_var, double momentum,
        double epsilon, const Tensor& counts) {
    TP_CHECK(mean.dim() == 2,
             "batch_norm_gather_stats_with_counts: expected mean to be "
             "2-dimensional (world_size, num_features)");
    TP_CHECK(static_cast<std::vector<int64_t>>(invstd.shape()) ==
                 static_cast<std::vector<int64_t>>(mean.shape()),
             "batch_norm_gather_stats_with_counts: expected invstd to have the "
             "same shape as mean");
    TP_CHECK(counts.numel() >= mean.size(0),
             "batch_norm_gather_stats_with_counts: expected counts to have at "
             "least one element per entry in mean's first dimension");

    const bool running_defined =
        running_mean.has_value() && running_mean->defined();
    const DType scalar_dtype =
        running_defined ? running_mean->dtype() : input.dtype();
    const DType acc_dtype = accumulate_dtype(mean.dtype());
    const int64_t features = mean.size(1);
    Tensor save_mean =
        Tensor::empty({features}, acc_dtype, mean.device());
    Tensor save_invstd =
        Tensor::empty({features}, acc_dtype, mean.device());
    TP_BN_DISPATCH_FLOATING(scalar_dtype, "batch_norm_gather_stats", {
        if (mean.numel() <=
            static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            launch_reduce_statistics<scalar_t, accscalar_t, int32_t>(
                mean, invstd, running_mean, running_var, momentum, epsilon,
                counts, save_mean, save_invstd);
        } else {
            launch_reduce_statistics<scalar_t, accscalar_t, int64_t>(
                mean, invstd, running_mean, running_var, momentum, epsilon,
                counts, save_mean, save_invstd);
        }
    });
    return std::make_tuple(save_mean, save_invstd);
}

std::tuple<Tensor, Tensor> batch_norm_gather_stats_cuda(
        const Tensor& input, const Tensor& mean, const Tensor& invstd,
        const std::optional<Tensor>& running_mean,
        const std::optional<Tensor>& running_var, double momentum,
        double epsilon, int64_t count) {
    TP_CHECK(mean.dim() == 2,
             "batch_norm_gather_stats: expected mean to be 2-dimensional "
             "(world_size, num_features)");
    const bool running_defined =
        running_mean.has_value() && running_mean->defined();
    const DType counts_dtype = running_defined ? running_mean->dtype()
                                               : input.dtype();
    Tensor counts = Tensor::full({mean.size(0)}, Scalar(count), counts_dtype,
                                 input.device());
    return batch_norm_gather_stats_with_counts_cuda(
        input, mean, invstd, running_mean, running_var, momentum, epsilon,
        counts);
}

// ---------------------------------------------------------------------------
// batch_norm_update_stats: batch statistics plus the running-buffer update
// ---------------------------------------------------------------------------

void batch_norm_mean_var(const Tensor& input, Tensor& save_mean,
                         Tensor& save_var) {
    // The variance transform is the identity here; epsilon is unused.
    if (choose_layout(input) == Layout::General) {
        std::vector<int64_t> reduce_dims;
        reduce_dims.push_back(0);
        for (int64_t d = 2; d < static_cast<int64_t>(input.dim()); ++d) {
            reduce_dims.push_back(d);
        }
        save_var = ops::var(input, reduce_dims, 0, false);
        save_mean = ops::mean(input, reduce_dims, false);
        return;
    }
    const double dummy_epsilon = 1e-5;
    TP_BN_DISPATCH_FLOATING(input.dtype(), "batch_norm_update_stats", {
        collect_statistics<scalar_t, accscalar_t, BnVar>(input, save_mean,
                                                         save_var,
                                                         dummy_epsilon);
    });
}

void batch_norm_update_running(const Tensor& save_mean, const Tensor& save_var,
                               const Tensor& running_mean,
                               const Tensor& running_var, double momentum,
                               int64_t n) {
    const int64_t count = save_mean.numel();
    const int threads = 256;
    const int blocks =
        static_cast<int>(bn_ceil_div(count, static_cast<int64_t>(threads)));
    cudaStream_t stream = getCurrentCUDAStream().stream();
    TP_BN_DISPATCH_FLOATING(running_mean.dtype(), "batch_norm_update_stats", {
        const accscalar_t bessel = static_cast<accscalar_t>(
            static_cast<double>(n) / static_cast<double>(n - 1));
        const accscalar_t mom = static_cast<accscalar_t>(momentum);
        bn_update_stats_kernel<scalar_t, scalar_t, accscalar_t>
            <<<blocks, threads, 0, stream>>>(
                count, save_mean.data_ptr<accscalar_t>(),
                save_var.data_ptr<accscalar_t>(),
                running_mean.data_ptr<scalar_t>(),
                running_var.data_ptr<scalar_t>(), bessel, mom);
    });
    check_bn_launch("batch_norm_update_stats");
}

std::tuple<Tensor, Tensor> batch_norm_update_stats_cuda(
        const Tensor& input, const std::optional<Tensor>& running_mean,
        const std::optional<Tensor>& running_var, double momentum) {
    const int64_t channels = input.size(1);
    TP_CHECK(input.numel() != 0,
             "input tensor must have at least one element");
    const DType acc_dtype = accumulate_dtype(input.dtype());
    Tensor save_mean = Tensor::empty({channels}, acc_dtype, input.device());
    Tensor save_var = Tensor::empty({channels}, acc_dtype, input.device());
    batch_norm_mean_var(input, save_mean, save_var);
    const bool running_defined =
        running_mean.has_value() && running_mean->defined();
    TP_CHECK(running_defined == (running_var.has_value() && running_var->defined()),
             "running_mean and running_var must both be defined or both undefined");
    if (running_defined) {
        const int64_t n = input.numel() / save_mean.numel();
        batch_norm_update_running(save_mean, save_var, *running_mean,
                                  *running_var, momentum, n);
    }
    return std::make_tuple(save_mean, save_var);
}

// ---------------------------------------------------------------------------
// batch_norm_backward_reduce: per-channel reduction for the elementwise pass
// ---------------------------------------------------------------------------

template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t, typename index_t>
void launch_backward_reduce(const Tensor& grad_output, const Tensor& input,
                            const Tensor& mean, const Tensor& invstd,
                            const std::optional<Tensor>& weight, bool input_g,
                            bool weight_g, bool bias_g, Tensor& sum_dy,
                            Tensor& sum_dy_xmu, Tensor& grad_weight,
                            Tensor& grad_bias) {
    const int64_t channels = input.size(1);
    const Tensor input_reshaped = merge_features(input);
    const Tensor grad_reshaped = merge_features(grad_output);
    if (input_g) {
        sum_dy = Tensor::empty_like(mean);
        sum_dy_xmu = Tensor::empty_like(mean);
    }
    if (weight_g) {
        grad_weight = Tensor::empty({channels}, weight->dtype(),
                                    weight->device());
    }
    if (bias_g) {
        grad_bias = Tensor::empty({channels}, weight->dtype(),
                                  weight->device());
    }

    auto in_acc = bn_acc3<input_scalar_t, index_t>(input_reshaped);
    auto go_acc = bn_acc3<input_scalar_t, index_t>(grad_reshaped);
    auto mean_acc = bn_acc_or_dummy<stat_accscalar_t, index_t>(
        std::make_optional(mean));
    auto invstd_acc = bn_acc_or_dummy<stat_accscalar_t, index_t>(
        std::make_optional(invstd));
    auto sum_dy_acc = bn_acc_or_dummy<stat_accscalar_t, index_t>(
        std::make_optional(sum_dy));
    auto sum_dy_xmu_acc = bn_acc_or_dummy<stat_accscalar_t, index_t>(
        std::make_optional(sum_dy_xmu));
    auto grad_weight_acc = bn_acc_or_dummy<stat_scalar_t, index_t>(
        std::make_optional(grad_weight));
    auto grad_bias_acc = bn_acc_or_dummy<stat_scalar_t, index_t>(
        std::make_optional(grad_bias));
    cudaStream_t stream = getCurrentCUDAStream().stream();

    const int64_t batch_size = in_acc.size(0);
    const int64_t feature_size = in_acc.size(2);
    int block_y = std::min<int>(bn_last_pow2(batch_size),
                                kBnMaxBlockSize / kBnWarpSize);
    int block_x = std::min<int>(
        std::max<int>(bn_get_num_threads(feature_size), kBnWarpSize),
        kBnMaxBlockSize / block_y);
    const dim3 block(static_cast<unsigned int>(block_x),
                     static_cast<unsigned int>(block_y));
    const dim3 grid(static_cast<unsigned int>(channels));
    bn_backward_reduce_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t,
                              index_t>
        <<<grid, block, 0, stream>>>(in_acc, go_acc, mean_acc, invstd_acc,
                                     sum_dy_acc, sum_dy_xmu_acc,
                                     grad_weight_acc, grad_bias_acc);
    check_bn_launch("batch_norm_backward_reduce");
}

template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t>
void launch_backward_reduce_channels_last(
        const Tensor& grad_output, const Tensor& input, const Tensor& mean,
        const Tensor& invstd, const std::optional<Tensor>& weight,
        Tensor& sum_dy, Tensor& sum_dy_xmu, Tensor& grad_weight,
        Tensor& grad_bias) {
    const int64_t stride = input.size(1);
    const int64_t reduction_size = input.numel() / std::max<int64_t>(stride, 1);
    const bool weight_defined = weight.has_value() && weight->defined();
    const DType param_dtype =
        weight_defined ? weight->dtype() : mean.dtype();
    sum_dy = Tensor::empty({stride}, mean.dtype(), mean.device());
    sum_dy_xmu = Tensor::empty({stride}, mean.dtype(), mean.device());
    if (weight_defined) {
        grad_weight = Tensor::empty({stride}, param_dtype, mean.device());
        grad_bias = Tensor::empty({stride}, param_dtype, mean.device());
    } else {
        grad_weight = Tensor::empty({0}, mean.dtype(), mean.device());
        grad_bias = Tensor::empty({0}, mean.dtype(), mean.device());
    }

    dim3 block;
    dim3 grid;
    flexible_launch_configs(reduction_size, stride, &block, &grid, true);

    Tensor staging_data;
    Tensor semaphores;
    if (grid.y > 1) {
        staging_data = Tensor::empty({2 * stride * grid.y}, mean.dtype(),
                                     mean.device());
        semaphores = Tensor::zeros({grid.x}, DType::Int32, input.device());
    }
    stat_accscalar_t* staging_ptr =
        grid.y > 1 ? staging_data.data_ptr<stat_accscalar_t>() : nullptr;
    int* semaphores_ptr = grid.y > 1 ? semaphores.data_ptr<int32_t>() : nullptr;
    cudaStream_t stream = getCurrentCUDAStream().stream();
    const stat_scalar_t* weight_ptr =
        weight_defined ? weight->data_ptr<stat_scalar_t>() : nullptr;
    stat_scalar_t* grad_weight_ptr =
        weight_defined ? grad_weight.data_ptr<stat_scalar_t>() : nullptr;
    stat_scalar_t* grad_bias_ptr =
        weight_defined ? grad_bias.data_ptr<stat_scalar_t>() : nullptr;
    bn_backward_reduce_channels_last_kernel<kBnElementsPerIter, input_scalar_t,
                                           stat_accscalar_t, stat_scalar_t>
        <<<grid, block, 0, stream>>>(
            input.data_ptr<input_scalar_t>(),
            grad_output.data_ptr<input_scalar_t>(),
            mean.data_ptr<stat_accscalar_t>(),
            invstd.data_ptr<stat_accscalar_t>(),
            sum_dy.data_ptr<stat_accscalar_t>(),
            sum_dy_xmu.data_ptr<stat_accscalar_t>(), grad_weight_ptr,
            grad_bias_ptr, staging_ptr, semaphores_ptr,
            static_cast<int>(reduction_size), static_cast<int>(stride));
    check_bn_launch("batch_norm_backward_reduce channels-last");
}

std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_cuda(
        const Tensor& grad_output, const Tensor& input, const Tensor& mean,
        const Tensor& invstd, const std::optional<Tensor>& weight,
        bool input_g, bool weight_g, bool bias_g) {
    TP_CHECK(mean.dtype() == invstd.dtype(),
             "mean and invstd need to have the same data types");
    const bool weight_defined = weight.has_value() && weight->defined();
    if (choose_layout(grad_output, input) == Layout::ChannelsLast &&
        (!weight_defined || weight->is_contiguous()) &&
        mean.is_contiguous() && invstd.is_contiguous()) {
        Tensor sum_dy_cl;
        Tensor sum_dy_xmu_cl;
        Tensor grad_weight_cl;
        Tensor grad_bias_cl;
        TP_BN_DISPATCH_FLOATING(
            grad_output.dtype(), "batch_norm_backward_reduce", {
                const bool mixed_cl = is_mixed_type(input, weight);
                if (mixed_cl) {
                    launch_backward_reduce_channels_last<scalar_t, accscalar_t,
                                                        accscalar_t>(
                        grad_output, input, mean, invstd, weight, sum_dy_cl,
                        sum_dy_xmu_cl, grad_weight_cl, grad_bias_cl);
                } else {
                    launch_backward_reduce_channels_last<scalar_t, scalar_t,
                                                        accscalar_t>(
                        grad_output, input, mean, invstd, weight, sum_dy_cl,
                        sum_dy_xmu_cl, grad_weight_cl, grad_bias_cl);
                }
            });
        return std::make_tuple(sum_dy_cl, sum_dy_xmu_cl, grad_weight_cl,
                               grad_bias_cl);
    }
    const bool mixed = is_mixed_type(input, weight);
    Tensor sum_dy;
    Tensor sum_dy_xmu;
    Tensor grad_weight;
    Tensor grad_bias;
    TP_BN_DISPATCH_FLOATING(grad_output.dtype(), "batch_norm_backward_reduce", {
        const bool use_32bit =
            grad_output.numel() <=
            static_cast<int64_t>(std::numeric_limits<int32_t>::max());
        if (mixed) {
            if (use_32bit) {
                launch_backward_reduce<scalar_t, accscalar_t, accscalar_t,
                                       int32_t>(grad_output, input, mean, invstd,
                                                weight, input_g, weight_g,
                                                bias_g, sum_dy, sum_dy_xmu,
                                                grad_weight, grad_bias);
            } else {
                launch_backward_reduce<scalar_t, accscalar_t, accscalar_t,
                                       int64_t>(grad_output, input, mean, invstd,
                                                weight, input_g, weight_g,
                                                bias_g, sum_dy, sum_dy_xmu,
                                                grad_weight, grad_bias);
            }
        } else {
            if (use_32bit) {
                launch_backward_reduce<scalar_t, scalar_t, accscalar_t,
                                       int32_t>(grad_output, input, mean, invstd,
                                                weight, input_g, weight_g,
                                                bias_g, sum_dy, sum_dy_xmu,
                                                grad_weight, grad_bias);
            } else {
                launch_backward_reduce<scalar_t, scalar_t, accscalar_t,
                                       int64_t>(grad_output, input, mean, invstd,
                                                weight, input_g, weight_g,
                                                bias_g, sum_dy, sum_dy_xmu,
                                                grad_weight, grad_bias);
            }
        }
    });
    return std::make_tuple(sum_dy, sum_dy_xmu, grad_weight, grad_bias);
}

// ---------------------------------------------------------------------------
// batch_norm_backward_elemt: input gradient from the reductions
// ---------------------------------------------------------------------------

template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t, typename index_t>
void launch_backward_elemt(const Tensor& grad_output, const Tensor& input,
                           const Tensor& mean, const Tensor& invstd,
                           const std::optional<Tensor>& weight,
                           const Tensor& sum_dy, const Tensor& sum_dy_xmu,
                           const Tensor& count, Tensor& grad_input) {
    const Tensor input_reshaped = merge_features(input);
    const Tensor grad_reshaped = merge_features(grad_output);
    const std::vector<int64_t> out_shape = {input_reshaped.size(0),
                                            input_reshaped.size(1),
                                            input_reshaped.size(2)};
    grad_input = Tensor::empty(out_shape, input.dtype(), input.device());
    auto in_acc = bn_acc3<input_scalar_t, index_t>(input_reshaped);
    auto go_acc = bn_acc3<input_scalar_t, index_t>(grad_reshaped);
    auto gi_acc = bn_acc3<input_scalar_t, index_t>(grad_input);
    auto mean_acc = bn_acc_or_dummy<stat_accscalar_t, index_t>(
        std::make_optional(mean));
    auto invstd_acc = bn_acc_or_dummy<stat_accscalar_t, index_t>(
        std::make_optional(invstd));
    auto weight_acc = bn_acc_or_dummy<stat_scalar_t, index_t>(weight);
    auto sum_dy_acc = bn_acc_or_dummy<stat_accscalar_t, index_t>(
        std::make_optional(sum_dy));
    auto sum_dy_xmu_acc = bn_acc_or_dummy<stat_accscalar_t, index_t>(
        std::make_optional(sum_dy_xmu));
    cudaStream_t stream = getCurrentCUDAStream().stream();

    const int tf = pointwise_threads_x(in_acc.size(2));
    const int tb = pointwise_threads_y(tf);
    const dim3 blocks =
        pointwise_blocks(in_acc.size(1), in_acc.size(0), tb);
    const dim3 threads(static_cast<unsigned int>(tf),
                       static_cast<unsigned int>(tb));
    bn_backward_elemt_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t,
                             index_t>
        <<<blocks, threads, 0, stream>>>(
            in_acc, go_acc, mean_acc, invstd_acc, weight_acc, sum_dy_acc,
            sum_dy_xmu_acc, gi_acc, count.data_ptr<int32_t>(),
            static_cast<int>(count.numel()));
    check_bn_launch("batch_norm_backward_elemt");
    grad_input = grad_input.view(input.shape());
}

template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t>
void backward_elemt_strided(const Tensor& grad_output, const Tensor& input,
                            const std::optional<Tensor>& weight,
                            const Tensor& mean, const Tensor& invstd,
                            const Tensor& sum_dy, const Tensor& sum_dy_xmu,
                            const Tensor& count, Tensor& grad_input) {
    const Tensor input_dense = input.contiguous();
    grad_input = Tensor::empty(input.shape(), input.dtype(), input.device());
    const BnStridedLayout layout = make_strided_layout(input, grad_input);
    const int threads = 256;
    const int blocks = static_cast<int>(
        bn_ceil_div(layout.numel, static_cast<int64_t>(threads)));
    cudaStream_t stream = getCurrentCUDAStream().stream();
    const stat_scalar_t* weight_p =
        (weight.has_value() && weight->defined()) ? weight->data_ptr<stat_scalar_t>()
                                                  : nullptr;
    bn_backward_elemt_strided_kernel<input_scalar_t, stat_scalar_t,
                                     stat_accscalar_t>
        <<<blocks, threads, 0, stream>>>(
            layout, grad_output.contiguous().data_ptr<input_scalar_t>(),
            input_dense.data_ptr<input_scalar_t>(),
            grad_input.data_ptr<input_scalar_t>(), weight_p,
            mean.data_ptr<stat_accscalar_t>(),
            invstd.data_ptr<stat_accscalar_t>(),
            sum_dy.data_ptr<stat_accscalar_t>(),
            sum_dy_xmu.data_ptr<stat_accscalar_t>(), count.data_ptr<int32_t>(),
            static_cast<int>(count.numel()));
    check_bn_launch("batch_norm_backward_elemt strided");
}

template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t>
void launch_backward_elemt_channels_last(
        const Tensor& grad_output, const Tensor& input, const Tensor& mean,
        const Tensor& invstd, const std::optional<Tensor>& weight,
        const Tensor& sum_dy, const Tensor& sum_dy_xmu, const Tensor& count,
        Tensor& grad_input) {
    const int64_t stride = input.size(1);
    const int64_t reduction_size = input.numel() / std::max<int64_t>(stride, 1);
    grad_input = Tensor::empty_like(input);
    dim3 block;
    dim3 grid;
    flexible_launch_configs(reduction_size, stride, &block, &grid);
    const bool weight_defined = weight.has_value() && weight->defined();
    const stat_scalar_t* weight_ptr =
        weight_defined ? weight->data_ptr<stat_scalar_t>() : nullptr;
    cudaStream_t stream = getCurrentCUDAStream().stream();
    // The normalization factor comes from the element tally carried by the
    // count operand, so a distributed reduction keeps its own scale.
    bn_backward_elemt_channels_last_kernel<kBnElementsPerIter, input_scalar_t,
                                           stat_accscalar_t, stat_scalar_t>
        <<<grid, block, 0, stream>>>(
            grad_output.data_ptr<input_scalar_t>(),
            input.data_ptr<input_scalar_t>(), mean.data_ptr<stat_accscalar_t>(),
            invstd.data_ptr<stat_accscalar_t>(), weight_ptr,
            sum_dy.data_ptr<stat_accscalar_t>(),
            sum_dy_xmu.data_ptr<stat_accscalar_t>(),
            count.data_ptr<int32_t>(), grad_input.data_ptr<input_scalar_t>(),
            static_cast<int64_t>(count.numel()),
            static_cast<int>(reduction_size), static_cast<int>(stride));
    check_bn_launch("batch_norm_backward_elemt channels-last");
}

Tensor batch_norm_backward_elemt_cuda(
        const Tensor& grad_output, const Tensor& input, const Tensor& mean,
        const Tensor& invstd, const std::optional<Tensor>& weight,
        const Tensor& sum_dy, const Tensor& sum_dy_xmu, const Tensor& count) {
    // The per-plane normalization factor comes from the element tally carried
    // by the count operand on every layout path.
    TP_CHECK(mean.dtype() == invstd.dtype(),
             "mean and invstd need to have the same data types");
    const bool mixed = is_mixed_type(input, weight) ||
                       is_mixed_type(grad_output, weight);
    Tensor grad_input;
    if (choose_layout(input, grad_output) == Layout::ChannelsLast &&
        count.dtype() == DType::Int32) {
        TP_BN_DISPATCH_FLOATING(input.dtype(), "batch_norm_backward_elemt", {
            if (mixed) {
                launch_backward_elemt_channels_last<scalar_t, accscalar_t,
                                                   accscalar_t>(
                    grad_output, input, mean, invstd, weight, sum_dy,
                    sum_dy_xmu, count, grad_input);
            } else {
                launch_backward_elemt_channels_last<scalar_t, scalar_t,
                                                   accscalar_t>(
                    grad_output, input, mean, invstd, weight, sum_dy,
                    sum_dy_xmu, count, grad_input);
            }
        });
        return grad_input;
    }
    if (choose_layout(input, grad_output) == Layout::General) {
        TP_BN_DISPATCH_FLOATING(input.dtype(), "batch_norm_backward_elemt", {
            if (mixed) {
                backward_elemt_strided<scalar_t, accscalar_t, accscalar_t>(
                    grad_output, input, weight, mean, invstd, sum_dy,
                    sum_dy_xmu, count, grad_input);
            } else {
                backward_elemt_strided<scalar_t, scalar_t, accscalar_t>(
                    grad_output, input, weight, mean, invstd, sum_dy,
                    sum_dy_xmu, count, grad_input);
            }
        });
        return grad_input;
    }
    const bool use_32bit =
        input.numel() <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    TP_BN_DISPATCH_FLOATING(input.dtype(), "batch_norm_backward_elemt", {
        if (mixed) {
            if (use_32bit) {
                launch_backward_elemt<scalar_t, accscalar_t, accscalar_t,
                                      int32_t>(grad_output, input, mean, invstd,
                                               weight, sum_dy, sum_dy_xmu, count,
                                               grad_input);
            } else {
                launch_backward_elemt<scalar_t, accscalar_t, accscalar_t,
                                      int64_t>(grad_output, input, mean, invstd,
                                               weight, sum_dy, sum_dy_xmu, count,
                                               grad_input);
            }
        } else {
            if (use_32bit) {
                launch_backward_elemt<scalar_t, scalar_t, accscalar_t,
                                      int32_t>(grad_output, input, mean, invstd,
                                               weight, sum_dy, sum_dy_xmu, count,
                                               grad_input);
            } else {
                launch_backward_elemt<scalar_t, scalar_t, accscalar_t,
                                      int64_t>(grad_output, input, mean, invstd,
                                               weight, sum_dy, sum_dy_xmu, count,
                                               grad_input);
            }
        }
    });
    return grad_input;
}

}  // namespace batch_norm

TENSORPLAY_LIBRARY_IMPL(CUDA, BatchNormKernels) {
    m.impl("batch_norm", batch_norm_cuda);
    m.impl("batch_norm_backward", batch_norm_backward_cuda);
    m.impl("batch_norm_stats", batch_norm::batch_norm_stats_cuda);
    m.impl("batch_norm_elemt", batch_norm::batch_norm_elemt_cuda);
    m.impl("batch_norm_elemt.out", batch_norm::batch_norm_elemt_cuda_out);
    m.impl("batch_norm_gather_stats",
           batch_norm::batch_norm_gather_stats_cuda);
    m.impl("batch_norm_gather_stats_with_counts",
           batch_norm::batch_norm_gather_stats_with_counts_cuda);
    m.impl("batch_norm_update_stats",
           batch_norm::batch_norm_update_stats_cuda);
    m.impl("batch_norm_backward_reduce",
           batch_norm::batch_norm_backward_reduce_cuda);
    m.impl("batch_norm_backward_elemt",
           batch_norm::batch_norm_backward_elemt_cuda);
}

}
}
