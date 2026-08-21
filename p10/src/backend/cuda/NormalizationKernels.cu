#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "CUDAContext.h"
#include "Exception.h"
#include "CUDNNUtils.h"

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace cuda {

#ifdef USE_CUDNN

namespace {

bool defined(const std::optional<Tensor>& value) {
    return value.has_value() && value->defined();
}

// cuDNN's spatial BN descriptor is 1xCx1x1.  A plain [C] descriptor would
// put C in the innermost dimension and is not the channel parameter layout.
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

} // namespace

Tensor batch_norm_cuda(
    const Tensor& input,
    std::optional<Tensor> weight_opt,
    std::optional<Tensor> bias_opt,
    std::optional<Tensor> running_mean_opt,
    std::optional<Tensor> running_var_opt,
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
    std::optional<Tensor> weight_opt,
    std::optional<Tensor> running_mean_opt,
    std::optional<Tensor> running_var_opt,
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

    // cuDNN's backward API consumes the batch mean and inverse variance saved
    // by its training forward.  Recompute those values from the saved input;
    // the public TensorPlay BN node intentionally stores only the input and
    // running-state handles, matching the CPU implementation.
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
        // The kernel only materializes the C-element inverse variance used by
        // cudnnBatchNormalizationBackward; running statistics stay untouched.
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
    const Tensor&, std::optional<Tensor>, std::optional<Tensor>,
    std::optional<Tensor>, std::optional<Tensor>, bool, double, double) {
    TP_THROW(NotImplementedError, "batch_norm CUDA requires cuDNN");
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda(
    const Tensor&, const Tensor&, std::optional<Tensor>, std::optional<Tensor>,
    std::optional<Tensor>, bool, double) {
    TP_THROW(NotImplementedError, "batch_norm_backward CUDA requires cuDNN");
}

#endif

TENSORPLAY_LIBRARY_IMPL(CUDA, NormalizationKernels) {
    m.impl("batch_norm", batch_norm_cuda);
    m.impl("batch_norm_backward", batch_norm_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay
