// multi_margin_loss / multilabel_margin_loss CUDA kernels.
//
// Port of aten/src/ATen/native/cuda/MultiMarginLoss.cu and
// MultiLabelMarginCriterion.cu: one block (128 threads) per batch row; each
// block computes its row's hinge sum through a shared-memory buffer and
// finalizes either a per-row output (reduction=none) or a per-row partial
// that a follow-up atomic reduction collapses to the scalar mean/sum
// (self-contained reduction -- does not go through the dispatched sum op).
// Arithmetic runs in double on device (same convention as
// Tier5LossesKernels.cu) and results are cast back to the input dtype.
// Out-of-range targets trip a device-side assert, mirroring ATen's
// CUDA_KERNEL_ASSERT.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "CUDARuntime.h"
#include "LinearAlgebraNames.h"

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <tuple>
#include <optional>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

namespace tensorplay {
namespace cuda {

namespace {

constexpr int kMarginThreads = 128;
constexpr int64_t kReductionNone = 0;
constexpr int64_t kReductionMean = 1;
constexpr int64_t kReductionSum = 2;

#define CUDA_CHECK(condition)                                              \
    do {                                                                   \
        cudaError_t error = condition;                                     \
        if (error != cudaSuccess) {                                        \
            TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
        }                                                                  \
    } while (0)

inline dim3 margin_grid(int64_t work) {
    return dim3(static_cast<unsigned>((work + kMarginThreads - 1) / kMarginThreads));
}

void check_reduction(int64_t reduction, const char* name) {
    if (reduction != kReductionNone && reduction != kReductionMean && reduction != kReductionSum)
        TP_THROW(ValueError, std::string(name) + ": invalid reduction, expected 0 (none), 1 (mean) or 2 (sum) but got " + std::to_string(reduction));
}

Tensor as_long_contiguous(const Tensor& target) {
    Tensor t = target.contiguous();
    if (t.dtype() != DType::Int64) t = t.to(DType::Int64);
    return t;
}

// ATen LossMulti.h multi_margin_loss_shape_check.
void multi_margin_shape_check(int64_t& nframe, int64_t& dim, const Tensor& input,
                              const Tensor& target, const std::optional<Tensor>& weight) {
    const int64_t ndims = input.dim();
    if (!((ndims == 2 && input.size(1) != 0) || (ndims == 1 && input.size(0) != 0) || ndims == 0))
        TP_THROW(RuntimeError, std::string("multi_margin_loss: Expected non-empty vector or matrix with optional 0-dim batch size, but got: ") + input.shape().toString());
    if (ndims <= 1) {
        nframe = 1;
        dim = ndims == 0 ? 1 : input.size(0);
    } else {
        nframe = input.size(0);
        dim = input.size(1);
    }
    if (!(target.dim() <= 1 && target.numel() == nframe))
        TP_THROW(RuntimeError, std::string("multi_margin_loss: target tensor should be 1-D with size equal to the number of input samples (batch size). Expected target size [") +
                     std::to_string(nframe) + "], but got " + target.shape().toString() +
                     ". Input has shape " + input.shape().toString() + ".");
    if (weight.has_value() && weight->defined()) {
        if (!(weight->dim() <= 1 && weight->numel() == dim))
            TP_THROW(RuntimeError, std::string("multi_margin_loss: inconsistent weight size, expected ") +
                         std::to_string(dim) + " but got " + weight->shape().toString());
    }
}

// ATen LossMulti.h multilabel_margin_loss_shape_check.
void multilabel_shape_check(int64_t& nframe, int64_t& dim, const Tensor& input,
                            const Tensor& target) {
    const int64_t ndims = input.dim();
    if (!((ndims == 2 && input.size(1) != 0) || (ndims == 1 && input.size(0) != 0) || ndims == 0))
        TP_THROW(RuntimeError, std::string("multilabel_margin_loss: Expected non-empty vector or matrix with optional 0-dim batch size, but got: ") + input.shape().toString());
    if (ndims <= 1) {
        nframe = 1;
        dim = ndims == 0 ? 1 : input.size(0);
        if (!(target.dim() <= 1 && target.numel() == dim))
            TP_THROW(RuntimeError, std::string("multilabel_margin_loss: inconsistent target size: ") +
                         target.shape().toString() + " for input of size: " + input.shape().toString());
    } else {
        nframe = input.size(0);
        dim = input.size(1);
        if (!(target.dim() == 2 && target.size(0) == nframe && target.size(1) == dim))
            TP_THROW(RuntimeError, std::string("multilabel_margin_loss: inconsistent target size: ") +
                         target.shape().toString() + " for input of size: " + input.shape().toString());
    }
}

// ---------------------------------------------------------------------------
// device kernels (one block per batch row)
// ---------------------------------------------------------------------------

template <int P>
__global__ void multi_margin_fwd_kernel(double* __restrict__ output, const double* __restrict__ input,
                                        const int64_t* __restrict__ target,
                                        const double* __restrict__ weight,
                                        int nframe, int dim, bool sizeAverage, double margin) {
    __shared__ double buffer[kMarginThreads];
    const int k = blockIdx.x;
    const double* input_k = input + static_cast<size_t>(k) * dim;
    double* output_k = output + k;
    const int target_k = static_cast<int>(target[k]);
    assert(target_k >= 0 && target_k < dim);
    const double input_target_k = input_k[target_k];

    buffer[threadIdx.x] = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (i == target_k) continue;
        const double z = margin - input_target_k + input_k[i];
        if (z > 0) {
            double h = (P == 1) ? z : z * z;
            if (weight != nullptr) h *= weight[target_k];
            buffer[threadIdx.x] += h;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double sum = 0;
        for (int i = 0; i < blockDim.x; ++i) sum += buffer[i];
        const int denom = sizeAverage ? nframe * dim : dim;
        *output_k = sum / denom;
    }
}

template <int P>
__global__ void multi_margin_bwd_kernel(double* __restrict__ gradInput, const double* __restrict__ gradOutput,
                                        const double* __restrict__ input, const int64_t* __restrict__ target,
                                        const double* __restrict__ weight,
                                        int nframe, int dim, bool sizeAverage, double margin, bool reduce) {
    __shared__ double buffer[kMarginThreads];
    const int k = blockIdx.x;
    const double* input_k = input + static_cast<size_t>(k) * dim;
    double* gradInput_k = gradInput + static_cast<size_t>(k) * dim;
    const int target_k = static_cast<int>(target[k]);
    assert(target_k >= 0 && target_k < dim);
    const double input_target_k = input_k[target_k];

    const double* gradOutput_k = gradOutput + (reduce ? 0 : k);
    const int denom = (sizeAverage && reduce) ? nframe * dim : dim;
    const double g = 1.0 / denom;

    buffer[threadIdx.x] = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (i == target_k) continue;
        const double z = margin - input_target_k + input_k[i];
        if (z > 0) {
            double h = (P == 1) ? g : 2 * g * z;
            if (weight != nullptr) h *= weight[target_k];
            buffer[threadIdx.x] -= h;
            gradInput_k[i] = h;
        } else {
            gradInput_k[i] = 0;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double s = 0;
        for (int i = 0; i < blockDim.x; ++i) s += buffer[i];
        gradInput_k[target_k] = s;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < dim; i += blockDim.x) gradInput_k[i] *= *gradOutput_k;
}

__global__ void multilabel_fwd_kernel(double* __restrict__ output, const double* __restrict__ input,
                                      const int64_t* __restrict__ target, double* __restrict__ is_target,
                                      int nframe, int dim, bool sizeAverage) {
    __shared__ double smem[kMarginThreads];
    const int k = blockIdx.x;
    const double* input_k = input + static_cast<size_t>(k) * dim;
    const int64_t* target_k = target + static_cast<size_t>(k) * dim;
    double* output_k = output + k;
    double* is_target_k = is_target + static_cast<size_t>(k) * dim;

    for (int d = threadIdx.x; d < dim; d += blockDim.x) is_target_k[d] = 0;
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int dt = 0; dt < dim; ++dt) {
            const int64_t idx = target_k[dt];
            if (idx < 0) break;
            assert(idx < dim);
            is_target_k[idx] = 1;
        }
    }
    __syncthreads();

    double sum = 0;
    for (int dt = 0; dt < dim; ++dt) {
        const int64_t idx = target_k[dt];
        if (idx < 0) break;
        const double input_target = input_k[idx];
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            if (is_target_k[d] == 0) {
                const double z = 1 - input_target + input_k[d];
                if (z > 0) sum += z;
            }
        }
    }

    smem[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        double total = 0;
        for (int i = 0; i < blockDim.x; ++i) total += smem[i];
        *output_k = sizeAverage ? total / dim / nframe : total / dim;
    }
}

__global__ void multilabel_bwd_kernel(double* __restrict__ gradInput, const double* __restrict__ gradOutput,
                                      const double* __restrict__ input, const int64_t* __restrict__ target,
                                      const double* __restrict__ is_target,
                                      int nframe, int dim, bool sizeAverage, bool reduce) {
    __shared__ double smem[kMarginThreads];
    const int k = blockIdx.x;
    const double* input_k = input + static_cast<size_t>(k) * dim;
    double* gradInput_k = gradInput + static_cast<size_t>(k) * dim;
    const int64_t* target_k = target + static_cast<size_t>(k) * dim;
    const double* is_target_k = is_target + static_cast<size_t>(k) * dim;

    const double* gradOutput_k = gradOutput + (reduce ? 0 : k);
    const double g = (sizeAverage && reduce) ? 1.0 / (nframe * dim) : 1.0 / dim;

    for (int d = threadIdx.x; d < dim; d += blockDim.x) gradInput_k[d] = 0;
    __syncthreads();

    for (int dt = 0; dt < dim; ++dt) {
        const int64_t idx = target_k[dt];
        if (idx < 0) break;
        assert(idx < dim);
        const double input_target = input_k[idx];
        double sum = 0;
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            if (is_target_k[d] == 0) {
                const double z = 1 - input_target + input_k[d];
                if (z > 0) {
                    sum -= g;
                    gradInput_k[d] += g;
                }
            }
        }
        smem[threadIdx.x] = sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            double total = 0;
            for (int i = 0; i < blockDim.x; ++i) total += smem[i];
            gradInput_k[idx] += total;
        }
        __syncthreads();
    }
    __syncthreads();

    for (int d = threadIdx.x; d < dim; d += blockDim.x) gradInput_k[d] *= *gradOutput_k;
}

__global__ void rows_sum_kernel(int64_t n, const double* in, double* total) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) atomicAdd(total, in[i]);
}

// Reduce the per-row double buffer to a host scalar and wrap it as a 0-D
// tensor of the requested dtype (self-contained -- no dispatched sum op).
Tensor scalar_from_rows(const Tensor& rows_f64, int64_t n, DType dt, const Device& dev) {
    Tensor total = Tensor::zeros({1}, DType::Float64, dev);
    if (n > 0) {
        auto stream = getCurrentCUDAStream().stream();
        rows_sum_kernel<<<margin_grid(n), kMarginThreads, 0, stream>>>(
            n, rows_f64.data_ptr<double>(), total.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    double h = 0;
    CUDA_CHECK(cudaMemcpy(&h, total.data_ptr<double>(), sizeof(double), cudaMemcpyDeviceToHost));
    return Tensor::full({}, Scalar(h), dt == DType::Float64 ? DType::Float64 : DType::Float32, dev);
}

} // namespace

Tensor multi_margin_loss_cuda(const Tensor& input, const Tensor& target, Scalar p,
                              Scalar margin, const std::optional<Tensor>& weight,
                              int64_t reduction) {
    check_reduction(reduction, "multi_margin_loss");
    const int64_t pint = p.to<int64_t>();
    if (pint != 1 && pint != 2)
        TP_THROW(RuntimeError, "multi_margin_loss: only p == 1 and p == 2 supported");

    int64_t nframe = 0, dim = 0;
    multi_margin_shape_check(nframe, dim, input, target, weight);

    const DType dt = input.dtype();
    const Device dev = input.device();
    const bool per_row = reduction == kReductionNone && target.dim() > 0;
    if (input.numel() == 0) {
        if (per_row) return Tensor::empty({nframe}, dt, dev);
        return Tensor::full({}, Scalar(0.0), dt == DType::Float64 ? DType::Float64 : DType::Float32, dev);
    }

    const Tensor input_f64 = input.contiguous().to(DType::Float64);
    const Tensor target_i64 = as_long_contiguous(target);
    Tensor weight_f64;
    if (weight.has_value() && weight->defined()) weight_f64 = weight->contiguous().to(DType::Float64);
    const double* weight_ptr = weight_f64.defined() ? weight_f64.data_ptr<double>() : nullptr;
    const double mg = margin.toDouble();
    auto stream = getCurrentCUDAStream().stream();

    const bool sizeAverage = reduction == kReductionMean;
    if (input.dim() <= 1) {
        Tensor tmp = Tensor::empty({1}, DType::Float64, dev);
        dim3 blocks(1);
        dim3 threads(kMarginThreads);
        const int dim1 = input.dim() == 0 ? 1 : static_cast<int>(input.size(0));
        if (pint == 1)
            multi_margin_fwd_kernel<1><<<blocks, threads, 0, stream>>>(
                tmp.data_ptr<double>(), input_f64.data_ptr<double>(),
                target_i64.data_ptr<int64_t>(), weight_ptr, 1, dim1, sizeAverage, mg);
        else
            multi_margin_fwd_kernel<2><<<blocks, threads, 0, stream>>>(
                tmp.data_ptr<double>(), input_f64.data_ptr<double>(),
                target_i64.data_ptr<int64_t>(), weight_ptr, 1, dim1, sizeAverage, mg);
        CUDA_CHECK(cudaGetLastError());
        if (per_row) return tmp.to(dt).reshape({nframe});
        return scalar_from_rows(tmp, 1, dt, dev);
    }

    if (per_row) {
        Tensor output = Tensor::empty({nframe}, DType::Float64, dev);
        dim3 blocks(static_cast<unsigned>(nframe));
        dim3 threads(kMarginThreads);
        if (pint == 1)
            multi_margin_fwd_kernel<1><<<blocks, threads, 0, stream>>>(
                output.data_ptr<double>(), input_f64.data_ptr<double>(),
                target_i64.data_ptr<int64_t>(), weight_ptr,
                static_cast<int>(nframe), static_cast<int>(dim), false, mg);
        else
            multi_margin_fwd_kernel<2><<<blocks, threads, 0, stream>>>(
                output.data_ptr<double>(), input_f64.data_ptr<double>(),
                target_i64.data_ptr<int64_t>(), weight_ptr,
                static_cast<int>(nframe), static_cast<int>(dim), false, mg);
        CUDA_CHECK(cudaGetLastError());
        return output.to(dt);
    }

    Tensor tmp = Tensor::empty({nframe}, DType::Float64, dev);
    dim3 blocks(static_cast<unsigned>(nframe));
    dim3 threads(kMarginThreads);
    if (pint == 1)
        multi_margin_fwd_kernel<1><<<blocks, threads, 0, stream>>>(
            tmp.data_ptr<double>(), input_f64.data_ptr<double>(),
            target_i64.data_ptr<int64_t>(), weight_ptr,
            static_cast<int>(nframe), static_cast<int>(dim), sizeAverage, mg);
    else
        multi_margin_fwd_kernel<2><<<blocks, threads, 0, stream>>>(
            tmp.data_ptr<double>(), input_f64.data_ptr<double>(),
            target_i64.data_ptr<int64_t>(), weight_ptr,
            static_cast<int>(nframe), static_cast<int>(dim), sizeAverage, mg);
    CUDA_CHECK(cudaGetLastError());
    return scalar_from_rows(tmp, nframe, dt, dev);
}

Tensor multi_margin_loss_cuda_backward(const Tensor& grad_output, const Tensor& input,
                                       const Tensor& target, Scalar p, Scalar margin,
                                       const std::optional<Tensor>& weight, int64_t reduction) {
    check_reduction(reduction, "multi_margin_loss_backward");
    const int64_t pint = p.to<int64_t>();
    if (pint != 1 && pint != 2)
        TP_THROW(RuntimeError, "multi_margin_loss_backward: only p == 1 and p == 2 supported");

    int64_t nframe = 0, dim = 0;
    multi_margin_shape_check(nframe, dim, input, target, weight);

    const DType dt = input.dtype();
    const Device dev = input.device();
    Tensor grad_input = Tensor::empty(input.shape(), DType::Float64, dev);
    if (input.numel() == 0) return grad_input.to(dt);

    const Tensor input_f64 = input.contiguous().to(DType::Float64);
    const Tensor target_i64 = as_long_contiguous(target);
    const Tensor grad_output_f64 = grad_output.contiguous().to(DType::Float64);
    Tensor weight_f64;
    if (weight.has_value() && weight->defined()) weight_f64 = weight->contiguous().to(DType::Float64);
    const double* weight_ptr = weight_f64.defined() ? weight_f64.data_ptr<double>() : nullptr;
    const double mg = margin.toDouble();
    auto stream = getCurrentCUDAStream().stream();

    const bool sizeAverage = reduction == kReductionMean;
    const bool reduce = reduction != kReductionNone;
    if (input.dim() <= 1) {
        dim3 blocks(1);
        dim3 threads(kMarginThreads);
        const int dim1 = input.dim() == 0 ? 1 : static_cast<int>(input.size(0));
        if (pint == 1)
            multi_margin_bwd_kernel<1><<<blocks, threads, 0, stream>>>(
                grad_input.data_ptr<double>(), grad_output_f64.data_ptr<double>(),
                input_f64.data_ptr<double>(), target_i64.data_ptr<int64_t>(), weight_ptr,
                1, dim1, sizeAverage, mg, reduce);
        else
            multi_margin_bwd_kernel<2><<<blocks, threads, 0, stream>>>(
                grad_input.data_ptr<double>(), grad_output_f64.data_ptr<double>(),
                input_f64.data_ptr<double>(), target_i64.data_ptr<int64_t>(), weight_ptr,
                1, dim1, sizeAverage, mg, reduce);
        CUDA_CHECK(cudaGetLastError());
        return grad_input.to(dt);
    }

    dim3 blocks(static_cast<unsigned>(nframe));
    dim3 threads(kMarginThreads);
    if (pint == 1)
        multi_margin_bwd_kernel<1><<<blocks, threads, 0, stream>>>(
            grad_input.data_ptr<double>(), grad_output_f64.data_ptr<double>(),
            input_f64.data_ptr<double>(), target_i64.data_ptr<int64_t>(), weight_ptr,
            static_cast<int>(nframe), static_cast<int>(dim), sizeAverage, mg, reduce);
    else
        multi_margin_bwd_kernel<2><<<blocks, threads, 0, stream>>>(
            grad_input.data_ptr<double>(), grad_output_f64.data_ptr<double>(),
            input_f64.data_ptr<double>(), target_i64.data_ptr<int64_t>(), weight_ptr,
            static_cast<int>(nframe), static_cast<int>(dim), sizeAverage, mg, reduce);
    CUDA_CHECK(cudaGetLastError());
    return grad_input.to(dt);
}

std::tuple<Tensor, Tensor> multilabel_margin_loss_forward_cuda(const Tensor& input,
                                                               const Tensor& target,
                                                               int64_t reduction) {
    check_reduction(reduction, "multilabel_margin_loss_forward");

    int64_t nframe = 0, dim = 0;
    multilabel_shape_check(nframe, dim, input, target);

    const DType dt = input.dtype();
    const Device dev = input.device();
    Tensor is_target = Tensor::zeros(target.shape(), DType::Float64, dev);
    const bool scalar_out = reduction != kReductionNone || target.dim() <= 1;
    if (input.numel() == 0) {
        Tensor output = scalar_out
            ? Tensor::full({}, Scalar(0.0), dt == DType::Float64 ? DType::Float64 : DType::Float32, dev)
            : Tensor::empty({nframe}, dt, dev);
        return std::make_tuple(output, is_target.to(dt));
    }

    const Tensor input_f64 = input.contiguous().to(DType::Float64);
    const Tensor target_i64 = as_long_contiguous(target);
    auto stream = getCurrentCUDAStream().stream();

    const bool sizeAverage = reduction == kReductionMean;
    if (input.dim() <= 1) {
        Tensor tmp = Tensor::empty({1}, DType::Float64, dev);
        dim3 blocks(1);
        dim3 threads(kMarginThreads);
        multilabel_fwd_kernel<<<blocks, threads, 0, stream>>>(
            tmp.data_ptr<double>(), input_f64.data_ptr<double>(),
            target_i64.data_ptr<int64_t>(), is_target.data_ptr<double>(),
            1, static_cast<int>(dim), sizeAverage);
        CUDA_CHECK(cudaGetLastError());
        return std::make_tuple(scalar_from_rows(tmp, 1, dt, dev), is_target.to(dt));
    }

    if (scalar_out) {
        Tensor tmp = Tensor::empty({nframe}, DType::Float64, dev);
        dim3 blocks(static_cast<unsigned>(nframe));
        dim3 threads(kMarginThreads);
        multilabel_fwd_kernel<<<blocks, threads, 0, stream>>>(
            tmp.data_ptr<double>(), input_f64.data_ptr<double>(),
            target_i64.data_ptr<int64_t>(), is_target.data_ptr<double>(),
            static_cast<int>(nframe), static_cast<int>(dim), sizeAverage);
        CUDA_CHECK(cudaGetLastError());
        return std::make_tuple(scalar_from_rows(tmp, nframe, dt, dev), is_target.to(dt));
    }

    Tensor output = Tensor::empty({nframe}, DType::Float64, dev);
    dim3 blocks(static_cast<unsigned>(nframe));
    dim3 threads(kMarginThreads);
    multilabel_fwd_kernel<<<blocks, threads, 0, stream>>>(
        output.data_ptr<double>(), input_f64.data_ptr<double>(),
        target_i64.data_ptr<int64_t>(), is_target.data_ptr<double>(),
        static_cast<int>(nframe), static_cast<int>(dim), false);
    CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(output.to(dt), is_target.to(dt));
}

Tensor multilabel_margin_loss_backward_cuda(const Tensor& grad_output, const Tensor& input,
                                            const Tensor& target, int64_t reduction,
                                            const Tensor& is_target) {
    check_reduction(reduction, "multilabel_margin_loss_backward");

    int64_t nframe = 0, dim = 0;
    multilabel_shape_check(nframe, dim, input, target);
    if (is_target.shape() != target.shape())
        TP_THROW(RuntimeError, "multilabel_margin_loss_backward: inconsistent is_target size");

    const DType dt = input.dtype();
    const Device dev = input.device();
    Tensor grad_input = Tensor::zeros(input.shape(), DType::Float64, dev);
    if (input.numel() == 0) return grad_input.to(dt);

    const Tensor input_f64 = input.contiguous().to(DType::Float64);
    const Tensor target_i64 = as_long_contiguous(target);
    const Tensor is_target_f64 = is_target.contiguous().to(DType::Float64);
    const Tensor grad_output_f64 = grad_output.contiguous().to(DType::Float64);
    auto stream = getCurrentCUDAStream().stream();

    const bool sizeAverage = reduction == kReductionMean;
    const bool reduce = reduction != kReductionNone;
    const int64_t blocks_count = input.dim() <= 1 ? 1 : nframe;
    dim3 blocks(static_cast<unsigned>(blocks_count));
    dim3 threads(kMarginThreads);
    multilabel_bwd_kernel<<<blocks, threads, 0, stream>>>(
        grad_input.data_ptr<double>(), grad_output_f64.data_ptr<double>(),
        input_f64.data_ptr<double>(), target_i64.data_ptr<int64_t>(),
        is_target_f64.data_ptr<double>(),
        static_cast<int>(input.dim() <= 1 ? 1 : nframe), static_cast<int>(dim), sizeAverage, reduce);
    CUDA_CHECK(cudaGetLastError());
    return grad_input.to(dt);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, MarginLossKernels) {
    m.impl("multi_margin_loss", multi_margin_loss_cuda);
    m.impl("multi_margin_loss_backward", multi_margin_loss_cuda_backward);
    m.impl("multilabel_margin_loss_forward", multilabel_margin_loss_forward_cuda);
    m.impl("multilabel_margin_loss_backward", multilabel_margin_loss_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay
