#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Half.h"
#include "BFloat16.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {

static void check_normalization_cuda_status(cudaError_t error,
                                           const char* operation) {
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string(operation) + ": " +
                                   cudaGetErrorString(error));
    }
}

static void check_normalization_cuda_launch(const char* operation) {
    check_normalization_cuda_status(cudaGetLastError(), operation);
}

namespace {

constexpr int kRmsWarpSize = 32;
constexpr int kRmsThreadsX = 32;
constexpr int kRmsThreadsY = 8;

template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) RmsAlignedVec {
    T val[VecSize];
};

template <typename Acc>
struct RmsWelfordData {
    Acc mean;
    Acc m2;
    int64_t n;
    Acc nf;
};

template <typename Acc>
__device__ inline RmsWelfordData<Acc> rms_welford_reduce(
        RmsWelfordData<Acc> acc, Acc value) {
    const Acc delta = value - acc.mean;
    const Acc nf = acc.nf + Acc(1);
    const Acc mean = acc.mean + delta / nf;
    return {mean, acc.m2 + delta * (value - mean), acc.n + 1, nf};
}

template <typename Acc>
__device__ inline RmsWelfordData<Acc> rms_welford_combine(
        RmsWelfordData<Acc> a, RmsWelfordData<Acc> b) {
    if (a.nf == Acc(0)) return b;
    if (b.nf == Acc(0)) return a;
    const Acc delta = b.mean - a.mean;
    const Acc nf = a.nf + b.nf;
    const Acc nb = b.nf / nf;
    return {
        a.mean + delta * nb,
        a.m2 + b.m2 + delta * delta * a.nf * nb,
        -1,
        nf,
    };
}

template <typename Acc>
__device__ inline RmsWelfordData<Acc> rms_welford_warp_reduce(
        RmsWelfordData<Acc> value) {
#pragma unroll
    for (int offset = kRmsWarpSize / 2; offset > 0; offset >>= 1) {
        RmsWelfordData<Acc> other;
        other.mean = __shfl_down_sync(0xffffffffffffffffull, value.mean, offset);
        other.m2 = __shfl_down_sync(0xffffffffffffffffull, value.m2, offset);
        other.n = __shfl_down_sync(0xffffffffffffffffull, value.n, offset);
        other.nf = __shfl_down_sync(0xffffffffffffffffull, value.nf, offset);
        value = rms_welford_combine(value, other);
    }
    return value;
}

template <typename Acc>
__device__ inline RmsWelfordData<Acc> rms_welford_block_reduce(
        RmsWelfordData<Acc> value, RmsWelfordData<Acc>* shared) {
    value = rms_welford_warp_reduce(value);
    if (threadIdx.x == 0) shared[threadIdx.y] = value;
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        RmsWelfordData<Acc> total{};
        for (int y = 0; y < blockDim.y; ++y) {
            total = rms_welford_combine(total, shared[y]);
        }
        shared[0] = total;
    }
    __syncthreads();
    return shared[0];
}

template <typename Acc>
__device__ inline Acc rms_sum_warp_reduce(Acc value) {
#pragma unroll
    for (int offset = kRmsWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
    }
    return value;
}

template <typename Acc>
__device__ inline Acc rms_sum_block_reduce(Acc value, Acc* shared) {
    value = rms_sum_warp_reduce(value);
    if (threadIdx.x == 0) shared[threadIdx.y] = value;
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        Acc total = Acc(0);
        for (int y = 0; y < blockDim.y; ++y) total += shared[y];
        shared[0] = total;
    }
    __syncthreads();
    return shared[0];
}

__device__ inline float rms_rsqrt(float value) {
    return rsqrtf(value);
}

__device__ inline double rms_rsqrt(double value) {
    return 1.0 / ::sqrt(value);
}

template <typename T>
bool rms_can_vectorize(const T* pointer, int alignment) {
    return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

template <typename T, typename Acc>
__global__ void rms_vectorized_forward_kernel(
        int64_t normalized_size,
        Acc eps,
        const T* __restrict__ input,
        const T* __restrict__ weight,
        T* __restrict__ output,
        Acc* __restrict__ rstd) {
    using Vec = RmsAlignedVec<T, 4>;
    __shared__ Acc shared[kRmsThreadsY];
    const int linear = threadIdx.x + threadIdx.y * blockDim.x;
    const int threads = blockDim.x * blockDim.y;
    const int64_t row = blockIdx.x;
    const int64_t vectors = normalized_size / 4;
    const Vec* input_vec = reinterpret_cast<const Vec*>(
        input + row * normalized_size);
    Vec* output_vec = reinterpret_cast<Vec*>(
        output + row * normalized_size);
    Acc sum = Acc(0);
    for (int64_t i = linear; i < vectors; i += threads) {
        const Vec values = input_vec[i];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const Acc value = static_cast<Acc>(values.val[k]);
            sum += value * value;
        }
    }
    sum = rms_sum_block_reduce(sum, shared);
    const Acc inverse = rms_rsqrt(
        sum / static_cast<Acc>(normalized_size) + eps);
    if (linear == 0) rstd[row] = inverse;
    for (int64_t i = linear; i < vectors; i += threads) {
        const Vec values = input_vec[i];
        Vec result;
        const Vec* weight_vec = weight == nullptr
            ? nullptr
            : reinterpret_cast<const Vec*>(weight);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const Acc gamma = weight_vec == nullptr
                ? Acc(1)
                : static_cast<Acc>(weight_vec[i].val[k]);
            result.val[k] = static_cast<T>(
                static_cast<Acc>(values.val[k]) * inverse * gamma);
        }
        output_vec[i] = result;
    }
}

template <typename T, typename Acc>
__global__ void rms_rowwise_moments_kernel(
        int64_t normalized_size,
        Acc eps,
        const T* __restrict__ input,
        Acc* __restrict__ rstd) {
    __shared__ RmsWelfordData<Acc> shared[kRmsThreadsY];
    const int linear = threadIdx.x + threadIdx.y * blockDim.x;
    const int threads = blockDim.x * blockDim.y;
    const int64_t row = blockIdx.x;
    const T* input_row = input + row * normalized_size;
    RmsWelfordData<Acc> value{};
    for (int64_t i = linear; i < normalized_size; i += threads) {
        value = rms_welford_reduce(
            value, static_cast<Acc>(input_row[i]));
    }
    value = rms_welford_block_reduce(value, shared);
    if (linear == 0) {
        const Acc variance = value.nf == Acc(0)
            ? Acc(0)
            : value.m2 / value.nf;
        rstd[row] = rms_rsqrt(
            variance + value.mean * value.mean + eps);
    }
}

template <typename T, typename Acc>
__global__ void rms_rowwise_forward_kernel(
        int64_t normalized_size,
        const T* __restrict__ input,
        const T* __restrict__ weight,
        const Acc* __restrict__ rstd,
        T* __restrict__ output) {
    const int linear = threadIdx.x + threadIdx.y * blockDim.x;
    const int threads = blockDim.x * blockDim.y;
    const int64_t row = blockIdx.x;
    const T* input_row = input + row * normalized_size;
    T* output_row = output + row * normalized_size;
    for (int64_t i = linear; i < normalized_size; i += threads) {
        const Acc gamma = weight == nullptr
            ? Acc(1)
            : static_cast<Acc>(weight[i]);
        output_row[i] = static_cast<T>(
            static_cast<Acc>(input_row[i]) * rstd[row] * gamma);
    }
}

template <typename T, typename Acc>
std::tuple<Tensor, Tensor> rms_norm_forward_dtype(
        const Tensor& input,
        const std::vector<int64_t>& normalized_shape,
        const std::optional<Tensor>& weight_opt,
        std::optional<double> eps_opt) {
    const int64_t input_ndim = input.dim();
    const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
    if (norm_ndim > input_ndim) {
        TP_THROW(RuntimeError,
                 "rms_norm: normalized_shape dim larger than input dim");
    }
    int64_t normalized_size = 1;
    for (int64_t i = 0; i < norm_ndim; ++i) {
        if (input.size(input_ndim - norm_ndim + i) != normalized_shape[i]) {
            TP_THROW(RuntimeError,
                     "rms_norm: Input shape mismatch with normalized_shape");
        }
        normalized_size *= normalized_shape[i];
    }
    const bool has_weight = weight_opt.has_value() && weight_opt->defined();
    if (has_weight && weight_opt->numel() != normalized_size) {
        TP_THROW(RuntimeError,
                 "rms_norm: weight shape mismatch with normalized_shape");
    }
    const int64_t rows = input.numel() /
        (normalized_size == 0 ? 1 : normalized_size);
    TP_CHECK(rows <= std::numeric_limits<int>::max(),
             "rms_norm: too many rows");
    TP_CHECK(normalized_size <= std::numeric_limits<int>::max(),
             "rms_norm: normalized shape is too large");

    Tensor input_contig = input.contiguous();
    Tensor weight = has_weight ? weight_opt->contiguous() : Tensor();
    Tensor output = Tensor::empty(
        static_cast<std::vector<int64_t>>(input_contig.shape()),
        input_contig.dtype(), input_contig.device());
    const DType acc_dtype = std::is_same<Acc, double>::value
        ? DType::Float64
        : DType::Float32;
    Tensor rstd = Tensor::empty({rows}, acc_dtype, input_contig.device());
    if (rows == 0 || normalized_size == 0) {
        return {output, rstd};
    }

    const Acc eps = static_cast<Acc>(eps_opt.value_or(
        std::is_same<Acc, double>::value
            ? std::numeric_limits<double>::epsilon()
            : std::numeric_limits<float>::epsilon()));
    const auto stream = getCurrentCUDAStream().stream();
    const dim3 threads(kRmsThreadsX, kRmsThreadsY);
    const int alignment = static_cast<int>(sizeof(T) * 4);
    const bool vectorized =
        !std::is_same<T, double>::value && normalized_size % 4 == 0 &&
        rms_can_vectorize(input_contig.data_ptr<T>(), alignment) &&
        rms_can_vectorize(output.data_ptr<T>(), alignment) &&
        (!has_weight || rms_can_vectorize(weight.data_ptr<T>(), alignment));
    if (vectorized) {
        rms_vectorized_forward_kernel<T, Acc>
            <<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
                normalized_size, eps, input_contig.data_ptr<T>(),
                has_weight ? weight.data_ptr<T>() : nullptr,
                output.data_ptr<T>(), rstd.data_ptr<Acc>());
    } else {
        const size_t shared_bytes =
            sizeof(RmsWelfordData<Acc>) * kRmsThreadsY;
        rms_rowwise_moments_kernel<T, Acc>
            <<<static_cast<unsigned int>(rows), threads, shared_bytes, stream>>>(
                normalized_size, eps, input_contig.data_ptr<T>(),
                rstd.data_ptr<Acc>());
        rms_rowwise_forward_kernel<T, Acc>
            <<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
                normalized_size, input_contig.data_ptr<T>(),
                has_weight ? weight.data_ptr<T>() : nullptr,
                rstd.data_ptr<Acc>(), output.data_ptr<T>());
    }
    check_normalization_cuda_launch("rms_norm_cuda");

    const int64_t axis = input_ndim - norm_ndim;
    std::vector<int64_t> stat_shape;
    for (int64_t i = 0; i < axis; ++i) {
        stat_shape.push_back(input_contig.size(i));
    }
    for (int64_t i = axis; i < input_ndim; ++i) stat_shape.push_back(1);
    int64_t stat_numel = 1;
    for (int64_t size : stat_shape) stat_numel *= size;
    if (rstd.numel() == stat_numel) rstd = rstd.view(stat_shape);
    return {output, rstd};
}

template <typename T, typename Acc>
__global__ void rms_grad_input_kernel(
        int64_t normalized_size,
        const T* __restrict__ grad_output,
        const T* __restrict__ input,
        const Acc* __restrict__ rstd,
        const T* __restrict__ weight,
        T* __restrict__ grad_input) {
    __shared__ Acc shared[kRmsThreadsY];
    const int linear = threadIdx.x + threadIdx.y * blockDim.x;
    const int threads = blockDim.x * blockDim.y;
    const int64_t row = blockIdx.x;
    const T* grad_row = grad_output + row * normalized_size;
    const T* input_row = input + row * normalized_size;
    T* grad_input_row = grad_input + row * normalized_size;
    const Acc inverse = rstd[row];
    const Acc f_h = static_cast<Acc>(normalized_size);
    Acc sum = Acc(0);
    int64_t index = 4 * linear;
    for (; index + 3 < normalized_size; index += 4 * threads) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const Acc grad = static_cast<Acc>(grad_row[index + k]);
            const Acc gamma = weight == nullptr
                ? Acc(1)
                : static_cast<Acc>(weight[index + k]);
            const Acc value = static_cast<Acc>(input_row[index + k]);
            sum += grad * gamma * value * inverse;
        }
    }
    for (; index < normalized_size; ++index) {
        const Acc grad = static_cast<Acc>(grad_row[index]);
        const Acc gamma = weight == nullptr
            ? Acc(1)
            : static_cast<Acc>(weight[index]);
        const Acc value = static_cast<Acc>(input_row[index]);
        sum += grad * gamma * value * inverse;
    }
    sum = rms_sum_block_reduce(sum, shared);
    const Acc scale = inverse / f_h;
    for (index = linear; index < normalized_size; index += threads) {
        const Acc grad = static_cast<Acc>(grad_row[index]);
        const Acc gamma = weight == nullptr
            ? Acc(1)
            : static_cast<Acc>(weight[index]);
        const Acc value = static_cast<Acc>(input_row[index]);
        grad_input_row[index] = static_cast<T>(
            (f_h * grad * gamma - value * inverse * sum) * scale);
    }
}

template <typename T, typename Acc>
__global__ void rms_grad_input_vectorized_kernel(
        int64_t normalized_size,
        const T* __restrict__ grad_output,
        const T* __restrict__ input,
        const Acc* __restrict__ rstd,
        const T* __restrict__ weight,
        T* __restrict__ grad_input) {
    using Vec = RmsAlignedVec<T, 4>;
    __shared__ Acc shared[kRmsThreadsY];
    const int linear = threadIdx.x + threadIdx.y * blockDim.x;
    const int threads = blockDim.x * blockDim.y;
    const int64_t row = blockIdx.x;
    const int64_t vectors = normalized_size / 4;
    const Vec* grad_vec = reinterpret_cast<const Vec*>(
        grad_output + row * normalized_size);
    const Vec* input_vec = reinterpret_cast<const Vec*>(
        input + row * normalized_size);
    const Vec* weight_vec = weight == nullptr
        ? nullptr
        : reinterpret_cast<const Vec*>(weight);
    Vec* grad_input_vec = reinterpret_cast<Vec*>(
        grad_input + row * normalized_size);
    const Acc inverse = rstd[row];
    const Acc f_h = static_cast<Acc>(normalized_size);
    Acc sum = Acc(0);
    for (int64_t i = linear; i < vectors; i += threads) {
        const Vec grad_values = grad_vec[i];
        const Vec input_values = input_vec[i];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const Acc grad = static_cast<Acc>(grad_values.val[k]);
            const Acc gamma = weight_vec == nullptr
                ? Acc(1)
                : static_cast<Acc>(weight_vec[i].val[k]);
            const Acc value = static_cast<Acc>(input_values.val[k]);
            sum += grad * gamma * value * inverse;
        }
    }
    sum = rms_sum_block_reduce(sum, shared);
    const Acc scale = inverse / f_h;
    for (int64_t i = linear; i < vectors; i += threads) {
        const Vec grad_values = grad_vec[i];
        const Vec input_values = input_vec[i];
        Vec result;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const Acc grad = static_cast<Acc>(grad_values.val[k]);
            const Acc gamma = weight_vec == nullptr
                ? Acc(1)
                : static_cast<Acc>(weight_vec[i].val[k]);
            const Acc value = static_cast<Acc>(input_values.val[k]);
            result.val[k] = static_cast<T>(
                (f_h * grad * gamma - value * inverse * sum) * scale);
        }
        grad_input_vec[i] = result;
    }
}

template <typename T, typename Acc, int BlockX, int BlockY, int RowsPerBlock>
__device__ __forceinline__ void rms_gamma_helper(
        int64_t row_start,
        int64_t rows,
        int64_t normalized_size,
        const T* __restrict__ grad_output,
        const T* __restrict__ input,
        const Acc* __restrict__ rstd,
        int64_t thread_x,
        bool check_x,
        bool check_y,
        Acc& sum) {
    constexpr int rows_per_thread = RowsPerBlock / BlockY;
    const int lane = threadIdx.x;
    const int64_t rstd_row = row_start + threadIdx.y * rows_per_thread;
    Acc warp_rstd = Acc(0);
    if (lane < rows_per_thread && rstd_row + lane < rows) {
        warp_rstd = rstd[rstd_row + lane];
    }
    __syncwarp();
    Acc grad_regs[rows_per_thread];
    Acc input_regs[rows_per_thread];
#pragma unroll
    for (int k = 0; k < rows_per_thread; ++k) {
        const int64_t row = row_start + threadIdx.y * rows_per_thread + k;
        const bool active =
            (!check_x || thread_x < normalized_size) &&
            (!check_y || row < rows);
        grad_regs[k] = active
            ? static_cast<Acc>(grad_output[row * normalized_size + thread_x])
            : Acc(0);
        input_regs[k] = active
            ? static_cast<Acc>(input[row * normalized_size + thread_x])
            : Acc(0);
    }
#pragma unroll
    for (int k = 0; k < rows_per_thread; ++k) {
        const Acc row_rstd = __shfl_sync(0xffffffffffffffffull, warp_rstd, k);
        sum += grad_regs[k] * input_regs[k] * row_rstd;
    }
}

template <typename T, typename Acc, int BlockX, int BlockY,
          int RowsPerBlock, bool Aligned>
__global__ void rms_gamma_kernel(
        int64_t rows,
        int64_t normalized_size,
        const T* __restrict__ grad_output,
        const T* __restrict__ input,
        const Acc* __restrict__ rstd,
        T* __restrict__ grad_weight) {
    extern __shared__ unsigned char shared_bytes[];
    Acc* shared = reinterpret_cast<Acc*>(shared_bytes);
    const int64_t thread_x =
        static_cast<int64_t>(blockIdx.x) * BlockX + threadIdx.x;
    Acc sum = Acc(0);
    for (int64_t row_start = blockIdx.y * RowsPerBlock;
         row_start < rows;
         row_start += RowsPerBlock * gridDim.y) {
        rms_gamma_helper<T, Acc, BlockX, BlockY, RowsPerBlock>(
            row_start, rows, normalized_size, grad_output, input, rstd,
            thread_x, !Aligned, !Aligned, sum);
    }
    if constexpr (BlockY == 1) {
        if (Aligned || thread_x < normalized_size) {
            grad_weight[static_cast<int64_t>(blockIdx.y) * normalized_size +
                        thread_x] = static_cast<T>(sum);
        }
    } else {
        const int padded_x = BlockX + 1;
        shared[threadIdx.y * padded_x + threadIdx.x] = sum;
        __syncthreads();
        const int thread_id = threadIdx.y * BlockX + threadIdx.x;
        const int warp_id = thread_id / kRmsWarpSize;
        const int lane = thread_id % kRmsWarpSize;
        const int warps = BlockX * BlockY / kRmsWarpSize;
        for (int x = warp_id; x < BlockX; x += warps) {
            Acc value = Acc(0);
            if (lane < BlockY) value = shared[lane * padded_x + x];
#pragma unroll
            for (int delta = BlockY / 2; delta > 0; delta >>= 1) {
                value += __shfl_xor_sync(0xffffffffffffffffull, value, delta);
            }
            const int64_t output_x =
                static_cast<int64_t>(blockIdx.x) * BlockX + x;
            if (threadIdx.x == 0 && (Aligned || output_x < normalized_size)) {
                grad_weight[output_x] = static_cast<T>(value);
            }
        }
    }
}

template <typename T, typename Acc, int BlockY, int RowsPerBlock>
void launch_rms_gamma_kernel(
        const T* grad_output,
        const T* input,
        const Acc* rstd,
        int64_t rows,
        int64_t normalized_size,
        Tensor& grad_weight,
        cudaStream_t stream) {
    constexpr int block_x = kRmsWarpSize;
    const bool aligned = rows % RowsPerBlock == 0 &&
        normalized_size % block_x == 0;
    dim3 blocks(
        static_cast<unsigned int>((normalized_size + block_x - 1) / block_x),
        1);
    dim3 threads(block_x, BlockY);
    const size_t shared_bytes =
        BlockY == 1 ? 0 : (block_x + 1) * BlockY * sizeof(Acc);
    if (aligned) {
        rms_gamma_kernel<T, Acc, block_x, BlockY, RowsPerBlock, true>
            <<<blocks, threads, shared_bytes, stream>>>(
                rows, normalized_size, grad_output, input, rstd,
                grad_weight.data_ptr<T>());
    } else {
        rms_gamma_kernel<T, Acc, block_x, BlockY, RowsPerBlock, false>
            <<<blocks, threads, shared_bytes, stream>>>(
                rows, normalized_size, grad_output, input, rstd,
                grad_weight.data_ptr<T>());
    }
    check_normalization_cuda_launch("fused_rms_norm_backward gamma");
}

template <typename T, typename Acc>
void launch_rms_gamma(
        const T* grad_output,
        const T* input,
        const Acc* rstd,
        int64_t rows,
        int64_t normalized_size,
        Tensor& grad_weight,
        cudaStream_t stream) {
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 1;
    cudaDeviceGetAttribute(
        &sm_count, cudaDevAttrMultiProcessorCount, device);
    const bool huge = rows > 65536 &&
        normalized_size / kRmsWarpSize < sm_count / 2;
    if (huge) {
        constexpr int block_y = 1;
        constexpr int rows_per_block = 32;
        const unsigned int block_x_count = static_cast<unsigned int>(
            (normalized_size + kRmsWarpSize - 1) / kRmsWarpSize);
        const unsigned int block_y_count = std::min<unsigned int>(
            32768u / block_x_count,
            static_cast<unsigned int>((rows + rows_per_block - 1) /
                                       rows_per_block));
        Tensor partials = Tensor::empty(
            {static_cast<int64_t>(block_y_count), normalized_size},
            grad_weight.dtype(), grad_weight.device());
        const bool aligned = rows % rows_per_block == 0 &&
            normalized_size % kRmsWarpSize == 0;
        if (aligned) {
            rms_gamma_kernel<T, Acc, kRmsWarpSize, block_y, rows_per_block, true>
                <<<dim3(block_x_count, block_y_count), dim3(kRmsWarpSize, 1),
                   0, stream>>>(
                    rows, normalized_size, grad_output, input, rstd,
                    partials.data_ptr<T>());
        } else {
            rms_gamma_kernel<T, Acc, kRmsWarpSize, block_y, rows_per_block, false>
                <<<dim3(block_x_count, block_y_count), dim3(kRmsWarpSize, 1),
                   0, stream>>>(
                    rows, normalized_size, grad_output, input, rstd,
                    partials.data_ptr<T>());
        }
        check_normalization_cuda_launch("fused_rms_norm_backward gamma partials");
        grad_weight = partials.sum(0);
        return;
    }
    if (rows < 64) {
        launch_rms_gamma_kernel<T, Acc, 1, 8>(
            grad_output, input, rstd, rows, normalized_size, grad_weight,
            stream);
    } else if (rows < 128) {
        launch_rms_gamma_kernel<T, Acc, 8, 64>(
            grad_output, input, rstd, rows, normalized_size, grad_weight,
            stream);
    } else if (rows < 256) {
        launch_rms_gamma_kernel<T, Acc, 16, 128>(
            grad_output, input, rstd, rows, normalized_size, grad_weight,
            stream);
    } else {
        launch_rms_gamma_kernel<T, Acc, 32, 256>(
            grad_output, input, rstd, rows, normalized_size, grad_weight,
            stream);
    }
}

template <typename T, typename Acc>
std::tuple<Tensor, Tensor> rms_norm_backward_dtype(
        const Tensor& grad_output,
        const Tensor& input,
        const std::vector<int64_t>& normalized_shape,
        const Tensor& rstd,
        const std::optional<Tensor>& weight_opt,
        const std::vector<bool>& output_mask) {
    const int64_t input_ndim = input.dim();
    const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
    if (norm_ndim > input_ndim) {
        TP_THROW(RuntimeError,
                 "fused_rms_norm_backward: normalized_shape dim larger than input dim");
    }
    int64_t normalized_size = 1;
    for (int64_t i = 0; i < norm_ndim; ++i) {
        if (input.size(input_ndim - norm_ndim + i) != normalized_shape[i]) {
            TP_THROW(RuntimeError,
                     "fused_rms_norm_backward: Input shape mismatch with normalized_shape");
        }
        normalized_size *= normalized_shape[i];
    }
    if (grad_output.dtype() != input.dtype() ||
        grad_output.numel() != input.numel()) {
        TP_THROW(RuntimeError,
                 "fused_rms_norm_backward: grad_output must match input");
    }
    const int64_t rows = input.numel() /
        (normalized_size == 0 ? 1 : normalized_size);
    if (rstd.numel() != rows) {
        TP_THROW(RuntimeError,
                 "fused_rms_norm_backward: saved rstd must have one value per row");
    }
    TP_CHECK(rows <= std::numeric_limits<int>::max(),
             "fused_rms_norm_backward: too many rows");
    TP_CHECK(normalized_size <= std::numeric_limits<int>::max(),
             "fused_rms_norm_backward: normalized shape is too large");

    const bool has_weight = weight_opt.has_value() && weight_opt->defined();
    if (has_weight && weight_opt->numel() != normalized_size) {
        TP_THROW(RuntimeError,
                 "fused_rms_norm_backward: weight shape mismatch");
    }
    Tensor grad_output_contig = grad_output.contiguous();
    Tensor input_contig = input.contiguous();
    Tensor weight = has_weight ? weight_opt->contiguous() : Tensor();
    Tensor rstd_flat = rstd.contiguous().reshape({rows});
    const DType acc_dtype = std::is_same<Acc, double>::value
        ? DType::Float64
        : DType::Float32;
    if (rstd_flat.dtype() != acc_dtype) {
        TP_THROW(RuntimeError,
                 "fused_rms_norm_backward: saved rstd dtype mismatch");
    }
    const bool need_grad_input = output_mask.empty() || output_mask[0];
    const bool need_grad_weight = output_mask.size() > 1 && output_mask[1];
    Tensor grad_input = need_grad_input
        ? Tensor::empty_like(input_contig)
        : Tensor();
    Tensor grad_weight = need_grad_weight && has_weight
        ? Tensor::empty({normalized_size}, input.dtype(), input.device())
        : Tensor();
    if (rows == 0 || normalized_size == 0) {
        if (need_grad_weight && !has_weight) grad_weight = Tensor();
        return {grad_input, grad_weight};
    }

    const auto stream = getCurrentCUDAStream().stream();
    const dim3 threads(kRmsThreadsX, kRmsThreadsY);
    if (grad_input.defined()) {
        const int alignment = static_cast<int>(sizeof(T) * 4);
        const bool vectorized =
            !std::is_same<T, double>::value && normalized_size % 4 == 0 &&
            rms_can_vectorize(grad_output_contig.data_ptr<T>(), alignment) &&
            rms_can_vectorize(input_contig.data_ptr<T>(), alignment) &&
            rms_can_vectorize(grad_input.data_ptr<T>(), alignment) &&
            (!has_weight || rms_can_vectorize(weight.data_ptr<T>(), alignment));
        if (vectorized) {
            rms_grad_input_vectorized_kernel<T, Acc>
                <<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
                    normalized_size, grad_output_contig.data_ptr<T>(),
                    input_contig.data_ptr<T>(), rstd_flat.data_ptr<Acc>(),
                    has_weight ? weight.data_ptr<T>() : nullptr,
                    grad_input.data_ptr<T>());
        } else {
            rms_grad_input_kernel<T, Acc>
                <<<static_cast<unsigned int>(rows), threads,
                   kRmsThreadsY * sizeof(Acc), stream>>>(
                    normalized_size, grad_output_contig.data_ptr<T>(),
                    input_contig.data_ptr<T>(), rstd_flat.data_ptr<Acc>(),
                    has_weight ? weight.data_ptr<T>() : nullptr,
                    grad_input.data_ptr<T>());
        }
        check_normalization_cuda_launch("fused_rms_norm_backward input");
    }
    if (grad_weight.defined()) {
        launch_rms_gamma<T, Acc>(
            grad_output_contig.data_ptr<T>(), input_contig.data_ptr<T>(),
            rstd_flat.data_ptr<Acc>(), rows, normalized_size, grad_weight,
            stream);
    }
    return {grad_input, grad_weight};
}

}

std::tuple<Tensor, Tensor> fused_rms_norm_cuda(
        const Tensor& input,
        const std::vector<int64_t>& normalized_shape,
        const std::optional<Tensor>& weight_opt,
        std::optional<double> eps_opt) {
    switch (input.dtype()) {
        case DType::Float32:
            return rms_norm_forward_dtype<float, float>(
                input, normalized_shape, weight_opt, eps_opt);
        case DType::Float64:
            return rms_norm_forward_dtype<double, double>(
                input, normalized_shape, weight_opt, eps_opt);
        case DType::Float16:
            return rms_norm_forward_dtype<tensorplay::Half, float>(
                input, normalized_shape, weight_opt, eps_opt);
        case DType::BFloat16:
            return rms_norm_forward_dtype<tensorplay::BFloat16, float>(
                input, normalized_shape, weight_opt, eps_opt);
        default:
            TP_THROW(NotImplementedError,
                     "fused RMSNorm CUDA supports floating-point dtypes only");
    }
}

std::tuple<Tensor, Tensor> fused_rms_norm_backward_cuda(
        const Tensor& grad_output,
        const Tensor& input,
        const std::vector<int64_t>& normalized_shape,
        const Tensor& rstd,
        const std::optional<Tensor>& weight_opt,
        const std::vector<bool>& output_mask) {
    switch (input.dtype()) {
        case DType::Float32:
            return rms_norm_backward_dtype<float, float>(
                grad_output, input, normalized_shape, rstd, weight_opt,
                output_mask);
        case DType::Float64:
            return rms_norm_backward_dtype<double, double>(
                grad_output, input, normalized_shape, rstd, weight_opt,
                output_mask);
        case DType::Float16:
            return rms_norm_backward_dtype<tensorplay::Half, float>(
                grad_output, input, normalized_shape, rstd, weight_opt,
                output_mask);
        case DType::BFloat16:
            return rms_norm_backward_dtype<tensorplay::BFloat16, float>(
                grad_output, input, normalized_shape, rstd, weight_opt,
                output_mask);
        default:
            TP_THROW(NotImplementedError,
                     "fused RMSNorm backward CUDA supports floating-point dtypes only");
    }
}

Tensor rms_norm_cuda(
        const Tensor& input,
        const std::vector<int64_t>& normalized_shape,
        const std::optional<Tensor>& weight_opt,
        std::optional<double> eps_opt) {
    return std::get<0>(fused_rms_norm_cuda(
        input, normalized_shape, weight_opt, eps_opt));
}

TENSORPLAY_LIBRARY_IMPL(CUDA, RMSNormKernels) {
    m.impl("rms_norm", rms_norm_cuda);
    m.impl("_fused_rms_norm", fused_rms_norm_cuda);
    m.impl("_fused_rms_norm_backward", fused_rms_norm_backward_cuda);
}

}
}
