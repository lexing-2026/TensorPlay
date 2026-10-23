#include "ReduceEngine.cuh"

#include <algorithm>

namespace tensorplay {
namespace cuda {

using namespace reduce_family;

namespace {

template <typename T>
Tensor norm_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim, double p) {
    using AccT = same_dtype_acc_t<T>;
    if (input.numel() == 0) {
        if (spec.reduced_numel == 0 &&
            (p < 0.0 || p == std::numeric_limits<double>::infinity())) {
            TP_THROW(RuntimeError,
                     "norm cannot reduce an empty dimension for this order");
        }
        return Tensor::zeros(
            reduction_output_shape(input, spec, keepdim), input.dtype(), input.device());
    }
    if (p == 0.0) {
        return run_reduction_typed<T, AccT, T>(
            input, spec, keepdim, input.dtype(), NormZeroOps<T, AccT, T>{}, AccT(0));
    }
    if (p == 1.0) {
        return run_reduction_typed<T, AccT, T>(
            input, spec, keepdim, input.dtype(), NormOneOps<T, AccT, T>{}, AccT(0));
    }
    if (p == 2.0) {
        return run_reduction_typed<T, AccT, T>(
            input, spec, keepdim, input.dtype(), NormTwoOps<AccT, T>{}, AccT(0));
    }
    if (p == std::numeric_limits<double>::infinity()) {
        return run_reduction_typed<T, AccT, T>(
            input, spec, keepdim, input.dtype(), AbsMaxOps<T, AccT, T>{}, AccT(0));
    }
    if (p == -std::numeric_limits<double>::infinity()) {
        return run_reduction_typed<T, AccT, T>(
            input, spec, keepdim, input.dtype(), AbsMinOps<T, AccT, T>{},
            std::numeric_limits<AccT>::infinity());
    }
    return run_reduction_typed<T, AccT, T>(
        input, spec, keepdim, input.dtype(), NormOps<T, AccT, T>{static_cast<AccT>(p)},
        AccT(0));
}

} // namespace

// Norm (L2)
//
// The generic reduction engine is deliberately flexible, but its global
// reduction configuration can split a single scalar across many CTAs and
// then allocate/finalize a partial buffer.  That overhead is visible in the
// hot Muon path, which normalizes one matrix per optimizer step.  Keep a
// native norm kernel: one coalesced grid pass and one small final reduction.
template <typename InputT, typename AccT>
__global__ void norm2_partial_kernel(
        int64_t n, const InputT* input, AccT* partials) {
    AccT value = AccT(0);
    const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = first; i < n; i += stride) {
        const AccT v = static_cast<AccT>(input[i]);
        value += v * v;
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
    }
    __shared__ AccT warp_values[32];
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();

    if (warp == 0) {
        const int warp_count = (blockDim.x + 31) / 32;
        value = lane < warp_count ? warp_values[lane] : AccT(0);
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
        }
        if (lane == 0) partials[blockIdx.x] = value;
    }
}

template <typename AccT, typename OutputT>
__global__ void norm2_finalize_kernel(
        int64_t count, const AccT* partials, OutputT* output) {
    AccT value = AccT(0);
    for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
        value += partials[i];
    }
    __shared__ AccT warp_values[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
    }
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    if (warp == 0) {
        const int warp_count = (blockDim.x + 31) / 32;
        value = lane < warp_count ? warp_values[lane] : AccT(0);
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
        }
        if (lane == 0) {
            if constexpr (std::is_same_v<AccT, float>) {
                output[0] = static_cast<OutputT>(sqrtf(value));
            } else {
                output[0] = static_cast<OutputT>(::sqrt(value));
            }
        }
    }
}

template <typename InputT, typename AccT, typename OutputT>
__global__ void norm2_single_block_kernel(
        int64_t n, const InputT* input, OutputT* output) {
    AccT value = AccT(0);
    for (int64_t base = static_cast<int64_t>(threadIdx.x) * 4;
         base < n;
         base += static_cast<int64_t>(blockDim.x) * 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t i = base + j;
            if (i < n) {
                const AccT v = static_cast<AccT>(input[i]);
                value += v * v;
            }
        }
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
    }
    __shared__ AccT warp_values[32];
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();

    if (warp == 0) {
        const int warp_count = (blockDim.x + 31) / 32;
        value = lane < warp_count ? warp_values[lane] : AccT(0);
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
        }
        if (lane == 0) {
            if constexpr (std::is_same_v<AccT, float>) {
                output[0] = static_cast<OutputT>(sqrtf(value));
            } else {
                output[0] = static_cast<OutputT>(::sqrt(value));
            }
        }
    }
}

template <typename InputT, typename AccT>
__global__ void norm2_atomic_kernel(
        int64_t n, const InputT* input, AccT* accumulator) {
    AccT value = AccT(0);
    for (int64_t base =
             static_cast<int64_t>(blockIdx.x) * blockDim.x * 4 +
             static_cast<int64_t>(threadIdx.x) * 4;
         base < n;
         base += static_cast<int64_t>(gridDim.x) * blockDim.x * 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t i = base + j;
            if (i < n) {
                const AccT v = static_cast<AccT>(input[i]);
                value += v * v;
            }
        }
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
    }
    __shared__ AccT warp_values[32];
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();

    if (warp == 0) {
        const int warp_count = (blockDim.x + 31) / 32;
        value = lane < warp_count ? warp_values[lane] : AccT(0);
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
        }
        if (lane == 0) atomicAdd(accumulator, value);
    }
}

template <typename AccT, typename OutputT>
__global__ void norm2_finalize_scalar_kernel(
        const AccT* accumulator, OutputT* output) {
    if (threadIdx.x == 0) {
        const AccT value = *accumulator;
        if constexpr (std::is_same_v<AccT, float>) {
            output[0] = static_cast<OutputT>(sqrtf(value));
        } else {
            output[0] = static_cast<OutputT>(::sqrt(value));
        }
    }
}

template <typename InputT, typename AccT>
Tensor norm2_global_fast_typed(const Tensor& self) {
    Tensor result = Tensor::empty({}, self.dtype(), self.device());
    const int64_t n = self.numel();
    if (n == 0) {
        result.zero_();
        return result;
    }

    constexpr int block_size = 256;
    const auto stream = getCurrentCUDAStream().stream();

    // Muon commonly normalizes matrices up to a few million elements.  A
    // single coalesced block avoids both the temporary partial allocation and
    // the second launch while retaining a grid-stride loop over the complete
    // tensor.  Use the two-stage path for very large reductions so bandwidth
    // remains the priority there.
    constexpr int64_t single_block_limit = 64 * 1024;
    if (n <= single_block_limit) {
        const int single_block_threads = n > 64 * 1024 ? 512 : block_size;
        norm2_single_block_kernel<InputT, AccT, InputT><<<1, single_block_threads, 0, stream>>>(
            n, static_cast<const InputT*>(self.data_ptr()),
            static_cast<InputT*>(result.data_ptr()));
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    constexpr int elements_per_thread = 8;
    const int64_t needed =
        (n + static_cast<int64_t>(block_size * elements_per_thread) - 1) /
        static_cast<int64_t>(block_size * elements_per_thread);
    // Keep the launch geometry out of the hot host path.  Querying
    // cudaGetDeviceProperties for every Muon step costs roughly 0.9 ms on
    // the target GPU, dwarfing the reduction itself.  256 CTAs is a safe
    // upper bound for this one-output reduction across the supported CUDA
    // devices; the grid-stride loop handles smaller tensors naturally.
    constexpr int64_t target = 256;
    const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed, target)));

    const DType accumulator_dtype =
        std::is_same_v<AccT, float> ? DType::Float32 : DType::Float64;
    Tensor accumulator = Tensor::empty({}, accumulator_dtype, self.device());
    CUDA_CHECK(cudaMemsetAsync(accumulator.data_ptr(), 0, sizeof(AccT), stream));
    norm2_atomic_kernel<InputT, AccT><<<blocks, block_size, 0, stream>>>(
        n, static_cast<const InputT*>(self.data_ptr()),
        static_cast<AccT*>(accumulator.data_ptr()));
    CUDA_CHECK(cudaGetLastError());
    norm2_finalize_scalar_kernel<AccT, InputT><<<1, 32, 0, stream>>>(
        static_cast<const AccT*>(accumulator.data_ptr()),
        static_cast<InputT*>(result.data_ptr()));
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor norm_global_kernel(const Tensor& self, double p) {
    if (isComplexType(self.dtype())) {
        Tensor areal = self.abs();
        return norm_global_kernel(areal, p);
    }
    // A transpose of a contiguous 2-D tensor is logically non-contiguous but
    // still occupies one dense storage span.  Since a global norm is
    // independent of element order, scan that span directly instead of
    // falling through to the generic strided reduction (or making a copy).
    const bool dense_storage =
        self.is_contiguous() ||
        (self.dim() == 2 && self.stride(0) == 1 &&
         self.stride(1) == self.size(0));
    if (p == 2.0 && dense_storage) {
        switch (self.dtype()) {
            case DType::Float32:
                return norm2_global_fast_typed<float, float>(self);
            case DType::Float64:
                return norm2_global_fast_typed<double, double>(self);
            case DType::Float16:
                return norm2_global_fast_typed<Half, float>(self);
            case DType::BFloat16:
                return norm2_global_fast_typed<BFloat16, float>(self);
            default:
                break;
        }
    }
    const ReductionSpec spec = make_reduction_spec(self, {});
    TP_DISPATCH_FLOAT_REDUCTION(norm_same_dtype, self.dtype(), self, spec, false, p);
}

Tensor norm_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, double p, bool keepdim) {
    if (isComplexType(self.dtype())) {
        Tensor areal = self.abs();
        return norm_dim_kernel(areal, dim, p, keepdim);
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_FLOAT_REDUCTION(norm_same_dtype, self.dtype(), self, spec, keepdim, p);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ReduceNormKernels) {
    m.impl("norm", norm_global_kernel);
    m.impl("norm.dim", norm_dim_kernel);
}

} // namespace cuda
} // namespace tensorplay
