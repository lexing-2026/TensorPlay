#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Scalar.h"
#include "SortingRadixSelect.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>

namespace tensorplay {
namespace cuda {

namespace {

template <typename T>
__device__ __forceinline__ bool median_value_is_nan(T value) {
    if constexpr (std::is_same<T, float>::value ||
                  std::is_same<T, double>::value) {
        return ::isnan(value);
    } else if constexpr (std::is_same<T, Half>::value ||
                         std::is_same<T, BFloat16>::value) {
        return ::isnan(static_cast<float>(value));
    } else {
        return false;
    }
}

template <typename T>
__global__ void median_select_kernel(int64_t n, const T* input, T* output) {
    __shared__ uint64_t radix_smem[32];
    __shared__ unsigned long long nan_count;
    if (threadIdx.x == 0) nan_count = 0;
    __syncthreads();

    unsigned long long local_nan_count = 0;
    for (uint64_t i = static_cast<uint64_t>(threadIdx.x);
         i < static_cast<uint64_t>(n);
         i += static_cast<uint64_t>(blockDim.x)) {
        local_nan_count += median_value_is_nan(input[i]) ? 1 : 0;
    }
    if (local_nan_count != 0) atomicAdd(&nan_count, local_nan_count);
    __syncthreads();

    const uint64_t k = nan_count != 0
        ? static_cast<uint64_t>(n)
        : static_cast<uint64_t>((n - 1) / 2 + 1);
    T median = static_cast<T>(0);
    topk_detail::topk_radix_select<T, uint64_t>(
        input, k, false, static_cast<uint64_t>(n), 1, radix_smem, &median);
    if (threadIdx.x == 0) output[0] = median;
}

}

Tensor median_kernel(const Tensor& self) {
    Tensor flat = self.contiguous().reshape({-1});
    const int64_t n = flat.numel();
    if (n == 0) {
        Scalar fill(std::numeric_limits<double>::quiet_NaN());
        switch (self.dtype()) {
            case DType::Bool: fill = Scalar(true); break;
            case DType::UInt8:
            case DType::UInt16:
            case DType::UInt32:
            case DType::UInt64:
                fill = Scalar(int64_t(0));
                break;
            case DType::Int8:
                fill = Scalar(int64_t(std::numeric_limits<int8_t>::lowest()));
                break;
            case DType::Int16:
                fill = Scalar(int64_t(std::numeric_limits<int16_t>::lowest()));
                break;
            case DType::Int32:
                fill = Scalar(int64_t(std::numeric_limits<int32_t>::lowest()));
                break;
            case DType::Int64:
                fill = Scalar(std::numeric_limits<int64_t>::lowest());
                break;
            default: break;
        }
        return Tensor::full({}, fill, self.dtype(), self.device());
    }
    const bool selection_supported =
        isIntegralType(flat.dtype()) ||
        flat.dtype() == DType::Float16 || flat.dtype() == DType::BFloat16 ||
        flat.dtype() == DType::Float32 || flat.dtype() == DType::Float64;
    if (selection_supported) {
        Tensor result = Tensor::empty({}, flat.dtype(), flat.device());
        auto stream = getCurrentCUDAStream().stream();
        switch (flat.dtype()) {
#define TP_MEDIAN_SELECT_CASE(ctype, name_)                                      \
    case DType::name_:                                                          \
        median_select_kernel<ctype><<<1, 256, 0, stream>>>(                     \
            n, static_cast<const ctype*>(flat.data_ptr()),                      \
            static_cast<ctype*>(result.data_ptr()));                            \
        break;
            TP_MEDIAN_SELECT_CASE(uint8_t, UInt8)
            TP_MEDIAN_SELECT_CASE(int8_t, Int8)
            TP_MEDIAN_SELECT_CASE(int16_t, Int16)
            TP_MEDIAN_SELECT_CASE(int32_t, Int32)
            TP_MEDIAN_SELECT_CASE(int64_t, Int64)
            TP_MEDIAN_SELECT_CASE(uint16_t, UInt16)
            TP_MEDIAN_SELECT_CASE(uint32_t, UInt32)
            TP_MEDIAN_SELECT_CASE(uint64_t, UInt64)
            TP_MEDIAN_SELECT_CASE(Half, Float16)
            TP_MEDIAN_SELECT_CASE(BFloat16, BFloat16)
            TP_MEDIAN_SELECT_CASE(float, Float32)
            TP_MEDIAN_SELECT_CASE(double, Float64)
#undef TP_MEDIAN_SELECT_CASE
            default:
                TP_THROW(TypeError, "median: unsupported selection dtype");
        }
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error));
        }
        return result;
    }
    extern std::tuple<Tensor, Tensor> sort_cuda(const Tensor& self, int64_t dim,
                                                bool descending);
    Tensor sorted = std::get<0>(sort_cuda(flat, 0, false));
    return sorted.select(0, (n - 1) / 2);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, MedianKernels) {
    m.impl("median", median_kernel);
}

}
}
