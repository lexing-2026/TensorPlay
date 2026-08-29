#pragma once

// foreach CUDA kernels fuse the tensor list horizontally: one launch walks
// chunks from many tensors instead of dispatching one elementwise kernel per
// Tensor.  This header keeps the same chunked metadata model while using
// TensorPlay's existing CUDA stream and scalar types.

#include "CUDARuntime.h"
#include "DType.h"
#include "Exception.h"
#include "Tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace foreach_mta {

// Keep the metadata conservative enough for CUDA's pre-13 kernel argument
// ILP=4 loop below: a 512-thread block covers a 64K chunk in 32 iterations
// instead of 256 scalar iterations with the old 256-thread kernel.
constexpr int32_t kMaxTensorsPerLaunch = 32;
constexpr int32_t kMaxBlocksPerLaunch = 320;
constexpr int64_t kChunkSize = 65536;
constexpr int32_t kILP = 4;
constexpr int32_t kBlockSize = 512;

template <typename T>
struct opmath_type {
    using type = T;
};
template <>
struct opmath_type<Half> {
    using type = float;
};
template <>
struct opmath_type<BFloat16> {
    using type = float;
};

template <typename T>
using opmath_t = typename opmath_type<T>::type;

template <typename T>
struct alignas(kILP * sizeof(T)) AlignedVec {
    T values[kILP];
};

template <typename T>
__device__ __forceinline__ bool is_aligned(const T* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) &
            ((kILP * sizeof(T)) - 1)) == 0;
}

template <typename T>
__device__ __forceinline__ void load_store(
        T* dst, const T* src, int64_t dst_offset, int64_t src_offset) {
    using Vec = AlignedVec<T>;
    reinterpret_cast<Vec*>(dst)[dst_offset] =
        reinterpret_cast<const Vec*>(src)[src_offset];
}

inline bool supported_dtype(DType dtype) {
    return dtype == DType::Float16 || dtype == DType::BFloat16 ||
           dtype == DType::Float32 || dtype == DType::Float64;
}

template <typename F>
bool dispatch_dtype(DType dtype, F&& fn) {
    switch (dtype) {
        case DType::Float16:
            fn.template operator()<Half, float>();
            return true;
        case DType::BFloat16:
            fn.template operator()<BFloat16, float>();
            return true;
        case DType::Float32:
            fn.template operator()<float, float>();
            return true;
        case DType::Float64:
            fn.template operator()<double, double>();
            return true;
        default:
            return false;
    }
}

inline bool eligible_list(const std::vector<Tensor>& values) {
    if (values.empty()) return true;
    const Tensor& first = values.front();
    if (!first.defined() || first.is_sparse() || !first.is_contiguous() ||
        !supported_dtype(first.dtype())) {
        return false;
    }
    for (const Tensor& value : values) {
        if (!value.defined() || value.is_sparse() || !value.is_contiguous() ||
            value.dtype() != first.dtype() || value.device() != first.device()) {
            return false;
        }
    }
    return true;
}

inline bool eligible_pair(const std::vector<Tensor>& lhs,
                          const std::vector<Tensor>& rhs) {
    if (lhs.size() != rhs.size() || !eligible_list(lhs)) return false;
    if (lhs.empty()) return true;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!rhs[i].defined() || rhs[i].is_sparse() ||
            !rhs[i].is_contiguous() || rhs[i].dtype() != lhs[0].dtype() ||
            rhs[i].device() != lhs[0].device() || rhs[i].shape() != lhs[i].shape()) {
            return false;
        }
    }
    return true;
}

inline bool eligible_ternary(const std::vector<Tensor>& first,
                             const std::vector<Tensor>& second,
                             const std::vector<Tensor>& third) {
    return eligible_pair(first, second) && eligible_pair(first, third);
}

template <int Depth>
struct TensorListMetadata {
    const void* addresses[Depth][kMaxTensorsPerLaunch]{};
    int64_t numel_for_tensor[kMaxTensorsPerLaunch]{};
    int32_t block_to_tensor[kMaxBlocksPerLaunch]{};
    int32_t block_to_chunk[kMaxBlocksPerLaunch]{};
    // Used only by ScalarList overloads.  Keeping it in the same metadata
    float scalar_values_float[kMaxTensorsPerLaunch]{};
    double scalar_values_double[kMaxTensorsPerLaunch]{};
    int32_t scalar_value_kind = 0;
};

template <int Depth, int OutputIndex, typename T, typename M, typename Op>
__global__ void multi_tensor_kernel(TensorListMetadata<Depth> metadata, Op op) {
    const int32_t block_index = static_cast<int32_t>(blockIdx.x);
    const int32_t tensor_index = metadata.block_to_tensor[block_index];
    const int64_t begin = static_cast<int64_t>(metadata.block_to_chunk[block_index]) *
                          kChunkSize;
    const int64_t end = metadata.numel_for_tensor[tensor_index];

    const T* inputs[Depth];
    T* output = const_cast<T*>(static_cast<const T*>(
        metadata.addresses[OutputIndex][tensor_index]));
#pragma unroll
    for (int depth = 0; depth < Depth; ++depth) {
        inputs[depth] = static_cast<const T*>(
            metadata.addresses[depth][tensor_index]);
    }

    const int64_t count = end - begin;
    bool aligned = (count % kILP == 0) && (kChunkSize % kILP == 0);
#pragma unroll
    for (int depth = 0; depth < Depth; ++depth) {
        aligned = aligned && is_aligned(inputs[depth] + begin);
    }
    aligned = aligned && is_aligned(output + begin);

    if (aligned) {
        // four scalar elements, so each thread performs four contiguous
        // loads/stores and the block needs only 32 loop rounds per chunk.
        alignas(kILP * sizeof(T)) T packed[Depth][kILP];
        for (int64_t vector_index = static_cast<int64_t>(threadIdx.x);
             vector_index * kILP < count &&
                 vector_index * kILP < kChunkSize;
             vector_index += static_cast<int64_t>(blockDim.x)) {
#pragma unroll
            for (int depth = 0; depth < Depth; ++depth) {
                load_store(packed[depth], inputs[depth] + begin,
                           0, vector_index);
            }
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) {
                M values[Depth];
#pragma unroll
                for (int depth = 0; depth < Depth; ++depth) {
                    values[depth] = static_cast<M>(packed[depth][lane]);
                }
                packed[OutputIndex][lane] = static_cast<T>(op(values));
            }
            load_store(output + begin, packed[OutputIndex],
                       vector_index, 0);
        }
    } else {
        for (int64_t i_start = 0; i_start < count && i_start < kChunkSize;
             i_start += static_cast<int64_t>(blockDim.x) * kILP) {
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) {
                const int64_t index = i_start + threadIdx.x +
                                      static_cast<int64_t>(lane) * blockDim.x;
                if (index >= count) continue;
                M values[Depth];
#pragma unroll
                for (int depth = 0; depth < Depth; ++depth) {
                    values[depth] = static_cast<M>(inputs[depth][begin + index]);
                }
                output[begin + index] = static_cast<T>(op(values));
            }
        }
    }
}

template <int Depth, int OutputIndex, typename T, typename M, typename Op>
__global__ void multi_tensor_scalar_list_kernel(
        TensorListMetadata<Depth> metadata, Op op) {
    const int32_t block_index = static_cast<int32_t>(blockIdx.x);
    const int32_t tensor_index = metadata.block_to_tensor[block_index];
    const int64_t begin = static_cast<int64_t>(metadata.block_to_chunk[block_index]) *
                          kChunkSize;
    const int64_t end = metadata.numel_for_tensor[tensor_index];
    const M scalar = metadata.scalar_value_kind == 1
        ? static_cast<M>(metadata.scalar_values_float[tensor_index])
        : static_cast<M>(metadata.scalar_values_double[tensor_index]);

    const T* inputs[Depth];
    T* output = const_cast<T*>(static_cast<const T*>(
        metadata.addresses[OutputIndex][tensor_index]));
#pragma unroll
    for (int depth = 0; depth < Depth; ++depth) {
        inputs[depth] = static_cast<const T*>(
            metadata.addresses[depth][tensor_index]);
    }

    const int64_t count = end - begin;
    bool aligned = (count % kILP == 0) && (kChunkSize % kILP == 0);
#pragma unroll
    for (int depth = 0; depth < Depth; ++depth) {
        aligned = aligned && is_aligned(inputs[depth] + begin);
    }
    aligned = aligned && is_aligned(output + begin);

    if (aligned) {
        alignas(kILP * sizeof(T)) T packed[Depth][kILP];
        for (int64_t vector_index = static_cast<int64_t>(threadIdx.x);
             vector_index * kILP < count &&
                 vector_index * kILP < kChunkSize;
             vector_index += static_cast<int64_t>(blockDim.x)) {
#pragma unroll
            for (int depth = 0; depth < Depth; ++depth) {
                load_store(packed[depth], inputs[depth] + begin,
                           0, vector_index);
            }
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) {
                M values[Depth];
#pragma unroll
                for (int depth = 0; depth < Depth; ++depth) {
                    values[depth] = static_cast<M>(packed[depth][lane]);
                }
                packed[OutputIndex][lane] =
                    static_cast<T>(op(values, scalar));
            }
            load_store(output + begin, packed[OutputIndex],
                       vector_index, 0);
        }
    } else {
        for (int64_t i_start = 0; i_start < count && i_start < kChunkSize;
             i_start += static_cast<int64_t>(blockDim.x) * kILP) {
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) {
                const int64_t index = i_start + threadIdx.x +
                                      static_cast<int64_t>(lane) * blockDim.x;
                if (index >= count) continue;
                M values[Depth];
#pragma unroll
                for (int depth = 0; depth < Depth; ++depth) {
                    values[depth] = static_cast<M>(inputs[depth][begin + index]);
                }
                output[begin + index] = static_cast<T>(op(values, scalar));
            }
        }
    }
}

template <bool HasScalarList, int Depth, int OutputIndex,
          typename T, typename M, typename Op>
void launch_impl(const std::array<const std::vector<Tensor>*, Depth>& lists,
                const std::vector<M>* scalar_values,
                Op op, const char* op_name) {
    const size_t tensor_count = lists[0]->size();
    for (size_t depth = 1; depth < Depth; ++depth) {
        if (lists[depth]->size() != tensor_count) {
            TP_THROW(ValueError, std::string(op_name) +
                ": tensor list arguments must have the same length");
        }
    }
    if (tensor_count == 0) return;
    if constexpr (HasScalarList) {
        if (scalar_values == nullptr || scalar_values->size() != tensor_count) {
            TP_THROW(ValueError, std::string(op_name) +
                ": scalar list must have the same length as the tensor list");
        }
    }

    size_t tensor_index = 0;
    int32_t next_chunk = 0;
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    while (tensor_index < tensor_count) {
        TensorListMetadata<Depth> metadata{};
        int32_t tensor_slots = 0;
        int32_t block_count = 0;

        while (tensor_index < tensor_count) {
            const Tensor& first = (*lists[0])[tensor_index];
            const int64_t numel = first.numel();
            if (numel == 0) {
                ++tensor_index;
                next_chunk = 0;
                continue;
            }
            const int32_t chunks = static_cast<int32_t>(
                (numel + kChunkSize - 1) / kChunkSize);
            if (tensor_slots == 0 || next_chunk == 0) {
                if (tensor_slots == kMaxTensorsPerLaunch ||
                    block_count == kMaxBlocksPerLaunch) {
                    break;
                }
                for (int depth = 0; depth < Depth; ++depth) {
                    metadata.addresses[depth][tensor_slots] =
                        (*lists[depth])[tensor_index].template data_ptr<T>();
                }
                metadata.numel_for_tensor[tensor_slots] = numel;
                if constexpr (HasScalarList) {
                    if constexpr (std::is_same_v<M, float>) {
                        metadata.scalar_values_float[tensor_slots] =
                            (*scalar_values)[tensor_index];
                        metadata.scalar_value_kind = 1;
                    } else {
                        metadata.scalar_values_double[tensor_slots] =
                            (*scalar_values)[tensor_index];
                        metadata.scalar_value_kind = 2;
                    }
                }
                ++tensor_slots;
            }

            const int32_t available = kMaxBlocksPerLaunch - block_count;
            const int32_t remaining = chunks - next_chunk;
            const int32_t take = available < remaining ? available : remaining;
            for (int32_t chunk = 0; chunk < take; ++chunk) {
                metadata.block_to_tensor[block_count + chunk] = tensor_slots - 1;
                metadata.block_to_chunk[block_count + chunk] = next_chunk + chunk;
            }
            block_count += take;
            next_chunk += take;
            if (next_chunk == chunks) {
                ++tensor_index;
                next_chunk = 0;
            } else {
                break;
            }
        }

        if (block_count == 0) {
            TP_THROW(ValueError, std::string(op_name) +
                ": unable to construct a CUDA foreach launch");
        }
        if constexpr (HasScalarList) {
            multi_tensor_scalar_list_kernel<Depth, OutputIndex, T, M, Op><<<
                static_cast<unsigned int>(block_count), kBlockSize, 0, stream>>>(
                    metadata, op);
        } else {
            multi_tensor_kernel<Depth, OutputIndex, T, M, Op><<<
                static_cast<unsigned int>(block_count), kBlockSize, 0, stream>>>(
                    metadata, op);
        }
        checkCuda(cudaGetLastError(), op_name);
    }
}

template <int Depth, int OutputIndex, typename T, typename M, typename Op>
void launch(const std::array<const std::vector<Tensor>*, Depth>& lists,
            Op op, const char* op_name) {
    launch_impl<false, Depth, OutputIndex, T, M, Op>(
        lists, nullptr, op, op_name);
}

template <int Depth, int OutputIndex, typename T, typename M, typename Op>
void launch_scalar_list(
        const std::array<const std::vector<Tensor>*, Depth>& lists,
        const std::vector<M>& scalar_values, Op op, const char* op_name) {
    launch_impl<true, Depth, OutputIndex, T, M, Op>(
        lists, &scalar_values, op, op_name);
}

template <typename M>
struct UnarySqrt {
    __device__ M operator()(M* values) const { return sqrt(values[0]); }
};
template <typename M>
struct UnaryRsqrt {
    __device__ M operator()(M* values) const { return M(1) / sqrt(values[0]); }
};
template <typename M>
struct UnaryNeg {
    __device__ M operator()(M* values) const { return -values[0]; }
};
template <typename M>
struct UnaryAbs {
    __device__ M operator()(M* values) const {
        return values[0] < M(0) ? -values[0] : values[0];
    }
};
template <typename M>
struct UnarySign {
    __device__ M operator()(M* values) const {
        return values[0] > M(0) ? M(1) : (values[0] < M(0) ? M(-1) : M(0));
    }
};
template <typename M>
struct UnaryReciprocal {
    __device__ M operator()(M* values) const { return M(1) / values[0]; }
};
template <typename M>
struct UnaryZero {
    __device__ M operator()(M*) const { return M(0); }
};

template <typename M>
struct BinaryAddScalar {
    M scalar;
    __device__ M operator()(M* values) const { return values[0] + scalar; }
};
template <typename M>
struct BinarySubScalar {
    M scalar;
    __device__ M operator()(M* values) const { return values[0] - scalar; }
};
template <typename M>
struct BinaryMulScalar {
    M scalar;
    __device__ M operator()(M* values) const { return values[0] * scalar; }
};
template <typename M>
struct BinaryDivScalar {
    M scalar;
    __device__ M operator()(M* values) const { return values[0] / scalar; }
};
template <typename M>
struct BinaryAddList {
    M alpha;
    __device__ M operator()(M* values) const { return values[0] + alpha * values[1]; }
};
template <typename M>
struct BinarySubList {
    M alpha;
    __device__ M operator()(M* values) const { return values[0] - alpha * values[1]; }
};
template <typename M>
struct BinaryMulList {
    __device__ M operator()(M* values) const { return values[0] * values[1]; }
};
template <typename M>
struct BinaryDivList {
    __device__ M operator()(M* values) const { return values[0] / values[1]; }
};
template <typename M>
struct TernaryAddcmul {
    M value;
    __device__ M operator()(M* values) const {
        if (value == M(1)) {
            return fma(values[1], values[2], values[0]);
        }
        return fma(value, values[1] * values[2], values[0]);
    }
};
template <typename M>
struct TernaryAddcdiv {
    M value;
    __device__ M operator()(M* values) const {
        const M quotient = values[1] / values[2];
        if (value == M(1)) {
            return values[0] + quotient;
        }
        return fma(value, quotient, values[0]);
    }
};
template <typename M>
struct BinaryLerp {
    M weight;
    __device__ M operator()(M* values) const {
        // weights subtract from the end value to avoid cancellation.
        return (weight > M(-0.5) && weight < M(0.5))
            ? values[0] + weight * (values[1] - values[0])
            : values[1] - (values[1] - values[0]) * (M(1) - weight);
    }
};
template <typename M>
struct BinaryMaximum {
    M scalar;
    __device__ M operator()(M* values) const {
        return values[0] > scalar ? values[0] : scalar;
    }
};
template <typename M>
struct BinaryMinimum {
    M scalar;
    __device__ M operator()(M* values) const {
        return values[0] < scalar ? values[0] : scalar;
    }
};
template <typename M>
struct BinaryMaximumList {
    __device__ M operator()(M* values) const {
        return values[0] > values[1] ? values[0] : values[1];
    }
};
template <typename M>
struct BinaryMinimumList {
    __device__ M operator()(M* values) const {
        return values[0] < values[1] ? values[0] : values[1];
    }
};
template <typename M>
struct BinaryMaximumScalarList {
    __device__ M operator()(M* values, M scalar) const {
        return values[0] > scalar ? values[0] : scalar;
    }
};
template <typename M>
struct BinaryMinimumScalarList {
    __device__ M operator()(M* values, M scalar) const {
        return values[0] < scalar ? values[0] : scalar;
    }
};
template <typename M>
struct UnaryPow {
    M exponent;
    __device__ M operator()(M* values) const { return pow(values[0], exponent); }
};
template <typename M>
struct UnaryPowScalarList {
    __device__ M operator()(M* values, M exponent) const {
        return pow(values[0], exponent);
    }
};
template <typename M>
struct BinaryPowList {
    __device__ M operator()(M* values) const {
        return pow(values[0], values[1]);
    }
};

#define TP_FOREACH_UNARY_MATH_FUNCTOR(NAME, BODY) \
template <typename M> \
struct NAME { \
    __device__ M operator()(M* values) const { return (BODY); } \
};

TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryExp, exp(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryExpm1, expm1(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryErf, erf(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryErfc, erfc(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryCeil, ceil(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryFloor, floor(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryFrac, values[0] - trunc(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryLgamma, lgamma(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryLog, log(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryLog10, log10(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryLog1p, log1p(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryLog2, log2(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryRound, round(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnarySin, sin(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnarySinh, sinh(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryCos, cos(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryCosh, cosh(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryTan, tan(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryTanh, tanh(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(
    UnarySigmoid, M(1) / (M(1) + exp(-values[0])))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryAcos, acos(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryAsin, asin(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryAtan, atan(values[0]))
TP_FOREACH_UNARY_MATH_FUNCTOR(UnaryTrunc, trunc(values[0]))

#undef TP_FOREACH_UNARY_MATH_FUNCTOR

template <typename T, typename M>
struct BinaryAddTensor {
    const T* scalar = nullptr;
    M alpha = M(1);
    __device__ M operator()(M* values) const {
        return values[0] + alpha * static_cast<M>(*scalar);
    }
};
template <typename T, typename M>
struct BinarySubTensor {
    const T* scalar = nullptr;
    M alpha = M(1);
    __device__ M operator()(M* values) const {
        return values[0] - alpha * static_cast<M>(*scalar);
    }
};
template <typename T, typename M>
struct BinaryMulTensor {
    const T* scalar = nullptr;
    __device__ M operator()(M* values) const {
        return values[0] * static_cast<M>(*scalar);
    }
};
template <typename T, typename M>
struct BinaryDivTensor {
    const T* scalar = nullptr;
    __device__ M operator()(M* values) const {
        return values[0] / static_cast<M>(*scalar);
    }
};

template <typename M>
struct BinaryAddScalarList {
    __device__ M operator()(M* values, M scalar) const {
        return values[0] + scalar;
    }
};
template <typename M>
struct BinarySubScalarList {
    __device__ M operator()(M* values, M scalar) const {
        return values[0] - scalar;
    }
};
template <typename M>
struct BinaryMulScalarList {
    __device__ M operator()(M* values, M scalar) const {
        return values[0] * scalar;
    }
};
template <typename M>
struct BinaryDivScalarList {
    __device__ M operator()(M* values, M scalar) const {
        return values[0] / scalar;
    }
};
template <typename M>
struct TernaryAddcmulScalarList {
    __device__ M operator()(M* values, M scalar) const {
        if (scalar == M(1)) {
            return fma(values[1], values[2], values[0]);
        }
        return fma(scalar, values[1] * values[2], values[0]);
    }
};
template <typename M>
struct TernaryAddcdivScalarList {
    __device__ M operator()(M* values, M scalar) const {
        const M quotient = values[1] / values[2];
        if (scalar == M(1)) {
            return values[0] + quotient;
        }
        return fma(scalar, quotient, values[0]);
    }
};
template <typename M>
struct BinaryLerpScalarList {
    __device__ M operator()(M* values, M scalar) const {
        return (scalar > M(-0.5) && scalar < M(0.5))
            ? values[0] + scalar * (values[1] - values[0])
            : values[1] - (values[1] - values[0]) * (M(1) - scalar);
    }
};

} // namespace foreach_mta
} // namespace cuda
} // namespace tensorplay
