#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include "TypePromotion.h"
#include "Utils.h"
#include "CUDARuntime.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cuda_runtime.h>
#include <thrust/complex.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace {

constexpr int kDistanceThreads = 256;
constexpr int kDistanceWarps = kDistanceThreads / 32;

#define DISTANCE_CUDA_CHECK(condition) \
    do { \
        const cudaError_t error = (condition); \
        if (error != cudaSuccess) { \
            TP_THROW(RuntimeError, "CUDA distance kernel failed: ", \
                     cudaGetErrorString(error)); \
        } \
    } while (0)

template <typename T>
__device__ __forceinline__ bool distance_isnan(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return ::isnan(value);
    } else {
        return ::isnan(value);
    }
}

template <typename T>
__device__ __forceinline__ T distance_abs(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return ::fabsf(value);
    } else {
        return ::fabs(value);
    }
}

template <typename T>
__device__ __forceinline__ T distance_pow(T value, T exponent) {
    if constexpr (std::is_same_v<T, float>) {
        return ::powf(value, exponent);
    } else {
        return ::pow(value, exponent);
    }
}

template <typename InputT, typename AccT>
__device__ __forceinline__ AccT distance_absdiff(
        InputT lhs, InputT rhs, AccT addend) {
    const InputT diff = static_cast<InputT>(
        static_cast<AccT>(lhs) - static_cast<AccT>(rhs) + addend);
    return distance_abs(static_cast<AccT>(diff));
}

template <typename Real>
__device__ __forceinline__ Real distance_absdiff(
        thrust::complex<Real> lhs, thrust::complex<Real> rhs, Real addend) {
    const thrust::complex<Real> diff =
        lhs - rhs + thrust::complex<Real>(addend, Real(0));
    return thrust::abs(diff);
}

template <typename T>
struct DistanceZero {
    static __device__ __forceinline__ void inc(T& acc, T value, T) {
        if (distance_isnan(value)) {
            acc = value;
        } else if (value != T(0)) {
            acc += T(1);
        }
    }

    static __device__ __forceinline__ T combine(T lhs, T rhs) {
        return lhs + rhs;
    }

    static __device__ __forceinline__ T finish(T value, T) {
        return value;
    }
};

template <typename T>
struct DistanceZeroCount {
    static __device__ __forceinline__ void inc(T& acc, T value, T) {
        if (value != T(0)) {
            acc += T(1);
        }
    }

    static __device__ __forceinline__ T combine(T lhs, T rhs) {
        return lhs + rhs;
    }

    static __device__ __forceinline__ T finish(T value, T) {
        return value;
    }
};

template <typename T>
struct DistanceOne {
    static __device__ __forceinline__ void inc(T& acc, T value, T) {
        acc += value;
    }

    static __device__ __forceinline__ T combine(T lhs, T rhs) {
        return lhs + rhs;
    }

    static __device__ __forceinline__ T finish(T value, T) {
        return value;
    }
};

template <typename T>
struct DistanceTwo {
    static __device__ __forceinline__ void inc(T& acc, T value, T) {
        acc += value * value;
    }

    static __device__ __forceinline__ T combine(T lhs, T rhs) {
        return lhs + rhs;
    }

    static __device__ __forceinline__ T finish(T value, T) {
        if constexpr (std::is_same_v<T, float>) {
            return ::sqrtf(value);
        } else {
            return ::sqrt(value);
        }
    }
};

template <typename T>
struct DistanceP {
    static __device__ __forceinline__ void inc(T& acc, T value, T p) {
        acc += distance_pow(value, p);
    }

    static __device__ __forceinline__ T combine(T lhs, T rhs) {
        return lhs + rhs;
    }

    static __device__ __forceinline__ T finish(T value, T p) {
        return distance_pow(value, T(1) / p);
    }
};

template <typename T>
struct DistanceMax {
    static __device__ __forceinline__ void inc(T& acc, T value, T) {
        acc = combine(acc, value);
    }

    static __device__ __forceinline__ T combine(T lhs, T rhs) {
        if (distance_isnan(lhs)) return lhs;
        if (distance_isnan(rhs)) return rhs;
        return lhs > rhs ? lhs : rhs;
    }

    static __device__ __forceinline__ T finish(T value, T) {
        return value;
    }
};

template <typename T>
struct DistanceMin {
    static __device__ __forceinline__ void inc(T& acc, T value, T) {
        acc = combine(acc, value);
    }

    static __device__ __forceinline__ T combine(T lhs, T rhs) {
        if (distance_isnan(lhs)) return lhs;
        if (distance_isnan(rhs)) return rhs;
        return lhs < rhs ? lhs : rhs;
    }

    static __device__ __forceinline__ T finish(T value, T) {
        return value;
    }
};

template <typename T>
struct DistanceMaxIgnoreNan {
    static __device__ __forceinline__ void inc(T& acc, T value, T) {
        if (value > acc) acc = value;
    }

    static __device__ __forceinline__ T combine(T lhs, T rhs) {
        return lhs > rhs ? lhs : rhs;
    }

    static __device__ __forceinline__ T finish(T value, T) {
        return value;
    }
};

template <typename AccT, typename Family>
__device__ __forceinline__ AccT distance_block_reduce(AccT value, AccT identity) {
    __shared__ AccT warp_values[kDistanceWarps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        value = Family::combine(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();

    if (warp == 0) {
        value = lane < kDistanceWarps ? warp_values[lane] : identity;
        for (int offset = kDistanceWarps / 2; offset > 0; offset >>= 1) {
            value = Family::combine(
                value, __shfl_down_sync(0xffffffffu, value, offset));
        }
    }
    return value;
}

template <typename InputT, typename AccT, typename OutputT, typename Family>
__global__ void distance_kernel(
        OutputT* output, const InputT* lhs, const InputT* rhs,
        int64_t rows, int64_t width, AccT p, AccT addend) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) return;

    const InputT* lhs_row = lhs + row * width;
    const InputT* rhs_row = rhs + row * width;
    AccT aggregate = std::is_same_v<Family, DistanceMin<AccT>>
        ? std::numeric_limits<AccT>::infinity()
        : AccT(0);
    for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        Family::inc(aggregate,
                    distance_absdiff(lhs_row[column], rhs_row[column], addend),
                    p);
    }

    const AccT identity = std::is_same_v<Family, DistanceMin<AccT>>
        ? std::numeric_limits<AccT>::infinity()
        : AccT(0);
    aggregate = distance_block_reduce<AccT, Family>(aggregate, identity);
    if (threadIdx.x == 0) {
        output[row] = static_cast<OutputT>(Family::finish(aggregate, p));
    }
}

template <typename T>
const T* distance_data_ptr(const Tensor& tensor) {
    return tensor.data_ptr<T>();
}

template <>
const thrust::complex<float>* distance_data_ptr<thrust::complex<float>>(
        const Tensor& tensor) {
    return reinterpret_cast<const thrust::complex<float>*>(
        tensor.data_ptr<std::complex<float>>());
}

template <>
const thrust::complex<double>* distance_data_ptr<thrust::complex<double>>(
        const Tensor& tensor) {
    return reinterpret_cast<const thrust::complex<double>*>(
        tensor.data_ptr<std::complex<double>>());
}

template <typename InputT, typename AccT, typename OutputT, typename Family>
void launch_distance_family(
        Tensor& output, const Tensor& lhs, const Tensor& rhs,
        int64_t rows, int64_t width, double p, double addend) {
    const dim3 grid(static_cast<unsigned>(rows));
    const dim3 block(kDistanceThreads);
    distance_kernel<InputT, AccT, OutputT, Family><<<
        grid, block, 0, getCurrentCUDAStream().stream()>>>(
        output.data_ptr<OutputT>(), distance_data_ptr<InputT>(lhs),
        distance_data_ptr<InputT>(rhs), rows, width,
        static_cast<AccT>(p), static_cast<AccT>(addend));
    DISTANCE_CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename AccT, typename OutputT>
void launch_distance(
        Tensor& output, const Tensor& lhs, const Tensor& rhs,
        int64_t rows, int64_t width, double p, double addend,
        bool propagate_max_nan) {
    if (p == 0.0) {
        launch_distance_family<InputT, AccT, OutputT, DistanceZeroCount<AccT>>(
            output, lhs, rhs, rows, width, p, addend);
    } else if (p == 1.0) {
        launch_distance_family<InputT, AccT, OutputT, DistanceOne<AccT>>(
            output, lhs, rhs, rows, width, p, addend);
    } else if (p == 2.0) {
        launch_distance_family<InputT, AccT, OutputT, DistanceTwo<AccT>>(
            output, lhs, rhs, rows, width, p, addend);
    } else if (p == std::numeric_limits<double>::infinity()) {
        if (propagate_max_nan) {
            launch_distance_family<InputT, AccT, OutputT, DistanceMax<AccT>>(
                output, lhs, rhs, rows, width, p, addend);
        } else {
            launch_distance_family<InputT, AccT, OutputT,
                                   DistanceMaxIgnoreNan<AccT>>(
                output, lhs, rhs, rows, width, p, addend);
        }
    } else if (p == -std::numeric_limits<double>::infinity()) {
        launch_distance_family<InputT, AccT, OutputT, DistanceMin<AccT>>(
            output, lhs, rhs, rows, width, p, addend);
    } else {
        launch_distance_family<InputT, AccT, OutputT, DistanceP<AccT>>(
            output, lhs, rhs, rows, width, p, addend);
    }
}

inline int64_t product_prefix(const std::vector<int64_t>& shape) {
    int64_t result = 1;
    for (int64_t index = 0; index + 1 < static_cast<int64_t>(shape.size()); ++index) {
        result *= shape[static_cast<size_t>(index)];
    }
    return result;
}

inline int64_t product_all(const std::vector<int64_t>& shape) {
    int64_t result = 1;
    for (int64_t extent : shape) result *= extent;
    return result;
}

inline bool empty_distance_has_no_identity(double p) {
    return p < 0.0 || p == std::numeric_limits<double>::infinity();
}

inline void check_empty_distance(double p) {
    if (empty_distance_has_no_identity(p)) {
        TP_THROW(RuntimeError,
                 "distance reduction cannot reduce an empty dimension for this order");
    }
}

template <typename InputT, typename AccT, typename OutputT>
void pairwise_distance_typed(
        Tensor& output, const Tensor& lhs, const Tensor& rhs,
        int64_t rows, int64_t width, double p, double eps) {
    launch_distance<InputT, AccT, OutputT>(
        output, lhs, rhs, rows, width, p, eps, true);
}

template <typename InputT, typename AccT, typename OutputT>
void dist_typed(
        Tensor& output, const Tensor& lhs, const Tensor& rhs,
        int64_t width, double p) {
    launch_distance<InputT, AccT, OutputT>(
        output, lhs, rhs, 1, width, p, 0.0, true);
}

template <typename InputT, typename AccT, typename OutputT, typename Family>
__global__ void pdist_kernel(
        OutputT* output, const InputT* input, int64_t n, int64_t width,
        AccT p, double n2, double n2_squared_minus_1) {
    const int64_t pair = static_cast<int64_t>(blockIdx.x);
    const int64_t i = static_cast<int64_t>(
        n2 - ::sqrt(n2_squared_minus_1 - 2.0 * static_cast<double>(pair)));
    const int64_t j = pair - n * i + i * (i + 1) / 2 + i + 1;

    const InputT* lhs = input + i * width;
    const InputT* rhs = input + j * width;
    AccT aggregate = AccT(0);
    for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        Family::inc(aggregate,
                    distance_absdiff(lhs[column], rhs[column], AccT(0)),
                    p);
    }

    const AccT identity = AccT(0);
    aggregate = distance_block_reduce<AccT, Family>(aggregate, identity);
    if (threadIdx.x == 0) {
        output[pair] = static_cast<OutputT>(Family::finish(aggregate, p));
    }
}

template <typename InputT, typename AccT, typename OutputT, typename Family>
void launch_pdist_family(
        Tensor& output, const Tensor& input, int64_t n, int64_t width, double p) {
    const dim3 grid(static_cast<unsigned>(output.numel()));
    const dim3 block(kDistanceThreads);
    const double n2 = static_cast<double>(n) - 0.5;
    const double n2_squared_minus_1 = n2 * n2 - 1.0;
    pdist_kernel<InputT, AccT, OutputT, Family><<<
        grid, block, 0, getCurrentCUDAStream().stream()>>>(
        output.data_ptr<OutputT>(), distance_data_ptr<InputT>(input),
        n, width, static_cast<AccT>(p), n2, n2_squared_minus_1);
    DISTANCE_CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename AccT, typename OutputT, typename Family>
__global__ void cdist_kernel(
        OutputT* output, const InputT* lhs, const InputT* rhs,
        int64_t batches, int64_t rows1, int64_t rows2, int64_t width,
        AccT p) {
    const int64_t pair_count = rows1 * rows2;
    const int64_t linear = static_cast<int64_t>(blockIdx.x);
    const int64_t batch = linear / pair_count;
    if (batch >= batches) return;
    const int64_t pair = linear % pair_count;
    const int64_t row1 = pair / rows2;
    const int64_t row2 = pair % rows2;

    const InputT* lhs_row = lhs + (batch * rows1 + row1) * width;
    const InputT* rhs_row = rhs + (batch * rows2 + row2) * width;
    AccT aggregate = std::is_same_v<Family, DistanceMin<AccT>>
        ? std::numeric_limits<AccT>::infinity()
        : AccT(0);
    for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        Family::inc(aggregate,
                    distance_absdiff(lhs_row[column], rhs_row[column], AccT(0)),
                    p);
    }

    const AccT identity = std::is_same_v<Family, DistanceMin<AccT>>
        ? std::numeric_limits<AccT>::infinity()
        : AccT(0);
    aggregate = distance_block_reduce<AccT, Family>(aggregate, identity);
    if (threadIdx.x == 0) {
        output[linear] = static_cast<OutputT>(Family::finish(aggregate, p));
    }
}

template <typename InputT, typename AccT, typename OutputT, typename Family>
void launch_cdist_family(
        Tensor& output, const Tensor& lhs, const Tensor& rhs,
        int64_t batches, int64_t rows1, int64_t rows2, int64_t width,
        double p) {
    const dim3 grid(static_cast<unsigned>(output.numel()));
    const dim3 block(kDistanceThreads);
    cdist_kernel<InputT, AccT, OutputT, Family><<<
        grid, block, 0, getCurrentCUDAStream().stream()>>>(
        output.data_ptr<OutputT>(), distance_data_ptr<InputT>(lhs),
        distance_data_ptr<InputT>(rhs), batches, rows1, rows2, width,
        static_cast<AccT>(p));
    DISTANCE_CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename AccT, typename OutputT>
void launch_pdist(
        Tensor& output, const Tensor& input, int64_t n, int64_t width, double p) {
    if (p == 0.0) {
        launch_pdist_family<InputT, AccT, OutputT, DistanceZeroCount<AccT>>(
            output, input, n, width, p);
    } else if (p == 1.0) {
        launch_pdist_family<InputT, AccT, OutputT, DistanceOne<AccT>>(
            output, input, n, width, p);
    } else if (p == 2.0) {
        launch_pdist_family<InputT, AccT, OutputT, DistanceTwo<AccT>>(
            output, input, n, width, p);
    } else if (p == std::numeric_limits<double>::infinity()) {
        launch_pdist_family<InputT, AccT, OutputT, DistanceMaxIgnoreNan<AccT>>(
            output, input, n, width, p);
    } else {
        launch_pdist_family<InputT, AccT, OutputT, DistanceP<AccT>>(
            output, input, n, width, p);
    }
}

template <typename InputT, typename AccT, typename OutputT>
void launch_cdist(
        Tensor& output, const Tensor& lhs, const Tensor& rhs,
        int64_t batches, int64_t rows1, int64_t rows2, int64_t width,
        double p) {
    if (p == 0.0) {
        launch_cdist_family<InputT, AccT, OutputT, DistanceZeroCount<AccT>>(
            output, lhs, rhs, batches, rows1, rows2, width, p);
    } else if (p == 1.0) {
        launch_cdist_family<InputT, AccT, OutputT, DistanceOne<AccT>>(
            output, lhs, rhs, batches, rows1, rows2, width, p);
    } else if (p == 2.0) {
        launch_cdist_family<InputT, AccT, OutputT, DistanceTwo<AccT>>(
            output, lhs, rhs, batches, rows1, rows2, width, p);
    } else if (p == std::numeric_limits<double>::infinity()) {
        launch_cdist_family<InputT, AccT, OutputT,
                           DistanceMaxIgnoreNan<AccT>>(
            output, lhs, rhs, batches, rows1, rows2, width, p);
    } else {
        launch_cdist_family<InputT, AccT, OutputT, DistanceP<AccT>>(
            output, lhs, rhs, batches, rows1, rows2, width, p);
    }
}

inline DType pairwise_compute_dtype(
        const Tensor& lhs, const Tensor& rhs, double eps) {
    DType dtype = promoteTypes(lhs.dtype(), rhs.dtype());
    if (!isFloatingOrComplexType(dtype)) {
        dtype = result_type(Scalar(eps), dtype);
    }
    if (!isFloatingOrComplexType(dtype)) dtype = DType::Float32;
    return dtype;
}

inline DType distance_output_dtype(DType dtype) {
    return isComplexType(dtype) ? toRealValueType(dtype) : dtype;
}

Tensor euclidean_distance_matmul(const Tensor& x1, const Tensor& x2) {
    Tensor x1_norm = x1.pow(Scalar(2)).sum(
        std::vector<int64_t>{-1}, true);
    Tensor x1_pad = Tensor::ones_like(x1_norm);
    Tensor x2_norm = x2.pow(Scalar(2)).sum(
        std::vector<int64_t>{-1}, true);
    Tensor x2_pad = Tensor::ones_like(x2_norm);
    Tensor x1_augmented = Tensor::cat(
        {x1.mul(Scalar(-2)), x1_norm, x1_pad}, -1);
    Tensor x2_augmented = Tensor::cat(
        {x2, x2_pad, x2_norm}, -1);
    Tensor result = tpx::ops::matmul(
        x1_augmented, x2_augmented.transpose(-2, -1));
    result.clamp_min_(Scalar(0));
    result.sqrt_();
    return result;
}

} // anonymous namespace

Tensor pairwise_distance_cuda(
        const Tensor& x1, const Tensor& x2, double p, double eps, bool keepdim) {
    const std::vector<int64_t> shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(x1.shape()),
        static_cast<std::vector<int64_t>>(x2.shape()));
    if (shape.empty()) {
        TP_THROW(RuntimeError,
                 "pairwise_distance: inputs must be at least 1-dimensional");
    }

    const DType compute_dtype = pairwise_compute_dtype(x1, x2, eps);
    const DType output_dtype = distance_output_dtype(compute_dtype);
    Tensor lhs = x1.to(compute_dtype).expand(shape).contiguous();
    Tensor rhs = x2.to(compute_dtype).expand(shape).contiguous();

    std::vector<int64_t> output_shape(shape.begin(), shape.end() - 1);
    if (keepdim) output_shape.push_back(1);
    Tensor output = Tensor::empty(output_shape, output_dtype, x1.device());
    const int64_t rows = product_prefix(shape);
    const int64_t width = shape.back();
    if (rows == 0) return output;
    if (width == 0) {
        check_empty_distance(p);
        return output.fill_(Scalar(0));
    }

    switch (compute_dtype) {
        case DType::Float32:
            pairwise_distance_typed<float, float, float>(
                output, lhs, rhs, rows, width, p, eps);
            break;
        case DType::Float64:
            pairwise_distance_typed<double, double, double>(
                output, lhs, rhs, rows, width, p, eps);
            break;
        case DType::Float16:
            pairwise_distance_typed<Half, float, Half>(
                output, lhs, rhs, rows, width, p, eps);
            break;
        case DType::BFloat16:
            pairwise_distance_typed<BFloat16, float, BFloat16>(
                output, lhs, rhs, rows, width, p, eps);
            break;
        case DType::ComplexFloat:
            pairwise_distance_typed<thrust::complex<float>, float, float>(
                output, lhs, rhs, rows, width, p, eps);
            break;
        case DType::ComplexDouble:
            pairwise_distance_typed<thrust::complex<double>, double, double>(
                output, lhs, rhs, rows, width, p, eps);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "pairwise_distance: unsupported CUDA dtype");
    }
    return output;
}

Tensor pdist_cuda(const Tensor& self, double p) {
    if (self.dim() != 2) {
        TP_THROW(RuntimeError, "pdist only supports 2D tensors, got: ", self.dim(), "D");
    }
    if (!isFloatingType(self.dtype())) {
        TP_THROW(TypeError, "pdist only supports floating-point dtypes");
    }
    if (!(p >= 0.0)) {
        TP_THROW(RuntimeError, "pdist only supports non-negative p values");
    }
    if (self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
        TP_THROW(NotImplementedError, "pdist: unsupported CUDA dtype");
    }

    const int64_t n = self.size(0);
    const int64_t width = self.size(1);
    const int64_t count = n * (n - 1) / 2;
    Tensor input = self.contiguous();
    Tensor output = Tensor::empty({count}, self.dtype(), self.device());
    if (count == 0) return output;
    if (width == 0) return output.fill_(Scalar(0));

    if (self.dtype() == DType::Float32) {
        launch_pdist<float, float, float>(output, input, n, width, p);
    } else {
        launch_pdist<double, double, double>(output, input, n, width, p);
    }
    return output;
}

Tensor dist_cuda(const Tensor& self, const Tensor& other, Scalar p_scalar) {
    return tpx::ops::norm(tpx::ops::sub(self, other), p_scalar.toDouble());
}

Tensor cdist_cuda(
        const Tensor& x1, const Tensor& x2, double p,
        std::optional<int64_t> compute_mode) {
    if (x1.dim() < 2) {
        TP_THROW(RuntimeError,
                 "cdist only supports at least 2D tensors, X1 got: ",
                 x1.dim(), "D");
    }
    if (x2.dim() < 2) {
        TP_THROW(RuntimeError,
                 "cdist only supports at least 2D tensors, X2 got: ",
                 x2.dim(), "D");
    }
    if (x1.size(-1) != x2.size(-1)) {
        TP_THROW(RuntimeError,
                 "X1 and X2 must have the same number of columns. X1: ",
                 x1.size(-1), " X2: ", x2.size(-1));
    }
    if (!isFloatingType(x1.dtype())) {
        TP_THROW(TypeError, "cdist only supports floating-point dtypes");
    }
    if (!isFloatingType(x2.dtype())) {
        TP_THROW(TypeError, "cdist only supports floating-point dtypes");
    }
    if (x1.dtype() != x2.dtype()) {
        TP_THROW(RuntimeError,
                 "expected scalar type ", toString(x1.dtype()),
                 " but found ", toString(x2.dtype()));
    }
    if (!(p >= 0.0)) {
        TP_THROW(RuntimeError, "cdist only supports non-negative p values");
    }
    const int64_t mode = compute_mode.value_or(0);
    if (mode < 0 || mode > 2) {
        TP_THROW(RuntimeError, "possible modes: 0, 1, 2, but was: ", mode);
    }

    const std::vector<int64_t> shape1 =
        static_cast<std::vector<int64_t>>(x1.shape());
    const std::vector<int64_t> shape2 =
        static_cast<std::vector<int64_t>>(x2.shape());
    const std::vector<int64_t> batch1(shape1.begin(), shape1.end() - 2);
    const std::vector<int64_t> batch2(shape2.begin(), shape2.end() - 2);
    const std::vector<int64_t> batch_shape = broadcast_shapes(batch1, batch2);
    const int64_t rows1 = x1.size(-2);
    const int64_t rows2 = x2.size(-2);
    const int64_t width = x1.size(-1);
    const int64_t batches = product_all(batch_shape);

    std::vector<int64_t> expanded1 = batch_shape;
    expanded1.push_back(rows1);
    expanded1.push_back(width);
    std::vector<int64_t> expanded2 = batch_shape;
    expanded2.push_back(rows2);
    expanded2.push_back(width);
    Tensor lhs = x1.expand(expanded1).contiguous();
    Tensor rhs = x2.expand(expanded2).contiguous();

    std::vector<int64_t> output_shape = batch_shape;
    output_shape.push_back(rows1);
    output_shape.push_back(rows2);
    Tensor output = Tensor::empty(output_shape, x1.dtype(), x1.device());
    if (rows1 == 0 || rows2 == 0 || batches == 0) return output;
    if (width == 0) return output.fill_(Scalar(0));

    if (p == 2.0 &&
        (mode == 1 || (mode == 0 && (rows1 > 25 || rows2 > 25)))) {
        return euclidean_distance_matmul(lhs, rhs);
    }

    switch (x1.dtype()) {
        case DType::Float32:
            launch_cdist<float, float, float>(
                output, lhs, rhs, batches, rows1, rows2, width, p);
            break;
        case DType::Float64:
            launch_cdist<double, double, double>(
                output, lhs, rhs, batches, rows1, rows2, width, p);
            break;
        default:
            TP_THROW(NotImplementedError, "cdist: unsupported CUDA dtype");
    }
    return output;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, DistanceKernels) {
    m.impl("cdist", cdist_cuda);
    m.impl("dist", dist_cuda);
    m.impl("pdist", pdist_cuda);
    m.impl("pairwise_distance", pairwise_distance_cuda);
}

} // namespace cuda
} // namespace tensorplay
