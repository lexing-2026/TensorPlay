// CUDA cross-product kernels kept in a separate translation unit. The kernel
// uses a stride-aware three-component formula with a grid-stride traversal
// over the dimensions outside dim.

#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDAContext.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Utils.h"

#include <cuda_runtime.h>
#include <thrust/complex.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <mutex>
#include <optional>
#include <type_traits>
#include <vector>

namespace tensorplay::cuda {

namespace {

constexpr int kCrossMaxDims = 64;
constexpr int kCrossBlock = 256;

#define CUDA_CHECK(condition) \
    do { \
        const cudaError_t error = (condition); \
        if (error != cudaSuccess) { \
            TP_THROW(RuntimeError, "CUDA Error: ", cudaGetErrorString(error)); \
        } \
    } while (0)

struct CrossTensorInfo {
    int ndim = 0;
    int64_t sizes[kCrossMaxDims]{};
    int64_t strides[kCrossMaxDims]{};
};

CrossTensorInfo make_cross_info(const Tensor& tensor,
                                const std::vector<int64_t>& output_shape) {
    if (output_shape.size() > kCrossMaxDims) {
        TP_THROW(RuntimeError, "cross: tensor rank exceeds ", kCrossMaxDims,
                 " dimensions on CUDA");
    }
    if (tensor.dim() != static_cast<int64_t>(output_shape.size())) {
        TP_THROW(RuntimeError, "cross: input rank does not match output rank");
    }

    CrossTensorInfo info;
    info.ndim = static_cast<int>(output_shape.size());
    for (int64_t dim = 0; dim < info.ndim; ++dim) {
        info.sizes[dim] = tensor.size(dim);
        info.strides[dim] = tensor.size(dim) == 1 ? 0 : tensor.stride(dim);
    }
    return info;
}

int64_t wrap_cross_dim(int64_t dim, int64_t ndim) {
    if (ndim <= 0 || dim < -ndim || dim >= ndim) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

int64_t default_cross_dim(const std::optional<int64_t>& dimension,
                          const Tensor& input) {
    if (dimension.has_value()) {
        return *dimension;
    }

    static std::once_flag warning_once;
    std::call_once(warning_once, [] {
        TP_WARN("Using cross without specifying the dim arg is deprecated.\n",
                "Please either pass the dim explicitly or use linalg_cross.\n",
                "The default value of dim will change to agree with that of linalg.cross in a future release.");
    });
    for (int64_t dim = 0; dim < input.dim(); ++dim) {
        if (input.size(dim) == 3) {
            return dim;
        }
    }
    TP_THROW(RuntimeError, "no dimension of size 3 in input");
}

std::vector<int64_t> cross_output_shape(const Tensor& input,
                                        const Tensor& other,
                                        int64_t& dim) {
    if (input.dim() != other.dim()) {
        TP_THROW(RuntimeError,
                 "linalg.cross: inputs must have the same number of dimensions.");
    }
    if (input.dim() == 0) {
        TP_THROW(IndexError,
                 "Dimension out of range (expected to be in range of [-1, 0], but got -1)");
    }
    dim = wrap_cross_dim(dim, input.dim());
    if (input.size(dim) != 3 || other.size(dim) != 3) {
        TP_THROW(RuntimeError, "linalg.cross: inputs dimension ", dim,
                 " must have length 3. Got ", input.size(dim), " and ",
                 other.size(dim));
    }
    if (input.dtype() != other.dtype()) {
        TP_THROW(RuntimeError, "expected scalar type ", toString(input.dtype()),
                 " but found ", toString(other.dtype()));
    }
    if (input.device() != other.device()) {
        TP_THROW(DeviceMismatchError,
                 "Expected all tensors to be on the same device, but got ",
                 input.device().toString(), " and ", other.device().toString());
    }
    if (!input.device().is_cuda()) {
        TP_THROW(DeviceMismatchError,
                 "cross CUDA kernel received non-CUDA tensors");
    }
    if (input.dtype() == DType::Bool) {
        TP_THROW(NotImplementedError, "\"cross\" not implemented for 'Bool'");
    }
    if (input.dtype() == DType::ComplexHalf ||
        input.dtype() == DType::BComplex32) {
        TP_THROW(NotImplementedError, "\"cross\" not implemented for reduced complex dtype ",
                 toString(input.dtype()));
    }

    return broadcast_shapes(static_cast<std::vector<int64_t>>(input.shape()),
                             static_cast<std::vector<int64_t>>(other.shape()));
}

__device__ __forceinline__ int64_t cross_offset(
    int64_t row, const CrossTensorInfo& info, int64_t dim) {
    int64_t offset = 0;
    for (int d = info.ndim - 1; d >= 0; --d) {
        if (d == dim) {
            continue;
        }
        const int64_t coordinate = row % info.sizes[d];
        row /= info.sizes[d];
        offset += coordinate * info.strides[d];
    }
    return offset;
}

template <typename scalar_t>
__global__ void cross_kernel(int64_t rows, scalar_t* out,
                             CrossTensorInfo out_info,
                             const scalar_t* x1, CrossTensorInfo x1_info,
                             const scalar_t* x2, CrossTensorInfo x2_info,
                             int64_t dim) {
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows; row += step) {
        scalar_t* out_row = out + cross_offset(row, out_info, dim);
        const scalar_t* x1_row = x1 + cross_offset(row, x1_info, dim);
        const scalar_t* x2_row = x2 + cross_offset(row, x2_info, dim);
        const int64_t ostride = out_info.strides[dim];
        const int64_t x1stride = x1_info.strides[dim];
        const int64_t x2stride = x2_info.strides[dim];

        const scalar_t val0 =
            x1_row[1 * x1stride] * x2_row[2 * x2stride] -
            x1_row[2 * x1stride] * x2_row[1 * x2stride];
        const scalar_t val1 =
            x1_row[2 * x1stride] * x2_row[0 * x2stride] -
            x1_row[0 * x1stride] * x2_row[2 * x2stride];
        const scalar_t val2 =
            x1_row[0 * x1stride] * x2_row[1 * x2stride] -
            x1_row[1 * x1stride] * x2_row[0 * x2stride];

        out_row[0 * ostride] = val0;
        out_row[1 * ostride] = val1;
        out_row[2 * ostride] = val2;
    }
}

template <typename scalar_t>
void launch_cross(const Tensor& result, const Tensor& a, const Tensor& b,
                  int64_t dim) {
    const int64_t rows = result.numel() / 3;
    if (rows == 0) {
        return;
    }
    const CrossTensorInfo result_info =
        make_cross_info(result, static_cast<std::vector<int64_t>>(result.shape()));
    const CrossTensorInfo a_info =
        make_cross_info(a, static_cast<std::vector<int64_t>>(result.shape()));
    const CrossTensorInfo b_info =
        make_cross_info(b, static_cast<std::vector<int64_t>>(result.shape()));
    const int64_t blocks64 = (rows + kCrossBlock - 1) / kCrossBlock;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<int64_t>(blocks64, 4096));
    cudaStream_t stream = getCurrentCUDAStream().stream();
    cross_kernel<scalar_t><<<blocks, kCrossBlock, 0, stream>>>(
        rows, result.data_ptr<scalar_t>(), result_info,
        a.data_ptr<scalar_t>(), a_info, b.data_ptr<scalar_t>(), b_info, dim);
    CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t>
void launch_cross_complex(const Tensor& result, const Tensor& a,
                          const Tensor& b, int64_t dim) {
    const int64_t rows = result.numel() / 3;
    if (rows == 0) {
        return;
    }
    const CrossTensorInfo result_info =
        make_cross_info(result, static_cast<std::vector<int64_t>>(result.shape()));
    const CrossTensorInfo a_info =
        make_cross_info(a, static_cast<std::vector<int64_t>>(result.shape()));
    const CrossTensorInfo b_info =
        make_cross_info(b, static_cast<std::vector<int64_t>>(result.shape()));
    const unsigned int blocks = static_cast<unsigned int>(std::min<int64_t>(
        (rows + kCrossBlock - 1) / kCrossBlock, 4096));
    cudaStream_t stream = getCurrentCUDAStream().stream();
    using device_complex = thrust::complex<scalar_t>;
    cross_kernel<device_complex><<<blocks, kCrossBlock, 0, stream>>>(
        rows,
        reinterpret_cast<device_complex*>(result.data_ptr<std::complex<scalar_t>>()),
        result_info,
        reinterpret_cast<const device_complex*>(a.data_ptr<std::complex<scalar_t>>()),
        a_info,
        reinterpret_cast<const device_complex*>(b.data_ptr<std::complex<scalar_t>>()),
        b_info, dim);
    CUDA_CHECK(cudaGetLastError());
}

Tensor cross_impl(const Tensor& input, const Tensor& other, int64_t dim) {
    const std::vector<int64_t> out_shape =
        cross_output_shape(input, other, dim);
    Tensor a = input.expand(out_shape);
    Tensor b = other.expand(out_shape);
    Tensor result = Tensor::empty(out_shape, input.dtype(), input.device());
    if (result.numel() == 0) {
        return result;
    }

    switch (input.dtype()) {
#define TP_CROSS_REAL_CASE(ctype, name) \
        case DType::name: \
            if constexpr (std::is_same_v<ctype, bool>) { \
                TP_THROW(NotImplementedError, "\"cross\" not implemented for 'Bool'"); \
            } else { \
                launch_cross<ctype>(result, a, b, dim); \
            } \
            break;
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_CROSS_REAL_CASE)
#undef TP_CROSS_REAL_CASE
        case DType::ComplexFloat:
            launch_cross_complex<float>(result, a, b, dim);
            break;
        case DType::ComplexDouble:
            launch_cross_complex<double>(result, a, b, dim);
            break;
        case DType::ComplexHalf:
        case DType::BComplex32:
            TP_THROW(NotImplementedError,
                     "cross: reduced complex dtypes are not supported");
        default:
            TP_THROW(NotImplementedError, "cross is not implemented for dtype ",
                     toString(input.dtype()));
    }
    return result;
}

} // namespace

Tensor linalg_cross_cuda(const Tensor& input, const Tensor& other, int64_t dim) {
    return cross_impl(input, other, dim);
}

Tensor cross_cuda(const Tensor& input, const Tensor& other,
                  std::optional<int64_t> dimension) {
    return cross_impl(input, other, default_cross_dim(dimension, input));
}

TENSORPLAY_LIBRARY_IMPL(CUDA, NativeCross) {
    m.impl("cross", cross_cuda);
    m.impl("linalg_cross", linalg_cross_cuda);
}

#undef CUDA_CHECK

} // namespace tensorplay::cuda
