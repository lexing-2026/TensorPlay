// Tensor comparison operators - CUDA kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "Exception.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

namespace {

constexpr int kThreads = 256;

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

void launch_ew(dim3& grid, dim3& block, int64_t n) {
    block = dim3(kThreads);
    grid = dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads));
}


template <typename T, typename Pred>
__global__ void ew_bool_unary_kernel(int64_t n, const T* a, bool* out, Pred pred) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = pred(a[i]);
}


template <typename Pred>
Tensor bool_unary_cuda(const Tensor& self, Pred pred, const char* name) {
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(self), DType::Bool, self.device());
    int64_t n = self.numel();
    if (n == 0) return out;
    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
#define TP_BU(ctype, name_) \
    case DType::name_: \
        ew_bool_unary_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, sc.data_ptr<ctype>(), out.data_ptr<bool>(), pred); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BU)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BU
    CUDA_CHECK(cudaGetLastError());
    return out;
}


__global__ void isclose_kernel(int64_t n, const double* ap, const double* bp,
                               bool* dp, double rtol, double atol, bool equal_nan) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        double x = ap[i], y = bp[i];
        bool close;
        if (x != x || y != y) {
            close = equal_nan && x != x && y != y;
        } else if (::isinf(x) || ::isinf(y)) {
            close = x == y;
        } else {
            double tol = atol + rtol * ::fabs(y);
            close = ::fabs(x - y) <= tol;
        }
        dp[i] = close;
    }
}

} // namespace

Tensor isclose_cuda(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
    std::vector<int64_t> out_shape = broadcast_shapes(shape_of(self), shape_of(other));
    Tensor a = self.to(DType::Float64).expand(out_shape).contiguous();
    Tensor b = other.to(DType::Float64).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, DType::Bool, self.device());
    int64_t n = out.numel();
    if (n == 0) return out;
    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
    isclose_kernel<<<grid, block, 0, stream>>>(
        n, a.data_ptr<double>(), b.data_ptr<double>(), out.data_ptr<bool>(),
        rtol, atol, equal_nan);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor isreal_cuda(const Tensor& self) {
    // Real dtypes are trivially real; complex tests imag==0.
    if (!isComplexType(self.dtype())) {
        return Tensor::ones(shape_of(self), DType::Bool, self.device());
    }
    return bool_unary_cuda(self, [] __device__ (auto x) -> bool {
        using T = decltype(x);
        if constexpr (std::is_same_v<T, std::complex<float>>) return x.imag() == 0.0f;
        else if constexpr (std::is_same_v<T, std::complex<double>>) return x.imag() == 0.0;
        else return true;
    }, "isreal");
}


TENSORPLAY_LIBRARY_IMPL(CUDA, TensorCompareKernels) {
    m.impl("isclose", isclose_cuda);
    m.impl("isreal", isreal_cuda);
}

} // namespace cuda
} // namespace tensorplay
