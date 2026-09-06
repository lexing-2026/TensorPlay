// Tensor comparison operators - CUDA kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "Exception.h"
#include "CUDARuntime.h"
#include "CUDALoops.cuh"

#include <cuda_runtime.h>
#include <thrust/complex.h>

#include <algorithm>
#include <cassert>
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

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

constexpr size_t kAssertMessageCapacity = 256;

struct AssertMessage {
    char text[kAssertMessageCapacity];
};

template <typename T>
__global__ void assert_async_kernel_impl(const T* input, AssertMessage message) {
    if (input[0] == static_cast<T>(0)) {
        printf("%s\n", message.text);
        assert(false);
    }
}

template <typename T>
__global__ void assert_async_complex_kernel_impl(
        const thrust::complex<T>* input, AssertMessage message) {
    if (input[0] == thrust::complex<T>(0, 0)) {
        printf("%s\n", message.text);
        assert(false);
    }
}

AssertMessage make_assert_message(const std::string& assert_msg) {
    if (assert_msg.size() >= kAssertMessageCapacity - 1) {
        TP_THROW(ValueError, "assert_async: message is too long");
    }
    AssertMessage message{};
    std::copy_n(assert_msg.data(), assert_msg.size(), message.text);
    return message;
}

void assert_async_msg_cuda(const Tensor& self, std::string assert_msg) {
    const int64_t n = self.numel();
    if (n == 0) {
        TP_THROW(RuntimeError,
                 "Boolean value of Tensor with no values is ambiguous");
    }
    if (n > 1) {
        TP_THROW(RuntimeError,
                 "Boolean value of Tensor with more than one value is ambiguous");
    }

    const AssertMessage message = make_assert_message(assert_msg);
    const auto stream = getCurrentCUDAStream().stream();
#define TP_ASSERT_CASE(ctype, name_) \
    case DType::name_: \
        assert_async_kernel_impl<ctype><<<1, 1, 0, stream>>>( \
            self.data_ptr<ctype>(), message); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_ASSERT_CASE)
        case DType::ComplexFloat:
            assert_async_complex_kernel_impl<float><<<1, 1, 0, stream>>>(
                reinterpret_cast<const thrust::complex<float>*>(
                    self.data_ptr<std::complex<float>>()), message);
            break;
        case DType::ComplexDouble:
            assert_async_complex_kernel_impl<double><<<1, 1, 0, stream>>>(
                reinterpret_cast<const thrust::complex<double>*>(
                    self.data_ptr<std::complex<double>>()), message);
            break;
        default:
            TP_THROW(TypeError, "assert_async: unsupported dtype");
    }
#undef TP_ASSERT_CASE
    CUDA_CHECK(cudaGetLastError());
}

void assert_async_cuda(const Tensor& self) {
    assert_async_msg_cuda(self, std::string());
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


} // namespace

Tensor isclose_cuda(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
    std::vector<int64_t> out_shape = broadcast_shapes(shape_of(self), shape_of(other));
    Tensor a = self.to(DType::Float64);
    Tensor b = other.to(DType::Float64);
    Tensor out = Tensor::empty(out_shape, DType::Bool, self.device());
    if (out.numel() == 0) return out;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();
    gpu_kernel(iter, [=] __device__(double x, double y) -> bool {
        if (x != x || y != y) {
            return equal_nan && x != x && y != y;
        }
        if (::isinf(x) || ::isinf(y)) {
            return x == y;
        }
        return ::fabs(x - y) <= atol + rtol * ::fabs(y);
    });
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
    m.impl("_assert_async", assert_async_cuda);
    m.impl("_assert_async.msg", assert_async_msg_cuda);
    m.impl("isclose", isclose_cuda);
    m.impl("isreal", isreal_cuda);
}

} // namespace cuda
} // namespace tensorplay
