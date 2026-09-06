// Tensor comparison operators - CUDA kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "Exception.h"
#include "CUDARuntime.h"
#include "Complex.h"
#include "CUDALoops.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
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
        const tensorplay::complex<T>* input, AssertMessage message) {
    if (static_cast<float>(input[0].real()) == 0.0f &&
        static_cast<float>(input[0].imag()) == 0.0f) {
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
        case DType::ComplexHalf:
            assert_async_complex_kernel_impl<Half><<<1, 1, 0, stream>>>(
                self.data_ptr<tensorplay::complex<Half>>(), message);
            break;
        case DType::ComplexFloat:
            assert_async_complex_kernel_impl<float><<<1, 1, 0, stream>>>(
                self.data_ptr<tensorplay::complex<float>>(), message);
            break;
        case DType::ComplexDouble:
            assert_async_complex_kernel_impl<double><<<1, 1, 0, stream>>>(
                self.data_ptr<tensorplay::complex<double>>(), message);
            break;
        case DType::BComplex32:
            assert_async_complex_kernel_impl<BFloat16><<<1, 1, 0, stream>>>(
                self.data_ptr<tensorplay::complex<BFloat16>>(), message);
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

template <typename T>
__device__ inline bool isclose_real_value(T lhs, T rhs, double rtol,
                                          double atol, bool equal_nan) {
    using math_t = std::conditional_t<std::is_same_v<T, double>, double, float>;
    const bool equal = lhs == rhs;
    const math_t lhs_math = static_cast<math_t>(lhs);
    const math_t rhs_math = static_cast<math_t>(rhs);
    bool close = equal;
    if (equal_nan && lhs_math != lhs_math && rhs_math != rhs_math) {
        close = true;
    }
    if (!close && (rtol != 0.0 || atol != 0.0)) {
        const math_t actual = ::fabs(lhs_math - rhs_math);
        const math_t allowed = static_cast<math_t>(atol) +
            static_cast<math_t>(rtol) * ::fabs(rhs_math);
        close = ::isfinite(actual) && actual <= allowed;
    }
    return close;
}

template <typename T>
void isclose_real_loop(TensorIterator& iter, double rtol, double atol,
                       bool equal_nan) {
    gpu_kernel(iter, [=] __device__(T lhs, T rhs) -> bool {
        return isclose_real_value(lhs, rhs, rtol, atol, equal_nan);
    });
}

template <typename ComplexT, typename RealT>
__device__ inline bool isclose_complex_value(
        ComplexT lhs, ComplexT rhs, double rtol, double atol, bool equal_nan) {
    const bool equal = lhs == rhs;
    const RealT lhs_real = static_cast<RealT>(lhs.real());
    const RealT lhs_imag = static_cast<RealT>(lhs.imag());
    const RealT rhs_real = static_cast<RealT>(rhs.real());
    const RealT rhs_imag = static_cast<RealT>(rhs.imag());
    const bool lhs_nan = lhs_real != lhs_real || lhs_imag != lhs_imag;
    const bool rhs_nan = rhs_real != rhs_real || rhs_imag != rhs_imag;
    bool close = equal || (equal_nan && lhs_nan && rhs_nan);
    if (!close && (rtol != 0.0 || atol != 0.0)) {
        const RealT actual = ::hypot(lhs_real - rhs_real, lhs_imag - rhs_imag);
        const RealT allowed = static_cast<RealT>(atol) +
            static_cast<RealT>(rtol) * ::hypot(rhs_real, rhs_imag);
        close = ::isfinite(actual) && actual <= allowed;
    }
    return close;
}

template <typename ComplexT, typename RealT>
void isclose_complex_loop(TensorIterator& iter, double rtol, double atol,
                          bool equal_nan) {
    gpu_kernel(iter, [=] __device__(ComplexT lhs, ComplexT rhs) -> bool {
        return isclose_complex_value<ComplexT, RealT>(
            lhs, rhs, rtol, atol, equal_nan);
    });
}

} // namespace

Tensor isclose_cuda(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError, toString(self.dtype()), " did not match ",
                 toString(other.dtype()));
    }
    TP_THROW_IF(rtol < 0, RuntimeError,
                "rtol must be greater than or equal to zero, but got ", rtol);
    TP_THROW_IF(atol < 0, RuntimeError,
                "atol must be greater than or equal to zero, but got ", atol);
    std::vector<int64_t> out_shape = broadcast_shapes(shape_of(self), shape_of(other));
    Tensor out = Tensor::empty(out_shape, DType::Bool, self.device());
    if (out.numel() == 0) return out;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(out)
        .add_const_input(self)
        .add_const_input(other)
        .build();
    switch (self.dtype()) {
#define TP_ISCLOSE_REAL_CASE(ctype, name) \
        case DType::name: \
            isclose_real_loop<ctype>(iter, rtol, atol, equal_nan); \
            break;
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_ISCLOSE_REAL_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_ISCLOSE_REAL_CASE)
#undef TP_ISCLOSE_REAL_CASE
        case DType::ComplexHalf:
            isclose_complex_loop<tensorplay::complex<Half>, float>(
                iter, rtol, atol, equal_nan);
            break;
        case DType::ComplexFloat:
            isclose_complex_loop<tensorplay::complex<float>, float>(
                iter, rtol, atol, equal_nan);
            break;
        case DType::ComplexDouble:
            isclose_complex_loop<tensorplay::complex<double>, double>(
                iter, rtol, atol, equal_nan);
            break;
        case DType::BComplex32:
            isclose_complex_loop<tensorplay::complex<BFloat16>, float>(
                iter, rtol, atol, equal_nan);
            break;
        default:
            TP_THROW(TypeError, "isclose: unsupported dtype ",
                     toString(self.dtype()));
    }
    return out;
}

Tensor isreal_cuda(const Tensor& self) {
    // Real dtypes are trivially real; complex tests imag==0.
    if (!isComplexType(self.dtype())) {
        return Tensor::ones(shape_of(self), DType::Bool, self.device());
    }
    Tensor out = Tensor::empty(shape_of(self), DType::Bool, self.device());
    if (out.numel() == 0) return out;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(out)
        .add_input(self)
        .build();
    switch (self.dtype()) {
        case DType::ComplexHalf:
            gpu_kernel(iter, [] __device__ (tensorplay::complex<Half> value) -> bool {
                return static_cast<float>(value.imag()) == 0.0f;
            });
            break;
        case DType::ComplexFloat:
            gpu_kernel(iter, [] __device__ (tensorplay::complex<float> value) -> bool {
                return value.imag() == 0.0f;
            });
            break;
        case DType::ComplexDouble:
            gpu_kernel(iter, [] __device__ (tensorplay::complex<double> value) -> bool {
                return value.imag() == 0.0;
            });
            break;
        case DType::BComplex32:
            gpu_kernel(iter, [] __device__ (tensorplay::complex<BFloat16> value) -> bool {
                return static_cast<float>(value.imag()) == 0.0f;
            });
            break;
        default:
            TP_THROW(TypeError, "isreal: unsupported dtype");
    }
    return out;
}


TENSORPLAY_LIBRARY_IMPL(CUDA, TensorCompareKernels) {
    m.impl("_assert_async", assert_async_cuda);
    m.impl("_assert_async.msg", assert_async_msg_cuda);
    m.impl("isclose", isclose_cuda);
    m.impl("isreal", isreal_cuda);
}

} // namespace cuda
} // namespace tensorplay
