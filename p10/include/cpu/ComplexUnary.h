// ComplexUnary.h -- shared elementwise complex-math dispatch.
#pragma once

#include <cmath>
#include <type_traits>

#include "Complex.h"
#include "Tensor.h"
#include "DType.h"
#include "Exception.h"
#include "Parallel.h"

namespace tensorplay {
namespace cpu {

inline constexpr int64_t kComplexGrain = 8192;

// --- Complex math helpers ----------------------------------------------------

template <typename T>
inline tensorplay::complex<T> cx_log1p(const tensorplay::complex<T>& z) {
    // Stable complex log1p formulation.
    tensorplay::complex<T> u = z + T(1);
    if (u == T(1)) return z;
    tensorplay::complex<T> log_u = tensorplay::log(u);
    if (u - T(1) == z) return log_u;
    return log_u * (z / (u - T(1)));
}

template <typename T>
inline tensorplay::complex<T> cx_expm1(const tensorplay::complex<T>& z) {
    // expm1(z) = expm1(x)*cos(y) - 2*sin(y/2)^2 + i * e^x * sin(y)
    T x = z.real();
    T y = z.imag();
    T a = std::sin(y / 2);
    T er = std::expm1(x) * std::cos(y) - T(2) * a * a;
    T ei = std::exp(x) * std::sin(y);
    return tensorplay::complex<T>(er, ei);
}

template <typename T>
inline tensorplay::complex<T> cx_log2(const tensorplay::complex<T>& z) {
    return tensorplay::log(z) / std::log(T(2));
}

template <typename T>
inline tensorplay::complex<T> cx_rsqrt(const tensorplay::complex<T>& z) {
    return T(1) / tensorplay::sqrt(z);
}

template <typename T>
inline tensorplay::complex<T> cx_sigmoid(const tensorplay::complex<T>& z) {
    return T(1) / (T(1) + tensorplay::exp(-z));
}

// Complex-capable unary kernel: `func` receives the local complex scalar type.
template <typename Func>
Tensor complex_unary_op_kernel(const Tensor& self, Func func);

}  // namespace cpu
}  // namespace tensorplay

// The template needs Tensor/TensorIterator-free elementwise access; keep the
// definition out of the include-heavy header translation units by defining it
// after its dependencies (Parallel.h gives parallel_for).
namespace tensorplay {
namespace cpu {

template <typename Func>
Tensor complex_unary_op_kernel(const Tensor& self, Func func) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                  self.dtype(), self.device());
    const int64_t n = self.numel();
    Tensor self_contig = self.is_contiguous() ? self : self.clone();

    switch (self.dtype()) {
        case DType::ComplexFloat: {
            auto* src = reinterpret_cast<const tensorplay::complex<float>*>(self_contig.data_ptr());
            auto* dst = reinterpret_cast<tensorplay::complex<float>*>(result.data_ptr());
            tensorplay::parallel::parallel_for(0, n, kComplexGrain, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    auto r = func(src[i]);
                    dst[i] = tensorplay::complex<float>(static_cast<float>(r.real()),
                                                        static_cast<float>(r.imag()));
                }
            });
            break;
        }
        case DType::ComplexDouble: {
            auto* src = reinterpret_cast<const tensorplay::complex<double>*>(self_contig.data_ptr());
            auto* dst = reinterpret_cast<tensorplay::complex<double>*>(result.data_ptr());
            tensorplay::parallel::parallel_for(0, n, kComplexGrain, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) dst[i] = func(src[i]);
            });
            break;
        }
        case DType::ComplexHalf:
        case DType::BComplex32: {
            if (self.dtype() == DType::ComplexHalf) {
                auto* src = reinterpret_cast<const tensorplay::complex<Half>*>(self_contig.data_ptr());
                auto* dst = reinterpret_cast<tensorplay::complex<Half>*>(result.data_ptr());
                tensorplay::parallel::parallel_for(0, n, kComplexGrain, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        tensorplay::complex<float> s(static_cast<float>(src[i].real()),
                                                     static_cast<float>(src[i].imag()));
                        tensorplay::complex<float> r = func(s);
                        dst[i] = tensorplay::complex<Half>(static_cast<Half>(r.real()),
                                                           static_cast<Half>(r.imag()));
                    }
                });
            } else {
                auto* src = reinterpret_cast<const tensorplay::complex<BFloat16>*>(self_contig.data_ptr());
                auto* dst = reinterpret_cast<tensorplay::complex<BFloat16>*>(result.data_ptr());
                tensorplay::parallel::parallel_for(0, n, kComplexGrain, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        tensorplay::complex<float> s(static_cast<float>(src[i].real()),
                                                     static_cast<float>(src[i].imag()));
                        auto rd = func(s);
                        tensorplay::complex<float> r(static_cast<float>(rd.real()),
                                                     static_cast<float>(rd.imag()));
                        dst[i] = tensorplay::complex<BFloat16>(static_cast<BFloat16>(r.real()),
                                                               static_cast<BFloat16>(r.imag()));
                    }
                });
            }
            break;
        }
        default: TP_THROW(TypeError, "complex unary op: unsupported dtype");
    }
    return result;
}

}  // namespace cpu
}  // namespace tensorplay
