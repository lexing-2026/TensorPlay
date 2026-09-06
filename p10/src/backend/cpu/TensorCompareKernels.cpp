// Tensor comparison operators - CPU kernels.
#include "Tensor.h"
#include "Complex.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "Exception.h"
#include "Parallel.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

namespace {
Tensor isclose_cpu(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
    // |a-b| <= atol + rtol*|b|; values also count as close when they are
    // equal, infinities match each other, and NaNs match under equal_nan.
    // Complex inputs keep complex arithmetic: both the equality check and
    // the error use the full two-component value.
    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError, toString(self.dtype()), " did not match ",
                 toString(other.dtype()));
    }
    TP_THROW_IF(rtol < 0, RuntimeError,
                "rtol must be greater than or equal to zero, but got ", rtol);
    TP_THROW_IF(atol < 0, RuntimeError,
                "atol must be greater than or equal to zero, but got ", atol);

    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    Tensor out = Tensor::empty(out_shape, DType::Bool, self.device());
    int64_t n = out.numel();
    bool* dp = out.data_ptr<bool>();

    if (isComplexType(self.dtype())) {
        Tensor a = self.to(DType::ComplexDouble).expand(out_shape).contiguous();
        Tensor b = other.to(DType::ComplexDouble).expand(out_shape).contiguous();
        const tensorplay::complex<double>* ap = a.data_ptr<tensorplay::complex<double>>();
        const tensorplay::complex<double>* bp = b.data_ptr<tensorplay::complex<double>>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                tensorplay::complex<double> x = ap[i], y = bp[i];
                bool close = x == y;
                if (equal_nan) {
                    auto is_nan = [](const tensorplay::complex<double>& v) {
                        return std::isnan(v.real()) || std::isnan(v.imag());
                    };
                    close = close || (is_nan(x) && is_nan(y));
                }
                if (!close) {
                    double actual = std::abs(x - y);
                    double allowed = atol + rtol * std::abs(y);
                    close = std::isfinite(actual) && actual <= allowed;
                }
                dp[i] = close;
            }
        });
        return out;
    }

    Tensor a = self.to(DType::Float64).expand(out_shape).contiguous();
    Tensor b = other.to(DType::Float64).expand(out_shape).contiguous();
    const double* ap = a.data_ptr<double>();
    const double* bp = b.data_ptr<double>();
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            double x = ap[i], y = bp[i];
            bool close = x == y;
            if (equal_nan && x != x && y != y) {
                close = true;
            }
            if (!close) {
                double actual = std::fabs(x - y);
                double allowed = atol + rtol * std::fabs(y);
                close = std::isfinite(actual) && actual <= allowed;
            }
            dp[i] = close;
        }
    });
    return out;
}

Tensor isreal_cpu(const Tensor& self) {
    // ComplexHelper: real dtypes are trivially real; complex tests imag==0.
    if (!isComplexType(self.dtype())) {
        return Tensor::ones(static_cast<std::vector<int64_t>>(self.shape()),
                            DType::Bool, self.device());
    }
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                               DType::Bool, self.device());
    int64_t n = out.numel();
    bool* dp = out.data_ptr<bool>();
    switch (self.dtype()) {
        case DType::ComplexHalf: {
            const auto* sp = sc.data_ptr<tensorplay::complex<Half>>();
            for (int64_t i = 0; i < n; ++i)
                dp[i] = static_cast<float>(sp[i].imag()) == 0.0f;
            break;
        }
        case DType::ComplexFloat: {
            const auto* sp = sc.data_ptr<tensorplay::complex<float>>();
            for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].imag() == 0.0f;
            break;
        }
        case DType::ComplexDouble: {
            const auto* sp = sc.data_ptr<tensorplay::complex<double>>();
            for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].imag() == 0.0;
            break;
        }
        case DType::BComplex32: {
            const auto* sp = sc.data_ptr<tensorplay::complex<BFloat16>>();
            for (int64_t i = 0; i < n; ++i)
                dp[i] = static_cast<float>(sp[i].imag()) == 0.0f;
            break;
        }
        default:
            TP_THROW(NotImplementedError, "isreal does not support this dtype");
    }
    return out;
}


}  // namespace

TENSORPLAY_LIBRARY_IMPL(CPU, TensorCompareKernels) {
    m.impl("isclose", isclose_cpu);
    m.impl("isreal", isreal_cpu);
}

} // namespace cpu
} // namespace tensorplay
