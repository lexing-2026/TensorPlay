#include "Tensor.h"
#include "Complex.h"
#include "Dispatcher.h"
#include "Generator.h"
#include "DistributionsHelper.h"
#include "DistributionDispatch.h"
#include "Scalar.h"
#include "Exception.h"
#include "Parallel.h"
#include "Context.h"
#include "cpu/vec/vec.h"
#include <vector>
#include <cmath>
#include <type_traits>
#include <algorithm>
#include <cstring>

namespace tensorplay {
namespace cpu {

Tensor& fill_kernel(Tensor& self, Scalar value);

Tensor allocate_cpu_tensor(const std::vector<int64_t>& size, DType dtype, bool pin_memory) {
#ifdef USE_CUDA
    if (pin_memory) {
        int64_t numel = 1;
        for (int64_t value : size) numel *= value;
        const size_t nbytes = static_cast<size_t>(numel) * elementSize(dtype);
        Storage storage(nbytes, getPinnedMemoryAllocator(), Device(DeviceType::CPU));
        return Tensor(storage, size, dtype);
    }
#else
    if (pin_memory) {
        TP_THROW(RuntimeError, "pin_memory requires a CUDA-enabled TensorPlay build");
    }
#endif
    return Tensor(size, dtype, Device(DeviceType::CPU));
}

Tensor rand_kernel(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    int64_t n = t.numel();
    auto& gen = default_generator();

    switch (dtype) {
        case DType::Float32: {
            float* data = t.data_ptr<float>();
            uniform_real_distribution<float> dist(0.0f, 1.0f);
            const float to_scalar = 1.0f;
            for (int64_t i = 0; i < n; ++i) {
                float value = static_cast<float>(dist(&gen));
                data[i] = value == to_scalar ? 0.0f : value;
            }
            break;
        }
        case DType::Float64: {
            double* data = t.data_ptr<double>();
            uniform_real_distribution<double> dist(0.0, 1.0);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = dist(&gen);
            }
            break;
        }
        case DType::Float16:
        case DType::BFloat16: {
            // uniform_kernel_cpu): Half/BFloat16 sample in opmath_t (float,
            // 24-bit mantissa mask) and cast to the storage dtype, clamping a
            // cast that rounded up to the upper bound back to 'from'.
            using math_t = float;
            const math_t lo = static_cast<math_t>(0.0);
            const math_t hi = static_cast<math_t>(1.0);
            const math_t to_scalar = hi;
            const math_t from_scalar = lo;
            if (dtype == DType::Float16) {
                Half* data = t.data_ptr<Half>();
                uniform_real_distribution<math_t> dist(lo, hi);
                for (int64_t i = 0; i < n; ++i) {
                    Half value = static_cast<Half>(dist(&gen));
                    data[i] = static_cast<math_t>(value) == to_scalar
                                  ? static_cast<Half>(from_scalar) : value;
                }
            } else {
                BFloat16* data = t.data_ptr<BFloat16>();
                uniform_real_distribution<math_t> dist(lo, hi);
                for (int64_t i = 0; i < n; ++i) {
                    BFloat16 value = static_cast<BFloat16>(dist(&gen));
                    data[i] = static_cast<math_t>(value) == to_scalar
                                  ? static_cast<BFloat16>(from_scalar) : value;
                }
            }
            break;
        }
        case DType::ComplexFloat: {
            tensorplay::complex<float>* data = t.data_ptr<tensorplay::complex<float>>();
            uniform_real_distribution<float> dist(0.0f, 1.0f);
            const float to_scalar = 1.0f;
            for (int64_t i = 0; i < n; ++i) {
                float re = static_cast<float>(dist(&gen));
                float im = static_cast<float>(dist(&gen));
                data[i] = tensorplay::complex<float>(
                    re == to_scalar ? 0.0f : re,
                    im == to_scalar ? 0.0f : im);
            }
            break;
        }
        case DType::ComplexDouble: {
            tensorplay::complex<double>* data = t.data_ptr<tensorplay::complex<double>>();
            uniform_real_distribution<double> dist(0.0, 1.0);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = tensorplay::complex<double>(dist(&gen), dist(&gen));
            }
            break;
        }
        default:
            TP_THROW(NotImplementedError, "rand() only supports floating dtypes for now");
    }
    return t;
}

namespace {
// E|z|^2 == 1; complex rand draws components uniform on [0, 1).
constexpr double kComplexComponentStd = 0.7071067811865476;  // 1/sqrt(2)
}  // namespace

Tensor zeros_kernel(const std::vector<int64_t>& size, DType dtype, Device device, bool pin_memory) {
    Tensor t = allocate_cpu_tensor(size, dtype, pin_memory);
    size_t nbytes = t.numel() * t.itemsize();
    if (t.data_ptr()) {
        std::memset(t.data_ptr(), 0, nbytes);
    }
    return t;
}

Tensor ones_kernel(const std::vector<int64_t>& size, DType dtype, Device device, bool pin_memory) {
    Tensor t = allocate_cpu_tensor(size, dtype, pin_memory);
    fill_kernel(t, 1);
    return t;
}

namespace {

DType infer_full_dtype(const Scalar& fill_value) {
    if (fill_value.isBoolean()) return DType::Bool;
    if (fill_value.isIntegral(false)) return DType::Int64;
    if (fill_value.isComplex()) {
        return globalContext().defaultDType() == DType::Float64
                   ? DType::ComplexDouble
                   : DType::ComplexFloat;
    }
    return globalContext().defaultDType();
}

} // namespace

Tensor full_kernel(const std::vector<int64_t>& size, Scalar fill_value, DType dtype, Device device, bool pin_memory) {
    DType inferred_dtype = dtype;
    if (inferred_dtype == DType::Undefined) {
        inferred_dtype = infer_full_dtype(fill_value);
    }
    Tensor t = allocate_cpu_tensor(size, inferred_dtype, pin_memory);
    fill_kernel(t, fill_value);
    return t;
}

Tensor& fill_kernel(Tensor& self, Scalar value) {
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        ctype* data = self.data_ptr<ctype>(); \
        int64_t n = self.numel(); \
        ctype val = value.to<ctype>(); \
        std::fill(data, data + n, val); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(OP_CASE)
        default: 
            std::cerr << "fill_kernel error: dtype=" << (int)self.dtype() << std::endl;
            TP_THROW(NotImplementedError, "fill_ not implemented for this dtype");
    }
    #undef OP_CASE
    return self;
}

#include <iostream>

Tensor arange_start_step_kernel(Scalar start, Scalar end, Scalar step, DType dtype, Device device) {
    // Better length calculation to avoid precision issues with large integers
    double s_d = start.toDouble();
    double e_d = end.toDouble();
    double st_d = step.toDouble();
    int64_t len;
    
    if (start.isIntegral() && end.isIntegral() && step.isIntegral()) {
         int64_t s = start.to<int64_t>();
         int64_t e = end.to<int64_t>();
         int64_t st = step.to<int64_t>();
         if (st == 0) TP_THROW(RuntimeError, "step must be nonzero");
         if ((st > 0 && s > e) || (st < 0 && s < e)) {
             len = 0;
         } else {
             // ceil((end-start)/step)
             double tmp = std::ceil((e_d - s_d) / st_d);
             len = static_cast<int64_t>(tmp);
         }
    } else {
         len = static_cast<int64_t>(std::ceil((e_d - s_d) / st_d));
    }

    if (len < 0) len = 0;
    
    // Type inference if Undefined
    if (dtype == DType::Undefined) {
        if (start.isFloatingPoint() || end.isFloatingPoint() || step.isFloatingPoint()) {
            dtype = globalContext().defaultDType();
        } else {
            dtype = DType::Int64;
        }
    }
    
    Tensor t({len}, dtype, device);

    // AT_DISPATCH_ALL_TYPES_AND2(Half, BFloat16) -- Bool, UInt16/32/64 and
    // complex are not implemented.  The sub-32-bit types supported below
    // were previously left uninitialized.
    switch (dtype) {
        case DType::Bool:
            TP_THROW(NotImplementedError,
                     "\"arange\" not implemented for '" + std::string(toString(dtype)) + "'");
#define ARANGE_CASE(ctype, name) \
        case DType::name: { \
            arange_fill<ctype>(t.data_ptr<ctype>(), len, s_d, st_d); \
            break; \
        }
        ARANGE_CASE(uint8_t, UInt8)
        ARANGE_CASE(int8_t, Int8)
        ARANGE_CASE(int16_t, Int16)
        ARANGE_CASE(int32_t, Int32)
        ARANGE_CASE(int64_t, Int64)
        ARANGE_CASE(float, Float32)
        ARANGE_CASE(double, Float64)
        ARANGE_CASE(tensorplay::Half, Float16)
        ARANGE_CASE(tensorplay::BFloat16, BFloat16)
#undef ARANGE_CASE
        default:
            TP_THROW(NotImplementedError,
                     "\"arange\" not implemented for '" + std::string(toString(dtype)) + "'");
    }
    return t;
}

Tensor arange_kernel(Scalar end, DType dtype, Device device) {
    return arange_start_step_kernel(Scalar(0), end, Scalar(1), dtype, device);
}

Tensor empty_kernel(const std::vector<int64_t>& size, DType dtype, Device device, bool pin_memory) {
    return allocate_cpu_tensor(size, dtype, pin_memory);
}

Tensor eye_kernel(int64_t n, int64_t m, DType dtype, Device device) {
    if (m < 0) m = n;
    Tensor t = Tensor::zeros({n, m}, dtype, device);
    const int64_t min_dim = std::min(n, m);
    if (min_dim == 0) return t;
#define TP_EYE_CASE(ctype, name)                                    \
    case DType::name: {                                             \
        ctype* data = t.data_ptr<ctype>();                          \
        for (int64_t i = 0; i < min_dim; ++i) data[i * m + i] = ctype(1); \
        break;                                                      \
    }
    switch (dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_EYE_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_EYE_CASE)
        case DType::ComplexFloat: {
            auto* data = t.data_ptr<tensorplay::complex<float>>();
            for (int64_t i = 0; i < min_dim; ++i) data[i * m + i] = {1.0f, 0.0f};
            break;
        }
        case DType::ComplexDouble: {
            auto* data = t.data_ptr<tensorplay::complex<double>>();
            for (int64_t i = 0; i < min_dim; ++i) data[i * m + i] = {1.0, 0.0};
            break;
        }
        case DType::ComplexHalf: {
            auto* data = t.data_ptr<tensorplay::complex<Half>>();
            for (int64_t i = 0; i < min_dim; ++i)
                data[i * m + i] =
                    tensorplay::complex<Half>(Half(1.0f), Half(0.0f));
            break;
        }
        case DType::BComplex32: {
            auto* data = t.data_ptr<tensorplay::complex<BFloat16>>();
            for (int64_t i = 0; i < min_dim; ++i)
                data[i * m + i] =
                    tensorplay::complex<BFloat16>(BFloat16(1.0f), BFloat16(0.0f));
            break;
        }
        default:
            TP_THROW(TypeError, "eye: unsupported dtype");
    }
#undef TP_EYE_CASE
    return t;
}

// Both sequence factories walk in from whichever endpoint is nearer: the first
// half accumulates forward from `start` and the second half backward from
// `end`, which pins both endpoints exactly and halves the error a single
// forward accumulation would reach at the far end.
//
// The step is carried in a wider domain than the element type: an integral
// output spans a range its own type may not represent, and a reduced-width
// float would round the increment away long before the end of the sequence.
namespace {

template <typename T>
void arange_fill(T* data, int64_t steps, double start, double step) {
    using Vec = tensorplay::vec::Vectorized<T>;
    constexpr int64_t width = Vec::size();
    parallel::parallel_for(0, steps, parallel::GRAIN_SIZE,
        [&](int64_t begin, int64_t last) {
            int64_t index = begin;
            const int64_t vector_end = begin +
                ((last - begin) / width) * width;
            for (; index < vector_end; index += width) {
                const T base = static_cast<T>(
                    start + static_cast<double>(index) * step);
                Vec::arange(base, step).store(data + index);
            }
            for (; index < last; ++index) {
                data[index] = static_cast<T>(
                    start + static_cast<double>(index) * step);
            }
        });
}

// The endpoints are first narrowed to the element type, so an integral
// sequence starts and ends where that type actually lands, and only then
// widened for the accumulation.  An integral element type steps through
// double, since its own range cannot express the increment.
template <typename T>
struct SequenceStepType {
    using type = std::conditional_t<std::is_integral_v<T>, double, T>;
};

template <typename T>
void linspace_fill(T* data, Scalar start, Scalar end, int64_t steps) {
    using step_t = typename SequenceStepType<T>::type;
    const T s = start.to<T>();
    const T e = end.to<T>();
    const step_t step =
        static_cast<step_t>((static_cast<step_t>(e) - static_cast<step_t>(s)) /
                            (steps - 1));
    const int64_t halfway = steps / 2;
    parallel::parallel_for(0, steps, parallel::GRAIN_SIZE, [&](int64_t begin, int64_t last) {
        for (int64_t i = begin; i < last; ++i) {
            data[i] = i < halfway ? static_cast<T>(s + step * i)
                                  : static_cast<T>(e - step * (steps - i - 1));
        }
    });
}

template <typename T>
void logspace_fill(T* data, Scalar start, Scalar end, int64_t steps,
                   double base) {
    const T s = start.to<T>();
    const T e = end.to<T>();
    const double step =
        static_cast<double>(e - s) / static_cast<double>(steps - 1);
    const int64_t halfway = steps / 2;
    parallel::parallel_for(0, steps, parallel::GRAIN_SIZE, [&](int64_t begin, int64_t last) {
        for (int64_t i = begin; i < last; ++i) {
            const double exponent = i < halfway ? s + step * i
                                                : e - step * (steps - i - 1);
            data[i] = static_cast<T>(std::pow(base, exponent));
        }
    });
}

// Complex sequences step through a complex domain of at least single
// precision; the reduced-width element types store the rounded result.
template <typename compute_t, typename store_t>
void linspace_fill_complex(store_t* data, Scalar start, Scalar end,
                           int64_t steps) {
    using value_t = typename compute_t::value_type;
    const auto start_host = start.to<tensorplay::complex<double>>();
    const auto end_host = end.to<tensorplay::complex<double>>();
    const compute_t s(static_cast<value_t>(start_host.real()),
                      static_cast<value_t>(start_host.imag()));
    const compute_t e(static_cast<value_t>(end_host.real()),
                      static_cast<value_t>(end_host.imag()));
    const compute_t step = (e - s) / static_cast<value_t>(steps - 1);
    const int64_t halfway = steps / 2;
    for (int64_t i = 0; i < steps; ++i) {
        const int64_t distance = i < halfway ? i : steps - i - 1;
        const compute_t value =
            i < halfway ? s + step * static_cast<value_t>(distance)
                        : e - step * static_cast<value_t>(distance);
        data[i] = store_t(value);
    }
}

template <typename compute_t, typename store_t>
void logspace_fill_complex(store_t* data, Scalar start, Scalar end,
                           int64_t steps, double base) {
    using value_t = typename compute_t::value_type;
    const auto start_host = start.to<tensorplay::complex<double>>();
    const auto end_host = end.to<tensorplay::complex<double>>();
    const compute_t s(static_cast<value_t>(start_host.real()),
                      static_cast<value_t>(start_host.imag()));
    const compute_t e(static_cast<value_t>(end_host.real()),
                      static_cast<value_t>(end_host.imag()));
    const compute_t base_value(static_cast<value_t>(base), value_t(0));
    const compute_t step = (e - s) / static_cast<value_t>(steps - 1);
    const int64_t halfway = steps / 2;
    for (int64_t i = 0; i < steps; ++i) {
        const int64_t distance = i < halfway ? i : steps - i - 1;
        const compute_t exponent =
            i < halfway ? s + step * static_cast<value_t>(distance)
                        : e - step * static_cast<value_t>(distance);
        data[i] = store_t(tensorplay::pow(base_value, exponent));
    }
}

}  // namespace

Tensor linspace_kernel(Scalar start, Scalar end, int64_t steps, DType dtype, Device device) {
    if (steps < 0) TP_THROW(RuntimeError, "number of steps must be non-negative");
    Tensor t({steps}, dtype, device);
    if (steps == 0) return t;
    // A single step is the start value itself; the increment is undefined.
    if (steps == 1) return fill_kernel(t, start);

#define TP_LINSPACE_CASE(ctype, name)                                        \
    case DType::name:                                                        \
        linspace_fill<ctype>(t.data_ptr<ctype>(), start, end, steps);        \
        break;
    switch (dtype) {
        TP_LINSPACE_CASE(uint8_t, UInt8)
        TP_LINSPACE_CASE(int8_t, Int8)
        TP_LINSPACE_CASE(int16_t, Int16)
        TP_LINSPACE_CASE(int32_t, Int32)
        TP_LINSPACE_CASE(int64_t, Int64)
        TP_LINSPACE_CASE(uint16_t, UInt16)
        TP_LINSPACE_CASE(uint32_t, UInt32)
        TP_LINSPACE_CASE(uint64_t, UInt64)
        TP_LINSPACE_CASE(float, Float32)
        TP_LINSPACE_CASE(double, Float64)
        TP_LINSPACE_CASE(Half, Float16)
        TP_LINSPACE_CASE(BFloat16, BFloat16)
        case DType::ComplexHalf:
            linspace_fill_complex<tensorplay::complex<float>,
                                  tensorplay::complex<Half>>(
                t.data_ptr<tensorplay::complex<Half>>(), start, end, steps);
            break;
        case DType::ComplexFloat:
            linspace_fill_complex<tensorplay::complex<float>,
                                  tensorplay::complex<float>>(
                t.data_ptr<tensorplay::complex<float>>(), start, end, steps);
            break;
        case DType::ComplexDouble:
            linspace_fill_complex<tensorplay::complex<double>,
                                  tensorplay::complex<double>>(
                t.data_ptr<tensorplay::complex<double>>(), start, end, steps);
            break;
        case DType::BComplex32:
            linspace_fill_complex<tensorplay::complex<float>,
                                  tensorplay::complex<BFloat16>>(
                t.data_ptr<tensorplay::complex<BFloat16>>(), start, end, steps);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "linspace does not support dtype '" +
                     std::string(toString(dtype)) + "'");
    }
#undef TP_LINSPACE_CASE
    return t;
}

Tensor logspace_kernel(Scalar start, Scalar end, int64_t steps, double base, DType dtype, Device device) {
    if (steps < 0) TP_THROW(RuntimeError, "number of steps must be non-negative");
    Tensor t({steps}, dtype, device);
    if (steps == 0) return t;
    if (steps == 1) {
        return isComplexType(dtype)
            ? fill_kernel(t, Scalar(tensorplay::pow(
                  tensorplay::complex<double>(base, 0.0),
                  start.to<tensorplay::complex<double>>())))
            : fill_kernel(t, Scalar(std::pow(base, start.toDouble())));
    }

#define TP_LOGSPACE_CASE(ctype, name)                                        \
    case DType::name:                                                        \
        logspace_fill<ctype>(t.data_ptr<ctype>(), start, end, steps, base);  \
        break;
    switch (dtype) {
        TP_LOGSPACE_CASE(uint8_t, UInt8)
        TP_LOGSPACE_CASE(int8_t, Int8)
        TP_LOGSPACE_CASE(int16_t, Int16)
        TP_LOGSPACE_CASE(int32_t, Int32)
        TP_LOGSPACE_CASE(int64_t, Int64)
        TP_LOGSPACE_CASE(uint16_t, UInt16)
        TP_LOGSPACE_CASE(uint32_t, UInt32)
        TP_LOGSPACE_CASE(uint64_t, UInt64)
        TP_LOGSPACE_CASE(float, Float32)
        TP_LOGSPACE_CASE(double, Float64)
        TP_LOGSPACE_CASE(Half, Float16)
        TP_LOGSPACE_CASE(BFloat16, BFloat16)
        case DType::ComplexHalf:
            logspace_fill_complex<tensorplay::complex<float>,
                                  tensorplay::complex<Half>>(
                t.data_ptr<tensorplay::complex<Half>>(), start, end, steps, base);
            break;
        case DType::ComplexFloat:
            logspace_fill_complex<tensorplay::complex<float>,
                                  tensorplay::complex<float>>(
                t.data_ptr<tensorplay::complex<float>>(), start, end, steps, base);
            break;
        case DType::ComplexDouble:
            logspace_fill_complex<tensorplay::complex<double>,
                                  tensorplay::complex<double>>(
                t.data_ptr<tensorplay::complex<double>>(), start, end, steps, base);
            break;
        case DType::BComplex32:
            logspace_fill_complex<tensorplay::complex<float>,
                                  tensorplay::complex<BFloat16>>(
                t.data_ptr<tensorplay::complex<BFloat16>>(), start, end, steps, base);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "logspace does not support dtype '" +
                     std::string(toString(dtype)) + "'");
    }
#undef TP_LOGSPACE_CASE
    return t;
}

// --- Random Factory Kernels ---

Tensor randn_kernel(const std::vector<int64_t>& size, DType dtype, Device device,
                    Generator* generator = nullptr, bool pin_memory = false) {
    Tensor t = device.type() == DeviceType::CPU
        ? allocate_cpu_tensor(size, dtype, pin_memory)
        : Tensor(size, dtype, device);
    int64_t n = t.numel();
    Generator* gen = generator != nullptr ? generator : &default_generator();

    switch (dtype) {
        case DType::Float32: {
            float* data = t.data_ptr<float>();
            if (n >= 16 && t.is_contiguous()) {
                normal_fill<float>(data, n, 0.0f, 1.0f, gen);
            } else {
                normal_distribution<double> dist(0.0, 1.0);
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = static_cast<float>(dist(gen));
                }
            }
            break;
        }
        case DType::Float64: {
            double* data = t.data_ptr<double>();
            if (n >= 16 && t.is_contiguous()) {
                normal_fill<double>(data, n, 0.0, 1.0, gen);
            } else {
                normal_distribution<double> dist(0.0, 1.0);
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = dist(gen);
                }
            }
            break;
        }
        case DType::Float16:
        case DType::BFloat16: {
            // normal_fill<scalar_t>): Half/BFloat16 draw uniforms in opmath
            // (float, 24-bit mantissa) through a 16-element stack buffer,
            // Box-Muller in float, then cast down to the storage dtype.
            if (n >= 16 && t.is_contiguous()) {
                if (dtype == DType::Float16) {
                    normal_fill_cast<Half>(t.data_ptr<Half>(), n, 0.0, 1.0, gen);
                } else {
                    normal_fill_cast<BFloat16>(t.data_ptr<BFloat16>(), n, 0.0, 1.0, gen);
                }
            } else {
                normal_distribution<double> dist(0.0, 1.0);
                if (dtype == DType::Float16) {
                    Half* data = t.data_ptr<Half>();
                    for (int64_t i = 0; i < n; ++i) {
                        data[i] = static_cast<Half>(dist(gen));
                    }
                } else {
                    BFloat16* data = t.data_ptr<BFloat16>();
                    for (int64_t i = 0; i < n; ++i) {
                        data[i] = static_cast<BFloat16>(dist(gen));
                    }
                }
            }
            break;
        }
        case DType::ComplexFloat:
        case DType::ComplexDouble: {
            // Each component uses a normal distribution scaled by sqrt(2) so
            // the complex value has unit expected squared magnitude.
            const double comp_std = kComplexComponentStd;
            if (dtype == DType::ComplexFloat) {
                tensorplay::complex<float>* data = t.data_ptr<tensorplay::complex<float>>();
                normal_distribution<double> dist(0.0, 1.0);
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = tensorplay::complex<float>(
                        static_cast<float>(dist(gen) * comp_std),
                        static_cast<float>(dist(gen) * comp_std));
                }
            } else {
                tensorplay::complex<double>* data = t.data_ptr<tensorplay::complex<double>>();
                normal_distribution<double> dist(0.0, 1.0);
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = tensorplay::complex<double>(dist(gen) * comp_std,
                                                          dist(gen) * comp_std);
                }
            }
            break;
        }
        default:
            TP_THROW(NotImplementedError, "randn() only supports floating dtypes for now");
    }
    return t;
}

Tensor randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& size, DType dtype, Device device) {
    TP_THROW_IF(high <= low, RuntimeError,
                "randint expects 'from' to be less than 'to', but got from=",
                low, " >= to=", high);

    Tensor t(size, dtype, device);
    auto& gen = default_generator();
    const uint64_t range = static_cast<uint64_t>(high) -
        static_cast<uint64_t>(low);
    const int64_t base = low;

    distribution::check_random_from_to_bounds(low, high, dtype);

    int64_t n = t.numel();
    distribution::dispatch_dtype(dtype, [&](auto tag) {
        using scalar_t = decltype(tag);
        scalar_t* data = t.data_ptr<scalar_t>();
        uniform_int_from_to_distribution<scalar_t> dist(range, base);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(&gen);
        }
    });
    return t;
}

void check_randperm_size(int64_t n, DType dtype) {
    if (n < 0) {
        TP_THROW(RuntimeError, "randperm(): n must be non-negative, got ", n);
    }
    uint64_t max_index;
    switch (dtype) {
        case DType::UInt8: max_index = std::numeric_limits<uint8_t>::max(); break;
        case DType::Int8: max_index = std::numeric_limits<int8_t>::max(); break;
        case DType::Int16: max_index = std::numeric_limits<int16_t>::max(); break;
        case DType::Int32: max_index = std::numeric_limits<int32_t>::max(); break;
        case DType::UInt16: max_index = std::numeric_limits<uint16_t>::max(); break;
        case DType::UInt32: max_index = std::numeric_limits<uint32_t>::max(); break;
        case DType::Float16: max_index = uint64_t{1} << 11; break;
        case DType::BFloat16: max_index = uint64_t{1} << 8; break;
        case DType::Float32: max_index = uint64_t{1} << 24; break;
        case DType::Float64: max_index = uint64_t{1} << 53; break;
        case DType::Bool: max_index = 1; break;
        case DType::Int64:
        case DType::UInt64:
            return;
        default:
            TP_THROW(NotImplementedError,
                     "randperm() does not support this output dtype");
    }
    if (static_cast<uint64_t>(n) > max_index + 1) {
        TP_THROW(RuntimeError,
                 "randperm(): n is too large for the requested output dtype");
    }
}

Tensor randperm_kernel(int64_t n, DType dtype, Device device) {
    check_randperm_size(n, dtype);
    Tensor t({n}, dtype, device);
    if (n == 0) return t;

    auto& gen = default_generator();
    distribution::dispatch_dtype(dtype, [&](auto tag) {
        using scalar_t = decltype(tag);
        scalar_t* data = t.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<scalar_t>(i);
        }
        for (int64_t i = 0; i < n - 1; ++i) {
            const uint64_t tail = static_cast<uint64_t>(n - i);
            const uint64_t draw = tail >= (1ULL << 32)
                ? gen.random64()
                : static_cast<uint64_t>(gen.random());
            const int64_t z = static_cast<int64_t>(draw % tail);
            scalar_t save = data[i];
            data[i] = data[z + i];
            data[z + i] = save;
        }
    });
    return t;
}

Tensor rand_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return rand_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype, dev);
}

Tensor randint_like_kernel(const Tensor& self, int64_t low, int64_t high, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return randint_kernel(low, high, static_cast<std::vector<int64_t>>(self.shape()), dtype, dev);
}

Tensor randn_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return randn_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype, dev);
}

Tensor empty_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return empty_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype, dev, false);
}

Tensor zeros_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return zeros_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype, dev, false);
}

Tensor ones_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return ones_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype, dev, false);
}

Tensor full_like_kernel(const Tensor& self, Scalar fill_value, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return full_kernel(static_cast<std::vector<int64_t>>(self.shape()), fill_value, dtype, dev, false);
}


Tensor& zero_kernel(Tensor& self) {
    return fill_kernel(self, 0);
}

// --- Stub-ABI adapters ------------------------------------------------------
// The dispatcher invokes kernels with schema-level argument types
// (std::optional<DType> / std::optional<Device>), while the raw kernels above
// keep concrete parameters for internal reuse.  These thin wrappers bridge the
namespace {

Device resolve_factory_device(const std::optional<Device>& device) {
    return device.has_value() ? *device : Device(globalContext().defaultDevice());
}

DType resolve_factory_dtype(const std::optional<DType>& dtype) {
    return (dtype.has_value() && *dtype != DType::Undefined)
               ? *dtype
               : globalContext().defaultDType();
}

Tensor rand_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                 std::optional<Device> device) {
    return rand_kernel(size, resolve_factory_dtype(dtype),
                       resolve_factory_device(device));
}

Tensor randn_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                  std::optional<Device> device) {
    return randn_kernel(size, resolve_factory_dtype(dtype),
                        resolve_factory_device(device));
}

Tensor randn_generator_stub(const std::vector<int64_t>& size,
                            std::optional<Generator> generator,
                            std::optional<DType> dtype,
                            std::optional<int64_t> layout,
                            std::optional<Device> device,
                            std::optional<bool> pin_memory) {
    if (layout.has_value() && *layout != 2) {
        TP_THROW(NotImplementedError,
                 "randn is only implemented for strided (dense) layout tensors");
    }
    Generator* generator_ptr = generator.has_value() ? &*generator : nullptr;
    return randn_kernel(size, resolve_factory_dtype(dtype),
                        resolve_factory_device(device), generator_ptr,
                        pin_memory.value_or(false));
}

Tensor randint_stub(int64_t low, int64_t high, const std::vector<int64_t>& size,
                    DType dtype, std::optional<Device> device) {
    return randint_kernel(low, high, size, dtype, resolve_factory_device(device));
}

Tensor randperm_stub(int64_t n, DType dtype, std::optional<Device> device) {
    return randperm_kernel(n, dtype, resolve_factory_device(device));
}

Tensor eye_stub(int64_t n, int64_t m, DType dtype, std::optional<Device> device) {
    return eye_kernel(n, m, dtype, resolve_factory_device(device));
}

Tensor arange_start_step_stub(Scalar start, Scalar end, Scalar step, DType dtype,
                              std::optional<Device> device) {
    return arange_start_step_kernel(start, end, step, dtype,
                                    resolve_factory_device(device));
}

Tensor arange_end_stub(Scalar end, DType dtype, std::optional<Device> device) {
    return arange_kernel(end, dtype, resolve_factory_device(device));
}

Tensor linspace_stub(Scalar start, Scalar end, int64_t steps, DType dtype,
                     std::optional<Device> device) {
    return linspace_kernel(start, end, steps, dtype, resolve_factory_device(device));
}

Tensor logspace_stub(Scalar start, Scalar end, int64_t steps, double base,
                     DType dtype, std::optional<Device> device) {
    return logspace_kernel(start, end, steps, base, dtype,
                           resolve_factory_device(device));
}

Tensor empty_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                  std::optional<Device> device, bool pin_memory) {
    return empty_kernel(size, resolve_factory_dtype(dtype),
                        resolve_factory_device(device), pin_memory);
}

Tensor zeros_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                  std::optional<Device> device, bool pin_memory) {
    return zeros_kernel(size, resolve_factory_dtype(dtype),
                        resolve_factory_device(device), pin_memory);
}

Tensor ones_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                 std::optional<Device> device, bool pin_memory) {
    return ones_kernel(size, resolve_factory_dtype(dtype),
                       resolve_factory_device(device), pin_memory);
}

Tensor full_stub(const std::vector<int64_t>& size, Scalar fill_value,
                 DType dtype, std::optional<Device> device, bool pin_memory) {
    return full_kernel(size, fill_value, dtype, resolve_factory_device(device),
                       pin_memory);
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(CPU, FactoryKernels) {
    m.impl("rand", rand_stub);
    m.impl("zeros", zeros_stub);
    m.impl("ones", ones_stub);
    m.impl("full", full_stub);
    m.impl("arange", arange_start_step_stub);
    m.impl("arange.end", arange_end_stub);
    m.impl("empty", empty_stub);
    m.impl("eye", eye_stub);
    m.impl("linspace", linspace_stub);
    m.impl("logspace", logspace_stub);
    m.impl("fill_.Scalar", fill_kernel);
    m.impl("zero_", zero_kernel);
    m.impl("randn", randn_stub);
    m.impl("randn.generator", randn_generator_stub);
    m.impl("randint", randint_stub);
    m.impl("randperm", randperm_stub);
    m.impl("rand_like", rand_like_kernel);
    m.impl("randint_like", randint_like_kernel);
    m.impl("randn_like", randn_like_kernel);
    m.impl("empty_like", empty_like_kernel);
    m.impl("zeros_like", zeros_like_kernel);
    m.impl("ones_like", ones_like_kernel);
    m.impl("full_like", full_like_kernel);
}

} // namespace cpu
} // namespace tensorplay
