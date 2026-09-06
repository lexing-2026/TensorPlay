// CPU kernels: range-factory out-variants (arange / linspace / logspace),
// eye out-variants, complex/polar out-variants, memory-format aware empty,
// the quantized-storage factory family, from_file, and the assert /
// dep-token kernels that need CPU-side semantics.
//
// Range-factory formulas:
//   arange:    length = ceil((end - start) / step); when the output dtype is
//              Int64 and every bound is an integer the length is computed with
//              exact integer arithmetic, ceil((end - start + step - sign(step))
//              / step), so no rounding error can appear for large ranges.
//   linspace:  element i < steps/2 is start + i*step, otherwise
//              end - (steps - i - 1)*step; both endpoints are exact.
//   logspace:  same placement, each element raised to `base`.
// All real-valued fills compute in double and cast to the storage dtype.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Context.h"
#include "Utils.h"
#include "DType.h"
#include "MemoryFormat.h"
#include "Quantizer.h"
#include "Complex.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tensorplay {
namespace composite {
// Defined in composite/TensorCompare.cpp: single-element truthiness.
bool is_nonzero_native(const Tensor& self);
}  // namespace composite

namespace cpu {

namespace ops = tensorplay::tpx::ops;

// Defined in FactoryKernels.cpp: uninitialized allocation with pin support.
Tensor allocate_cpu_tensor(const std::vector<int64_t>& size, DType dtype, bool pin_memory);

namespace {

// Layout encoding: 5 == strided (dense); sparse layouts are not supported by
// the dense factories below.
void require_strided_layout(const char* op_name, const std::optional<int64_t>& layout) {
    if (layout.has_value() && *layout != 5) {
        TP_THROW(NotImplementedError, std::string(op_name) +
                 " is only implemented for strided (dense) layout tensors");
    }
}

DType resolve_dtype_or_default(const std::optional<DType>& dtype, DType fallback) {
    return (dtype.has_value() && *dtype != DType::Undefined) ? *dtype : fallback;
}

DType require_explicit_dtype(const std::optional<DType>& dtype) {
    if (!dtype.has_value() || *dtype == DType::Undefined) {
        TP_THROW(RuntimeError, "Must provide data type for Tensor creation functions.");
    }
    return *dtype;
}

void check_size_nonnegative(const std::vector<int64_t>& size) {
    for (const int64_t s : size) {
        if (s < 0) {
            TP_THROW(RuntimeError,
                     "Trying to create tensor with negative dimension ", s);
        }
    }
}

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

// Uninitialized allocation, with channels-last layouts realized as an
// as_strided view over the flat buffer (the buffer is uninitialized, so any
// dense stride interpretation of it is a valid empty tensor of that format).
Tensor empty_raw(const std::vector<int64_t>& size, DType dtype,
                 const std::optional<Device>& device, bool pin_memory,
                 const std::optional<int64_t>& memory_format) {
    check_size_nonnegative(size);
    const Device dev = device.value_or(globalContext().defaultDevice());
    Tensor result = allocate_cpu_tensor(size, dtype, pin_memory);
    const MemoryFormat mf = static_cast<MemoryFormat>(
        memory_format.value_or(static_cast<int64_t>(MemoryFormat::Contiguous)));
    if (mf == MemoryFormat::ChannelsLast || mf == MemoryFormat::ChannelsLast3d) {
        result = result.as_strided(size, get_channels_last_strides(size), 0);
    }
    return result;
}

// ---------------------------------------------------------------------------
// arange
// ---------------------------------------------------------------------------

template <typename scalar_t>
void arange_fill_data(scalar_t* data, int64_t size, const Scalar& start,
                      const Scalar& step, bool exact_integer) {
    if (exact_integer) {
        const int64_t s = start.to<int64_t>();
        const int64_t st = step.to<int64_t>();
        for (int64_t i = 0; i < size; ++i) {
            data[i] = static_cast<scalar_t>(s + i * st);
        }
    } else {
        const double s = start.toDouble();
        const double st = step.toDouble();
        for (int64_t i = 0; i < size; ++i) {
            data[i] = static_cast<scalar_t>(s + static_cast<double>(i) * st);
        }
    }
}

Tensor& arange_out_kernel(Scalar start, Scalar end, Scalar step, Tensor& result) {
    const double dstart = start.toDouble();
    const double dend = end.toDouble();
    const double dstep = step.toDouble();
    if (!(dstep > 0 || dstep < 0)) {
        TP_THROW(RuntimeError, "step must be nonzero");
    }
    if (!(std::isfinite(dstart) && std::isfinite(dend))) {
        TP_THROW(RuntimeError, "unsupported range: ", dstart, " -> ", dend);
    }
    if (!((dstep > 0 && dend >= dstart) || (dstep < 0 && dend <= dstart))) {
        TP_THROW(RuntimeError, "upper bound and lower bound inconsistent with step sign");
    }

    const bool exact_integer = result.dtype() == DType::Int64 &&
                               start.isIntegral(false) && end.isIntegral(false) &&
                               step.isIntegral(false);
    int64_t size;
    if (exact_integer) {
        const int64_t s = start.to<int64_t>();
        const int64_t e = end.to<int64_t>();
        const int64_t st = step.to<int64_t>();
        const int64_t sgn = (st > 0) - (st < 0);
        size = (e - s + st - sgn) / st;
    } else {
        const double size_d = std::ceil((dend - dstart) / dstep);
        if (!(size_d >= 0 &&
              size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()))) {
            TP_THROW(RuntimeError, "invalid size, possible overflow?");
        }
        size = static_cast<int64_t>(size_d);
    }

    const int64_t numel = result.numel();
    if (numel != size) {
        if (numel > 0) {
            TP_WARN("The number of elements in the out tensor of shape ",
                    result.shape().toString(), " is ", numel,
                    " which does not match the computed number of elements ", size,
                    ". Note that this may occur as a result of rounding error. ",
                    "The out tensor will be resized to a tensor of shape (", size, ",).");
        }
        ops::resize_(result, {size});
    }
    if (size == 0) {
        return result;
    }

    switch (result.dtype()) {
        case DType::UInt8:
            arange_fill_data(result.data_ptr<uint8_t>(), size, start, step, exact_integer);
            break;
        case DType::Int8:
            arange_fill_data(result.data_ptr<int8_t>(), size, start, step, exact_integer);
            break;
        case DType::Int16:
            arange_fill_data(result.data_ptr<int16_t>(), size, start, step, exact_integer);
            break;
        case DType::Int32:
            arange_fill_data(result.data_ptr<int32_t>(), size, start, step, exact_integer);
            break;
        case DType::Int64:
            arange_fill_data(result.data_ptr<int64_t>(), size, start, step, exact_integer);
            break;
        case DType::Float32:
            arange_fill_data(result.data_ptr<float>(), size, start, step, exact_integer);
            break;
        case DType::Float64:
            arange_fill_data(result.data_ptr<double>(), size, start, step, exact_integer);
            break;
        case DType::Float16:
            arange_fill_data(result.data_ptr<Half>(), size, start, step, exact_integer);
            break;
        case DType::BFloat16:
            arange_fill_data(result.data_ptr<BFloat16>(), size, start, step, exact_integer);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "\"arange\" not implemented for '", toString(result.dtype()), "'");
    }
    return result;
}

Tensor arange_start_step_options_kernel(Scalar start, Scalar end, Scalar step,
                                        std::optional<DType> dtype,
                                        std::optional<int64_t> layout,
                                        std::optional<Device> device,
                                        std::optional<bool> pin_memory) {
    require_strided_layout("arange", layout);
    const bool all_integral =
        start.isIntegral(true) && end.isIntegral(true) && step.isIntegral(true);
    DType dt;
    if (dtype.has_value() && *dtype != DType::Undefined) {
        dt = *dtype;
    } else if (all_integral) {
        dt = DType::Int64;
    } else {
        dt = globalContext().defaultDType();
    }
    Tensor result = allocate_cpu_tensor({0}, dt, pin_memory.value_or(false));
    return arange_out_kernel(start, end, step, result);
}

// Overload wrappers over the shared implementation above.
Tensor arange_start_step_options_kernel_start(Scalar start, Scalar end,
                                              std::optional<DType> dtype,
                                              std::optional<int64_t> layout,
                                              std::optional<Device> device,
                                              std::optional<bool> pin_memory) {
    return arange_start_step_options_kernel(start, end, Scalar(static_cast<int64_t>(1)),
                                            dtype, layout, device, pin_memory);
}

Tensor& arange_out_end_kernel(Scalar end, Tensor& out) {
    return arange_out_kernel(Scalar(static_cast<int64_t>(0)), end,
                             Scalar(static_cast<int64_t>(1)), out);
}

// ---------------------------------------------------------------------------
// linspace / logspace (out-variant fills)
// ---------------------------------------------------------------------------

template <typename store_t>
void linspace_fill_real(store_t* data, const Scalar& start, const Scalar& end,
                        int64_t steps) {
    const double start_v = start.toDouble();
    const double end_v = end.toDouble();
    const double step = (end_v - start_v) / static_cast<double>(steps - 1);
    const int64_t halfway = steps / 2;
    for (int64_t i = 0; i < steps; ++i) {
        data[i] = static_cast<store_t>(
            i < halfway ? start_v + step * static_cast<double>(i)
                        : end_v - step * static_cast<double>(steps - i - 1));
    }
}

template <typename compute_t, typename store_t>
void linspace_fill_complex(store_t* data, const Scalar& start, const Scalar& end,
                           int64_t steps) {
    const compute_t start_v = start.to<compute_t>();
    const compute_t end_v = end.to<compute_t>();
    const compute_t step = (end_v - start_v) / static_cast<compute_t>(steps - 1);
    const int64_t halfway = steps / 2;
    for (int64_t i = 0; i < steps; ++i) {
        const int64_t k = i < halfway ? i : steps - i - 1;
        const compute_t base = i < halfway ? start_v : end_v;
        const compute_t value = i < halfway ? base + step * static_cast<compute_t>(static_cast<float>(k))
                                            : base - step * static_cast<compute_t>(static_cast<float>(k));
        data[i] = static_cast<store_t>(value);
    }
}

void linspace_fill_dispatch(Tensor& r, const Scalar& start, const Scalar& end,
                            int64_t steps) {
    switch (r.dtype()) {
        case DType::UInt8:
            linspace_fill_real(r.data_ptr<uint8_t>(), start, end, steps);
            break;
        case DType::Int8:
            linspace_fill_real(r.data_ptr<int8_t>(), start, end, steps);
            break;
        case DType::Int16:
            linspace_fill_real(r.data_ptr<int16_t>(), start, end, steps);
            break;
        case DType::Int32:
            linspace_fill_real(r.data_ptr<int32_t>(), start, end, steps);
            break;
        case DType::Int64:
            linspace_fill_real(r.data_ptr<int64_t>(), start, end, steps);
            break;
        case DType::Float32:
            linspace_fill_real(r.data_ptr<float>(), start, end, steps);
            break;
        case DType::Float64:
            linspace_fill_real(r.data_ptr<double>(), start, end, steps);
            break;
        case DType::Float16:
            linspace_fill_real(r.data_ptr<Half>(), start, end, steps);
            break;
        case DType::BFloat16:
            linspace_fill_real(r.data_ptr<BFloat16>(), start, end, steps);
            break;
        case DType::ComplexHalf:
            linspace_fill_complex<complex<float>, complex<Half>>(
                r.data_ptr<complex<Half>>(), start, end, steps);
            break;
        case DType::ComplexFloat:
            linspace_fill_complex<complex<float>, complex<float>>(
                r.data_ptr<complex<float>>(), start, end, steps);
            break;
        case DType::ComplexDouble:
            linspace_fill_complex<complex<double>, complex<double>>(
                r.data_ptr<complex<double>>(), start, end, steps);
            break;
        case DType::BComplex32:
            linspace_fill_complex<complex<float>, complex<BFloat16>>(
                r.data_ptr<complex<BFloat16>>(), start, end, steps);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "\"linspace\" not implemented for '", toString(r.dtype()), "'");
    }
}

Tensor& linspace_out_scalar_kernel(Scalar start, Scalar end, int64_t steps, Tensor& result) {
    if (steps < 0) {
        TP_THROW(RuntimeError, "number of steps must be non-negative");
    }
    if (result.numel() != steps) {
        ops::resize_(result, {steps});
    }
    if (steps == 0) {
        return result;
    }
    if (steps == 1) {
        ops::fill_(result, start);
        return result;
    }
    Tensor r = result.is_contiguous() ? result : ops::contiguous(result);
    linspace_fill_dispatch(r, start, end, steps);
    if (!result.is_contiguous()) {
        ops::copy_(result, r);
    }
    return result;
}

template <typename store_t>
void logspace_fill_real(store_t* data, const Scalar& start, const Scalar& end,
                        int64_t steps, double base) {
    const double start_v = start.toDouble();
    const double end_v = end.toDouble();
    const double step = (end_v - start_v) / static_cast<double>(steps - 1);
    const int64_t halfway = steps / 2;
    for (int64_t i = 0; i < steps; ++i) {
        data[i] = static_cast<store_t>(
            i < halfway ? std::pow(base, start_v + step * static_cast<double>(i))
                        : std::pow(base, end_v - step * static_cast<double>(steps - i - 1)));
    }
}

template <typename compute_t, typename store_t>
void logspace_fill_complex(store_t* data, const Scalar& start, const Scalar& end,
                           int64_t steps, double base) {
    const compute_t scalar_base = static_cast<compute_t>(base);
    const compute_t scalar_start = start.to<compute_t>();
    const compute_t scalar_end = end.to<compute_t>();
    const compute_t step =
        (scalar_end - scalar_start) / static_cast<compute_t>(steps - 1);
    const int64_t halfway = steps / 2;
    for (int64_t i = 0; i < steps; ++i) {
        const int64_t k = i < halfway ? i : steps - i - 1;
        const compute_t v = i < halfway
            ? scalar_start + step * static_cast<compute_t>(static_cast<float>(k))
            : scalar_end - step * static_cast<compute_t>(static_cast<float>(k));
        data[i] = static_cast<store_t>(tensorplay::pow(scalar_base, v));
    }
}

void logspace_fill_dispatch(Tensor& r, const Scalar& start, const Scalar& end,
                            int64_t steps, double base) {
    switch (r.dtype()) {
        case DType::UInt8:
            logspace_fill_real(r.data_ptr<uint8_t>(), start, end, steps, base);
            break;
        case DType::Int8:
            logspace_fill_real(r.data_ptr<int8_t>(), start, end, steps, base);
            break;
        case DType::Int16:
            logspace_fill_real(r.data_ptr<int16_t>(), start, end, steps, base);
            break;
        case DType::Int32:
            logspace_fill_real(r.data_ptr<int32_t>(), start, end, steps, base);
            break;
        case DType::Int64:
            logspace_fill_real(r.data_ptr<int64_t>(), start, end, steps, base);
            break;
        case DType::Float32:
            logspace_fill_real(r.data_ptr<float>(), start, end, steps, base);
            break;
        case DType::Float64:
            logspace_fill_real(r.data_ptr<double>(), start, end, steps, base);
            break;
        case DType::Float16:
            logspace_fill_real(r.data_ptr<Half>(), start, end, steps, base);
            break;
        case DType::BFloat16:
            logspace_fill_real(r.data_ptr<BFloat16>(), start, end, steps, base);
            break;
        case DType::ComplexHalf:
            logspace_fill_complex<complex<float>, complex<Half>>(
                r.data_ptr<complex<Half>>(), start, end, steps, base);
            break;
        case DType::ComplexFloat:
            logspace_fill_complex<complex<float>, complex<float>>(
                r.data_ptr<complex<float>>(), start, end, steps, base);
            break;
        case DType::ComplexDouble:
            logspace_fill_complex<complex<double>, complex<double>>(
                r.data_ptr<complex<double>>(), start, end, steps, base);
            break;
        case DType::BComplex32:
            logspace_fill_complex<complex<float>, complex<BFloat16>>(
                r.data_ptr<complex<BFloat16>>(), start, end, steps, base);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "\"logspace\" not implemented for '", toString(r.dtype()), "'");
    }
}

Tensor& logspace_out_scalar_kernel(Scalar start, Scalar end, int64_t steps,
                                   double base, Tensor& result) {
    if (steps < 0) {
        TP_THROW(RuntimeError, "number of steps must be non-negative");
    }
    if (result.numel() != steps) {
        ops::resize_(result, {steps});
    }

    Tensor r = result.is_contiguous() ? result : ops::contiguous(result);

    if (steps == 0) {
        // nothing to fill
    } else if (steps == 1) {
        if (isComplexType(r.dtype())) {
            ops::fill_(r, Scalar(tensorplay::pow(base, start.to<complex<double>>())));
        } else {
            ops::fill_(r, Scalar(std::pow(base, start.toDouble())));
        }
    } else {
        logspace_fill_dispatch(r, start, end, steps, base);
    }

    if (!result.is_contiguous()) {
        ops::copy_(result, r);
    }
    return result;
}

// Tensor-argument out variants: the 0-dim bounds are unwrapped to scalars and
// forwarded to the scalar fill kernel.
Tensor& linspace_out_tt_kernel(const Tensor& start, const Tensor& end,
                               int64_t steps, Tensor& out) {
    if (!(start.dim() == 0 && end.dim() == 0)) {
        TP_THROW(RuntimeError,
                 "linspace only supports 0-dimensional start and end tensors, "
                 "but got start with ", start.dim(), " dimension(s) and end with ",
                 end.dim(), " dimension(s).");
    }
    return linspace_out_scalar_kernel(start.item(), end.item(), steps, out);
}

Tensor& linspace_out_ts_kernel(const Tensor& start, Scalar end, int64_t steps, Tensor& out) {
    if (start.dim() != 0) {
        TP_THROW(RuntimeError,
                 "linspace only supports 0-dimensional start and end tensors, "
                 "but got start with ", start.dim(), " dimension(s).");
    }
    return linspace_out_scalar_kernel(start.item(), end, steps, out);
}

Tensor& linspace_out_st_kernel(Scalar start, const Tensor& end, int64_t steps, Tensor& out) {
    if (end.dim() != 0) {
        TP_THROW(RuntimeError,
                 "linspace only supports 0-dimensional start and end tensors, "
                 "but got end with ", end.dim(), " dimension(s).");
    }
    return linspace_out_scalar_kernel(start, end.item(), steps, out);
}

Tensor& logspace_out_tt_kernel(const Tensor& start, const Tensor& end,
                               int64_t steps, double base, Tensor& out) {
    if (!(start.dim() == 0 && end.dim() == 0)) {
        TP_THROW(RuntimeError,
                 "logspace only supports 0-dimensional start and end tensors, "
                 "but got start with ", start.dim(), " dimension(s) and end with ",
                 end.dim(), " dimension(s).");
    }
    return logspace_out_scalar_kernel(start.item(), end.item(), steps, base, out);
}

Tensor& logspace_out_ts_kernel(const Tensor& start, Scalar end, int64_t steps,
                               double base, Tensor& out) {
    if (start.dim() != 0) {
        TP_THROW(RuntimeError,
                 "logspace only supports 0-dimensional start and end tensors, "
                 "but got start with ", start.dim(), " dimension(s).");
    }
    return logspace_out_scalar_kernel(start.item(), end, steps, base, out);
}

Tensor& logspace_out_st_kernel(Scalar start, const Tensor& end, int64_t steps,
                               double base, Tensor& out) {
    if (end.dim() != 0) {
        TP_THROW(RuntimeError,
                 "logspace only supports 0-dimensional start and end tensors, "
                 "but got end with ", end.dim(), " dimension(s).");
    }
    return logspace_out_scalar_kernel(start, end.item(), steps, base, out);
}

// ---------------------------------------------------------------------------
// eye
// ---------------------------------------------------------------------------

Tensor& eye_out_full_kernel(int64_t n, int64_t m, Tensor& result) {
    if (n < 0) {
        TP_THROW(RuntimeError, "n must be greater or equal to 0, got ", n);
    }
    if (m < 0) {
        TP_THROW(RuntimeError, "m must be greater or equal to 0, got ", m);
    }
    ops::resize_(result, {n, m});
    ops::zero_(result);

    const int64_t sz = std::min<int64_t>(n, m);
    const std::vector<int64_t> strides = result.strides();
    const int64_t diag_stride = strides[0] + strides[1];

#define TP_EYE_FILL_CASE(ctype, name)                                        \
    case DType::name: {                                                      \
        ctype* result_data = result.data_ptr<ctype>();                       \
        for (int64_t i = 0; i < sz; ++i) {                                   \
            result_data[i * diag_stride] = static_cast<ctype>(1);            \
        }                                                                    \
        break;                                                               \
    }
    switch (result.dtype()) {
        TP_EYE_FILL_CASE(uint8_t, UInt8)
        TP_EYE_FILL_CASE(int8_t, Int8)
        TP_EYE_FILL_CASE(int16_t, Int16)
        TP_EYE_FILL_CASE(int32_t, Int32)
        TP_EYE_FILL_CASE(int64_t, Int64)
        TP_EYE_FILL_CASE(uint16_t, UInt16)
        TP_EYE_FILL_CASE(uint32_t, UInt32)
        TP_EYE_FILL_CASE(uint64_t, UInt64)
        TP_EYE_FILL_CASE(float, Float32)
        TP_EYE_FILL_CASE(double, Float64)
        TP_EYE_FILL_CASE(tensorplay::Half, Float16)
        TP_EYE_FILL_CASE(tensorplay::BFloat16, BFloat16)
        TP_EYE_FILL_CASE(bool, Bool)
        TP_EYE_FILL_CASE(complex<tensorplay::Half>, ComplexHalf)
        TP_EYE_FILL_CASE(complex<float>, ComplexFloat)
        TP_EYE_FILL_CASE(complex<double>, ComplexDouble)
        TP_EYE_FILL_CASE(complex<tensorplay::BFloat16>, BComplex32)
        default:
            TP_THROW(NotImplementedError,
                     "\"eye\" not implemented for '", toString(result.dtype()), "'");
    }
#undef TP_EYE_FILL_CASE
    return result;
}

Tensor eye_m_kernel(int64_t n, int64_t m, std::optional<DType> dtype,
                    std::optional<int64_t> layout,
                    std::optional<Device> device,
                    std::optional<bool> pin_memory) {
    require_strided_layout("eye", layout);
    const DType dt = resolve_dtype_or_default(dtype, globalContext().defaultDType());
    Tensor tensor = allocate_cpu_tensor({0}, dt, pin_memory.value_or(false));
    return eye_out_full_kernel(n, m, tensor);
}

Tensor& eye_out_n_kernel(int64_t n, Tensor& out) {
    return eye_out_full_kernel(n, n, out);
}

// ---------------------------------------------------------------------------
// complex / polar out-variants
// ---------------------------------------------------------------------------

// The inputs must be Half/Float/Double of equal dtype; the output dtype is the
// matching complex type.
void complex_check_floating(const Tensor& a, const Tensor& b) {
    const auto is_real_float = [](DType t) {
        return t == DType::Float32 || t == DType::Float64 || t == DType::Float16;
    };
    if (!(is_real_float(a.dtype()) && is_real_float(b.dtype()))) {
        TP_THROW(NotImplementedError,
                 "Expected both inputs to be Half, Float or Double tensors but got ",
                 toString(a.dtype()), " and ", toString(b.dtype()));
    }
}

void complex_check_dtype(const Tensor& result, const Tensor& a, const Tensor& b) {
    complex_check_floating(a, b);
    if (a.dtype() != b.dtype()) {
        TP_THROW(RuntimeError,
                 "Expected object of scalar type ", toString(a.dtype()),
                 " but got scalar type ", toString(b.dtype()), " for second argument");
    }
    if (result.dtype() != toComplexType(a.dtype())) {
        TP_THROW(RuntimeError,
                 "Expected object of scalar type ", toString(toComplexType(a.dtype())),
                 " but got scalar type ", toString(result.dtype()), " for argument 'out'");
    }
}

Tensor& complex_out_kernel(const Tensor& real, const Tensor& imag, Tensor& result) {
    complex_check_dtype(result, real, imag);
    const std::vector<int64_t> shape = broadcast_shapes(shape_of(real), shape_of(imag));
    Tensor rc = ops::expand(real, shape).contiguous();
    Tensor ic = ops::expand(imag, shape).contiguous();
    ops::resize_(result, shape);

    const int64_t n = result.numel();
    switch (real.dtype()) {
        case DType::Float16: {
            const Half* rp = rc.data_ptr<Half>();
            const Half* ip = ic.data_ptr<Half>();
            complex<Half>* dp = result.data_ptr<complex<Half>>();
            for (int64_t i = 0; i < n; ++i) dp[i] = complex<Half>(rp[i], ip[i]);
            break;
        }
        case DType::Float32: {
            const float* rp = rc.data_ptr<float>();
            const float* ip = ic.data_ptr<float>();
            complex<float>* dp = result.data_ptr<complex<float>>();
            for (int64_t i = 0; i < n; ++i) dp[i] = complex<float>(rp[i], ip[i]);
            break;
        }
        case DType::Float64: {
            const double* rp = rc.data_ptr<double>();
            const double* ip = ic.data_ptr<double>();
            complex<double>* dp = result.data_ptr<complex<double>>();
            for (int64_t i = 0; i < n; ++i) dp[i] = complex<double>(rp[i], ip[i]);
            break;
        }
        default:
            TP_THROW(NotImplementedError,
                     "\"complex\" not implemented for '", toString(real.dtype()), "'");
    }
    return result;
}

Tensor& polar_out_kernel(const Tensor& abs, const Tensor& angle, Tensor& result) {
    complex_check_dtype(result, abs, angle);
    const std::vector<int64_t> shape = broadcast_shapes(shape_of(abs), shape_of(angle));
    Tensor ac = ops::expand(abs, shape).contiguous();
    Tensor thc = ops::expand(angle, shape).contiguous();
    ops::resize_(result, shape);

    const int64_t n = result.numel();
    switch (abs.dtype()) {
        case DType::Float16: {
            const Half* ap = ac.data_ptr<Half>();
            const Half* tp = thc.data_ptr<Half>();
            complex<Half>* dp = result.data_ptr<complex<Half>>();
            for (int64_t i = 0; i < n; ++i) {
                const complex<float> v =
                    tensorplay::polar(static_cast<float>(ap[i]), static_cast<float>(tp[i]));
                dp[i] = complex<Half>(v.real(), v.imag());
            }
            break;
        }
        case DType::Float32: {
            const float* ap = ac.data_ptr<float>();
            const float* tp = thc.data_ptr<float>();
            complex<float>* dp = result.data_ptr<complex<float>>();
            for (int64_t i = 0; i < n; ++i) dp[i] = tensorplay::polar(ap[i], tp[i]);
            break;
        }
        case DType::Float64: {
            const double* ap = ac.data_ptr<double>();
            const double* tp = thc.data_ptr<double>();
            complex<double>* dp = result.data_ptr<complex<double>>();
            for (int64_t i = 0; i < n; ++i) dp[i] = tensorplay::polar(ap[i], tp[i]);
            break;
        }
        default:
            TP_THROW(NotImplementedError,
                     "\"polar\" not implemented for '", toString(abs.dtype()), "'");
    }
    return result;
}

// ---------------------------------------------------------------------------
// empty with memory format
// ---------------------------------------------------------------------------

Tensor empty_memory_format_kernel(const std::vector<int64_t>& size,
                                  std::optional<DType> dtype,
                                  std::optional<int64_t> layout,
                                  std::optional<Device> device,
                                  std::optional<bool> pin_memory,
                                  std::optional<int64_t> memory_format) {
    require_strided_layout("empty", layout);
    const DType dt = resolve_dtype_or_default(dtype, globalContext().defaultDType());
    return empty_raw(size, dt, device, pin_memory.value_or(false), memory_format);
}

// ---------------------------------------------------------------------------
// quantized-storage factories
// ---------------------------------------------------------------------------
// Quantized factories return tensors of a quantized dtype carrying an
// attached quantizer: the affine parameters live on the tensor and are read
// back through q_scale()/q_per_channel_scales()/dequantize().

Tensor empty_affine_quantized_kernel(const std::vector<int64_t>& size,
                                     std::optional<DType> dtype,
                                     std::optional<int64_t> layout,
                                     std::optional<Device> device,
                                     std::optional<bool> pin_memory,
                                     double scale, int64_t zero_point,
                                     std::optional<int64_t> memory_format) {
    require_strided_layout("_empty_affine_quantized", layout);
    const DType dt = require_explicit_dtype(dtype);
    if (!isQuantizedType(dt)) {
        TP_THROW(TypeError,
                 "_empty_affine_quantized(): dtype must be a quantized "
                 "dtype, got ", toString(dt));
    }
    Tensor out = empty_raw(size, dt, device, pin_memory.value_or(false), memory_format);
    out.impl()->set_quantizer(
        make_per_tensor_affine_quantizer(scale, zero_point, dt));
    return out;
}

Tensor empty_per_channel_affine_quantized_kernel(
    const std::vector<int64_t>& size, const Tensor& scales, const Tensor& zero_points,
    int64_t axis, std::optional<DType> dtype, std::optional<int64_t> layout,
    std::optional<Device> device, std::optional<bool> pin_memory,
    std::optional<int64_t> memory_format) {
    require_strided_layout("_empty_per_channel_affine_quantized", layout);
    const DType dt = require_explicit_dtype(dtype);
    if (!isQuantizedType(dt)) {
        TP_THROW(TypeError,
                 "_empty_per_channel_affine_quantized(): dtype must be a "
                 "quantized dtype, got ", toString(dt));
    }
    if (scales.dim() != 1) {
        TP_THROW(ValueError, "per-channel quantized empty tensor requires 1-D scales, got ",
                 scales.dim(), " dimension(s)");
    }
    if (zero_points.dim() != 1) {
        TP_THROW(ValueError, "per-channel quantized empty tensor requires 1-D zero_points, got ",
                 zero_points.dim(), " dimension(s)");
    }
    if (scales.numel() != zero_points.numel()) {
        TP_THROW(ValueError, "scales and zero_points must have the same number of elements");
    }
    if (axis < 0) axis += static_cast<int64_t>(size.size());
    if (axis < 0 || axis >= static_cast<int64_t>(size.size())) {
        TP_THROW(ValueError, "axis must be between 0 and number of dimensions, got ", axis);
    }
    if (scales.size(0) != size[static_cast<size_t>(axis)]) {
        TP_THROW(ValueError,
                 "per-channel quantized empty tensor requires one qparam per channel");
    }
    const Device target = device.value_or(Device(DeviceType::CPU));
    if (scales.device() != target || zero_points.device() != target) {
        TP_THROW(RuntimeError,
                 "per-channel quantization parameters must be on the output device");
    }
    Tensor out = empty_raw(size, dt, device, pin_memory.value_or(false), memory_format);
    out.impl()->set_quantizer(
        make_per_channel_affine_quantizer(scales, zero_points, axis, dt));
    return out;
}

Tensor empty_quantized_kernel(const std::vector<int64_t>& size, const Tensor& qtensor,
                              std::optional<DType> dtype,
                              std::optional<int64_t> layout,
                              std::optional<Device> device,
                              std::optional<bool> pin_memory,
                              std::optional<int64_t> memory_format) {
    require_strided_layout("empty_quantized", layout);
    quantized::require_quantized(qtensor, "empty_quantized");
    const DType dt = resolve_dtype_or_default(dtype, qtensor.dtype());
    if (dt != qtensor.dtype()) {
        TP_THROW(RuntimeError,
                 "empty_quantized(): dtype must match the source quantized tensor");
    }
    Tensor out = empty_raw(size, dt, device, pin_memory.value_or(false), memory_format);
    const QuantizerPtr source_quantizer = quantized::quantizer_of(qtensor);
    switch (source_quantizer->qscheme()) {
        case kPerTensorAffine:
            out.impl()->set_quantizer(make_per_tensor_affine_quantizer(
                source_quantizer->scale(), source_quantizer->zero_point(), dt));
            break;
        case kPerChannelAffine:
        case kPerChannelAffineFloatQParams: {
            const Device target = device.value_or(qtensor.device());
            Tensor scales = source_quantizer->scales();
            Tensor zero_points = source_quantizer->zero_points();
            if (scales.device() != target) scales = scales.to(target);
            if (zero_points.device() != target) zero_points = zero_points.to(target);
            out.impl()->set_quantizer(make_per_channel_affine_quantizer(
                scales, zero_points, source_quantizer->axis(), dt));
            break;
        }
        default:
            TP_THROW(ValueError,
                     "empty_quantized(): unsupported quantization scheme");
    }
    return out;
}

// ---------------------------------------------------------------------------
// from_file
// ---------------------------------------------------------------------------

namespace {

struct MappedRegion {
    void* addr;
    size_t length;
};

void unmap_mapped_region(void* ctx) {
    auto* region = static_cast<MappedRegion*>(ctx);
    if (region != nullptr) {
        if (region->addr != nullptr && region->addr != MAP_FAILED) {
            ::munmap(region->addr, region->length);
        }
        delete region;
    }
}

}  // namespace

Tensor from_file_kernel(std::string filename, std::optional<bool> shared,
                        std::optional<int64_t> size,
                        std::optional<DType> dtype,
                        std::optional<int64_t> layout,
                        std::optional<Device> device,
                        std::optional<bool> pin_memory) {
    (void)device;
    if (pin_memory.has_value() && *pin_memory) {
        TP_THROW(RuntimeError, "tensors constructed from a file cannot be pinned");
    }
    require_strided_layout("from_file", layout);

    const int64_t my_size = size.value_or(0);
    if (my_size < 0) {
        TP_THROW(RuntimeError, "from_file: size must be non-negative, got ", my_size);
    }
    const DType dt = resolve_dtype_or_default(dtype, globalContext().defaultDType());
    const size_t size_bytes = static_cast<size_t>(my_size) * elementSize(dt);
    const bool shared_map = shared.value_or(false);

    if (shared_map) {
        // Memory-map the file so that writes go through to it. The mapping is
        // released when the last tensor referencing the storage is freed.
        if (size_bytes == 0) {
            Storage storage;
            return Tensor(storage, {my_size}, dt);
        }
        const int fd = ::open(filename.c_str(), O_RDWR);
        if (fd < 0) {
            TP_THROW(RuntimeError, "from_file: cannot open file '", filename, "'");
        }
        struct stat st;
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            TP_THROW(RuntimeError, "from_file: cannot stat file '", filename, "'");
        }
        if (static_cast<uint64_t>(st.st_size) < static_cast<uint64_t>(size_bytes)) {
            ::close(fd);
            TP_THROW(RuntimeError,
                     "from_file: file '", filename, "' is smaller (", st.st_size,
                     " bytes) than the requested tensor size (", size_bytes, " bytes)");
        }
        void* addr = ::mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        if (addr == MAP_FAILED) {
            TP_THROW(RuntimeError, "from_file: cannot memory-map file '", filename, "'");
        }
        auto* region = new MappedRegion{addr, size_bytes};
        DataPtr data_ptr(addr, region, &unmap_mapped_region, Device(DeviceType::CPU));
        Storage storage(std::move(data_ptr), size_bytes, nullptr);
        return Tensor(storage, {my_size}, dt);
    }

    // Private copy: allocate CPU storage and read the leading bytes of the file.
    Storage storage(size_bytes);
    if (size_bytes > 0) {
        std::FILE* file = std::fopen(filename.c_str(), "rb");
        if (file == nullptr) {
            TP_THROW(RuntimeError, "from_file: cannot open file '", filename, "'");
        }
        const size_t nread = std::fread(storage.data(), 1, size_bytes, file);
        std::fclose(file);
        if (nread != size_bytes) {
            TP_THROW(RuntimeError,
                     "from_file: file '", filename, "' is smaller than the requested ",
                     "tensor size (read ", nread, " of ", size_bytes, " bytes)");
        }
    }
    return Tensor(storage, {my_size}, dt);
}

// ---------------------------------------------------------------------------
// assert / dep-token kernels
// ---------------------------------------------------------------------------

void assert_async_kernel(const Tensor& self) {
    if (!composite::is_nonzero_native(self)) {
        TP_THROW(RuntimeError, "Expected Tensor with single nonzero value, but got zero");
    }
}

void assert_async_msg_kernel(const Tensor& self, std::string assert_msg) {
    if (!composite::is_nonzero_native(self)) {
        TP_THROW(RuntimeError,
                 !assert_msg.empty() ? assert_msg : std::string("Assertion is failed"));
    }
}

Tensor functional_assert_async_msg_kernel(const Tensor& self, std::string assert_msg,
                                          const Tensor& dep_token) {
    assert_async_msg_kernel(self, assert_msg);
    return ops::clone(dep_token);
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CPU, FactoryToolsOps) {
    m.impl("arange.start", arange_start_step_options_kernel_start);
    m.impl("arange.start_step", arange_start_step_options_kernel);
    m.impl("arange.out", arange_out_end_kernel);
    m.impl("arange.start_out", arange_out_kernel);
    m.impl("linspace.out", linspace_out_scalar_kernel);
    m.impl("linspace.Tensor_Tensor_out", linspace_out_tt_kernel);
    m.impl("linspace.Tensor_Scalar_out", linspace_out_ts_kernel);
    m.impl("linspace.Scalar_Tensor_out", linspace_out_st_kernel);
    m.impl("logspace.out", logspace_out_scalar_kernel);
    m.impl("logspace.Tensor_Tensor_out", logspace_out_tt_kernel);
    m.impl("logspace.Tensor_Scalar_out", logspace_out_ts_kernel);
    m.impl("logspace.Scalar_Tensor_out", logspace_out_st_kernel);
    m.impl("eye.m", eye_m_kernel);
    m.impl("eye.out", eye_out_n_kernel);
    m.impl("eye.m_out", eye_out_full_kernel);
    m.impl("complex.out", complex_out_kernel);
    m.impl("polar.out", polar_out_kernel);
    m.impl("empty.memory_format", empty_memory_format_kernel);
    m.impl("_empty_affine_quantized", empty_affine_quantized_kernel);
    m.impl("_empty_per_channel_affine_quantized", empty_per_channel_affine_quantized_kernel);
    m.impl("empty_quantized", empty_quantized_kernel);
    m.impl("from_file", from_file_kernel);
    m.impl("_assert_async", assert_async_kernel);
    m.impl("_assert_async.msg", assert_async_msg_kernel);
    m.impl("_functional_assert_async.msg", functional_assert_async_msg_kernel);
}

}  // namespace cpu
}  // namespace tensorplay
