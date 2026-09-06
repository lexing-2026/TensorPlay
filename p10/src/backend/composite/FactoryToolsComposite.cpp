// Backend-neutral composite kernels: window functions (bartlett / blackman /
// hann / hamming and their periodic variants), linspace/logspace overloads
// with tensor bounds, the *_out fill wrappers for ones/zeros/full/empty, the
// new_* factories, and small utility ops (metadata asserts, symbolic range
// constraints, dep tokens, internal-overlap debugging, reshape-from-tensor).
//
// Window weight formulas over the index ramp n = 0..L-1 (L incremented by one
// for periodic windows, then truncated back):
//   bartlett  w[n] = 1 - |n/c - 1| with c = (L-1)/2, realized as
//             (2/L') * n on the first half and 2 - (2/L') * n on the second.
//   blackman  w[n] = 0.42 + 0.08*cos(4*pi*n/(L-1)) - 0.5*cos(2*pi*n/(L-1)).
//   hamming   w[n] = alpha - beta*cos(2*pi*n/(L-1)); hann is hamming with
//             alpha = beta = 0.5.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Context.h"
#include "SizesAndStrides.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Layout encoding: 2 == strided (dense); the dense factories below reject the
// sparse layouts (0 = COO, 1 = CSR).
void require_strided_layout(const char* op_name, const std::optional<int64_t>& layout) {
    if (layout.has_value() && *layout != 2) {
        TP_THROW(NotImplementedError, std::string(op_name) +
                 " is only implemented for strided (dense) layout tensors");
    }
}

int64_t layout_of(const Tensor& tensor) {
    if (!tensor.is_sparse()) return 2;
    return tensor.is_sparse_csr() ? 1 : 0;
}

void check_size_nonnegative(const std::vector<int64_t>& size) {
    for (const int64_t s : size) {
        if (s < 0) {
            TP_THROW(RuntimeError,
                     "Trying to create tensor with negative dimension ", s);
        }
    }
}

std::string int_vec_to_string(const std::vector<int64_t>& values) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) os << ", ";
        os << values[i];
    }
    os << "]";
    return os.str();
}

// ---------------------------------------------------------------------------
// window functions
// ---------------------------------------------------------------------------

void window_checks(const char* function_name, DType dtype, int64_t window_length) {
    if (!isFloatingType(dtype) && !isComplexType(dtype)) {
        TP_THROW(RuntimeError, function_name,
                 " expects floating point dtypes, got: ", toString(dtype));
    }
    if (window_length < 0) {
        TP_THROW(RuntimeError, function_name,
                 " requires non-negative window_length, got window_length=",
                 window_length);
    }
}

// Index ramp shared by all windows; periodic windows use L+1 points and are
// truncated afterwards.
Tensor window_ramp(int64_t length, DType dtype, const std::optional<int64_t>& layout,
                   const std::optional<Device>& device,
                   const std::optional<bool>& pin_memory) {
    return ops::arange(Scalar(static_cast<int64_t>(0)), Scalar(length),
                       Scalar(static_cast<int64_t>(1)), dtype, layout, device,
                       pin_memory);
}

Tensor bartlett_window_impl(int64_t window_length, bool periodic,
                            DType dtype, const std::optional<int64_t>& layout,
                            const std::optional<Device>& device,
                            const std::optional<bool>& pin_memory) {
    window_checks("bartlett_window", dtype, window_length);
    require_strided_layout("bartlett_window", layout);
    if (window_length == 0) {
        return ops::empty({0}, dtype, device, pin_memory.value_or(false), false);
    }
    if (window_length == 1) {
        return ops::ones({1}, dtype, device, pin_memory.value_or(false), false);
    }
    int64_t length = window_length;
    if (periodic) {
        length += 1;
    }
    Tensor window = window_ramp(length, dtype, layout, device, pin_memory);
    ops::mul_(window, Scalar(2.0 / static_cast<double>(length - 1)));
    const int64_t first_half_size = ((length - 1) >> 1) + 1;
    Tensor second_half = ops::narrow(window, 0, first_half_size, length - first_half_size);
    ops::mul_(second_half, Scalar(-1.0));
    ops::add_(second_half, Scalar(2.0));
    return periodic ? ops::narrow(window, 0, 0, length - 1) : window;
}

Tensor blackman_window_impl(int64_t window_length, bool periodic,
                            DType dtype, const std::optional<int64_t>& layout,
                            const std::optional<Device>& device,
                            const std::optional<bool>& pin_memory) {
    window_checks("blackman_window", dtype, window_length);
    require_strided_layout("blackman_window", layout);
    if (window_length == 0) {
        return ops::empty({0}, dtype, device, pin_memory.value_or(false), false);
    }
    if (window_length == 1) {
        return ops::ones({1}, dtype, device, pin_memory.value_or(false), false);
    }
    int64_t length = window_length;
    if (periodic) {
        length += 1;
    }
    Tensor window = window_ramp(length, dtype, layout, device, pin_memory);
    ops::mul_(window, Scalar(kPi / static_cast<double>(length - 1)));
    Tensor four = ops::mul(window, Scalar(4.0));
    ops::cos_(four);
    ops::mul_(four, Scalar(0.08));
    Tensor two = ops::mul(window, Scalar(2.0));
    ops::cos_(two);
    ops::mul_(two, Scalar(0.5));
    window = ops::sub(four, two);
    ops::add_(window, Scalar(0.42));
    return periodic ? ops::narrow(window, 0, 0, length - 1) : window;
}

Tensor hamming_window_impl(int64_t window_length, bool periodic, double alpha,
                           double beta, DType dtype,
                           const std::optional<int64_t>& layout,
                           const std::optional<Device>& device,
                           const std::optional<bool>& pin_memory) {
    window_checks("hamming_window", dtype, window_length);
    require_strided_layout("hamming_window", layout);
    if (window_length == 0) {
        return ops::empty({0}, dtype, device, pin_memory.value_or(false), false);
    }
    if (window_length == 1) {
        return ops::ones({1}, dtype, device, pin_memory.value_or(false), false);
    }
    int64_t length = window_length;
    if (periodic) {
        length += 1;
    }
    Tensor window = window_ramp(length, dtype, layout, device, pin_memory);
    ops::mul_(window, Scalar(2.0 * kPi / static_cast<double>(length - 1)));
    ops::cos_(window);
    ops::mul_(window, Scalar(-beta));
    ops::add_(window, Scalar(alpha));
    return periodic ? ops::narrow(window, 0, 0, length - 1) : window;
}

// ---------------------------------------------------------------------------
// linspace / logspace with tensor bounds
// ---------------------------------------------------------------------------

DType infer_linspace_logspace_dtype(const Scalar& start, const Scalar& end,
                                    const std::optional<DType>& dtype,
                                    const char* fn_name) {
    if (start.isComplex() || end.isComplex()) {
        const DType default_complex = globalContext().defaultDType() == DType::Float64
                                          ? DType::ComplexDouble
                                          : DType::ComplexFloat;
        if (dtype.has_value() && *dtype != DType::Undefined) {
            if (!isComplexType(*dtype)) {
                TP_THROW(RuntimeError, fn_name, ": inferred dtype ",
                         toString(default_complex), " can't be safely cast to passed dtype ",
                         toString(*dtype));
            }
            return *dtype;
        }
        return default_complex;
    }
    return (dtype.has_value() && *dtype != DType::Undefined)
               ? *dtype
               : globalContext().defaultDType();
}

Tensor linspace_scalar_impl(const Scalar& start, const Scalar& end, int64_t steps,
                            const std::optional<DType>& dtype,
                            const std::optional<int64_t>& layout,
                            const std::optional<Device>& device,
                            const std::optional<bool>& pin_memory) {
    if (steps < 0) {
        TP_THROW(RuntimeError, "number of steps must be non-negative");
    }
    require_strided_layout("linspace", layout);
    const DType dt = infer_linspace_logspace_dtype(start, end, dtype, "linspace");
    Tensor result = ops::empty({steps}, dt, device, pin_memory.value_or(false), false);
    return ops::linspace(start, end, steps, result);
}

Tensor logspace_scalar_impl(const Scalar& start, const Scalar& end, int64_t steps,
                            double base, const std::optional<DType>& dtype,
                            const std::optional<int64_t>& layout,
                            const std::optional<Device>& device,
                            const std::optional<bool>& pin_memory) {
    if (steps < 0) {
        TP_THROW(RuntimeError, "number of steps must be non-negative");
    }
    require_strided_layout("logspace", layout);
    const DType dt = infer_linspace_logspace_dtype(start, end, dtype, "logspace");
    Tensor result = ops::empty({steps}, dt, device, pin_memory.value_or(false), false);
    return ops::logspace(start, end, steps, base, result);
}

DType infer_arange_dtype(const Scalar& start, const Scalar& end,
                         const Scalar& step,
                         const std::optional<DType>& dtype) {
    if (dtype.has_value() && *dtype != DType::Undefined) {
        return *dtype;
    }
    const bool all_integral =
        start.isIntegral(true) && end.isIntegral(true) && step.isIntegral(true);
    return all_integral ? DType::Int64 : globalContext().defaultDType();
}

Tensor arange_options_impl(const Scalar& start, const Scalar& end,
                           const Scalar& step,
                           const std::optional<DType>& dtype,
                           const std::optional<int64_t>& layout,
                           const std::optional<Device>& device,
                           const std::optional<bool>& pin_memory) {
    require_strided_layout("arange", layout);
    const DType dt = infer_arange_dtype(start, end, step, dtype);
    Tensor result = ops::empty({0}, dt, device, pin_memory.value_or(false), false);
    return ops::arange(start, end, step, result);
}

Tensor eye_options_impl(int64_t n, int64_t m,
                        const std::optional<DType>& dtype,
                        const std::optional<int64_t>& layout,
                        const std::optional<Device>& device,
                        const std::optional<bool>& pin_memory) {
    if (n < 0) {
        TP_THROW(RuntimeError, "n must be greater or equal to 0, got ", n);
    }
    if (m < 0) {
        TP_THROW(RuntimeError, "m must be greater or equal to 0, got ", m);
    }
    require_strided_layout("eye", layout);
    const DType dt = (dtype.has_value() && *dtype != DType::Undefined)
                         ? *dtype
                         : globalContext().defaultDType();
    Tensor result = ops::empty({0}, dt, device, pin_memory.value_or(false), false);
    return ops::eye(n, m, result);
}

// ---------------------------------------------------------------------------
// new_* factories
// ---------------------------------------------------------------------------

Tensor new_empty_impl(const Tensor& self, const std::vector<int64_t>& size,
                      const std::optional<DType>& dtype,
                      const std::optional<int64_t>& layout,
                      const std::optional<Device>& device,
                      const std::optional<bool>& pin_memory) {
    require_strided_layout("new_empty", layout);
    const DType dt = (dtype.has_value() && *dtype != DType::Undefined)
                         ? *dtype
                         : self.dtype();
    const Device dev = device.value_or(self.device());
    return ops::empty(size, dt, dev, pin_memory.value_or(false), false);
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// window functions (registered impls)
// ---------------------------------------------------------------------------

Tensor bartlett_window_periodic_native(int64_t window_length, bool periodic,
                                       std::optional<DType> dtype,
                                       std::optional<int64_t> layout,
                                       std::optional<Device> device,
                                       std::optional<bool> pin_memory) {
    return bartlett_window_impl(window_length, periodic,
                                dtype.value_or(globalContext().defaultDType()),
                                layout, device, pin_memory);
}

Tensor blackman_window_periodic_native(int64_t window_length, bool periodic,
                                       std::optional<DType> dtype,
                                       std::optional<int64_t> layout,
                                       std::optional<Device> device,
                                       std::optional<bool> pin_memory) {
    return blackman_window_impl(window_length, periodic,
                                dtype.value_or(globalContext().defaultDType()),
                                layout, device, pin_memory);
}

Tensor hann_window_periodic_native(int64_t window_length, bool periodic,
                                   std::optional<DType> dtype,
                                   std::optional<int64_t> layout,
                                   std::optional<Device> device,
                                   std::optional<bool> pin_memory) {
    // hann is the symmetric-coefficient hamming window: alpha = beta = 0.5.
    return hamming_window_impl(window_length, periodic, 0.5, 0.5,
                               dtype.value_or(globalContext().defaultDType()),
                               layout, device, pin_memory);
}

Tensor hamming_window_periodic_native(int64_t window_length, bool periodic,
                                      std::optional<DType> dtype,
                                      std::optional<int64_t> layout,
                                      std::optional<Device> device,
                                      std::optional<bool> pin_memory) {
    return hamming_window_impl(window_length, periodic, 0.54, 0.46,
                               dtype.value_or(globalContext().defaultDType()),
                               layout, device, pin_memory);
}

Tensor hamming_window_periodic_alpha_native(int64_t window_length, bool periodic,
                                            double alpha, std::optional<DType> dtype,
                                            std::optional<int64_t> layout,
                                            std::optional<Device> device,
                                            std::optional<bool> pin_memory) {
    return hamming_window_impl(window_length, periodic, alpha, 0.46,
                               dtype.value_or(globalContext().defaultDType()),
                               layout, device, pin_memory);
}

Tensor hamming_window_periodic_alpha_beta_native(int64_t window_length, bool periodic,
                                                 double alpha, double beta,
                                                 std::optional<DType> dtype,
                                                 std::optional<int64_t> layout,
                                                 std::optional<Device> device,
                                                 std::optional<bool> pin_memory) {
    return hamming_window_impl(window_length, periodic, alpha, beta,
                               dtype.value_or(globalContext().defaultDType()),
                               layout, device, pin_memory);
}

Tensor arange_start_native(Scalar start, Scalar end, std::optional<DType> dtype,
                           std::optional<int64_t> layout,
                           std::optional<Device> device,
                           std::optional<bool> pin_memory) {
    return arange_options_impl(start, end, Scalar(static_cast<int64_t>(1)),
                               dtype, layout, device, pin_memory);
}

Tensor arange_start_step_native(Scalar start, Scalar end, Scalar step,
                                std::optional<DType> dtype,
                                std::optional<int64_t> layout,
                                std::optional<Device> device,
                                std::optional<bool> pin_memory) {
    return arange_options_impl(start, end, step, dtype, layout, device,
                               pin_memory);
}

Tensor eye_m_native(int64_t n, int64_t m, std::optional<DType> dtype,
                    std::optional<int64_t> layout,
                    std::optional<Device> device,
                    std::optional<bool> pin_memory) {
    return eye_options_impl(n, m, dtype, layout, device, pin_memory);
}

// ---------------------------------------------------------------------------
// linspace / logspace (registered impls)
// ---------------------------------------------------------------------------

Tensor linspace_tensor_tensor_native(const Tensor& start, const Tensor& end,
                                     int64_t steps, std::optional<DType> dtype,
                                     std::optional<int64_t> layout,
                                     std::optional<Device> device,
                                     std::optional<bool> pin_memory) {
    if (!(start.dim() == 0 && end.dim() == 0)) {
        TP_THROW(RuntimeError,
                 "linspace only supports 0-dimensional start and end tensors, "
                 "but got start with ", start.dim(), " dimension(s) and end with ",
                 end.dim(), " dimension(s).");
    }
    return linspace_scalar_impl(start.item(), end.item(), steps, dtype, layout,
                                device, pin_memory);
}

Tensor linspace_tensor_scalar_native(const Tensor& start, Scalar end, int64_t steps,
                                     std::optional<DType> dtype,
                                     std::optional<int64_t> layout,
                                     std::optional<Device> device,
                                     std::optional<bool> pin_memory) {
    if (start.dim() != 0) {
        TP_THROW(RuntimeError,
                 "linspace only supports 0-dimensional start and end tensors, "
                 "but got start with ", start.dim(), " dimension(s).");
    }
    return linspace_scalar_impl(start.item(), end, steps, dtype, layout, device,
                                pin_memory);
}

Tensor linspace_scalar_tensor_native(Scalar start, const Tensor& end, int64_t steps,
                                     std::optional<DType> dtype,
                                     std::optional<int64_t> layout,
                                     std::optional<Device> device,
                                     std::optional<bool> pin_memory) {
    if (end.dim() != 0) {
        TP_THROW(RuntimeError,
                 "linspace only supports 0-dimensional start and end tensors, "
                 "but got end with ", end.dim(), " dimension(s).");
    }
    return linspace_scalar_impl(start, end.item(), steps, dtype, layout, device,
                                pin_memory);
}

Tensor& linspace_tensor_tensor_out_native(const Tensor& start, const Tensor& end,
                                          int64_t steps, Tensor& out) {
    if (!(start.dim() == 0 && end.dim() == 0)) {
        TP_THROW(RuntimeError,
                 "linspace only supports 0-dimensional start and end tensors, "
                 "but got start with ", start.dim(), " dimension(s) and end with ",
                 end.dim(), " dimension(s).");
    }
    return ops::linspace(start.item(), end.item(), steps, out);
}

Tensor& linspace_tensor_scalar_out_native(const Tensor& start, Scalar end,
                                          int64_t steps, Tensor& out) {
    if (start.dim() != 0) {
        TP_THROW(RuntimeError,
                 "linspace only supports 0-dimensional start and end tensors, "
                 "but got start with ", start.dim(), " dimension(s).");
    }
    return ops::linspace(start.item(), end, steps, out);
}

Tensor& linspace_scalar_tensor_out_native(Scalar start, const Tensor& end,
                                          int64_t steps, Tensor& out) {
    if (end.dim() != 0) {
        TP_THROW(RuntimeError,
                 "linspace only supports 0-dimensional start and end tensors, "
                 "but got end with ", end.dim(), " dimension(s).");
    }
    return ops::linspace(start, end.item(), steps, out);
}

Tensor logspace_tensor_tensor_native(const Tensor& start, const Tensor& end,
                                     int64_t steps, double base,
                                     std::optional<DType> dtype,
                                     std::optional<int64_t> layout,
                                     std::optional<Device> device,
                                     std::optional<bool> pin_memory) {
    if (!(start.dim() == 0 && end.dim() == 0)) {
        TP_THROW(RuntimeError,
                 "logspace only supports 0-dimensional start and end tensors, "
                 "but got start with ", start.dim(), " dimension(s) and end with ",
                 end.dim(), " dimension(s).");
    }
    return logspace_scalar_impl(start.item(), end.item(), steps, base, dtype, layout,
                                device, pin_memory);
}

Tensor logspace_tensor_scalar_native(const Tensor& start, Scalar end, int64_t steps,
                                     double base, std::optional<DType> dtype,
                                     std::optional<int64_t> layout,
                                     std::optional<Device> device,
                                     std::optional<bool> pin_memory) {
    if (start.dim() != 0) {
        TP_THROW(RuntimeError,
                 "logspace only supports 0-dimensional start and end tensors, "
                 "but got start with ", start.dim(), " dimension(s).");
    }
    return logspace_scalar_impl(start.item(), end, steps, base, dtype, layout, device,
                                pin_memory);
}

Tensor logspace_scalar_tensor_native(Scalar start, const Tensor& end, int64_t steps,
                                     double base, std::optional<DType> dtype,
                                     std::optional<int64_t> layout,
                                     std::optional<Device> device,
                                     std::optional<bool> pin_memory) {
    if (end.dim() != 0) {
        TP_THROW(RuntimeError,
                 "logspace only supports 0-dimensional start and end tensors, "
                 "but got end with ", end.dim(), " dimension(s).");
    }
    return logspace_scalar_impl(start, end.item(), steps, base, dtype, layout, device,
                                pin_memory);
}

Tensor& logspace_tensor_tensor_out_native(const Tensor& start, const Tensor& end,
                                          int64_t steps, double base, Tensor& out) {
    if (!(start.dim() == 0 && end.dim() == 0)) {
        TP_THROW(RuntimeError,
                 "logspace only supports 0-dimensional start and end tensors, "
                 "but got start with ", start.dim(), " dimension(s) and end with ",
                 end.dim(), " dimension(s).");
    }
    return ops::logspace(start.item(), end.item(), steps, base, out);
}

Tensor& logspace_tensor_scalar_out_native(const Tensor& start, Scalar end,
                                          int64_t steps, double base, Tensor& out) {
    if (start.dim() != 0) {
        TP_THROW(RuntimeError,
                 "logspace only supports 0-dimensional start and end tensors, "
                 "but got start with ", start.dim(), " dimension(s).");
    }
    return ops::logspace(start.item(), end, steps, base, out);
}

Tensor& logspace_scalar_tensor_out_native(Scalar start, const Tensor& end,
                                          int64_t steps, double base, Tensor& out) {
    if (end.dim() != 0) {
        TP_THROW(RuntimeError,
                 "logspace only supports 0-dimensional start and end tensors, "
                 "but got end with ", end.dim(), " dimension(s).");
    }
    return ops::logspace(start, end.item(), steps, base, out);
}

// ---------------------------------------------------------------------------
// *_out fill wrappers
// ---------------------------------------------------------------------------

Tensor& ones_out_native(const std::vector<int64_t>& size, Tensor& out) {
    ops::resize_(out, size);
    ops::fill_(out, Scalar(1.0));
    return out;
}

Tensor& zeros_out_native(const std::vector<int64_t>& size, Tensor& out) {
    ops::resize_(out, size);
    ops::zero_(out);
    return out;
}

Tensor& full_out_native(const std::vector<int64_t>& size, Scalar fill_value, Tensor& out) {
    ops::resize_(out, size);
    ops::fill_(out, fill_value);
    return out;
}

Tensor& empty_out_native(const std::vector<int64_t>& size,
                         std::optional<int64_t> memory_format, Tensor& out) {
    if (memory_format.has_value()) {
        TP_THROW(RuntimeError,
                 "'memory_format' argument is incompatible with 'out' tensor argument");
    }
    check_size_nonnegative(size);
    if (out.is_sparse()) {
        TP_THROW(NotImplementedError,
                 "empty.out is not implemented for sparse output tensors");
    }
    ops::resize_(out, size);
    return out;
}

// ---------------------------------------------------------------------------
// new_* factories (registered impls)
// ---------------------------------------------------------------------------

Tensor new_empty_native(const Tensor& self, const std::vector<int64_t>& size,
                        std::optional<DType> dtype, std::optional<int64_t> layout,
                        std::optional<Device> device, std::optional<bool> pin_memory) {
    return new_empty_impl(self, size, dtype, layout, device, pin_memory);
}

Tensor new_empty_strided_native(const Tensor& self, const std::vector<int64_t>& size,
                                const std::vector<int64_t>& stride,
                                std::optional<DType> dtype, std::optional<int64_t> layout,
                                std::optional<Device> device,
                                std::optional<bool> pin_memory) {
    require_strided_layout("new_empty_strided", layout);
    const DType dt = (dtype.has_value() && *dtype != DType::Undefined)
                         ? *dtype
                         : self.dtype();
    const Device dev = device.value_or(self.device());
    return ops::empty_strided(size, stride, dt, dev, pin_memory.value_or(false));
}

Tensor new_full_native(const Tensor& self, const std::vector<int64_t>& size,
                       Scalar fill_value, std::optional<DType> dtype,
                       std::optional<int64_t> layout, std::optional<Device> device,
                       std::optional<bool> pin_memory) {
    Tensor r = new_empty_impl(self, size, dtype, layout, device, pin_memory);
    ops::fill_(r, fill_value);
    return r;
}

Tensor new_ones_native(const Tensor& self, const std::vector<int64_t>& size,
                       std::optional<DType> dtype, std::optional<int64_t> layout,
                       std::optional<Device> device, std::optional<bool> pin_memory) {
    Tensor r = new_empty_impl(self, size, dtype, layout, device, pin_memory);
    ops::fill_(r, Scalar(1.0));
    return r;
}

Tensor new_zeros_native(const Tensor& self, const std::vector<int64_t>& size,
                        std::optional<DType> dtype, std::optional<int64_t> layout,
                        std::optional<Device> device, std::optional<bool> pin_memory) {
    Tensor r = new_empty_impl(self, size, dtype, layout, device, pin_memory);
    ops::zero_(r);
    return r;
}

// A zero-filled buffer carrying `other`'s sizes/strides/offset and feature
// metadata; `self_num_batch_dims` batch dimensions taken from `self` are
// prepended, with the batch slices laid out contiguously behind the storage.
Tensor new_zeros_with_same_feature_meta_native(const Tensor& self, const Tensor& other,
                                               int64_t self_num_batch_dims) {
    const std::vector<int64_t> other_sizes = other.shape();
    const std::vector<int64_t> other_strides = other.strides();
    const int64_t other_offset = static_cast<int64_t>(other.impl()->storage_offset());
    const int64_t other_storage_numel = other.impl()->has_storage()
        ? static_cast<int64_t>(other.impl()->storage().nbytes() / other.itemsize())
        : 0;

    if (self_num_batch_dims == 0) {
        Tensor new_tensor = ops::zeros({other_storage_numel}, other.dtype(), other.device());
        return new_tensor.as_strided(other_sizes, other_strides, other_offset);
    }

    const std::vector<int64_t> self_sizes = self.shape();

    // The batch dims of self are prepended to the sizes of other (they need
    // not match: the inplace-over-view case relies on other's shape alone).
    std::vector<int64_t> out_sizes(static_cast<size_t>(other.dim() + self_num_batch_dims));
    std::copy(self_sizes.begin(), self_sizes.begin() + self_num_batch_dims,
              out_sizes.begin());
    std::copy(other_sizes.begin(), other_sizes.end(),
              out_sizes.begin() + self_num_batch_dims);

    // other's strides are reused, and the batch strides stack the slices
    // contiguously behind the storage.
    std::vector<int64_t> out_strides(static_cast<size_t>(other.dim() + self_num_batch_dims));
    int64_t prod = other_storage_numel;
    for (int64_t i = self_num_batch_dims - 1; i >= 0; --i) {
        out_strides[static_cast<size_t>(i)] = prod;
        prod *= self_sizes[static_cast<size_t>(i)];
    }
    std::copy(other_strides.begin(), other_strides.end(),
              out_strides.begin() + self_num_batch_dims);

    Tensor new_tensor = ops::zeros({prod}, other.dtype(), other.device());
    return new_tensor.as_strided(out_sizes, out_strides, other_offset);
}

bool has_same_storage_numel_native(const Tensor& base, const Tensor& other) {
    const auto storage_numel = [](const Tensor& t) -> int64_t {
        if (!t.impl() || !t.impl()->has_storage()) return 0;
        return static_cast<int64_t>(t.impl()->storage().nbytes() / t.itemsize());
    };
    return storage_numel(base) == storage_numel(other);
}

Tensor lazy_clone_native(const Tensor& self) {
    // Copy-on-write storage is not modeled: materialize an eager copy with the
    // same sizes/strides. Values and metadata match; the storage is disjoint.
    return ops::clone(self);
}

Tensor normal_functional_native(const Tensor& self, double mean, double std,
                                std::optional<Generator> generator) {
    Tensor result = ops::clone(self);
    ops::normal_(result, mean, std, std::move(generator));
    return result;
}

// ---------------------------------------------------------------------------
// assert / debug / constraint utilities
// ---------------------------------------------------------------------------

void assert_tensor_metadata_native(const Tensor& a,
                                   std::optional<std::vector<int64_t>> size,
                                   std::optional<std::vector<int64_t>> stride,
                                   std::optional<DType> dtype,
                                   std::optional<Device> device,
                                   std::optional<int64_t> layout) {
    if (size.has_value()) {
        const std::vector<int64_t> actual = a.shape();
        if (actual != *size) {
            TP_THROW(RuntimeError, "Tensor sizes mismatch! Expected: ",
                     int_vec_to_string(*size), ", Got: ", int_vec_to_string(actual));
        }
    }
    if (stride.has_value()) {
        const std::vector<int64_t> actual = a.strides();
        if (actual != *stride) {
            TP_THROW(RuntimeError, "Tensor strides mismatch! Expected: ",
                     int_vec_to_string(*stride), ", Got: ", int_vec_to_string(actual));
        }
    }
    if (dtype.has_value() && a.dtype() != *dtype) {
        TP_THROW(RuntimeError, "Tensor dtype mismatch! Expected: ", toString(*dtype),
                 ", Got: ", toString(a.dtype()));
    }
    if (device.has_value()) {
        // Only the device type is required; an index is compared when both
        // sides carry one.
        if (a.device().type() != device->type() ||
            (device->index() >= 0 && a.device().index() >= 0 &&
             device->index() != a.device().index())) {
            TP_THROW(RuntimeError, "Tensor device mismatch! Expected: ",
                     device->toString(), ", Got: ", a.device().toString());
        }
    }
    if (layout.has_value() && layout_of(a) != *layout) {
        TP_THROW(RuntimeError, "Tensor layout mismatch! Expected: ", *layout,
                 ", Got: ", layout_of(a));
    }
}

int64_t debug_has_internal_overlap_native(const Tensor& self) {
    // Codes: 0 = no internal overlap, 1 = a zero stride with size > 1 makes
    // elements alias, 2 = cannot decide cheaply.
    constexpr int64_t kNoOverlap = 0;
    constexpr int64_t kOverlapping = 1;
    constexpr int64_t kTooHard = 2;

    const std::vector<int64_t> sizes = self.shape();
    const std::vector<int64_t> strides = self.strides();
    if (SizesAndStrides::is_non_overlapping_and_dense(sizes, strides)) {
        return kNoOverlap;
    }
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] > 1 && strides[i] == 0) {
            return kOverlapping;
        }
    }
    return kTooHard;
}

Tensor reshape_from_tensor_native(const Tensor& self, const Tensor& shape_tensor) {
    if (shape_tensor.dim() != 1) {
        TP_THROW(RuntimeError, "shape tensor must be 1-dimensional");
    }
    if (shape_tensor.dtype() != DType::Int64) {
        TP_THROW(TypeError, "shape tensor must have dtype Int64, got ",
                 toString(shape_tensor.dtype()));
    }
    const Tensor flat =
        shape_tensor.is_contiguous() ? shape_tensor : ops::contiguous(shape_tensor);
    const int64_t* data = flat.data_ptr<int64_t>();
    const std::vector<int64_t> shape(data, data + flat.numel());
    return ops::reshape(self, shape);
}

Tensor is_all_true_native(const Tensor& self) {
    if (self.dtype() != DType::Bool) {
        TP_THROW(RuntimeError, "_is_all_true expects a Bool tensor, got ",
                 toString(self.dtype()));
    }
    return ops::all(self);
}

Tensor is_any_true_native(const Tensor& self) {
    if (self.dtype() != DType::Bool) {
        TP_THROW(RuntimeError, "_is_any_true expects a Bool tensor, got ",
                 toString(self.dtype()));
    }
    return ops::any(self);
}

void assert_scalar_native(Scalar self, std::string assert_msg) {
    if (!self.to<bool>()) {
        TP_THROW(RuntimeError,
                 !assert_msg.empty() ? assert_msg : std::string("Assertion is failed"));
    }
}

Tensor functional_assert_scalar_native(Scalar self, std::string assert_msg,
                                       const Tensor& dep_token) {
    assert_scalar_native(self, assert_msg);
    return ops::clone(dep_token);
}

void print_native(std::string s) {
    std::cout << s << '\n';
}

// ---------------------------------------------------------------------------
// symbolic size constraints
// ---------------------------------------------------------------------------

void sym_constrain_range_native(Scalar size, std::optional<int64_t> min,
                                std::optional<int64_t> max) {
    const int64_t min_val =
        min.has_value() ? *min : std::numeric_limits<int64_t>::min();
    const int64_t max_val =
        max.has_value() ? *max : std::numeric_limits<int64_t>::max();
    const int64_t size_as_int = size.to<int64_t>();

    if (max_val < min_val) {
        TP_THROW(RuntimeError,
                 "Max must be greater than or equal to min. Got min=", min_val,
                 " max=", max_val);
    }
    if (!(min_val <= size_as_int && size_as_int <= max_val)) {
        TP_THROW(RuntimeError, "Invalid value range for ", size_as_int, " between [",
                 min_val, ", ", max_val, "].");
    }
}

void sym_constrain_range_for_size_native(Scalar size, std::optional<int64_t> min,
                                         std::optional<int64_t> max) {
    const int64_t min_val = min.has_value() ? *min : 0;
    if (max.has_value() && *max <= 2) {
        TP_THROW(RuntimeError,
                 "Max value to constrain_range_for_size must be greater than 2. got: ",
                 *max);
    }
    sym_constrain_range_native(size, min_val, max);
}

Tensor functional_sym_constrain_range_native(Scalar size, std::optional<int64_t> min,
                                             std::optional<int64_t> max,
                                             const Tensor& dep_token) {
    sym_constrain_range_native(size, min, max);
    return ops::clone(dep_token);
}

Tensor functional_sym_constrain_range_for_size_native(Scalar size,
                                                      std::optional<int64_t> min,
                                                      std::optional<int64_t> max,
                                                      const Tensor& dep_token) {
    sym_constrain_range_for_size_native(size, min, max);
    return ops::clone(dep_token);
}

Tensor make_dep_token_native(std::optional<DType> dtype, std::optional<int64_t> layout,
                             std::optional<Device> device,
                             std::optional<bool> pin_memory,
                             std::optional<int64_t> memory_format) {
    // memory_format is accepted for signature fidelity: a dep token is a
    // zero-element scalar, so only the default dense layout is produced.
    (void)memory_format;
    require_strided_layout("_make_dep_token", layout);
    return ops::empty({}, dtype, device, pin_memory.value_or(false), false);
}

TENSORPLAY_LIBRARY_IMPL(Composite, FactoryToolsComposite) {
    m.impl("bartlett_window.periodic", bartlett_window_periodic_native);
    m.impl("blackman_window.periodic", blackman_window_periodic_native);
    m.impl("hann_window.periodic", hann_window_periodic_native);
    m.impl("hamming_window.periodic", hamming_window_periodic_native);
    m.impl("hamming_window.periodic_alpha", hamming_window_periodic_alpha_native);
    m.impl("hamming_window.periodic_alpha_beta",
           hamming_window_periodic_alpha_beta_native);
    m.impl("arange.start", arange_start_native);
    m.impl("arange.start_step", arange_start_step_native);
    m.impl("eye.m", eye_m_native);
    m.impl("linspace.Tensor_Tensor", linspace_tensor_tensor_native);
    m.impl("linspace.Tensor_Scalar", linspace_tensor_scalar_native);
    m.impl("linspace.Scalar_Tensor", linspace_scalar_tensor_native);
    m.impl("linspace.Tensor_Tensor_out", linspace_tensor_tensor_out_native);
    m.impl("linspace.Tensor_Scalar_out", linspace_tensor_scalar_out_native);
    m.impl("linspace.Scalar_Tensor_out", linspace_scalar_tensor_out_native);
    m.impl("logspace.Tensor_Tensor", logspace_tensor_tensor_native);
    m.impl("logspace.Tensor_Scalar", logspace_tensor_scalar_native);
    m.impl("logspace.Scalar_Tensor", logspace_scalar_tensor_native);
    m.impl("logspace.Tensor_Tensor_out", logspace_tensor_tensor_out_native);
    m.impl("logspace.Tensor_Scalar_out", logspace_tensor_scalar_out_native);
    m.impl("logspace.Scalar_Tensor_out", logspace_scalar_tensor_out_native);
    m.impl("ones.out", ones_out_native);
    m.impl("zeros.out", zeros_out_native);
    m.impl("full.out", full_out_native);
    m.impl("empty.out", empty_out_native);
    m.impl("new_empty", new_empty_native);
    m.impl("new_empty_strided", new_empty_strided_native);
    m.impl("new_full", new_full_native);
    m.impl("new_ones", new_ones_native);
    m.impl("new_zeros", new_zeros_native);
    m.impl("_new_zeros_with_same_feature_meta",
           new_zeros_with_same_feature_meta_native);
    m.impl("_has_same_storage_numel", has_same_storage_numel_native);
    m.impl("_lazy_clone", lazy_clone_native);
    m.impl("normal_functional", normal_functional_native);
    m.impl("_assert_tensor_metadata", assert_tensor_metadata_native);
    m.impl("_debug_has_internal_overlap", debug_has_internal_overlap_native);
    m.impl("_reshape_from_tensor", reshape_from_tensor_native);
    m.impl("_is_all_true", is_all_true_native);
    m.impl("_is_any_true", is_any_true_native);
    m.impl("_assert_scalar", assert_scalar_native);
    m.impl("_functional_assert_scalar", functional_assert_scalar_native);
    m.impl("_print", print_native);
    m.impl("sym_constrain_range", sym_constrain_range_native);
    m.impl("sym_constrain_range_for_size", sym_constrain_range_for_size_native);
    m.impl("_functional_sym_constrain_range", functional_sym_constrain_range_native);
    m.impl("_functional_sym_constrain_range_for_size",
           functional_sym_constrain_range_for_size_native);
    m.impl("_make_dep_token", make_dep_token_native);
}

}  // namespace composite
}  // namespace tensorplay
