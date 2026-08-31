// Composite kernels: tril_indices / triu_indices / scalar_tensor /
// kaiser_window.
// fill helpers and the cpu/UnaryOpsKernel.cpp kaiser formula):
//   w[n] = i0(beta * sqrt(|1 - ((n - alpha) / alpha)^2|)) / i0(beta),
//   alpha = (L - 1) / 2 with L incremented for periodic windows.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

int64_t get_tril_size(int64_t row, int64_t col, int64_t offset) {
    if (row == 0 || col == 0) return 0;
    const int64_t m_first_row = offset > 0 ? std::min<int64_t>(col, 1 + offset)
                                           : (row + offset > 0);
    const int64_t m_last_row = std::max<int64_t>(0, std::min<int64_t>(col, row + offset));
    const int64_t n_row_all = std::max<int64_t>(0, std::min<int64_t>(row, row + offset));
    const int64_t n_row_trapezoid = m_last_row - m_first_row + 1;
    int64_t tril_size = (m_first_row + m_last_row) * n_row_trapezoid >> 1;
    const int64_t diff_row = n_row_all - n_row_trapezoid;
    if (diff_row > 0) tril_size += diff_row * col;
    return tril_size;
}

void check_tril_args(int64_t row, int64_t col) {
    if (row < 0) TP_THROW(RuntimeError, "row must be non-negative, got", row);
    if (col < 0) TP_THROW(RuntimeError, "col must be non-negative, got", col);
}

template <typename index_t>
void fill_tril(index_t* rows, index_t* cols, int64_t tril_size,
               int64_t col, int64_t offset) {
    int64_t i = 0;
    index_t r = static_cast<index_t>(std::max<int64_t>(0, -offset)), c = 0;
    while (i < tril_size) {
        rows[i] = r;
        cols[i++] = c;
        c += 1;
        if (c > r + offset || c >= col) {
            r += 1;
            c = 0;
        }
    }
}

template <typename index_t>
void fill_triu(index_t* rows, index_t* cols, int64_t triu_size,
               int64_t col, int64_t offset) {
    int64_t i = 0;
    index_t c = static_cast<index_t>(std::max<int64_t>(0, offset)), r = 0;
    while (i < triu_size) {
        rows[i] = r;
        cols[i++] = c;
        c += 1;
        if (c >= col) {
            r += 1;
            c = static_cast<index_t>(std::max<int64_t>(0, r + offset));
        }
    }
}

Tensor tril_family(int64_t row, int64_t col, int64_t offset, DType dtype,
                   std::optional<Device> device, bool pin_memory,
                   bool upper) {
    check_tril_args(row, col);
    if (dtype != DType::Int64 && dtype != DType::Int32) {
        TP_THROW(RuntimeError, "tril_indices/triu_indices: dtype must be int32 or int64");
    }
    const int64_t count = upper
        ? row * col - get_tril_size(row, col, offset - 1)
        : get_tril_size(row, col, offset);
    Tensor result = ops::empty({2, count}, dtype,
                               std::optional<Device>(Device(DeviceType::CPU)),
                               pin_memory);
    if (dtype == DType::Int64) {
        auto* data = result.data_ptr<int64_t>();
        if (upper) fill_triu(data, data + count, count, col, offset);
        else fill_tril(data, data + count, count, col, offset);
    } else {
        auto* data = result.data_ptr<int32_t>();
        if (upper) fill_triu(data, data + count, count, col, offset);
        else fill_tril(data, data + count, count, col, offset);
    }
    const Device target = device.value_or(Device(DeviceType::CPU));
    if (!target.is_cpu()) result = result.to(target);
    return result;
}

Tensor kaiser_window_impl(int64_t window_length, bool periodic, double beta,
                          std::optional<DType> dtype,
                          std::optional<Device> device, bool pin_memory) {
    DType dt = dtype.value_or(DType::Undefined);
    if (dt == DType::Undefined) dt = DType::Float32;
    if (!isFloatingType(dt) && !isComplexType(dt)) {
        TP_THROW(RuntimeError,
                 "kaiser_window expects floating point dtype, got ",
                 toString(dt));
    }
    if (window_length < 0) {
        TP_THROW(RuntimeError, "window_length must be non-negative");
    }
    if (window_length == 0) return ops::empty({0}, dt, device, pin_memory);
    if (window_length == 1) return ops::ones({1}, dt, device, pin_memory);

    const int64_t length = periodic ? window_length + 1 : window_length;
    const double alpha = static_cast<double>(length - 1) / 2.0;
    const Tensor n = ops::arange(Scalar(int64_t(0)), Scalar(length),
                                 Scalar(int64_t(1)), dt, device);
    const Tensor x = ops::div(ops::sub(n, Scalar(alpha)), Scalar(alpha));
    const Tensor x2 = ops::mul(x, x);
    const Tensor rad = ops::sqrt(ops::abs(ops::neg(ops::sub(x2, Scalar(1)))));
    const Tensor arg = ops::mul(rad, Scalar(beta));
    const double denom = std::cyl_bessel_i(0.0, beta);
    Tensor window = ops::div(ops::i0(arg), Scalar(denom));
    if (periodic) window = ops::narrow(window, 0, 0, window_length);
    return window;
}

} // anonymous namespace

Tensor tril_indices_native(int64_t row, int64_t col, int64_t offset,
                           DType dtype, std::optional<Device> device,
                           bool pin_memory) {
    return tril_family(row, col, offset, dtype, device, pin_memory, false);
}

Tensor triu_indices_native(int64_t row, int64_t col, int64_t offset,
                           DType dtype, std::optional<Device> device,
                           bool pin_memory) {
    return tril_family(row, col, offset, dtype, device, pin_memory, true);
}

Tensor scalar_tensor_native(const Scalar& s, std::optional<DType> dtype,
                            std::optional<Device> device, bool pin_memory) {
    DType dt = dtype.value_or(DType::Undefined);
    if (dt == DType::Undefined) dt = scalar_natural_dtype(s);
    return ops::full({}, s, dt, device, pin_memory);
}

Tensor kaiser_window_native(int64_t window_length, std::optional<DType> dtype,
                            std::optional<Device> device, bool pin_memory) {
    return kaiser_window_impl(window_length, true, 12.0, dtype, device,
                              pin_memory);
}

Tensor kaiser_window_periodic_native(int64_t window_length, bool periodic,
                                     std::optional<DType> dtype,
                                     std::optional<Device> device,
                                     bool pin_memory) {
    return kaiser_window_impl(window_length, periodic, 12.0, dtype, device,
                              pin_memory);
}

Tensor kaiser_window_beta_native(int64_t window_length, bool periodic,
                                 double beta, std::optional<DType> dtype,
                                 std::optional<Device> device,
                                 bool pin_memory) {
    return kaiser_window_impl(window_length, periodic, beta, dtype, device,
                              pin_memory);
}

// Inclusive interval: the end participates, realized as arange over
// [start, end + step) so integer and float steps both terminate on the
// last representable value below end + step.
Tensor range_native(Scalar start, Scalar end, Scalar step,
                    std::optional<DType> dtype, std::optional<Device> device) {
    Scalar end_plus_step;
    if (end.isFloatingPoint() || step.isFloatingPoint()) {
        end_plus_step = Scalar(end.toDouble() + step.toDouble());
    } else {
        end_plus_step = Scalar(end.to<int64_t>() + step.to<int64_t>());
    }
    return ops::arange(start, end_plus_step, step,
                       dtype.value_or(DType::Undefined), device);
}

// Uninitialized allocation with explicit strides: the base dense buffer is
// handed the requested layout via as_strided, matching the factory contract
// (values undefined until written).
Tensor empty_strided_native(const std::vector<int64_t>& size,
                            const std::vector<int64_t>& stride,
                            std::optional<DType> dtype,
                            std::optional<Device> device, bool pin_memory) {
    Tensor base = ops::empty(size, dtype, device, pin_memory);
    return base.as_strided(size, stride, 0);
}

TENSORPLAY_LIBRARY_IMPL(Composite, TensorFactoriesComposite) {
    m.impl("tril_indices", tril_indices_native);
    m.impl("triu_indices", triu_indices_native);
    m.impl("scalar_tensor", scalar_tensor_native);
    m.impl("kaiser_window", kaiser_window_native);
    m.impl("kaiser_window.periodic", kaiser_window_periodic_native);
    m.impl("kaiser_window.beta", kaiser_window_beta_native);
    m.impl("range", range_native);
    m.impl("empty_strided", empty_strided_native);
}

} // namespace composite
} // namespace tensorplay
