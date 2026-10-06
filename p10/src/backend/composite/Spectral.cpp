// n-dimensional, Hermitian and frequency-grid FFT entry points.
//
// The transforms resize their input to the requested signal shape and run
// one of the primitive transforms (_fft_c2c, _fft_r2c, _fft_c2r), which
// carry the gradients, forward-mode tangents and every transformed dimension
// in one call.  NumPy's normalization names map onto the primitives' scaling
// by direction: "backward" scales only the inverse transform by 1/n,
// "forward" only the forward one, and "ortho" both by 1/sqrt(n).

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Context.h"
#include "OutWrite.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "Autograd.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

using OptDims = std::optional<std::vector<int64_t>>;
using OptNorm = std::optional<std::string>;

// The primitives' scaling of the whole signal.
enum class fft_norm_mode : int64_t { none = 0, by_root_n = 1, by_n = 2 };

int64_t norm_from_string(const OptNorm& norm, bool forward) {
    fft_norm_mode mode = fft_norm_mode::none;
    if (!norm.has_value() || *norm == "backward") {
        mode = forward ? fft_norm_mode::none : fft_norm_mode::by_n;
    } else if (*norm == "forward") {
        mode = forward ? fft_norm_mode::by_n : fft_norm_mode::none;
    } else if (*norm == "ortho") {
        mode = fft_norm_mode::by_root_n;
    } else {
        TP_THROW(RuntimeError, "Invalid normalization mode: \"", *norm, "\"");
    }
    return static_cast<int64_t>(mode);
}

// Integral inputs become the default floating dtype; require_complex then
// moves real inputs to the complex dtype of the same precision.
Tensor promote_fft(const Tensor& input, bool require_complex) {
    DType dtype = input.dtype();
    if (isComplexType(dtype)) {
        TP_CHECK_NOT_IMPLEMENTED(
            dtype == DType::ComplexFloat || dtype == DType::ComplexDouble,
            "Unsupported dtype ", dtype);
        return input;
    }
    if (!isFloatingType(dtype)) dtype = globalContext().defaultDType();
    TP_CHECK_NOT_IMPLEMENTED(dtype == DType::Float32 || dtype == DType::Float64,
                             "Unsupported dtype ", dtype);
    if (require_complex) {
        dtype = dtype == DType::Float64 ? DType::ComplexDouble : DType::ComplexFloat;
    }
    return dtype == input.dtype() ? input : tpx::to(input, dtype);
}

int64_t wrap_fft_dim(int64_t dim, int64_t ndim) {
    TP_CHECK_INDEX(dim >= -ndim && dim < ndim,
                   "Dimension out of range (expected to be in range of [",
                   -ndim, ", ", ndim - 1, "], but got ", dim, ")");
    return dim < 0 ? dim + ndim : dim;
}

std::vector<int64_t> wrap_fft_dims(const std::vector<int64_t>& dims, int64_t ndim) {
    std::vector<int64_t> result;
    result.reserve(dims.size());
    for (int64_t dim : dims) result.push_back(wrap_fft_dim(dim, ndim));
    return result;
}

// Gives x the length sizes[i] along dims[i] (-1 keeps it), slicing from the
// start or zero-padding at the end.
Tensor resize_fft_input(Tensor x, const std::vector<int64_t>& dims,
                        const std::vector<int64_t>& sizes) {
    const int64_t ndim = x.dim();
    std::vector<int64_t> pad(static_cast<size_t>(2 * ndim), 0);
    bool must_pad = false;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (sizes[i] == -1) continue;
        const int64_t have = x.size(dims[i]);
        if (have < sizes[i]) {
            must_pad = true;
            pad[static_cast<size_t>(2 * (ndim - 1 - dims[i]) + 1)] = sizes[i] - have;
        } else if (have > sizes[i]) {
            x = ops::narrow(x, dims[i], 0, sizes[i]);
        }
    }
    return must_pad ? ops::constant_pad_nd(x, pad, 0) : x;
}

// Dimensions to transform, and the signal length in each of them.
struct ShapeAndDims {
    std::vector<int64_t> shape;
    std::vector<int64_t> dims;
};

// Wraps `dim`, defaults it from `s` (trailing axes) or to every axis, maps an
// entry of -1 in `s` to the input's length, and checks that the dimensions
// are unique and every length is positive.
ShapeAndDims canonicalize_fft_shape_and_dims(const Tensor& input,
                                             const OptDims& shape,
                                             const OptDims& dim) {
    const int64_t ndim = input.dim();
    ShapeAndDims ret;
    if (dim.has_value()) {
        ret.dims = wrap_fft_dims(*dim, ndim);
        std::vector<int64_t> sorted = ret.dims;
        std::sort(sorted.begin(), sorted.end());
        TP_CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
                 "FFT dims must be unique");
    }
    if (shape.has_value()) {
        TP_CHECK(!dim.has_value() || dim->size() == shape->size(),
                 "When given, dim and shape arguments must have the same length");
        const int64_t transform_ndim = static_cast<int64_t>(shape->size());
        TP_CHECK(transform_ndim <= ndim, "Got shape with ", transform_ndim,
                 " values but input tensor only has ", ndim, " dimensions.");
        if (!dim.has_value()) {
            ret.dims.resize(static_cast<size_t>(transform_ndim));
            std::iota(ret.dims.begin(), ret.dims.end(), ndim - transform_ndim);
        }
        ret.shape.resize(static_cast<size_t>(transform_ndim));
        for (size_t i = 0; i < ret.shape.size(); ++i) {
            const int64_t n = (*shape)[i];
            ret.shape[i] = n == -1 ? input.size(ret.dims[i]) : n;
        }
    } else if (!dim.has_value()) {
        ret.dims.resize(static_cast<size_t>(ndim));
        std::iota(ret.dims.begin(), ret.dims.end(), int64_t{0});
        const auto& sizes = input.sizes();
        ret.shape.assign(sizes.begin(), sizes.end());
    } else {
        ret.shape.reserve(ret.dims.size());
        for (int64_t d : ret.dims) ret.shape.push_back(input.size(d));
    }
    for (int64_t n : ret.shape) {
        TP_CHECK(n > 0, "Invalid number of data points (", n, ") specified");
    }
    return ret;
}

// For a complex-to-real transform: the real output length along the last
// (Hermitian) dimension -- s[-1] when given, otherwise 2 * (n - 1) -- and
// the one-sided input length it needs there.
ShapeAndDims canonicalize_fft_c2r_shape_and_dims(const char* fname, const Tensor& input,
                                                 const OptDims& shape, const OptDims& dim,
                                                 int64_t& last_dim_size) {
    ShapeAndDims desc = canonicalize_fft_shape_and_dims(input, shape, dim);
    TP_CHECK(!desc.shape.empty(), fname, " must transform at least one axis");
    last_dim_size = !shape.has_value() || shape->back() == -1
        ? 2 * (input.size(desc.dims.back()) - 1)
        : desc.shape.back();
    TP_CHECK(last_dim_size >= 1, "Invalid number of data points (", last_dim_size,
             ") specified");
    desc.shape.back() = last_dim_size / 2 + 1;
    return desc;
}

std::vector<int64_t> leading(const std::vector<int64_t>& dims) {
    return std::vector<int64_t>(dims.begin(), dims.end() - 1);
}

void check_real_input(const Tensor& input, const char* fname) {
    TP_CHECK_TYPE(!isComplexType(input.dtype()), fname,
                  " expects a real input tensor, but got ", input.dtype());
}

// out= writes: a complex result needs a complex `out`, a real one a
// floating-point `out`.
Tensor& write_fft_out(Tensor& out, const Tensor& result, const char* fname) {
    if (isComplexType(result.dtype())) {
        TP_CHECK(isComplexType(out.dtype()), fname,
                 " expects a complex output tensor, but got ", out.dtype());
    } else {
        TP_CHECK(isFloatingType(out.dtype()), fname,
                 " expects a floating point output tensor, but got ", out.dtype());
    }
    return write_out(out, result);
}

// ---------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------

// Complex-to-real along `dim` with a Hermitian input of n / 2 + 1 values
// (hfft runs the forward transform of the conjugated input).
Tensor fft_c2r(const char* fname, const Tensor& self, std::optional<int64_t> n_opt,
               int64_t unwrapped_dim, const OptNorm& norm_str, bool forward) {
    Tensor input = promote_fft(self, /*require_complex=*/true);
    TP_CHECK(input.dim() > 0, fname, " expects an input with at least one dimension");
    const int64_t dim = wrap_fft_dim(unwrapped_dim, input.dim());
    const int64_t n = n_opt.value_or(2 * (input.size(dim) - 1));
    TP_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
    if (n_opt.has_value()) input = resize_fft_input(input, {dim}, {n / 2 + 1});
    const int64_t norm = norm_from_string(norm_str, forward);
    if (forward) input = ops::conj(input);
    return ops::_fft_c2r(input, {dim}, norm, n);
}

// Real-to-complex along `dim` (ihfft runs the inverse transform, the
// conjugate of the forward one for a real signal).
Tensor fft_r2c(const char* fname, const Tensor& self, std::optional<int64_t> n_opt,
               int64_t unwrapped_dim, const OptNorm& norm_str, bool forward,
               bool onesided) {
    check_real_input(self, fname);
    Tensor input = promote_fft(self, /*require_complex=*/false);
    TP_CHECK(input.dim() > 0, fname, " expects an input with at least one dimension");
    const int64_t dim = wrap_fft_dim(unwrapped_dim, input.dim());
    const int64_t n = n_opt.value_or(input.size(dim));
    TP_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
    if (n_opt.has_value()) input = resize_fft_input(input, {dim}, {n});
    const int64_t norm = norm_from_string(norm_str, forward);
    Tensor ret = ops::_fft_r2c(input, {dim}, norm, onesided);
    return forward ? ret : ops::conj(ret);
}

Tensor fftn_c2c(const Tensor& self, const OptDims& s, const OptDims& dim,
                const OptNorm& norm_str, bool forward) {
    const auto desc = canonicalize_fft_shape_and_dims(self, s, dim);
    const Tensor input = promote_fft(self, /*require_complex=*/true);
    const Tensor x = resize_fft_input(input, desc.dims, desc.shape);
    return ops::_fft_c2c(x, desc.dims, norm_from_string(norm_str, forward), forward);
}

Tensor rfftn_impl(const Tensor& self, const OptDims& s, const OptDims& dim,
                  const OptNorm& norm_str) {
    check_real_input(self, "rfftn");
    const auto desc = canonicalize_fft_shape_and_dims(self, s, dim);
    TP_CHECK(!desc.shape.empty(), "rfftn must transform at least one axis");
    const Tensor input = promote_fft(self, /*require_complex=*/false);
    const Tensor x = resize_fft_input(input, desc.dims, desc.shape);
    return ops::_fft_r2c(x, desc.dims, norm_from_string(norm_str, /*forward=*/true),
                         /*onesided=*/true);
}

Tensor irfftn_impl(const Tensor& self, const OptDims& s, const OptDims& dim,
                   const OptNorm& norm_str) {
    int64_t last_dim_size = 0;
    const auto desc =
        canonicalize_fft_c2r_shape_and_dims("irfftn", self, s, dim, last_dim_size);
    const Tensor input = promote_fft(self, /*require_complex=*/true);
    const Tensor x = resize_fft_input(input, desc.dims, desc.shape);
    return ops::_fft_c2r(x, desc.dims, norm_from_string(norm_str, /*forward=*/false),
                         last_dim_size);
}

// The forward transform along the leading dimensions, then the
// complex-to-real one of the conjugate along the last.
Tensor hfftn_impl(const Tensor& self, const OptDims& s, const OptDims& dim,
                  const OptNorm& norm_str) {
    int64_t last_dim_size = 0;
    const auto desc =
        canonicalize_fft_c2r_shape_and_dims("hfftn", self, s, dim, last_dim_size);
    const Tensor input = promote_fft(self, /*require_complex=*/true);
    Tensor x = resize_fft_input(input, desc.dims, desc.shape);
    const int64_t norm = norm_from_string(norm_str, /*forward=*/true);
    if (desc.dims.size() > 1) {
        x = ops::_fft_c2c(x, leading(desc.dims), norm, /*forward=*/true);
    }
    return ops::_fft_c2r(ops::conj(x), {desc.dims.back()}, norm, last_dim_size);
}

// The conjugated one-sided transform along the last dimension, then the
// inverse transform along the leading ones.
Tensor ihfftn_impl(const Tensor& self, const OptDims& s, const OptDims& dim,
                   const OptNorm& norm_str) {
    check_real_input(self, "ihfftn");
    const auto desc = canonicalize_fft_shape_and_dims(self, s, dim);
    TP_CHECK(!desc.shape.empty(), "ihfftn must transform at least one axis");
    const Tensor input = promote_fft(self, /*require_complex=*/false);
    const Tensor x = resize_fft_input(input, desc.dims, desc.shape);
    const int64_t norm = norm_from_string(norm_str, /*forward=*/false);
    Tensor tmp = ops::conj(ops::_fft_r2c(x, {desc.dims.back()}, norm, /*onesided=*/true));
    if (desc.dims.size() == 1) return tmp;
    return ops::_fft_c2c(tmp, leading(desc.dims), norm, /*forward=*/false);
}

Tensor fft_hfft_native(const Tensor& self, std::optional<int64_t> n, int64_t dim,
                       const OptNorm& norm) {
    return fft_c2r("hfft", self, n, dim, norm, /*forward=*/true);
}

Tensor fft_ihfft_native(const Tensor& self, std::optional<int64_t> n, int64_t dim,
                        const OptNorm& norm) {
    return fft_r2c("ihfft", self, n, dim, norm, /*forward=*/false, /*onesided=*/true);
}

Tensor fft_fftn_native(const Tensor& self, const OptDims& s, const OptDims& dim,
                       const OptNorm& norm) {
    return fftn_c2c(self, s, dim, norm, /*forward=*/true);
}

Tensor fft_ifftn_native(const Tensor& self, const OptDims& s, const OptDims& dim,
                        const OptNorm& norm) {
    return fftn_c2c(self, s, dim, norm, /*forward=*/false);
}

Tensor fft_hfft2_native(const Tensor& self, const OptDims& s,
                        const std::vector<int64_t>& dim, const OptNorm& norm) {
    return hfftn_impl(self, s, dim, norm);
}

Tensor fft_ihfft2_native(const Tensor& self, const OptDims& s,
                         const std::vector<int64_t>& dim, const OptNorm& norm) {
    return ihfftn_impl(self, s, dim, norm);
}

Tensor& fft_hfft_out_native(const Tensor& self, std::optional<int64_t> n,
                            int64_t dim, const OptNorm& norm, Tensor& out) {
    return write_fft_out(out, fft_hfft_native(self, n, dim, norm), "hfft");
}

Tensor& fft_ihfft_out_native(const Tensor& self, std::optional<int64_t> n,
                             int64_t dim, const OptNorm& norm, Tensor& out) {
    return write_fft_out(out, fft_ihfft_native(self, n, dim, norm), "ihfft");
}

Tensor& fft_hfft2_out_native(const Tensor& self, const OptDims& s,
                             const std::vector<int64_t>& dim, const OptNorm& norm,
                             Tensor& out) {
    return write_fft_out(out, hfftn_impl(self, s, dim, norm), "hfft2");
}

Tensor& fft_ihfft2_out_native(const Tensor& self, const OptDims& s,
                              const std::vector<int64_t>& dim, const OptNorm& norm,
                              Tensor& out) {
    return write_fft_out(out, ihfftn_impl(self, s, dim, norm), "ihfft2");
}

Tensor& fft_fftn_out_native(const Tensor& self, const OptDims& s, const OptDims& dim,
                            const OptNorm& norm, Tensor& out) {
    return write_fft_out(out, fftn_c2c(self, s, dim, norm, true), "fftn");
}

Tensor& fft_ifftn_out_native(const Tensor& self, const OptDims& s, const OptDims& dim,
                             const OptNorm& norm, Tensor& out) {
    return write_fft_out(out, fftn_c2c(self, s, dim, norm, false), "ifftn");
}

Tensor& fft_rfftn_out_native(const Tensor& self, const OptDims& s, const OptDims& dim,
                             const OptNorm& norm, Tensor& out) {
    return write_fft_out(out, rfftn_impl(self, s, dim, norm), "rfftn");
}

Tensor& fft_irfftn_out_native(const Tensor& self, const OptDims& s, const OptDims& dim,
                              const OptNorm& norm, Tensor& out) {
    return write_fft_out(out, irfftn_impl(self, s, dim, norm), "irfftn");
}

Tensor& fft_hfftn_out_native(const Tensor& self, const OptDims& s, const OptDims& dim,
                             const OptNorm& norm, Tensor& out) {
    return write_fft_out(out, hfftn_impl(self, s, dim, norm), "hfftn");
}

Tensor& fft_ihfftn_out_native(const Tensor& self, const OptDims& s, const OptDims& dim,
                              const OptNorm& norm, Tensor& out) {
    return write_fft_out(out, ihfftn_impl(self, s, dim, norm), "ihfftn");
}

// ---------------------------------------------------------------------------
// Spectrum re-ordering
// ---------------------------------------------------------------------------

Tensor fftshift_impl(const Tensor& self, const OptDims& dim, bool inverse) {
    std::vector<int64_t> dims;
    if (dim.has_value()) {
        dims = wrap_fft_dims(*dim, self.dim());
    } else {
        dims.resize(static_cast<size_t>(self.dim()));
        std::iota(dims.begin(), dims.end(), int64_t{0});
    }
    if (dims.empty()) return ops::clone(self);
    std::vector<int64_t> shifts;
    shifts.reserve(dims.size());
    for (int64_t d : dims) {
        const int64_t n = self.size(d);
        shifts.push_back(inverse ? (n + 1) / 2 : n / 2);
    }
    return ops::roll(self, shifts, dims);
}

Tensor fft_fftshift_native(const Tensor& self, const OptDims& dim) {
    return fftshift_impl(self, dim, /*inverse=*/false);
}

Tensor fft_ifftshift_native(const Tensor& self, const OptDims& dim) {
    return fftshift_impl(self, dim, /*inverse=*/true);
}

// ---------------------------------------------------------------------------
// Frequency grids
// ---------------------------------------------------------------------------

void check_frequency_dtype(DType dtype, const char* fname) {
    TP_CHECK(isFloatingType(dtype) || isComplexType(dtype), fname,
             " requires a floating point or complex dtype");
}

// [0, 1, ..., (n - 1) / 2, -(n / 2), ..., -1] / (n * d) when `full`, else
// [0, 1, ..., n / 2] / (n * d), computed in the real dtype of `dtype`.
Tensor frequency_values(int64_t n, double d, bool full, DType dtype, Device device) {
    TP_CHECK(n >= 0, "Trying to create tensor with negative dimension ", n);
    const DType real = isComplexType(dtype) ? toRealValueType(dtype) : dtype;
    const std::optional<Device> where(device);
    Tensor values;
    if (full) {
        Tensor positive = ops::arange(Scalar(0), Scalar((n + 1) / 2), Scalar(1), real, where);
        Tensor negative = ops::arange(Scalar(-(n / 2)), Scalar(0), Scalar(1), real, where);
        values = ops::cat({positive, negative}, 0);
    } else {
        values = ops::arange(Scalar(0), Scalar(n / 2 + 1), Scalar(1), real, where);
    }
    values = ops::mul(values, Scalar(1.0 / (static_cast<double>(n) * d)));
    return real == dtype ? values : tpx::to(values, dtype);
}

Tensor frequency_factory(int64_t n, double d, bool full, const char* fname,
                         std::optional<DType> dtype, std::optional<int64_t> layout,
                         std::optional<Device> device, std::optional<bool> pin_memory) {
    const DType out_dtype = dtype.value_or(globalContext().defaultDType());
    check_frequency_dtype(out_dtype, fname);
    Tensor values = frequency_values(n, d, full, out_dtype, device.value_or(Device()));
    if (!pin_memory.value_or(false)) return values;
    Tensor out = ops::empty(values.sizes().vec(), out_dtype, layout, values.device(),
                            pin_memory, std::nullopt);
    ops::copy_(out, values);
    return out;
}

Tensor fft_fftfreq_native(int64_t n, double d, std::optional<DType> dtype,
                          std::optional<int64_t> layout, std::optional<Device> device,
                          std::optional<bool> pin_memory) {
    return frequency_factory(n, d, /*full=*/true, "fftfreq", dtype, layout, device,
                             pin_memory);
}

Tensor fft_rfftfreq_native(int64_t n, double d, std::optional<DType> dtype,
                           std::optional<int64_t> layout, std::optional<Device> device,
                           std::optional<bool> pin_memory) {
    return frequency_factory(n, d, /*full=*/false, "rfftfreq", dtype, layout, device,
                             pin_memory);
}

Tensor& fft_fftfreq_out_native(int64_t n, double d, Tensor& out) {
    check_frequency_dtype(out.dtype(), "fftfreq");
    return write_fft_out(out, frequency_values(n, d, true, out.dtype(), out.device()),
                         "fftfreq");
}

Tensor& fft_rfftfreq_out_native(int64_t n, double d, Tensor& out) {
    check_frequency_dtype(out.dtype(), "rfftfreq");
    return write_fft_out(out, frequency_values(n, d, false, out.dtype(), out.device()),
                         "rfftfreq");
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(Composite, SpectralOps) {
    m.impl("fft_hfft", fft_hfft_native);
    m.impl("fft_hfft.out", fft_hfft_out_native);
    m.impl("fft_ihfft", fft_ihfft_native);
    m.impl("fft_ihfft.out", fft_ihfft_out_native);
    m.impl("fft_hfft2", fft_hfft2_native);
    m.impl("fft_hfft2.out", fft_hfft2_out_native);
    m.impl("fft_ihfft2", fft_ihfft2_native);
    m.impl("fft_ihfft2.out", fft_ihfft2_out_native);
    m.impl("fft_fftn", fft_fftn_native);
    m.impl("fft_fftn.out", fft_fftn_out_native);
    m.impl("fft_ifftn", fft_ifftn_native);
    m.impl("fft_ifftn.out", fft_ifftn_out_native);
    m.impl("fft_rfftn", rfftn_impl);
    m.impl("fft_rfftn.out", fft_rfftn_out_native);
    m.impl("fft_irfftn", irfftn_impl);
    m.impl("fft_irfftn.out", fft_irfftn_out_native);
    m.impl("fft_hfftn", hfftn_impl);
    m.impl("fft_hfftn.out", fft_hfftn_out_native);
    m.impl("fft_ihfftn", ihfftn_impl);
    m.impl("fft_ihfftn.out", fft_ihfftn_out_native);
    m.impl("fft_fftshift", fft_fftshift_native);
    m.impl("fft_ifftshift", fft_ifftshift_native);
    m.impl("fft_fftfreq", fft_fftfreq_native);
    m.impl("fft_fftfreq.out", fft_fftfreq_out_native);
    m.impl("fft_rfftfreq", fft_rfftfreq_native);
    m.impl("fft_rfftfreq.out", fft_rfftfreq_out_native);
}

}  // namespace composite
}  // namespace tensorplay
