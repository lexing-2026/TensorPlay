#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Context.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

struct FFTShapeAndDims {
    std::vector<int64_t> shape;
    std::vector<int64_t> dims;
};

bool is_complex_dtype(DType dtype) {
    return dtype == DType::ComplexFloat || dtype == DType::ComplexDouble ||
           dtype == DType::ComplexHalf || dtype == DType::BComplex32;
}

bool is_supported_complex_dtype(DType dtype) {
    return dtype == DType::ComplexFloat || dtype == DType::ComplexDouble;
}

Tensor promote_fft_complex(const Tensor& input) {
    if (is_supported_complex_dtype(input.dtype())) return input;
    TP_CHECK(input.dtype() == DType::Float32 || input.dtype() == DType::Float64,
             "FFT expects a floating-point or supported complex input tensor");
    return input.to(input.dtype() == DType::Float64
                        ? DType::ComplexDouble
                        : DType::ComplexFloat);
}

int64_t wrap_fft_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    TP_CHECK(dim >= 0 && dim < ndim, "FFT dimension out of range");
    return dim;
}

FFTShapeAndDims canonicalize_fft_shape_and_dims(
    const Tensor& input,
    const std::optional<std::vector<int64_t>>& shape_opt,
    const std::optional<std::vector<int64_t>>& dims_opt) {
    const int64_t ndim = input.dim();
    FFTShapeAndDims result;

    if (dims_opt.has_value()) {
        result.dims = *dims_opt;
        for (int64_t& dim : result.dims) dim = wrap_fft_dim(dim, ndim);
        std::vector<int64_t> sorted_dims = result.dims;
        std::sort(sorted_dims.begin(), sorted_dims.end());
        TP_CHECK(std::adjacent_find(sorted_dims.begin(), sorted_dims.end()) ==
                     sorted_dims.end(),
                 "FFT dimensions must be unique");
    }

    if (shape_opt.has_value()) {
        const auto& requested_shape = *shape_opt;
        TP_CHECK(!dims_opt.has_value() || requested_shape.size() == result.dims.size(),
                 "FFT shape and dimension arguments must have the same length");
        TP_CHECK(requested_shape.size() <= static_cast<size_t>(ndim),
                 "FFT shape has more values than the input has dimensions");
        if (!dims_opt.has_value()) {
            result.dims.resize(requested_shape.size());
            std::iota(result.dims.begin(), result.dims.end(),
                      ndim - static_cast<int64_t>(requested_shape.size()));
        }
        result.shape.resize(requested_shape.size());
        for (size_t i = 0; i < requested_shape.size(); ++i) {
            const int64_t requested = requested_shape[i];
            result.shape[i] = requested == -1
                ? input.size(result.dims[i])
                : requested;
        }
    } else if (!dims_opt.has_value()) {
        result.dims.resize(static_cast<size_t>(ndim));
        std::iota(result.dims.begin(), result.dims.end(), 0);
        result.shape = static_cast<std::vector<int64_t>>(input.shape());
    } else {
        result.shape.reserve(result.dims.size());
        for (int64_t dim : result.dims) result.shape.push_back(input.size(dim));
    }

    for (int64_t size : result.shape) {
        TP_CHECK(size > 0, "Invalid number of data points specified");
    }
    return result;
}

std::string norm_string(const std::optional<std::string>& norm) {
    const std::string value = norm.value_or("backward");
    TP_CHECK(value == "backward" || value == "forward" || value == "ortho",
             "Invalid normalization mode: ", value);
    return value;
}

std::string reverse_norm(const std::optional<std::string>& norm) {
    const std::string value = norm_string(norm);
    if (value == "backward") return "forward";
    if (value == "forward") return "backward";
    return value;
}

int64_t hermitian_output_size(
    const Tensor& input,
    const FFTShapeAndDims& desc,
    const std::optional<std::vector<int64_t>>& shape_opt) {
    const int64_t last_dim = desc.dims.back();
    const bool infer_last = !shape_opt.has_value() || shape_opt->back() == -1;
    const int64_t size = infer_last
        ? 2 * (input.size(last_dim) - 1)
        : desc.shape.back();
    TP_CHECK(size >= 1, "Invalid number of data points specified");
    return size;
}

Tensor& write_fft_out(Tensor& out, const Tensor& result, const char* name) {
    TP_CHECK(out.defined(), name, " output must be a defined tensor");
    TP_CHECK(out.dtype() == result.dtype(), name,
             " output and result must have the same dtype");
    TP_CHECK(out.device() == result.device(), name,
             " output and result must be on the same device");
    ops::resize_(out, static_cast<std::vector<int64_t>>(result.shape()));
    ops::copy_(out, result);
    return out;
}

Tensor fftn_impl(const Tensor& input,
                 const std::optional<std::vector<int64_t>>& shape_opt,
                 const std::optional<std::vector<int64_t>>& dims_opt,
                 const std::optional<std::string>& norm,
                 bool forward) {
    const auto desc = canonicalize_fft_shape_and_dims(input, shape_opt, dims_opt);
    Tensor result = promote_fft_complex(input);
    const std::string mode = norm_string(norm);
    for (size_t i = 0; i < desc.dims.size(); ++i) {
        result = forward
            ? ops::fft_fft(result, desc.shape[i], desc.dims[i], mode)
            : ops::fft_ifft(result, desc.shape[i], desc.dims[i], mode);
    }
    return result;
}

Tensor rfftn_impl(const Tensor& input,
                  const std::optional<std::vector<int64_t>>& shape_opt,
                  const std::optional<std::vector<int64_t>>& dims_opt,
                  const std::optional<std::string>& norm) {
    TP_CHECK(!is_complex_dtype(input.dtype()),
             "rfftn expects a real-valued input tensor");
    const auto desc = canonicalize_fft_shape_and_dims(input, shape_opt, dims_opt);
    TP_CHECK(!desc.dims.empty(), "rfftn must transform at least one axis");
    const std::string mode = norm_string(norm);
    Tensor result = ops::fft_rfft(input, desc.shape.back(), desc.dims.back(), mode);
    for (size_t i = 0; i + 1 < desc.dims.size(); ++i) {
        result = ops::fft_fft(result, desc.shape[i], desc.dims[i], mode);
    }
    return result;
}

Tensor irfftn_impl(const Tensor& input,
                   const std::optional<std::vector<int64_t>>& shape_opt,
                   const std::optional<std::vector<int64_t>>& dims_opt,
                   const std::optional<std::string>& norm) {
    TP_CHECK(is_supported_complex_dtype(input.dtype()),
             "irfftn expects a complex-valued input tensor");
    const auto desc = canonicalize_fft_shape_and_dims(input, shape_opt, dims_opt);
    TP_CHECK(!desc.dims.empty(), "irfftn must transform at least one axis");
    const std::string mode = norm_string(norm);
    Tensor result = input;
    for (size_t i = 0; i + 1 < desc.dims.size(); ++i) {
        result = ops::fft_ifft(result, desc.shape[i], desc.dims[i], mode);
    }
    return ops::fft_irfft(result, hermitian_output_size(input, desc, shape_opt),
                          desc.dims.back(), mode);
}

Tensor hfftn_impl(const Tensor& input,
                  const std::optional<std::vector<int64_t>>& shape_opt,
                  const std::optional<std::vector<int64_t>>& dims_opt,
                  const std::optional<std::string>& norm) {
    TP_CHECK(is_supported_complex_dtype(input.dtype()),
             "hfftn expects a complex-valued input tensor");
    const auto desc = canonicalize_fft_shape_and_dims(input, shape_opt, dims_opt);
    TP_CHECK(!desc.dims.empty(), "hfftn must transform at least one axis");
    const std::string mode = norm_string(norm);
    Tensor result = input;
    for (size_t i = 0; i + 1 < desc.dims.size(); ++i) {
        result = ops::fft_fft(result, desc.shape[i], desc.dims[i], mode);
    }
    result = ops::conj(result);
    return ops::fft_irfft(result, hermitian_output_size(input, desc, shape_opt),
                          desc.dims.back(), reverse_norm(norm));
}

Tensor ihfftn_impl(const Tensor& input,
                   const std::optional<std::vector<int64_t>>& shape_opt,
                   const std::optional<std::vector<int64_t>>& dims_opt,
                   const std::optional<std::string>& norm) {
    TP_CHECK(!is_complex_dtype(input.dtype()),
             "ihfftn expects a real-valued input tensor");
    const auto desc = canonicalize_fft_shape_and_dims(input, shape_opt, dims_opt);
    TP_CHECK(!desc.dims.empty(), "ihfftn must transform at least one axis");
    const std::string mode = norm_string(norm);
    Tensor result = ops::fft_rfft(input, desc.shape.back(), desc.dims.back(),
                                  reverse_norm(norm));
    result = ops::conj_physical(result);
    for (size_t i = 0; i + 1 < desc.dims.size(); ++i) {
        result = ops::fft_ifft(result, desc.shape[i], desc.dims[i], mode);
    }
    return result;
}

Tensor fft_hfft_native(const Tensor& input, std::optional<int64_t> n,
                       int64_t dim, std::optional<std::string> norm) {
    TP_CHECK(is_supported_complex_dtype(input.dtype()),
             "hfft expects a complex-valued input tensor");
    return ops::fft_irfft(ops::conj(input), n.value_or(-1), dim,
                          reverse_norm(norm));
}

Tensor& fft_hfft_out_native(const Tensor& input, std::optional<int64_t> n,
                            int64_t dim, std::optional<std::string> norm,
                            Tensor& out) {
    return write_fft_out(out, fft_hfft_native(input, n, dim, norm), "hfft");
}

Tensor fft_ihfft_native(const Tensor& input, std::optional<int64_t> n,
                        int64_t dim, std::optional<std::string> norm) {
    TP_CHECK(!is_complex_dtype(input.dtype()),
             "ihfft expects a real-valued input tensor");
    return ops::conj_physical(ops::fft_rfft(input, n.value_or(-1), dim,
                                            reverse_norm(norm)));
}

Tensor& fft_ihfft_out_native(const Tensor& input, std::optional<int64_t> n,
                             int64_t dim, std::optional<std::string> norm,
                             Tensor& out) {
    return write_fft_out(out, fft_ihfft_native(input, n, dim, norm), "ihfft");
}

Tensor& fftn_out_native(const Tensor& input,
                        const std::optional<std::vector<int64_t>>& shape_opt,
                        const std::optional<std::vector<int64_t>>& dims_opt,
                        const std::optional<std::string>& norm, bool forward,
                        Tensor& out) {
    return write_fft_out(out, fftn_impl(input, shape_opt, dims_opt, norm, forward),
                         forward ? "fftn" : "ifftn");
}

Tensor& rfftn_out_native(const Tensor& input,
                         const std::optional<std::vector<int64_t>>& shape_opt,
                         const std::optional<std::vector<int64_t>>& dims_opt,
                         const std::optional<std::string>& norm, Tensor& out) {
    return write_fft_out(out, rfftn_impl(input, shape_opt, dims_opt, norm), "rfftn");
}

Tensor& irfftn_out_native(const Tensor& input,
                          const std::optional<std::vector<int64_t>>& shape_opt,
                          const std::optional<std::vector<int64_t>>& dims_opt,
                          const std::optional<std::string>& norm, Tensor& out) {
    return write_fft_out(out, irfftn_impl(input, shape_opt, dims_opt, norm), "irfftn");
}

Tensor& hfftn_out_native(const Tensor& input,
                         const std::optional<std::vector<int64_t>>& shape_opt,
                         const std::optional<std::vector<int64_t>>& dims_opt,
                         const std::optional<std::string>& norm, Tensor& out) {
    return write_fft_out(out, hfftn_impl(input, shape_opt, dims_opt, norm), "hfftn");
}

Tensor& ihfftn_out_native(const Tensor& input,
                          const std::optional<std::vector<int64_t>>& shape_opt,
                          const std::optional<std::vector<int64_t>>& dims_opt,
                          const std::optional<std::string>& norm, Tensor& out) {
    return write_fft_out(out, ihfftn_impl(input, shape_opt, dims_opt, norm), "ihfftn");
}

Tensor fft_fftshift_native(const Tensor& input,
                           const std::optional<std::vector<int64_t>>& dims_opt,
                           bool inverse) {
    std::vector<int64_t> dims;
    if (dims_opt.has_value()) {
        dims = *dims_opt;
        for (int64_t& dim : dims) dim = wrap_fft_dim(dim, input.dim());
    } else {
        dims.resize(static_cast<size_t>(input.dim()));
        std::iota(dims.begin(), dims.end(), 0);
    }
    if (dims.empty()) return input;
    std::vector<int64_t> shifts;
    shifts.reserve(dims.size());
    for (int64_t dim : dims) {
        const int64_t size = input.size(dim);
        shifts.push_back(inverse ? (size + 1) / 2 : size / 2);
    }
    return ops::roll(input, shifts, dims);
}

Tensor fft_hfft2_native(const Tensor& input,
                        const std::optional<std::vector<int64_t>>& shape_opt,
                        const std::vector<int64_t>& dims,
                        const std::optional<std::string>& norm) {
    return hfftn_impl(input, shape_opt, dims, norm);
}

Tensor& fft_hfft2_out_native(const Tensor& input,
                             const std::optional<std::vector<int64_t>>& shape_opt,
                             const std::vector<int64_t>& dims,
                             const std::optional<std::string>& norm, Tensor& out) {
    return write_fft_out(out, fft_hfft2_native(input, shape_opt, dims, norm), "hfft2");
}

Tensor fft_ihfft2_native(const Tensor& input,
                         const std::optional<std::vector<int64_t>>& shape_opt,
                         const std::vector<int64_t>& dims,
                         const std::optional<std::string>& norm) {
    return ihfftn_impl(input, shape_opt, dims, norm);
}

Tensor& fft_ihfft2_out_native(const Tensor& input,
                              const std::optional<std::vector<int64_t>>& shape_opt,
                              const std::vector<int64_t>& dims,
                              const std::optional<std::string>& norm, Tensor& out) {
    return write_fft_out(out, fft_ihfft2_native(input, shape_opt, dims, norm), "ihfft2");
}

Tensor fftfreq_values_native(int64_t n, double d, const Tensor& out) {
    const DType dtype = out.dtype();
    const DType real_dtype = toRealValueType(dtype);
    const std::optional<Device> device(out.device());
    Tensor values = ops::arange(
        Scalar(0), Scalar(n), Scalar(1), real_dtype, device);
    if (n > 0) {
        Tensor positive = ops::narrow(values, 0, 0, (n + 1) / 2);
        Tensor negative = ops::arange(
            Scalar(-(n / 2)), Scalar(0), Scalar(1), real_dtype, device);
        values = ops::cat({positive, negative}, 0);
    }
    if (dtype != real_dtype) values = values.to(dtype);
    const double scale = n == 0 ? 0.0 : 1.0 / (n * d);
    return ops::mul(values, Scalar(scale));
}

Tensor fftfreq_native(
    int64_t n, double d, std::optional<DType> dtype,
    std::optional<int64_t> layout, std::optional<Device> device,
    std::optional<bool> pin_memory) {
    TP_CHECK(n >= 0, "fftfreq: n must be non-negative");
    const DType output_dtype = dtype.value_or(globalContext().defaultDType());
    TP_CHECK(isFloatingOrComplexType(output_dtype),
             "fftfreq requires a floating point or complex dtype");
    Tensor out = ops::empty({n}, output_dtype, layout, device, pin_memory,
                            std::nullopt);
    return ops::copy_(out, fftfreq_values_native(n, d, out), false);
}

Tensor rfftfreq_values_native(int64_t n, double d, const Tensor& out) {
    const DType dtype = out.dtype();
    const DType real_dtype = toRealValueType(dtype);
    Tensor values = ops::arange(
        Scalar(0), Scalar(n / 2 + 1), Scalar(1), real_dtype,
        std::optional<Device>(out.device()));
    if (dtype != real_dtype) values = values.to(dtype);
    const double scale = n == 0 ? 0.0 : 1.0 / (n * d);
    return ops::mul(values, Scalar(scale));
}

Tensor rfftfreq_native(
    int64_t n, double d, std::optional<DType> dtype,
    std::optional<int64_t> layout, std::optional<Device> device,
    std::optional<bool> pin_memory) {
    TP_CHECK(n >= 0, "rfftfreq: n must be non-negative");
    const DType output_dtype = dtype.value_or(globalContext().defaultDType());
    TP_CHECK(isFloatingOrComplexType(output_dtype),
             "rfftfreq requires a floating point or complex dtype");
    Tensor out = ops::empty({n / 2 + 1}, output_dtype, layout, device,
                            pin_memory, std::nullopt);
    return ops::copy_(out, rfftfreq_values_native(n, d, out), false);
}

Tensor& fftfreq_out_native(int64_t n, double d, Tensor& out) {
    TP_CHECK(n >= 0, "fftfreq: n must be non-negative");
    TP_CHECK(isFloatingOrComplexType(out.dtype()),
             "fftfreq requires a floating point or complex dtype");
    return write_fft_out(out, fftfreq_values_native(n, d, out), "fftfreq");
}

Tensor& rfftfreq_out_native(int64_t n, double d, Tensor& out) {
    TP_CHECK(n >= 0, "rfftfreq: n must be non-negative");
    TP_CHECK(isFloatingOrComplexType(out.dtype()),
             "rfftfreq requires a floating point or complex dtype");
    return write_fft_out(out, rfftfreq_values_native(n, d, out), "rfftfreq");
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
    m.impl("fft_fftn", [](const Tensor& input,
                           std::optional<std::vector<int64_t>> shape,
                           std::optional<std::vector<int64_t>> dims,
                           std::optional<std::string> norm) {
        return fftn_impl(input, shape, dims, norm, true);
    });
    m.impl("fft_fftn.out", [](const Tensor& input,
                               std::optional<std::vector<int64_t>> shape,
                               std::optional<std::vector<int64_t>> dims,
                               std::optional<std::string> norm, Tensor& out) {
        return fftn_out_native(input, shape, dims, norm, true, out);
    });
    m.impl("fft_ifftn", [](const Tensor& input,
                            std::optional<std::vector<int64_t>> shape,
                            std::optional<std::vector<int64_t>> dims,
                            std::optional<std::string> norm) {
        return fftn_impl(input, shape, dims, norm, false);
    });
    m.impl("fft_ifftn.out", [](const Tensor& input,
                                std::optional<std::vector<int64_t>> shape,
                                std::optional<std::vector<int64_t>> dims,
                                std::optional<std::string> norm, Tensor& out) {
        return fftn_out_native(input, shape, dims, norm, false, out);
    });
    m.impl("fft_rfftn", rfftn_impl);
    m.impl("fft_rfftn.out", rfftn_out_native);
    m.impl("fft_irfftn", irfftn_impl);
    m.impl("fft_irfftn.out", irfftn_out_native);
    m.impl("fft_hfftn", hfftn_impl);
    m.impl("fft_hfftn.out", hfftn_out_native);
    m.impl("fft_ihfftn", ihfftn_impl);
    m.impl("fft_ihfftn.out", ihfftn_out_native);
    m.impl("fft_fftshift", [](const Tensor& input,
                               std::optional<std::vector<int64_t>> dims) {
        return fft_fftshift_native(input, dims, false);
    });
    m.impl("fft_ifftshift", [](const Tensor& input,
                                std::optional<std::vector<int64_t>> dims) {
        return fft_fftshift_native(input, dims, true);
    });
    m.impl("fft_fftfreq", fftfreq_native);
    m.impl("fft_fftfreq.out", fftfreq_out_native);
    m.impl("fft_rfftfreq", rfftfreq_native);
    m.impl("fft_rfftfreq.out", rfftfreq_out_native);
}

}  // namespace composite
}  // namespace tensorplay
