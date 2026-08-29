#include "Spectral2DKernels.h"

#include "Dispatcher.h"
#include "Exception.h"
#include "pocketfft_hdronly.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cpu {

namespace {

struct FFT2Args {
    int64_t first_dim;
    int64_t last_dim;
    int64_t first_size;
    int64_t last_size;
};

enum class FFTNorm { none, by_root_n, by_n };

int64_t wrap_fft_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    TP_CHECK(dim >= 0 && dim < ndim, "FFT dimension is out of range");
    return dim;
}

FFT2Args canonicalize_fft2(const Tensor& self,
                           const std::optional<std::vector<int64_t>>& s,
                           const std::vector<int64_t>& dim,
                           bool inverse_real) {
    TP_CHECK(self.dim() >= 2, "FFT2 expects an input with at least two dimensions");
    TP_CHECK(dim.size() == 2, "FFT2 expects exactly two transform dimensions");
    const int64_t first_dim = wrap_fft_dim(dim[0], self.dim());
    const int64_t last_dim = wrap_fft_dim(dim[1], self.dim());
    TP_CHECK(first_dim != last_dim, "FFT transform dimensions must be unique");
    if (s.has_value()) {
        TP_CHECK(s->size() == 2,
                 "FFT2 shape and dimension arguments must have the same length");
    }

    auto resolve_size = [&](size_t index, int64_t source_dim) {
        int64_t size = s.has_value() ? (*s)[index] : -1;
        if (size == -1) {
            size = inverse_real && index == 1
                ? 2 * (self.size(source_dim) - 1)
                : self.size(source_dim);
        }
        TP_CHECK(size > 0, "FFT size must be positive");
        return size;
    };

    return {first_dim, last_dim,
            resolve_size(0, first_dim), resolve_size(1, last_dim)};
}

FFTNorm norm_from_string(const std::string& norm, bool forward) {
    if (norm == "backward") return forward ? FFTNorm::none : FFTNorm::by_n;
    if (norm == "forward") return forward ? FFTNorm::by_n : FFTNorm::none;
    if (norm == "ortho") return FFTNorm::by_root_n;
    TP_THROW(RuntimeError, "Invalid normalization mode: \"", norm, "\"");
    return FFTNorm::none;
}

template <typename T>
T norm_factor(FFTNorm mode, int64_t n) {
    switch (mode) {
        case FFTNorm::none: return T(1);
        case FFTNorm::by_root_n: return T(1) / std::sqrt(T(n));
        case FFTNorm::by_n: return T(1) / T(n);
    }
    return T(1);
}

bool is_complex(DType dtype) {
    return dtype == DType::ComplexFloat || dtype == DType::ComplexDouble;
}

DType complex_dtype(DType dtype) {
    TP_CHECK(dtype == DType::Float32 || dtype == DType::Float64,
             "Unsupported real dtype for FFT");
    return dtype == DType::Float64 ? DType::ComplexDouble : DType::ComplexFloat;
}

DType real_dtype(DType dtype) {
    TP_CHECK(dtype == DType::ComplexFloat || dtype == DType::ComplexDouble,
             "Unsupported complex dtype for FFT");
    return dtype == DType::ComplexDouble ? DType::Float64 : DType::Float32;
}

std::vector<int64_t> sizes_of(const Tensor& tensor) {
    return static_cast<std::vector<int64_t>>(tensor.shape());
}

std::vector<std::ptrdiff_t> byte_strides(const std::vector<int64_t>& sizes,
                                         size_t item_size) {
    std::vector<std::ptrdiff_t> result(sizes.size());
    std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(item_size);
    for (int64_t i = static_cast<int64_t>(sizes.size()) - 1; i >= 0; --i) {
        result[static_cast<size_t>(i)] = stride;
        stride *= sizes[static_cast<size_t>(i)];
    }
    return result;
}

pocketfft::shape_t pocket_shape(const std::vector<int64_t>& sizes) {
    return pocketfft::shape_t(sizes.begin(), sizes.end());
}

std::pair<Tensor, std::vector<int64_t>> move_fft_dims_last(const Tensor& input,
                                                            int64_t first_dim,
                                                            int64_t last_dim) {
    const int64_t ndim = input.dim();
    std::vector<int64_t> permutation;
    permutation.reserve(static_cast<size_t>(ndim));
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != first_dim && i != last_dim) permutation.push_back(i);
    }
    permutation.push_back(first_dim);
    permutation.push_back(last_dim);

    bool identity = true;
    for (int64_t i = 0; i < ndim; ++i) {
        if (permutation[static_cast<size_t>(i)] != i) {
            identity = false;
            break;
        }
    }
    if (identity) return {input, {}};

    Tensor moved = input.permute(permutation).contiguous();
    std::vector<int64_t> inverse(static_cast<size_t>(ndim));
    for (int64_t i = 0; i < ndim; ++i) {
        inverse[static_cast<size_t>(permutation[static_cast<size_t>(i)])] = i;
    }
    return {std::move(moved), std::move(inverse)};
}

Tensor finish_fft_layout(Tensor output, const std::vector<int64_t>& inverse) {
    if (inverse.empty()) return output;
    return output.permute(inverse).contiguous();
}

template <typename T>
Tensor materialize_complex(const Tensor& input) {
    using C = std::complex<T>;
    Tensor source = input.contiguous();
    Tensor output(sizes_of(source), complex_dtype(source.dtype()), source.device());
    const T* source_ptr = static_cast<const T*>(source.data_ptr());
    C* output_ptr = static_cast<C*>(output.data_ptr());
    for (int64_t i = 0; i < source.numel(); ++i) {
        output_ptr[i] = C(source_ptr[i], T(0));
    }
    return output;
}

template <typename T>
Tensor extract_real(const Tensor& input) {
    using C = std::complex<T>;
    Tensor source = input.contiguous();
    Tensor output(sizes_of(source), real_dtype(source.dtype()), source.device());
    const C* source_ptr = static_cast<const C*>(source.data_ptr());
    T* output_ptr = static_cast<T*>(output.data_ptr());
    for (int64_t i = 0; i < source.numel(); ++i) {
        output_ptr[i] = source_ptr[i].real();
    }
    return output;
}

Tensor promote_complex(const Tensor& input) {
    TP_CHECK(input.dtype() == DType::Float32 || input.dtype() == DType::Float64 ||
                 is_complex(input.dtype()),
             "Unsupported dtype for FFT");
    if (is_complex(input.dtype())) return input.contiguous();
    return input.dtype() == DType::Float64
        ? materialize_complex<double>(input)
        : materialize_complex<float>(input);
}

Tensor resize_fft_plane(const Tensor& input, int64_t height, int64_t width) {
    std::vector<int64_t> input_sizes = sizes_of(input);
    const int64_t input_height = input_sizes[input_sizes.size() - 2];
    const int64_t input_width = input_sizes.back();
    if (input_height == height && input_width == width) return input;
    TP_CHECK(height > 0 && width > 0, "FFT size must be positive");

    std::vector<int64_t> output_sizes = input_sizes;
    output_sizes[output_sizes.size() - 2] = height;
    output_sizes.back() = width;
    const bool needs_zero_fill = height > input_height || width > input_width;
    Tensor output = needs_zero_fill
        ? Tensor::zeros(output_sizes, input.dtype(), input.device())
        : Tensor(output_sizes, input.dtype(), input.device());

    int64_t batch = 1;
    for (size_t i = 0; i + 2 < input_sizes.size(); ++i) batch *= input_sizes[i];
    const int64_t copy_height = std::min(input_height, height);
    const int64_t copy_width = std::min(input_width, width);
    const size_t item_size = input.itemsize();
    const char* source = static_cast<const char*>(input.data_ptr());
    char* destination = static_cast<char*>(output.data_ptr());
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t row = 0; row < copy_height; ++row) {
            const int64_t source_offset = (b * input_height + row) * input_width;
            const int64_t destination_offset = (b * height + row) * width;
            std::memcpy(destination + destination_offset * item_size,
                        source + source_offset * item_size,
                        static_cast<size_t>(copy_width) * item_size);
        }
    }
    return output;
}

template <typename T>
Tensor c2c_plane(const Tensor& input, bool forward, FFTNorm norm,
                 int64_t transform_size, const pocketfft::shape_t& axes) {
    using C = std::complex<T>;
    const std::vector<int64_t> sizes = sizes_of(input);
    Tensor output(sizes, input.dtype(), input.device());
    const auto shape = pocket_shape(sizes);
    const auto strides = byte_strides(sizes, sizeof(C));
    pocketfft::c2c<T>(shape, strides, strides, axes, forward,
                      static_cast<const C*>(input.data_ptr()),
                      static_cast<C*>(output.data_ptr()),
                      norm_factor<T>(norm, transform_size), 1);
    return output;
}

template <typename T>
Tensor r2c_plane(const Tensor& input, FFTNorm norm, int64_t transform_size) {
    using C = std::complex<T>;
    const std::vector<int64_t> input_sizes = sizes_of(input);
    const int64_t width = input_sizes.back();
    std::vector<int64_t> output_sizes = input_sizes;
    output_sizes.back() = width / 2 + 1;
    Tensor output(output_sizes, complex_dtype(input.dtype()), input.device());
    pocketfft::r2c<T>(
        pocket_shape(input_sizes), byte_strides(input_sizes, sizeof(T)),
        byte_strides(output_sizes, sizeof(C)), input_sizes.size() - 1, true,
        static_cast<const T*>(input.data_ptr()), static_cast<C*>(output.data_ptr()),
        norm_factor<T>(norm, transform_size), 1);
    return output;
}

template <typename T>
Tensor c2r_plane(const Tensor& input, int64_t output_width, FFTNorm norm,
                 int64_t transform_size) {
    using C = std::complex<T>;
    const std::vector<int64_t> input_sizes = sizes_of(input);
    std::vector<int64_t> output_sizes = input_sizes;
    output_sizes.back() = output_width;
    Tensor output(output_sizes, real_dtype(input.dtype()), input.device());
    pocketfft::c2r<T>(
        pocket_shape(output_sizes), byte_strides(input_sizes, sizeof(C)),
        byte_strides(output_sizes, sizeof(T)), output_sizes.size() - 1, false,
        static_cast<const C*>(input.data_ptr()), static_cast<T*>(output.data_ptr()),
        norm_factor<T>(norm, transform_size), 1);
    return output;
}

template <typename T>
Tensor fft2_c2c_impl(const Tensor& self, const FFT2Args& args,
                     FFTNorm norm, bool forward) {
    Tensor input = promote_complex(self);
    auto [moved, inverse] = move_fft_dims_last(input, args.first_dim, args.last_dim);
    Tensor resized = resize_fft_plane(moved, args.first_size, args.last_size);
    const pocketfft::shape_t axes{
        static_cast<size_t>(resized.dim() - 2),
        static_cast<size_t>(resized.dim() - 1)};
    Tensor output = c2c_plane<T>(resized, forward, norm,
                                 args.first_size * args.last_size, axes);
    return finish_fft_layout(std::move(output), inverse);
}

template <typename T>
Tensor fft2_r2c_impl(const Tensor& self, const FFT2Args& args,
                     FFTNorm norm) {
    TP_CHECK(!is_complex(self.dtype()), "RFFT2 expects a real input");
    TP_CHECK(self.dtype() == DType::Float32 || self.dtype() == DType::Float64,
             "Unsupported real dtype for FFT");
    auto [moved, inverse] = move_fft_dims_last(self.contiguous(),
                                               args.first_dim, args.last_dim);
    Tensor resized = resize_fft_plane(moved, args.first_size, args.last_size);
    const int64_t transform_size = args.first_size * args.last_size;
    Tensor spectrum = r2c_plane<T>(resized, norm, transform_size);
    const pocketfft::shape_t axes{static_cast<size_t>(spectrum.dim() - 2)};
    Tensor output = c2c_plane<T>(spectrum, true, FFTNorm::none, 1, axes);
    return finish_fft_layout(std::move(output), inverse);
}

template <typename T>
Tensor fft2_c2r_impl(const Tensor& self, const FFT2Args& args,
                     FFTNorm norm) {
    TP_CHECK(is_complex(self.dtype()), "IRFFT2 expects a complex input");
    auto [moved, inverse] = move_fft_dims_last(self.contiguous(),
                                               args.first_dim, args.last_dim);
    Tensor resized = resize_fft_plane(moved, args.first_size, args.last_size / 2 + 1);
    const int64_t transform_size = args.first_size * args.last_size;
    const pocketfft::shape_t axes{static_cast<size_t>(resized.dim() - 2)};
    Tensor transformed = c2c_plane<T>(resized, false, FFTNorm::none, 1, axes);
    Tensor output = c2r_plane<T>(transformed, args.last_size, norm, transform_size);
    return finish_fft_layout(std::move(output), inverse);
}

Tensor resize_fft_support(const Tensor& input, int64_t first_dim,
                          int64_t last_dim, int64_t first_size,
                          int64_t last_size) {
    auto [moved, inverse] = move_fft_dims_last(input.contiguous(), first_dim, last_dim);
    Tensor resized = resize_fft_plane(moved, first_size, last_size);
    return finish_fft_layout(std::move(resized), inverse);
}

template <typename T>
Tensor fft2_c2c_backward_impl(const Tensor& grad, const Tensor& self,
                              const FFT2Args& args, const std::string& norm,
                              bool forward_was) {
    TP_CHECK(is_complex(grad.dtype()), "FFT2 backward expects a complex gradient");
    auto [moved, inverse] = move_fft_dims_last(grad.contiguous(),
                                               args.first_dim, args.last_dim);
    const pocketfft::shape_t axes{
        static_cast<size_t>(moved.dim() - 2),
        static_cast<size_t>(moved.dim() - 1)};
    const int64_t transform_size = moved.size(-2) * moved.size(-1);
    Tensor transformed = c2c_plane<T>(moved, !forward_was,
                                      norm_from_string(norm, forward_was),
                                      transform_size, axes);
    Tensor output = finish_fft_layout(std::move(transformed), inverse);
    output = resize_fft_support(output, args.first_dim, args.last_dim,
                                self.size(args.first_dim), self.size(args.last_dim));
    return is_complex(self.dtype()) ? output : extract_real<T>(output);
}

template <typename T>
Tensor fft2_rfft_backward_impl(const Tensor& grad, const Tensor& self,
                               const FFT2Args& args, const std::string& norm) {
    TP_CHECK(is_complex(grad.dtype()), "RFFT2 backward expects a complex gradient");
    auto [moved, inverse] = move_fft_dims_last(grad.contiguous(),
                                               args.first_dim, args.last_dim);
    Tensor full = resize_fft_plane(moved, args.first_size, args.last_size);
    const pocketfft::shape_t axes{
        static_cast<size_t>(full.dim() - 2),
        static_cast<size_t>(full.dim() - 1)};
    Tensor transformed = c2c_plane<T>(full, false,
                                      norm_from_string(norm, true),
                                      args.first_size * args.last_size, axes);
    Tensor output = extract_real<T>(transformed);
    output = finish_fft_layout(std::move(output), inverse);
    return resize_fft_support(output, args.first_dim, args.last_dim,
                              self.size(args.first_dim), self.size(args.last_dim));
}

template <typename T>
Tensor fft2_irfft_backward_impl(const Tensor& grad, const Tensor& self,
                                const FFT2Args& args, const std::string& norm) {
    TP_CHECK(!is_complex(grad.dtype()), "IRFFT2 backward expects a real gradient");
    Tensor spectrum = fft2_r2c_impl<T>(grad, args,
                                       norm_from_string(norm, false));
    auto [moved, inverse] = move_fft_dims_last(spectrum, args.first_dim,
                                               args.last_dim);
    Tensor resized = resize_fft_plane(moved, self.size(args.first_dim),
                                      self.size(args.last_dim));
    using C = std::complex<T>;
    C* data = static_cast<C*>(resized.data_ptr());
    const int64_t width = resized.size(-1);
    const int64_t interior_end = (args.last_size + 1) / 2;
    const int64_t count = resized.numel();
    for (int64_t index = 0; index < count; ++index) {
        const int64_t column = index % width;
        if (column > 0 && column < interior_end) data[index] *= T(2);
    }
    return finish_fft_layout(std::move(resized), inverse);
}

}  // namespace

Tensor fft_fft2_cpu(const Tensor& self, std::optional<std::vector<int64_t>> s,
                   const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, false);
    return self.dtype() == DType::ComplexDouble || self.dtype() == DType::Float64
        ? fft2_c2c_impl<double>(self, args, norm_from_string(norm, true), true)
        : fft2_c2c_impl<float>(self, args, norm_from_string(norm, true), true);
}

Tensor fft_ifft2_cpu(const Tensor& self, std::optional<std::vector<int64_t>> s,
                    const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, false);
    return self.dtype() == DType::ComplexDouble || self.dtype() == DType::Float64
        ? fft2_c2c_impl<double>(self, args, norm_from_string(norm, false), false)
        : fft2_c2c_impl<float>(self, args, norm_from_string(norm, false), false);
}

Tensor fft_rfft2_cpu(const Tensor& self, std::optional<std::vector<int64_t>> s,
                    const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, false);
    return self.dtype() == DType::Float64
        ? fft2_r2c_impl<double>(self, args, norm_from_string(norm, true))
        : fft2_r2c_impl<float>(self, args, norm_from_string(norm, true));
}

Tensor fft_irfft2_cpu(const Tensor& self, std::optional<std::vector<int64_t>> s,
                     const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, true);
    return self.dtype() == DType::ComplexDouble
        ? fft2_c2r_impl<double>(self, args, norm_from_string(norm, false))
        : fft2_c2r_impl<float>(self, args, norm_from_string(norm, false));
}

Tensor fft_fft2_backward_cpu(const Tensor& grad, const Tensor& self,
                             std::optional<std::vector<int64_t>> s,
                             const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, false);
    return grad.dtype() == DType::ComplexDouble
        ? fft2_c2c_backward_impl<double>(grad, self, args, norm, true)
        : fft2_c2c_backward_impl<float>(grad, self, args, norm, true);
}

Tensor fft_ifft2_backward_cpu(const Tensor& grad, const Tensor& self,
                              std::optional<std::vector<int64_t>> s,
                              const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, false);
    return grad.dtype() == DType::ComplexDouble
        ? fft2_c2c_backward_impl<double>(grad, self, args, norm, false)
        : fft2_c2c_backward_impl<float>(grad, self, args, norm, false);
}

Tensor fft_rfft2_backward_cpu(const Tensor& grad, const Tensor& self,
                              std::optional<std::vector<int64_t>> s,
                              const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, false);
    return grad.dtype() == DType::ComplexDouble
        ? fft2_rfft_backward_impl<double>(grad, self, args, norm)
        : fft2_rfft_backward_impl<float>(grad, self, args, norm);
}

Tensor fft_irfft2_backward_cpu(const Tensor& grad, const Tensor& self,
                               std::optional<std::vector<int64_t>> s,
                               const std::vector<int64_t>& dim, std::string norm) {
    const FFT2Args args = canonicalize_fft2(self, s, dim, true);
    return grad.dtype() == DType::Float64
        ? fft2_irfft_backward_impl<double>(grad, self, args, norm)
        : fft2_irfft_backward_impl<float>(grad, self, args, norm);
}

TENSORPLAY_LIBRARY_IMPL(CPU, Spectral2DKernels) {
    m.impl("fft_fft2", fft_fft2_cpu);
    m.impl("fft_ifft2", fft_ifft2_cpu);
    m.impl("fft_rfft2", fft_rfft2_cpu);
    m.impl("fft_irfft2", fft_irfft2_cpu);
    m.impl("fft_fft2_backward", fft_fft2_backward_cpu);
    m.impl("fft_ifft2_backward", fft_ifft2_backward_cpu);
    m.impl("fft_rfft2_backward", fft_rfft2_backward_cpu);
    m.impl("fft_irfft2_backward", fft_irfft2_backward_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
