// The primitive transforms the n-dimensional and Hermitian FFT ops build on.
//
// Each transforms every listed dimension in one pocketfft call, reading the
// input through its strides.  `normalization` scales the whole signal once,
// whatever the direction: 0 leaves it unscaled, 1 divides by the square root
// of the signal size and 2 by the signal size.

#include "Tensor.h"
#include "DType.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "OutWrite.h"
#include "pocketfft_hdronly.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

namespace tensorplay {
namespace cpu {

namespace {

std::vector<int64_t> wrap_transform_dims(const Tensor& self,
                                         const std::vector<int64_t>& dims) {
    const int64_t ndim = self.dim();
    std::vector<int64_t> result;
    result.reserve(dims.size());
    for (int64_t dim : dims) {
        TP_CHECK_INDEX(dim >= -ndim && dim < ndim,
                       "Dimension out of range (expected to be in range of [",
                       -ndim, ", ", ndim - 1, "], but got ", dim, ")");
        if (dim < 0) dim += ndim;
        TP_CHECK(std::find(result.begin(), result.end(), dim) == result.end(),
                 "FFT dims must be unique");
        result.push_back(dim);
    }
    return result;
}

template <typename T>
T compute_fct(const std::vector<int64_t>& sizes, const std::vector<int64_t>& dims,
              int64_t normalization) {
    if (normalization == 0) return T(1);
    TP_CHECK(normalization == 1 || normalization == 2,
             "Unsupported normalization type ", normalization);
    int64_t n = 1;
    for (int64_t dim : dims) n *= sizes[static_cast<size_t>(dim)];
    return normalization == 1 ? T(1) / std::sqrt(T(n)) : T(1) / T(n);
}

pocketfft::shape_t shape_of(const std::vector<int64_t>& sizes) {
    return pocketfft::shape_t(sizes.begin(), sizes.end());
}

pocketfft::stride_t byte_strides(const Tensor& t) {
    const auto strides = t.strides();
    pocketfft::stride_t result(strides.begin(), strides.end());
    const auto item = static_cast<std::ptrdiff_t>(t.itemsize());
    for (auto& stride : result) stride *= item;
    return result;
}

pocketfft::shape_t axes_of(const std::vector<int64_t>& dims) {
    return pocketfft::shape_t(dims.begin(), dims.end());
}

template <typename T>
const std::complex<T>* complex_data(const Tensor& t) {
    return reinterpret_cast<const std::complex<T>*>(t.data_ptr());
}

template <typename T>
std::complex<T>* complex_data(Tensor& t) {
    return reinterpret_cast<std::complex<T>*>(t.data_ptr());
}

// A real signal's spectrum is Hermitian: X[k] = conj(X[-k]) with every
// transformed index negated modulo its length.  `out` (contiguous, full
// length along the last transformed dimension) holds the first n / 2 + 1
// entries there; this writes the rest from them.
template <typename T>
void fill_with_conjugate_symmetry(Tensor& out, const std::vector<int64_t>& dims) {
    using C = std::complex<T>;
    const std::vector<int64_t> sizes = out.sizes().vec();
    const int64_t ndim = static_cast<int64_t>(sizes.size());
    const int64_t last = dims.back();
    const int64_t n = sizes[static_cast<size_t>(last)];
    const int64_t half = n / 2 + 1;
    if (half >= n || out.numel() == 0) return;

    std::vector<int64_t> stride(static_cast<size_t>(ndim), 1);
    for (int64_t d = ndim - 2; d >= 0; --d) {
        stride[static_cast<size_t>(d)] =
            stride[static_cast<size_t>(d + 1)] * sizes[static_cast<size_t>(d + 1)];
    }
    std::vector<bool> mirrored(static_cast<size_t>(ndim), false);
    for (int64_t d : dims) mirrored[static_cast<size_t>(d)] = true;

    C* data = complex_data<T>(out);
    const int64_t step = stride[static_cast<size_t>(last)];
    std::vector<int64_t> index(static_cast<size_t>(ndim), 0);
    const int64_t rows = out.numel() / n;
    for (int64_t row = 0; row < rows; ++row) {
        int64_t target = 0;
        int64_t source = 0;
        for (int64_t d = 0; d < ndim; ++d) {
            if (d == last) continue;
            const size_t u = static_cast<size_t>(d);
            const int64_t i = index[u];
            target += i * stride[u];
            source += (mirrored[u] && i != 0 ? sizes[u] - i : i) * stride[u];
        }
        for (int64_t k = half; k < n; ++k) {
            data[target + k * step] = std::conj(data[source + (n - k) * step]);
        }
        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (d == last) continue;
            const size_t u = static_cast<size_t>(d);
            if (++index[u] < sizes[u]) break;
            index[u] = 0;
        }
    }
}

void check_complex_input(const Tensor& self) {
    TP_CHECK(self.dtype() == DType::ComplexFloat || self.dtype() == DType::ComplexDouble,
             "Only supports complex dtypes, but found: ", self.dtype());
}

}  // namespace

Tensor _fft_c2c_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                    int64_t normalization, bool forward) {
    check_complex_input(self);
    if (dim.empty()) return self.clone();
    const auto dims = wrap_transform_dims(self, dim);
    const std::vector<int64_t> sizes = self.sizes().vec();
    Tensor out(sizes, self.dtype(), self.device());
    const auto shape = shape_of(sizes);
    const auto axes = axes_of(dims);
    if (self.dtype() == DType::ComplexFloat) {
        pocketfft::c2c<float>(shape, byte_strides(self), byte_strides(out), axes, forward,
                              complex_data<float>(self), complex_data<float>(out),
                              compute_fct<float>(sizes, dims, normalization));
    } else {
        pocketfft::c2c<double>(shape, byte_strides(self), byte_strides(out), axes, forward,
                               complex_data<double>(self), complex_data<double>(out),
                               compute_fct<double>(sizes, dims, normalization));
    }
    return out;
}

Tensor _fft_r2c_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                    int64_t normalization, bool onesided) {
    TP_CHECK_TYPE(isFloatingType(self.dtype()),
                  "Only supports floating-point dtypes, but found: ", self.dtype());
    TP_CHECK_NOT_IMPLEMENTED(self.dtype() == DType::Float32 || self.dtype() == DType::Float64,
                             "real-to-complex FFT is not implemented for ", self.dtype(),
                             " on CPU");
    TP_CHECK(!dim.empty(), "_fft_r2c must transform at least one dimension");
    const auto dims = wrap_transform_dims(self, dim);
    const std::vector<int64_t> sizes = self.sizes().vec();
    std::vector<int64_t> out_sizes = sizes;
    if (onesided) {
        auto& last = out_sizes[static_cast<size_t>(dims.back())];
        last = last / 2 + 1;
    }
    const bool is_double = self.dtype() == DType::Float64;
    Tensor out(out_sizes, is_double ? DType::ComplexDouble : DType::ComplexFloat,
               self.device());
    const auto shape = shape_of(sizes);
    const auto axes = axes_of(dims);
    if (is_double) {
        pocketfft::r2c<double>(shape, byte_strides(self), byte_strides(out), axes, true,
                               static_cast<const double*>(self.data_ptr()),
                               complex_data<double>(out),
                               compute_fct<double>(sizes, dims, normalization));
        if (!onesided) fill_with_conjugate_symmetry<double>(out, dims);
    } else {
        pocketfft::r2c<float>(shape, byte_strides(self), byte_strides(out), axes, true,
                              static_cast<const float*>(self.data_ptr()),
                              complex_data<float>(out),
                              compute_fct<float>(sizes, dims, normalization));
        if (!onesided) fill_with_conjugate_symmetry<float>(out, dims);
    }
    return out;
}

// The input holds the first last_dim_size / 2 + 1 entries of a Hermitian
// signal along the last transformed dimension; any further entries are
// ignored.
Tensor _fft_c2r_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                    int64_t normalization, int64_t last_dim_size) {
    check_complex_input(self);
    TP_CHECK(!dim.empty(), "_fft_c2r must transform at least one dimension");
    TP_CHECK(last_dim_size >= 1, "Invalid number of data points (", last_dim_size,
             ") specified");
    const auto dims = wrap_transform_dims(self, dim);
    const int64_t last = dims.back();
    TP_CHECK(self.size(last) >= last_dim_size / 2 + 1, "_fft_c2r expects at least ",
             last_dim_size / 2 + 1, " values along dimension ", last,
             " for a signal of length ", last_dim_size, ", but got ", self.size(last));
    std::vector<int64_t> out_sizes = self.sizes().vec();
    out_sizes[static_cast<size_t>(last)] = last_dim_size;
    const bool is_double = self.dtype() == DType::ComplexDouble;
    Tensor out(out_sizes, is_double ? DType::Float64 : DType::Float32, self.device());
    const auto shape = shape_of(out_sizes);
    const auto axes = axes_of(dims);
    if (is_double) {
        pocketfft::c2r<double>(shape, byte_strides(self), byte_strides(out), axes, false,
                               complex_data<double>(self),
                               static_cast<double*>(out.data_ptr()),
                               compute_fct<double>(out_sizes, dims, normalization));
    } else {
        pocketfft::c2r<float>(shape, byte_strides(self), byte_strides(out), axes, false,
                              complex_data<float>(self),
                              static_cast<float*>(out.data_ptr()),
                              compute_fct<float>(out_sizes, dims, normalization));
    }
    return out;
}

Tensor& _fft_r2c_out_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                         int64_t normalization, bool onesided, Tensor& out) {
    return write_out(out, _fft_r2c_cpu(self, dim, normalization, onesided));
}

Tensor& _fft_c2r_out_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                         int64_t normalization, int64_t last_dim_size, Tensor& out) {
    return write_out(out, _fft_c2r_cpu(self, dim, normalization, last_dim_size));
}

Tensor& _fft_c2c_out_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                         int64_t normalization, bool forward, Tensor& out) {
    return write_out(out, _fft_c2c_cpu(self, dim, normalization, forward));
}

TENSORPLAY_LIBRARY_IMPL(CPU, SpectralInteropKernels) {
    m.impl("_fft_r2c", _fft_r2c_cpu);
    m.impl("_fft_r2c.out", _fft_r2c_out_cpu);
    m.impl("_fft_c2r", _fft_c2r_cpu);
    m.impl("_fft_c2r.out", _fft_c2r_out_cpu);
    m.impl("_fft_c2c", _fft_c2c_cpu);
    m.impl("_fft_c2c.out", _fft_c2c_out_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
