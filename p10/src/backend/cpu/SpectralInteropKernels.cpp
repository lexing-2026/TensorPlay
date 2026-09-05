#include "Tensor.h"
#include "DType.h"
#include "Dispatcher.h"
#include "Exception.h"

#include <algorithm>
#include <string>
#include <vector>

namespace tensorplay {
namespace cpu {

Tensor fft_fft_cpu(const Tensor& self, int64_t n, int64_t dim, std::string norm);
Tensor fft_ifft_cpu(const Tensor& self, int64_t n, int64_t dim, std::string norm);
Tensor fft_rfft_cpu(const Tensor& self, int64_t n, int64_t dim, std::string norm);
Tensor fft_irfft_cpu(const Tensor& self, int64_t n, int64_t dim, std::string norm);

namespace {

std::vector<int64_t> canonicalize_dims(const Tensor& self,
                                       const std::vector<int64_t>& dims,
                                       const char* name) {
    TP_CHECK(!dims.empty(), name, ": expected at least one transform dimension");
    TP_CHECK(self.dim() > 0, name, ": expected a non-scalar input");
    const int64_t ndim = self.dim();
    std::vector<int64_t> result;
    result.reserve(dims.size());
    for (int64_t dim : dims) {
        if (dim < 0) dim += ndim;
        TP_CHECK(dim >= 0 && dim < ndim, name,
                 ": dimension out of range: ", dim);
        TP_CHECK(std::find(result.begin(), result.end(), dim) == result.end(),
                 name, ": transform dimensions must be unique");
        result.push_back(dim);
    }
    return result;
}

std::string normalization_name(int64_t normalization, const char* name) {
    switch (normalization) {
        case 0: return "backward";
        case 1: return "forward";
        case 2: return "ortho";
        default:
            TP_THROW(RuntimeError, name, ": invalid normalization mode: ",
                     normalization);
    }
    return "backward";
}

Tensor& assign_fft_out(Tensor& out, const Tensor& result, const char* name) {
    TP_CHECK(out.defined(), name, ": output must be a defined tensor");
    TP_CHECK(out.device() == result.device(), name,
             ": output and input must be on the same device");
    TP_CHECK(out.dtype() == result.dtype(), name,
             ": output and result must have the same dtype");
    if (out.shape() == result.shape()) {
        out.copy_(result);
    } else {
        out.unsafeGetTensorImpl()->copy_metadata_from(*result.unsafeGetTensorImpl());
    }
    return out;
}

void check_real_fft_input(const Tensor& self, const char* name) {
    TP_CHECK(self.dtype() == DType::Float32 || self.dtype() == DType::Float64,
             name, ": expected a floating-point input");
}

void check_complex_fft_input(const Tensor& self, const char* name) {
    TP_CHECK(isComplexType(self.dtype()), name,
             ": expected a complex input");
}

}  // namespace

Tensor _fft_r2c_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                    int64_t normalization, bool onesided) {
    check_real_fft_input(self, "_fft_r2c");
    const auto dims = canonicalize_dims(self, dim, "_fft_r2c");
    const std::string norm = normalization_name(normalization, "_fft_r2c");
    const int64_t last_dim = dims.back();

    Tensor result = onesided
        ? fft_rfft_cpu(self, -1, last_dim, norm)
        : fft_fft_cpu(self, -1, last_dim, norm);
    for (size_t i = 0; i + 1 < dims.size(); ++i) {
        result = fft_fft_cpu(result, -1, dims[i], norm);
    }
    return result;
}

Tensor& _fft_r2c_out_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                         int64_t normalization, bool onesided, Tensor& out) {
    return assign_fft_out(out,
                          _fft_r2c_cpu(self, dim, normalization, onesided),
                          "_fft_r2c");
}

Tensor _fft_c2r_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                    int64_t normalization, int64_t last_dim_size) {
    check_complex_fft_input(self, "_fft_c2r");
    const auto dims = canonicalize_dims(self, dim, "_fft_c2r");
    TP_CHECK(last_dim_size >= 1, "_fft_c2r: invalid number of data points: ",
             last_dim_size);
    const std::string norm = normalization_name(normalization, "_fft_c2r");
    Tensor result = self;
    for (size_t i = 0; i + 1 < dims.size(); ++i) {
        result = fft_ifft_cpu(result, -1, dims[i], norm);
    }
    return fft_irfft_cpu(result, last_dim_size, dims.back(), norm);
}

Tensor& _fft_c2r_out_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                         int64_t normalization, int64_t last_dim_size,
                         Tensor& out) {
    return assign_fft_out(out,
                          _fft_c2r_cpu(self, dim, normalization, last_dim_size),
                          "_fft_c2r");
}

Tensor _fft_c2c_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                    int64_t normalization, bool forward) {
    check_complex_fft_input(self, "_fft_c2c");
    if (dim.empty()) return self.clone();
    const auto dims = canonicalize_dims(self, dim, "_fft_c2c");
    const std::string norm = normalization_name(normalization, "_fft_c2c");
    Tensor result = self;
    for (int64_t transform_dim : dims) {
        result = forward
            ? fft_fft_cpu(result, -1, transform_dim, norm)
            : fft_ifft_cpu(result, -1, transform_dim, norm);
    }
    return result;
}

Tensor& _fft_c2c_out_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                         int64_t normalization, bool forward, Tensor& out) {
    return assign_fft_out(out,
                          _fft_c2c_cpu(self, dim, normalization, forward),
                          "_fft_c2c");
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
