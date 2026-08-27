// Native spectral kernels for the audio stack.
//
// Structure mirrors ATen's native/SpectralOps.cpp + SpectralOpsUtils.h:
// public fft_* entry points follow upstream's norm_from_string /
// resize_fft_input flow, stft/istft follow upstream framing and overlap-add
// exactly. The FFT engine is vendored pocketfft (pocketfft_hdronly.h) — the
// same backend ATen's CPU kernels use.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include "pocketfft_hdronly.h"

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>

namespace tensorplay {
namespace cpu {

namespace {

constexpr double kPiD = 3.141592653589793238463;

inline int64_t wrap_dim(int64_t idx, int64_t ndim) {
    if (idx < 0) idx += ndim;
    TP_CHECK(idx >= 0 && idx < ndim, "Dimension out of range");
    return idx;
}

inline std::vector<int64_t> sizes_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

inline bool is_cplx(DType dt) {
    return dt == DType::ComplexFloat || dt == DType::ComplexDouble;
}

inline DType complex_of_real(DType real_dt) {
    TP_CHECK(real_dt == DType::Float32 || real_dt == DType::Float64,
             "Unsupported real dtype for spectral op");
    return real_dt == DType::Float64 ? DType::ComplexDouble : DType::ComplexFloat;
}

inline DType real_of_complex(DType cdt) {
    TP_CHECK(cdt == DType::ComplexFloat || cdt == DType::ComplexDouble,
             "Unsupported complex dtype for spectral op");
    return cdt == DType::ComplexDouble ? DType::Float64 : DType::Float32;
}

std::vector<std::ptrdiff_t> contig_byte_strides(const std::vector<int64_t>& sizes,
                                                size_t isz) {
    std::vector<std::ptrdiff_t> strides(sizes.size(), 0);
    std::ptrdiff_t acc = isz;
    for (int i = (int)sizes.size() - 1; i >= 0; --i) {
        strides[i] = acc;
        acc *= sizes[i];
    }
    return strides;
}

pocketfft::shape_t to_pshape(const std::vector<int64_t>& v) {
    return pocketfft::shape_t(v.begin(), v.end());
}

// ---------------------------------------------------------------------------
// Normalization copied from ATen native/SpectralOpsUtils.h
// ---------------------------------------------------------------------------
enum class fft_norm_mode {
    none,       // No normalization
    by_root_n,  // Divide by sqrt(signal_size)
    by_n,       // Divide by signal_size
};

// Copy of ATen norm_from_string: strings have direction-specific meaning.
inline fft_norm_mode norm_from_string(const std::string& norm, bool forward) {
    if (norm == "backward") return forward ? fft_norm_mode::none : fft_norm_mode::by_n;
    if (norm == "forward")  return forward ? fft_norm_mode::by_n : fft_norm_mode::none;
    if (norm == "ortho")    return fft_norm_mode::by_root_n;
    TP_THROW(RuntimeError, "Invalid normalization mode: \"", norm, "\"");
    return fft_norm_mode::none;
}

template <typename T>
inline T norm_factor(fft_norm_mode mode, int64_t n) {
    switch (mode) {
        case fft_norm_mode::none:      return T(1);
        case fft_norm_mode::by_root_n: return T(1) / std::sqrt(T(n));
        case fft_norm_mode::by_n:      return T(1) / T(n);
    }
    return T(1);
}

inline int64_t infer_ft_real_to_complex_onesided_size(int64_t real_size) {
    return (real_size / 2) + 1;
}

}  // namespace

// Copy of ATen resize_fft_input for one dim on a contiguous tensor:
// slice from 0 when larger, zero-pad at the end when smaller.
Tensor resize_input_dim(const Tensor& contig, int64_t dim, int64_t want) {
    std::vector<int64_t> sizes = sizes_of(contig);
    const int64_t have = sizes[dim];
    if (have == want) return contig;
    TP_CHECK(want > 0, "resize_fft_input: invalid target length");
    const size_t elem = contig.itemsize();
    std::vector<int64_t> out_sizes = sizes;
    out_sizes[dim] = want;
    Tensor out = have > want ? Tensor(out_sizes, contig.dtype())
                             : Tensor::zeros(out_sizes, contig.dtype(), contig.device());
    const char* src = static_cast<const char*>(contig.data_ptr());
    char* dst = static_cast<char*>(out.data_ptr());
    const int nd = (int)sizes.size();
    std::vector<int64_t> strides(nd, 0);
    { int64_t acc = 1; for (int i = nd - 1; i >= 0; --i) { strides[i] = acc; acc *= sizes[i]; } }
    int64_t outer = 1;
    for (int i = 0; i < nd; ++i) if (i != dim) outer *= sizes[i];
    const int64_t copy_n = std::min(have, want);
    std::vector<int64_t> counter(nd, 0);
    for (int64_t o = 0; o < outer; ++o) {
        int64_t off = 0;
        for (int i = 0; i < nd; ++i) if (i != dim) off += counter[i] * strides[i];
        std::memcpy(dst + off * elem, src + off * elem, elem * copy_n);
        for (int i = nd - 1; i >= 0; --i) {
            if (i == dim) continue;
            if (++counter[i] < sizes[i]) break;
            counter[i] = 0;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Transform cores over contiguous tensors (pocketfft batched calls)
// ---------------------------------------------------------------------------

template <typename T>
Tensor core_c2c(const Tensor& contig, int64_t dim, int64_t out_len,
                fft_norm_mode mode, bool forward) {
    using C = std::complex<T>;
    Tensor src = resize_input_dim(contig, dim, out_len);
    std::vector<int64_t> sizes = sizes_of(src);
    std::vector<int64_t> out_sizes = sizes;

    // When truncating we must not read beyond out_len: pocketfft reads the
    // full input shape, so shrink a copy of the source along dim first.
    Tensor out(out_sizes, contig.dtype());
    auto pshape = to_pshape(sizes);
    auto sin_ = contig_byte_strides(sizes, sizeof(C));
    auto sout = contig_byte_strides(out_sizes, sizeof(C));
    pocketfft::shape_t axes{size_t(dim)};
    const T fct = norm_factor<T>(mode, out_len);
    pocketfft::c2c<T>(pshape, sin_, sout, axes, forward,
                      static_cast<const C*>(src.data_ptr()),
                      static_cast<C*>(out.data_ptr()), fct, /*nthreads=*/1);
    return out;
}

template <typename T>
Tensor core_r2c(const Tensor& contig, int64_t dim, fft_norm_mode mode, bool onesided) {
    using C = std::complex<T>;
    std::vector<int64_t> sizes = sizes_of(contig);
    const int64_t N = sizes[dim];
    const int64_t out_len = onesided ? infer_ft_real_to_complex_onesided_size(N) : N;
    std::vector<int64_t> out_sizes = sizes;
    out_sizes[dim] = out_len;
    Tensor out(out_sizes, complex_of_real(contig.dtype()));

    auto pshape = to_pshape(sizes);
    auto sin_ = contig_byte_strides(sizes, sizeof(T));
    auto sout = contig_byte_strides(out_sizes, sizeof(C));
    pocketfft::shape_t axes{size_t(dim)};
    const T fct = norm_factor<T>(mode, N);
    if (onesided) {
        pocketfft::r2c<T>(pshape, sin_, sout, size_t(dim), /*forward=*/true,
                          static_cast<const T*>(contig.data_ptr()),
                          static_cast<C*>(out.data_ptr()), fct, 1);
    } else {
        // twosided real transform: promote to complex and run c2c
        std::vector<int64_t> csizes = sizes;
        Tensor csrc(csizes, complex_of_real(contig.dtype()));
        const T* rp = static_cast<const T*>(contig.data_ptr());
        C* cp = static_cast<C*>(csrc.data_ptr());
        const int64_t total = contig.numel();
        for (int64_t i = 0; i < total; ++i) cp[i] = C(rp[i], T(0));
        pocketfft::c2c<T>(pshape, sin_, contig_byte_strides(out_sizes, sizeof(C)),
                          axes, true, static_cast<const C*>(csrc.data_ptr()),
                          static_cast<C*>(out.data_ptr()), fct, 1);
    }
    return out;
}

// _fft_c2r analogue: input holds n/2+1 Hermitian bins, output length out_len.
template <typename T>
Tensor core_c2r(const Tensor& contig, int64_t dim, int64_t out_len, fft_norm_mode mode) {
    using C = std::complex<T>;
    std::vector<int64_t> isizes = sizes_of(contig);
    const int64_t bins_needed = out_len / 2 + 1;
    Tensor src = isizes[dim] == bins_needed
                     ? contig
                     : resize_input_dim(contig, dim, bins_needed);
    std::vector<int64_t> in_sizes = sizes_of(src);
    std::vector<int64_t> out_sizes = in_sizes;
    out_sizes[dim] = out_len;
    Tensor out(out_sizes, real_of_complex(contig.dtype()));

    auto pshape_out = to_pshape(out_sizes);
    auto sin_ = contig_byte_strides(in_sizes, sizeof(C));
    auto sout = contig_byte_strides(out_sizes, sizeof(T));
    pocketfft::shape_t axes{size_t(dim)};
    const T fct = norm_factor<T>(mode, out_len);
    pocketfft::c2r<T>(pshape_out, sin_, sout, size_t(dim), /*forward=*/false,
                      static_cast<const C*>(src.data_ptr()),
                      static_cast<T*>(out.data_ptr()), fct, 1);
    return out;
}

// ---------------------------------------------------------------------------
// Public fft entry points (mirror ATen fft_c2c / fft_r2c / fft_c2r wrappers)
// ---------------------------------------------------------------------------

namespace {
// ATen parity: torch.fft.fft/ifft accept real input and promote it to complex
// with a zero imaginary part (SpectralOps.cpp fft_r2c "fft"/"ifft" entry).
template <typename T>
Tensor materialize_real_as_complex(const Tensor& x) {
    using C = std::complex<T>;
    Tensor out(sizes_of(x), complex_of_real(x.dtype()));
    const T* src = static_cast<const T*>(x.data_ptr());
    C* dst = static_cast<C*>(out.data_ptr());
    const int64_t n = x.numel();
    for (int64_t i = 0; i < n; ++i) dst[i] = C(src[i], T(0));
    return out;
}

Tensor promote_real_input_for_c2c(const Tensor& self) {
    TP_CHECK(self.dtype() == DType::Float32 || self.dtype() == DType::Float64,
             "Unsupported input dtype for spectral op");
    Tensor x = self.contiguous();
    if (x.dtype() == DType::Float32)
        return materialize_real_as_complex<float>(x);
    return materialize_real_as_complex<double>(x);
}
}  // namespace

Tensor fft_fft_cpu(const Tensor& self, int64_t n, int64_t dim, std::string norm) {
    Tensor x_in = is_cplx(self.dtype()) ? self.contiguous() : promote_real_input_for_c2c(self);
    TP_CHECK(x_in.dim() >= 1, "fft expects at least 1 dimension");
    dim = wrap_dim(dim, x_in.dim());
    const int64_t N = sizes_of(x_in)[dim];
    const int64_t n_eff = n > 0 ? n : N;
    TP_CHECK(n_eff >= 1, "Invalid number of data points specified");
    const auto mode = norm_from_string(norm, /*forward=*/true);
    if (x_in.dtype() == DType::ComplexFloat)
        return core_c2c<float>(x_in, dim, n_eff, mode, true);
    return core_c2c<double>(x_in, dim, n_eff, mode, true);
}

Tensor fft_ifft_cpu(const Tensor& self, int64_t n, int64_t dim, std::string norm) {
    Tensor x_in = is_cplx(self.dtype()) ? self.contiguous() : promote_real_input_for_c2c(self);
    TP_CHECK(x_in.dim() >= 1, "ifft expects at least 1 dimension");
    dim = wrap_dim(dim, x_in.dim());
    const int64_t N = sizes_of(x_in)[dim];
    const int64_t n_eff = n > 0 ? n : N;
    TP_CHECK(n_eff >= 1, "Invalid number of data points specified");
    const auto mode = norm_from_string(norm, false);
    if (x_in.dtype() == DType::ComplexFloat)
        return core_c2c<float>(x_in, dim, n_eff, mode, false);
    return core_c2c<double>(x_in, dim, n_eff, mode, false);
}

Tensor fft_rfft_cpu(const Tensor& self, int64_t n, int64_t dim, std::string norm) {
    TP_CHECK(!is_cplx(self.dtype()), "torch.fft.rfft expects a real input");
    TP_CHECK(self.dim() >= 1, "rfft expects at least 1 dimension");
    dim = wrap_dim(dim, self.dim());
    Tensor x = self.contiguous();
    const int64_t N = sizes_of(x)[dim];
    const int64_t n_eff = n > 0 ? n : N;
    TP_CHECK(n_eff >= 1, "Invalid number of data points specified");
    if (n > 0 && n != N) x = resize_input_dim(x, dim, n);
    const auto mode = norm_from_string(norm, true);
    if (x.dtype() == DType::Float32)
        return core_r2c<float>(x, dim, mode, /*onesided=*/true);
    return core_r2c<double>(x, dim, mode, true);
}

Tensor fft_irfft_cpu(const Tensor& self, int64_t n, int64_t dim, std::string norm) {
    TP_CHECK(is_cplx(self.dtype()), "torch.fft.irfft expects a complex input");
    TP_CHECK(self.dim() >= 1, "irfft expects at least 1 dimension");
    dim = wrap_dim(dim, self.dim());
    Tensor x = self.contiguous();
    const int64_t F = sizes_of(x)[dim];
    const int64_t n_eff = n > 0 ? n : 2 * (F - 1);
    TP_CHECK(n_eff >= 1, "Invalid number of data points specified");
    const auto mode = norm_from_string(norm, false);
    if (x.dtype() == DType::ComplexFloat)
        return core_c2r<float>(x, dim, n_eff, mode);
    return core_c2r<double>(x, dim, n_eff, mode);
}

// ---------------------------------------------------------------------------
// Backward helpers — adjoint = same internal transform with the flipped
// direction and the identical normalization enum (verified against ATen's
// convention), followed by resize back to the primal input support.
// ---------------------------------------------------------------------------

namespace {
// ATen parity: the adjoint of "materialize real as complex" is taking the
// real part, so fft/ifft on a real primal yields a real gradient.
template <typename T>
Tensor extract_real_part(const Tensor& z) {
    using C = std::complex<T>;
    Tensor out(sizes_of(z), real_of_complex(z.dtype()));
    Tensor zc = z.contiguous();
    const C* src = static_cast<const C*>(zc.data_ptr());
    T* dst = static_cast<T*>(out.data_ptr());
    const int64_t n = z.numel();
    for (int64_t i = 0; i < n; ++i) dst[i] = src[i].real();
    return out;
}

template <typename T>
Tensor c2c_backward_core(const Tensor& grad, int64_t input_len, int64_t dim,
                         const std::string& norm, bool forward_was) {
    Tensor g = grad.contiguous();
    const auto mode = norm_from_string(norm, forward_was);
    Tensor t = core_c2c<T>(g, dim, g.size(dim), mode, /*forward=*/!forward_was);
    return resize_input_dim(t, dim, input_len);
}
}  // namespace

Tensor fft_fft_backward_cpu(const Tensor& grad, const Tensor& self, int64_t dim, std::string norm) {
    dim = wrap_dim(dim, self.dim());
    const int64_t input_len = self.size(dim);
    const bool real_primal = !is_cplx(self.dtype());
    if (grad.dtype() == DType::ComplexFloat) {
        Tensor g = c2c_backward_core<float>(grad, input_len, dim, norm, true);
        return real_primal ? extract_real_part<float>(g) : g;
    }
    Tensor g = c2c_backward_core<double>(grad, input_len, dim, norm, true);
    return real_primal ? extract_real_part<double>(g) : g;
}

Tensor fft_ifft_backward_cpu(const Tensor& grad, const Tensor& self, int64_t dim, std::string norm) {
    dim = wrap_dim(dim, self.dim());
    const int64_t input_len = self.size(dim);
    const bool real_primal = !is_cplx(self.dtype());
    if (grad.dtype() == DType::ComplexFloat) {
        Tensor g = c2c_backward_core<float>(grad, input_len, dim, norm, false);
        return real_primal ? extract_real_part<float>(g) : g;
    }
    Tensor g = c2c_backward_core<double>(grad, input_len, dim, norm, false);
    return real_primal ? extract_real_part<double>(g) : g;
}

namespace {
// rfft adjoint — ATen fft_r2c_backward (torch/csrc/autograd/FunctionsManual.cpp
// :5135): view onesided r2c as [zero-fill imag, c2c forward, drop half], so the
// backward is [zero-fill the twosided spectrum, c2c INVERSE with the forward's
// normalization, take the real part].
template <typename T>
Tensor rfft_backward_core(const Tensor& grad, int64_t input_len, int64_t dim,
                          fft_norm_mode mode) {
    Tensor g = grad.contiguous();
    std::vector<int64_t> gsizes = sizes_of(g);
    const int64_t bins = gsizes[dim];
    std::vector<int64_t> full_sizes = gsizes;
    full_sizes[dim] = input_len;
    Tensor full(full_sizes, g.dtype());
    full.zero_();
    if (bins == input_len) {
        // grad already covers every bin: plain inverse c2c.
        Tensor t = core_c2c<T>(g, dim, input_len, mode, /*forward=*/false);
        return extract_real_part<T>(t);
    }
    full.slice(dim, 0, bins).copy_(g);
    Tensor t = core_c2c<T>(full, dim, input_len, mode, /*forward=*/false);
    return extract_real_part<T>(t);
}
}  // namespace

Tensor fft_rfft_backward_cpu(const Tensor& grad, const Tensor& self, int64_t dim, std::string norm) {
    dim = wrap_dim(dim, self.dim());
    const auto mode = norm_from_string(norm, true);
    if (grad.dtype() == DType::ComplexFloat)
        return rfft_backward_core<float>(grad, self.size(dim), dim, mode);
    return rfft_backward_core<double>(grad, self.size(dim), dim, mode);
}

namespace {
// irfft adjoint — ATen fft_c2r_backward (FunctionsManual.cpp :5095):
// r2c of the real gradient with the forward's normalization, then double the
// bins whose conjugate mirror fell outside the onesided range
// (indices 1 .. N - onesided_length).
template <typename T>
Tensor irfft_backward_core(const Tensor& grad, int64_t freq_bins, int64_t dim,
                           fft_norm_mode mode) {
    Tensor g = grad.contiguous();
    Tensor t = core_r2c<T>(g, dim, mode, /*onesided=*/true);
    const int64_t got_bins = sizes_of(t)[dim];
    const int64_t double_length = freq_bins - got_bins;
    if (double_length > 0) {
        // bins 1 .. N - onesided_length receive their conjugate mirror twice.
        Tensor scaled = t.slice(dim, 1, 1 + double_length).mul(Scalar(2.0));
        t.slice(dim, 1, 1 + double_length).copy_(scaled);
    }
    return t;
}
}  // namespace

Tensor fft_irfft_backward_cpu(const Tensor& grad, const Tensor& self, int64_t dim, std::string norm) {
    dim = wrap_dim(dim, self.dim());
    const auto mode = norm_from_string(norm, false);
    if (grad.dtype() == DType::Float32)
        return irfft_backward_core<float>(grad, self.size(dim), dim, mode);
    return irfft_backward_core<double>(grad, self.size(dim), dim, mode);
}

// ---------------------------------------------------------------------------
// Window factories — formulas mirror ATen native/TensorFactories.cpp
// ---------------------------------------------------------------------------

namespace {
Tensor window_tensor(int64_t out_len, int64_t formula_len, std::optional<DType> dtype_opt,
                     const char* name, const std::function<double(int64_t)>& formula) {
    if (out_len < 0) {
        TP_THROW(ValueError, name, ": window_length must be non-negative");
    }
    // Python layer passes DType::Undefined for "no dtype given" (torch
    // default-floating semantics); normalize before the support check.
    DType dt = dtype_opt.value_or(DType::Float32);
    if (dt == DType::Undefined) dt = DType::Float32;
    if (dt != DType::Float32 && dt != DType::Float64) {
        TP_THROW(NotImplementedError, name, ": only float32/float64 windows are supported");
    }
    if (out_len == 0) return Tensor(std::vector<int64_t>{int64_t(0)}, dt);
    Tensor w({out_len}, dt);
    if (out_len == 1) {
        if (dt == DType::Float64) w.data_ptr<double>()[0] = 1.0;
        else w.data_ptr<float>()[0] = 1.0f;
        return w;
    }
    if (dt == DType::Float64) {
        double* p = w.data_ptr<double>();
        for (int64_t n = 0; n < out_len; ++n) p[n] = formula(n);
    } else {
        float* p = w.data_ptr<float>();
        for (int64_t n = 0; n < out_len; ++n) p[n] = float(formula(n));
    }
    return w;
}
}  // namespace

// ATen denominator semantics: periodic -> N, symmetric -> N - 1.
inline int64_t window_denominator(int64_t window_length, bool periodic) {
    return window_length - (periodic ? 0 : 1);
}

Tensor hann_window_cpu(int64_t window_length, bool periodic, std::optional<DType> dtype) {
    const int64_t L = window_denominator(window_length, periodic);
    return window_tensor(window_length, L, dtype, "hann_window", [L](int64_t n) {
        return 0.5 - 0.5 * std::cos(2.0 * kPiD * n / L);
    });
}

Tensor hamming_window_cpu(int64_t window_length, bool periodic, double alpha, double beta, std::optional<DType> dtype) {
    const int64_t L = window_denominator(window_length, periodic);
    return window_tensor(window_length, L, dtype, "hamming_window", [L, alpha, beta](int64_t n) {
        return alpha - beta * std::cos(2.0 * kPiD * n / L);
    });
}

Tensor bartlett_window_cpu(int64_t window_length, bool periodic, std::optional<DType> dtype) {
    const int64_t L = window_denominator(window_length, periodic);
    return window_tensor(window_length, L, dtype, "bartlett_window", [L](int64_t n) {
        const double num = 2.0 * static_cast<double>(n);
        if (num < static_cast<double>(L)) return num / static_cast<double>(L);
        if (num > static_cast<double>(L)) return 2.0 - num / static_cast<double>(L);
        return 1.0;
    });
}

Tensor blackman_window_cpu(int64_t window_length, bool periodic, std::optional<DType> dtype) {
    const int64_t L = window_denominator(window_length, periodic);
    return window_tensor(window_length, L, dtype, "blackman_window", [L](int64_t n) {
        // ATen blackman_window: a0 - a1*cos(x) + a2*cos(2x), x = 2*pi*n/L.
        const double x = 2.0 * kPiD * n / static_cast<double>(L);
        return 0.42 - 0.5 * std::cos(x) + 0.08 * std::cos(2 * x);
    });
}

// ---------------------------------------------------------------------------
// stft / istft / stft_backward — algorithms mirror ATen SpectralOps.cpp
// ---------------------------------------------------------------------------

namespace {

// Reflect/constant padding along the last axis of a contiguous (batch, len)
// real tensor. Mirrors ATen at::pad semantics ("reflect" excludes the edge).
template <typename T>
Tensor pad_time_axis(const Tensor& contig, int64_t pad, const std::string& mode) {
    std::vector<int64_t> sizes = sizes_of(contig);
    const int64_t batch = sizes[0];
    const int64_t len = sizes[1];
    const int64_t out_len = len + 2 * pad;
    Tensor out({batch, out_len}, contig.dtype());
    const T* src = static_cast<const T*>(contig.data_ptr());
    T* dst = static_cast<T*>(out.data_ptr());
    for (int64_t b = 0; b < batch; ++b) {
        const T* in_row = src + b * len;
        T* out_row = dst + b * out_len;
        if (mode == "constant") {
            std::memset(out_row, 0, sizeof(T) * pad);
            std::memcpy(out_row + pad, in_row, sizeof(T) * len);
            std::memset(out_row + pad + len, 0, sizeof(T) * pad);
        } else {  // reflect
            for (int64_t i = 0; i < pad; ++i) {
                int64_t idx = (i + 1) % std::max<int64_t>(2 * len - 2, 1);
                if (idx >= len) idx = 2 * len - 2 - idx;
                out_row[pad - 1 - i] = in_row[idx];
            }
            std::memcpy(out_row + pad, in_row, sizeof(T) * len);
            for (int64_t i = 0; i < pad; ++i) {
                int64_t idx = len - 2 - i;
                idx = -idx % std::max<int64_t>(2 * len - 2, 1);
                if (idx < 0) idx += std::max<int64_t>(2 * len - 2, 1);
                if (idx >= len) idx = 2 * len - 2 - idx;
                out_row[pad + len + i] = in_row[idx];
            }
        }
    }
    return out;
}

// Adjoint of pad_time_axis: crop (constant) or mirror-scatter (reflect).
template <typename T>
void unpad_scatter_time_axis(const T* padded_grad, int64_t batch, int64_t padded_len,
                             int64_t pad, const std::string& mode, T* out_grad) {
    const int64_t len = padded_len - 2 * pad;
    for (int64_t b = 0; b < batch; ++b) {
        const T* prow = padded_grad + b * padded_len;
        T* orow = out_grad + b * len;
        if (mode == "constant") {
            std::memcpy(orow, prow + pad, sizeof(T) * len);
        } else {  // reflect adjoint scatters mirrored values back
            for (int64_t j = 0; j < len; ++j) orow[j] = prow[pad + j];
            for (int64_t i = 0; i < pad; ++i) {
                int64_t idx = (i + 1) % std::max<int64_t>(2 * len - 2, 1);
                if (idx >= len) idx = 2 * len - 2 - idx;
                orow[idx] += prow[pad - 1 - i];
                int64_t idx2 = len - 2 - i;
                idx2 = -idx2 % std::max<int64_t>(2 * len - 2, 1);
                if (idx2 < 0) idx2 += std::max<int64_t>(2 * len - 2, 1);
                if (idx2 >= len) idx2 = 2 * len - 2 - idx2;
                orow[idx2] += prow[pad + len + i];
            }
        }
    }
}

template <typename T>
void fill_win_full(std::vector<T>& win_full, const std::optional<Tensor>& window,
                   int64_t win_length, int64_t n_fft) {
    // ATen SpectralOps.cpp stft: no window -> rectangular (ones); a defined
    // window of win_length < n_fft is zero-padded on both sides.
    if (!window.has_value()) {
        win_full.assign(n_fft, T(1));
        return;
    }
    Tensor w = window->contiguous();
    TP_CHECK(w.dim() == 1 && w.size(0) == win_length, "window must be 1D of size win_length");
    std::vector<T> tmp(win_length);
    if (w.dtype() == DType::Float64 && !std::is_same_v<T, double>) {
        const double* dp = static_cast<const double*>(w.data_ptr());
        for (int64_t i = 0; i < win_length; ++i) tmp[i] = T(dp[i]);
    } else if (w.dtype() == DType::Float32 && std::is_same_v<T, double>) {
        const float* fp = static_cast<const float*>(w.data_ptr());
        for (int64_t i = 0; i < win_length; ++i) tmp[i] = T(fp[i]);
    } else {
        const T* p = static_cast<const T*>(w.data_ptr());
        for (int64_t i = 0; i < win_length; ++i) tmp[i] = p[i];
    }
    win_full.assign(n_fft, T(0));
    const int64_t left = (n_fft - win_length) / 2;
    for (int64_t i = 0; i < win_length; ++i) win_full[left + i] = tmp[i];
}

template <typename T>
Tensor stft_impl(const Tensor& work, int64_t n_fft, int64_t hop, int64_t win,
                 const std::optional<Tensor>& window, bool normalized, bool onesided,
                 bool return_complex, bool was_1d) {
    using C = std::complex<T>;
    std::vector<int64_t> wsizes = sizes_of(work);  // (batch, plen)
    const int64_t batch = wsizes[0];
    const int64_t plen = wsizes[1];
    const int64_t n_frames = 1 + (plen - n_fft) / hop;
    const int64_t n_freq = onesided ? infer_ft_real_to_complex_onesided_size(n_fft) : n_fft;

    std::vector<T> win_full;
    fill_win_full(win_full, window, win, n_fft);

    // frames buffer (batch*n_frames, n_fft) — ATen's time2col as_strided step
    Tensor frames({batch * n_frames, n_fft}, work.dtype());
    {
        const T* srcp = static_cast<const T*>(work.data_ptr());
        T* frp = static_cast<T*>(frames.data_ptr());
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t t = 0; t < n_frames; ++t) {
                T* dst = frp + (b * n_frames + t) * n_fft;
                const T* row = srcp + b * plen + t * hop;
                for (int64_t k = 0; k < n_fft; ++k) dst[k] = row[k] * win_full[k];
            }
        }
    }

    // batched FFT over all frames via pocketfft
    auto mode = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::none;
    const T fct = norm_factor<T>(mode, n_fft);
    Tensor spec({batch * n_frames, n_freq}, complex_of_real(work.dtype()));
    {
        auto pshape = pocketfft::shape_t{size_t(batch * n_frames), size_t(n_fft)};
        auto sin_ = contig_byte_strides({batch * n_frames, n_fft}, sizeof(T));
        auto sout = contig_byte_strides({batch * n_frames, n_freq}, sizeof(C));
        if (onesided) {
            pocketfft::r2c<T>(pshape, sin_, sout, 1, true,
                              static_cast<const T*>(frames.data_ptr()),
                              static_cast<C*>(spec.data_ptr()), fct, 1);
        } else {
            // promote frames to complex then c2c
            Tensor cframes(std::vector<int64_t>{batch * n_frames, n_fft}, complex_of_real(work.dtype()));
            const T* rp = static_cast<const T*>(frames.data_ptr());
            C* cp = static_cast<C*>(cframes.data_ptr());
            for (int64_t i = 0; i < frames.numel(); ++i) cp[i] = C(rp[i], T(0));
            pocketfft::c2c<T>(pshape, contig_byte_strides({batch * n_frames, n_fft}, sizeof(C)),
                              sout, {1}, true,
                              static_cast<const C*>(cframes.data_ptr()),
                              static_cast<C*>(spec.data_ptr()), fct, 1);
        }
    }

    // output layout (batch, freq, frames); squeeze batch when input was 1D
    std::vector<int64_t> out_sizes = was_1d ? std::vector<int64_t>{n_freq, n_frames}
                                            : std::vector<int64_t>{batch, n_freq, n_frames};
    Tensor out;
    const C* sp = static_cast<const C*>(spec.data_ptr());
    if (return_complex) {
        out = Tensor(out_sizes, complex_of_real(work.dtype()));
        C* op = static_cast<C*>(out.data_ptr());
        for (int64_t b = 0; b < batch; ++b)
            for (int64_t t = 0; t < n_frames; ++t)
                for (int64_t k = 0; k < n_freq; ++k)
                    op[(size_t(b) * n_freq + k) * n_frames + t] =
                        sp[(size_t(b) * n_frames + t) * n_freq + k];
    } else {
        out_sizes.push_back(2);
        out = Tensor(out_sizes, work.dtype());
        T* op = static_cast<T*>(out.data_ptr());
        for (int64_t b = 0; b < batch; ++b)
            for (int64_t t = 0; t < n_frames; ++t)
                for (int64_t k = 0; k < n_freq; ++k) {
                    const size_t base = ((size_t(b) * n_freq + k) * n_frames + t) * 2;
                    const C& v = sp[(size_t(b) * n_frames + t) * n_freq + k];
                    op[base] = v.real();
                    op[base + 1] = v.imag();
                }
    }
    return out;
}

}  // namespace

Tensor stft_cpu(const Tensor& self, int64_t n_fft, std::optional<int64_t> hop_length,
                std::optional<int64_t> win_length, const std::optional<Tensor>& window,
                bool center, std::string pad_mode, bool normalized, bool onesided,
                bool return_complex) {
    TP_CHECK(!is_cplx(self.dtype()), "stft: complex input not supported; use a real waveform");
    TP_CHECK(self.dtype() == DType::Float32 || self.dtype() == DType::Float64,
             "stft: expected a floating point input");
    TP_CHECK(self.dim() >= 1 && self.dim() <= 2, "stft: expected 1D or 2D input");
    TP_CHECK(pad_mode == "constant" || pad_mode == "reflect", "stft: unsupported pad_mode");

    const int64_t hop = hop_length.value_or(n_fft >> 2);
    const int64_t win = win_length.value_or(n_fft);
    TP_CHECK(hop > 0, "stft: expected hop_length > 0");
    TP_CHECK(win > 0 && win <= n_fft, "stft: expected 0 < win_length <= n_fft");
    if (window.has_value()) {
        TP_CHECK(window->dim() == 1 && window->size(0) == win,
                 "stft: expected a 1D window tensor of size equal to win_length");
    }

    Tensor x = self.contiguous();
    const bool was_1d = x.dim() == 1;
    if (was_1d) x = x.unsqueeze(0);

    if (center) {
        const int64_t pad = n_fft / 2;
        if (x.dtype() == DType::Float32) x = pad_time_axis<float>(x, pad, pad_mode);
        else x = pad_time_axis<double>(x, pad, pad_mode);
    }
    const int64_t plen = x.size(1);
    TP_CHECK(n_fft > 0 && n_fft <= plen, "stft: expected 0 < n_fft <= signal length");

    if (x.dtype() == DType::Float32)
        return stft_impl<float>(x, n_fft, hop, win, window, normalized, onesided,
                                return_complex, was_1d);
    return stft_impl<double>(x, n_fft, hop, win, window, normalized, onesided,
                             return_complex, was_1d);
}

namespace {

// istft core: input (batch, freq, frames) complex; mirrors ATen istft.
template <typename T>
Tensor istft_impl(const Tensor& input, int64_t n_fft, int64_t hop, int64_t win,
                  const std::optional<Tensor>& window, bool center, bool normalized,
                  bool onesided, std::optional<int64_t> length) {
    using C = std::complex<T>;
    const auto mode = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::by_n;
    std::vector<int64_t> isizes = sizes_of(input);
    // ATen istft checks the *real view* (3 or 4 dims); on the complex tensor
    // this is 2D (freq, frames) -> (len,) or 3D (batch, freq, frames) -> (B, len).
    TP_CHECK(isizes.size() == 2 || isizes.size() == 3,
             "istft: expected a complex tensor with 2 or 3 dimensions");
    const bool was_2d = isizes.size() == 2;
    const int64_t frames = isizes.back();
    const int64_t fft_size = isizes[isizes.size() - 2];
    const int64_t batch = was_2d ? 1 : isizes[0];
    const int64_t expected_len = n_fft + hop * (frames - 1);
    if (onesided) {
        TP_CHECK(fft_size == n_fft / 2 + 1,
                 "istft: frequency dim must equal n_fft/2+1 when onesided");
    } else {
        TP_CHECK(fft_size == n_fft, "istft: frequency dim must equal n_fft when onesided=False");
    }

    // window center-padded to n_fft with zeros when defined (ATen constant_pad_nd),
    // rectangular ones only when the window is absent.
    std::vector<T> win_full(n_fft, window.has_value() ? T(0) : T(1));
    {
        std::vector<T> tmp(win);
        if (window.has_value()) {
            Tensor w = window->contiguous();
            TP_CHECK(w.dim() == 1 && w.size(0) == win,
                     "istft: Invalid window shape; window has to be 1D and of length win_length");
            if (w.dtype() == DType::Float64 && !std::is_same_v<T, double>) {
                const double* dp = static_cast<const double*>(w.data_ptr());
                for (int64_t i = 0; i < win; ++i) tmp[i] = T(dp[i]);
            } else if (w.dtype() == DType::Float32 && std::is_same_v<T, double>) {
                const float* fp = static_cast<const float*>(w.data_ptr());
                for (int64_t i = 0; i < win; ++i) tmp[i] = T(fp[i]);
            } else {
                const T* p = static_cast<const T*>(w.data_ptr());
                for (int64_t i = 0; i < win; ++i) tmp[i] = p[i];
            }
            const int64_t left = (n_fft - win) / 2;
            for (int64_t i = 0; i < win; ++i) win_full[left + i] = tmp[i];
        }
    }
    // gather columns into (batch*frames, bins) — ATen's transpose(1,2) step.
    // For c2r pocketfft reads only the first n/2+1 Hermitian bins.
    const int64_t bins = n_fft / 2 + 1;
    Tensor cols({batch * frames, bins}, input.dtype());
    {
        const C* src = static_cast<const C*>(input.contiguous().data_ptr());
        C* dst = static_cast<C*>(cols.data_ptr());
        for (int64_t b = 0; b < batch; ++b)
            for (int64_t t = 0; t < frames; ++t)
                for (int64_t k = 0; k < bins; ++k)
                    dst[(size_t(b) * frames + t) * bins + k] =
                        src[(size_t(b) * fft_size + k) * frames + t];
    }

    // batched inverse rfft: fct implements ATen's by_n/by_root_n choice
    Tensor time_frames({batch * frames, n_fft}, real_of_complex(input.dtype()));
    {
        auto pshape_out = pocketfft::shape_t{size_t(batch * frames), size_t(n_fft)};
        auto sin_ = contig_byte_strides({batch * frames, bins}, sizeof(C));
        auto sout = contig_byte_strides({batch * frames, n_fft}, sizeof(T));
        pocketfft::c2r<T>(pshape_out, sin_, sout, 1, /*forward=*/false,
                          static_cast<const C*>(cols.data_ptr()),
                          static_cast<T*>(time_frames.data_ptr()),
                          norm_factor<T>(mode, n_fft), 1);
    }

    // overlap-add with the window and envelope accumulation
    const T* tfp = static_cast<const T*>(time_frames.data_ptr());
    std::vector<T> y(size_t(batch) * expected_len, T(0));
    std::vector<T> env(size_t(batch) * expected_len, T(0));
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < frames; ++t) {
            const T* fr = tfp + (size_t(b) * frames + t) * n_fft;
            T* yr = y.data() + size_t(b) * expected_len;
            T* er = env.data() + size_t(b) * expected_len;
            const int64_t off = t * hop;
            for (int64_t n = 0; n < n_fft; ++n) {
                yr[off + n] += fr[n] * win_full[n];
                er[off + n] += win_full[n] * win_full[n];
            }
        }
    }

    // crop / length handling (mirrors ATen)
    const int64_t start = center ? n_fft / 2 : 0;
    int64_t end;
    if (length.has_value()) end = start + *length;
    else if (center) end = expected_len - n_fft / 2;
    else end = expected_len;
    end = std::min(end, expected_len);
    TP_CHECK(end > start, "istft: requested output length is too small");

    std::vector<int64_t> out_sizes = was_2d ? std::vector<int64_t>{end - start}
                                            : std::vector<int64_t>{batch, end - start};
    Tensor out(out_sizes, real_of_complex(input.dtype()));
    T* op = static_cast<T*>(out.data_ptr());
    for (int64_t b = 0; b < batch; ++b) {
        const T* yr = y.data() + size_t(b) * expected_len;
        const T* er = env.data() + size_t(b) * expected_len;
        for (int64_t i = start; i < end; ++i) {
            TP_CHECK(er[i] >= T(1e-11), "istft: window overlap-add envelope collapsed");
            op[size_t(b) * (end - start) + (i - start)] = yr[i] / er[i];
        }
    }
    return out;
}

// stft adjoint. grad_output: (batch?, freq, frames); self: original waveform.
template <typename T>
Tensor stft_backward_impl(const Tensor& grad_output, const Tensor& self, int64_t n_fft,
                          int64_t hop, int64_t win_length, const std::optional<Tensor>& window,
                          bool center, bool normalized, bool onesided, const std::string& pad_mode) {
    using C = std::complex<T>;
    const auto mode = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::none;
    const int64_t n_freq = onesided ? infer_ft_real_to_complex_onesided_size(n_fft) : n_fft;

    std::vector<int64_t> gsizes = sizes_of(grad_output);
    const int64_t frames = gsizes.back();
    const int64_t gfreq = gsizes[gsizes.size() - 2];
    TP_CHECK(gfreq == n_freq, "stft_backward: frequency dim mismatch");
    const bool was_1d = gsizes.size() == 2;
    const int64_t batch = was_1d ? 1 : gsizes[0];

    std::vector<T> win_full;
    fill_win_full(win_full, window, win_length, n_fft);

    // adjoint scale equals the primal forward factor
    const T s_fwd = norm_factor<T>(mode, n_fft);

    // gather grad columns into (batch*frames, n_freq)
    Tensor cols({batch * frames, n_freq}, complex_of_real(self.dtype()));
    {
        const C* gsrc = static_cast<const C*>(grad_output.contiguous().data_ptr());
        C* dst = static_cast<C*>(cols.data_ptr());
        for (int64_t b = 0; b < batch; ++b)
            for (int64_t k = 0; k < n_freq; ++k)
                for (int64_t t = 0; t < frames; ++t)
                    dst[(size_t(b) * frames + t) * n_freq + k] =
                        gsrc[(size_t(b) * n_freq + k) * frames + t];
    }

    // ATen composes the spectral adjoint via fft_r2c_backward: zero-fill the
    // twosided spectrum from the onesided grad, run the INVERSE c2c carrying
    // the forward's normalization, then project to the real part.
    Tensor full({batch * frames, n_fft}, complex_of_real(self.dtype()));
    full.zero_();
    if (n_freq < n_fft) full.slice(1, 0, n_freq).copy_(cols);
    else full.copy_(cols);
    Tensor ctime({batch * frames, n_fft}, complex_of_real(self.dtype()));
    {
        auto pshape = pocketfft::shape_t{size_t(batch * frames), size_t(n_fft)};
        auto strides = contig_byte_strides({batch * frames, n_fft}, sizeof(C));
        pocketfft::c2c<T>(pshape, strides, strides, {1}, /*forward=*/false,
                          static_cast<const C*>(full.data_ptr()),
                          static_cast<C*>(ctime.data_ptr()), s_fwd, 1);
    }
    Tensor time_frames({batch * frames, n_fft}, self.dtype());
    {
        const C* cp = static_cast<const C*>(ctime.data_ptr());
        T* rp = static_cast<T*>(time_frames.data_ptr());
        const int64_t total = batch * frames * n_fft;
        for (int64_t i = 0; i < total; ++i) rp[i] = cp[i].real();
    }

    // multiply window and overlap-add scatter into padded positions
    const int64_t orig_len = self.size(self.dim() - 1);
    const int64_t padded_len = orig_len + (center ? (n_fft / 2) * 2 : 0);
    std::vector<T> xg(size_t(batch) * padded_len, T(0));
    const T* tfp = static_cast<const T*>(time_frames.data_ptr());
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < frames; ++t) {
            const T* fr = tfp + (size_t(b) * frames + t) * n_fft;
            T* row = xg.data() + size_t(b) * padded_len + size_t(t) * hop;
            for (int64_t n = 0; n < n_fft; ++n) row[n] += fr[n] * win_full[n];
        }
    }

    // undo center padding (adjoint)
    std::vector<int64_t> out_sizes = sizes_of(self);
    Tensor out(out_sizes, self.dtype());
    if (center) {
        unpad_scatter_time_axis<T>(xg.data(), batch, padded_len, n_fft / 2, pad_mode,
                                   static_cast<T*>(out.data_ptr()));
    } else {
        std::memcpy(out.data_ptr(), xg.data(), sizeof(T) * xg.size());
    }
    return out;
}

}  // namespace

Tensor istft_cpu(const Tensor& input, int64_t n_fft, std::optional<int64_t> hop_length,
                 std::optional<int64_t> win_length, const std::optional<Tensor>& window,
                 bool center, bool normalized, bool onesided, std::optional<int64_t> length,
                 bool return_complex) {
    TP_CHECK(is_cplx(input.dtype()),
             "istft requires a complex input matching stft(return_complex=True)");
    TP_CHECK(!return_complex, "istft: complex output path not supported");
    const int64_t hop = hop_length.value_or(n_fft >> 2);
    const int64_t win = win_length.value_or(n_fft);
    TP_CHECK(hop > 0 && hop <= win, "istft: expected 0 < hop_length <= win_length");
    TP_CHECK(win > 0 && win <= n_fft, "istft: expected 0 < win_length <= n_fft");

    if (input.dtype() == DType::ComplexFloat)
        return istft_impl<float>(input, n_fft, hop, win, window, center, normalized,
                                 onesided, length);
    return istft_impl<double>(input, n_fft, hop, win, window, center, normalized,
                              onesided, length);
}

Tensor stft_backward_cpu(const Tensor& grad_output, const Tensor& self, int64_t n_fft,
                         std::optional<int64_t> hop_length, std::optional<int64_t> win_length,
                         const std::optional<Tensor>& window, bool center, std::string pad_mode,
                         bool normalized, bool onesided) {
    TP_CHECK(!is_cplx(self.dtype()), "stft_backward: expected real input");
    TP_CHECK(!center || pad_mode == "constant" || pad_mode == "reflect",
             "stft_backward: unsupported pad_mode (use constant|reflect)");
    const int64_t hop_r = hop_length.value_or(n_fft >> 2);
    const int64_t win_r = win_length.value_or(n_fft);
    Tensor x = self.contiguous();
    if (grad_output.dtype() == DType::ComplexFloat)
        return stft_backward_impl<float>(grad_output, x, n_fft, hop_r, win_r, window,
                                         center, normalized, onesided, pad_mode);
    return stft_backward_impl<double>(grad_output, x, n_fft, hop_r, win_r, window,
                                      center, normalized, onesided, pad_mode);
}

TENSORPLAY_LIBRARY_IMPL(CPU, SpectralKernels) {
    m.impl("fft_fft", fft_fft_cpu);
    m.impl("fft_ifft", fft_ifft_cpu);
    m.impl("fft_rfft", fft_rfft_cpu);
    m.impl("fft_irfft", fft_irfft_cpu);
    m.impl("fft_fft_backward", fft_fft_backward_cpu);
    m.impl("fft_ifft_backward", fft_ifft_backward_cpu);
    m.impl("fft_rfft_backward", fft_rfft_backward_cpu);
    m.impl("fft_irfft_backward", fft_irfft_backward_cpu);
    m.impl("hann_window", hann_window_cpu);
    m.impl("hamming_window", hamming_window_cpu);
    m.impl("bartlett_window", bartlett_window_cpu);
    m.impl("blackman_window", blackman_window_cpu);
    m.impl("stft", stft_cpu);
    m.impl("istft", istft_cpu);
    m.impl("stft_backward", stft_backward_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
