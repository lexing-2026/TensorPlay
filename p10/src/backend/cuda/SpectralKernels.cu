// CUDA spectral kernels for the audio stack — cuFFT backend.
//
// batched cuFFT plans execute the transforms; CUFFT_CHECK is adapted from
// the vendored CuFFTUtils.h. Transforms along an interior dim gather lines
// into a packed (lines, n) buffer, run a batched cufftPlan1d, and scatter
// back — semantically identical to the CPU pocketfft path in
// native/SpectralOpsUtils.h).

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <mutex>
#include <map>

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = (condition); \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

inline std::string cufft_error_string(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
        default: return "unknown cuFFT error";
    }
}
#define CUFFT_CHECK(condition) \
  do { \
    cufftResult error = (condition); \
    if (error != CUFFT_SUCCESS) { \
      TP_THROW(RuntimeError, std::string("cuFFT error: ") + cufft_error_string(error)); \
    } \
  } while (0)

namespace {

constexpr int kThreads = 256;

inline std::vector<int64_t> sizes_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

inline int64_t wrap_dim(int64_t idx, int64_t ndim) {
    if (idx < 0) idx += ndim;
    TP_CHECK(idx >= 0 && idx < ndim, "Dimension out of range");
    return idx;
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

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
enum class fft_norm_mode { none, by_root_n, by_n };

template <typename T>
__host__ __device__ inline T norm_factor(fft_norm_mode mode, int64_t n) {
    switch (mode) {
        case fft_norm_mode::none:      return T(1);
        case fft_norm_mode::by_root_n: return T(1) / std::sqrt(T(n));
        case fft_norm_mode::by_n:      return T(1) / T(n);
    }
    return T(1);
}

inline fft_norm_mode norm_from_string(const std::string& norm, bool forward) {
    if (norm == "backward") return forward ? fft_norm_mode::none : fft_norm_mode::by_n;
    if (norm == "forward")  return forward ? fft_norm_mode::by_n : fft_norm_mode::none;
    if (norm == "ortho")    return fft_norm_mode::by_root_n;
    TP_THROW(RuntimeError, "Invalid normalization mode: \"", norm, "\"");
    return fft_norm_mode::none;
}

inline int64_t infer_onesided(int64_t real_size) { return real_size / 2 + 1; }

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
cufftHandle acquire_plan(cufftType type, int64_t n, int64_t batch) {
    struct Key {
        int type; long long n; long long b;
        bool operator<(const Key& o) const {
            if (type != o.type) return type < o.type;
            if (n != o.n) return n < o.n;
            return b < o.b;
        }
    };
    static std::mutex mu;
    static std::map<Key, cufftHandle> cache;
    const Key key{(int)type, n, batch};
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, (int)n, type, (int)batch));
    CUFFT_CHECK(cufftSetStream(plan, getCurrentCUDAStream().stream()));
    cache.emplace(key, plan);
    return plan;
}

}  // namespace

namespace {

// Because every spectral input is made contiguous first and the transform
// dim is moved last via permute+contiguous (dispatcher ops), packed buffers
// are plain (lines, n) with lines = numel / len.
struct LineInfo {
    int64_t lines;
    int64_t len;
};

template <bool IsDouble>
struct CudaTypes {
    using R = std::conditional_t<IsDouble, double, float>;
    using C = std::conditional_t<IsDouble, cufftDoubleComplex, cufftComplex>;
};

LineInfo last_dim_lines(const std::vector<int64_t>& sizes) {
    LineInfo li;
    li.len = sizes.back();
    li.lines = 1;
    for (size_t i = 0; i + 1 < sizes.size(); ++i) li.lines *= sizes[i];
    return li;
}

// permute dims so that `dim` moves to the end; returns (tensor, inverse_perm)
std::pair<Tensor, std::vector<int64_t>> move_dim_last(const Tensor& contig, int64_t dim) {
    std::vector<int64_t> sizes = sizes_of(contig);
    const int nd = (int)sizes.size();
    if (dim == nd - 1) return {contig, {}};
    std::vector<int64_t> perm;
    for (int i = 0; i < nd; ++i) if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    Tensor t = contig.permute(perm).contiguous();
    // inverse permutation
    std::vector<int64_t> inv(nd);
    for (int i = 0; i < nd; ++i) inv[perm[i]] = i;
    return {t, inv};
}

template <typename T>
Tensor resize_last_dim(const Tensor& contig, int64_t want) {
    std::vector<int64_t> sizes = sizes_of(contig);
    const int64_t have = sizes.back();
    if (have == want) return contig;
    TP_CHECK(want > 0, "resize: invalid length");
    std::vector<int64_t> out_sizes = sizes;
    out_sizes.back() = want;
    Tensor out = have > want ? Tensor(out_sizes, contig.dtype())
                             : Tensor::zeros(out_sizes, contig.dtype(), contig.device());
    const int64_t lines = have > want ? out.numel() / want : contig.numel() / have;
    auto stream = getCurrentCUDAStream().stream();
    const size_t elem = contig.itemsize();
    CUDA_CHECK(cudaMemcpy2DAsync(
        out.data_ptr(), (size_t)want * elem,
        contig.data_ptr(), (size_t)have * elem,
        std::min(have, want) * elem, (size_t)lines,
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// gather / scatter / scale kernels
// ---------------------------------------------------------------------------

namespace {

template <typename T>
__global__ void gather_pad_r2r_kernel(int64_t lines, int64_t n_src, int64_t n_dst,
                                      const T* __restrict__ src, T* __restrict__ dst) {
    const int64_t line = blockIdx.x;
    if (line >= lines) return;
    const T* s = src + line * n_src;
    T* d = dst + line * n_dst;
    for (int64_t i = threadIdx.x; i < n_dst; i += blockDim.x) {
        d[i] = (i < n_src) ? s[i] : T(0);
    }
}

template <typename C>
__global__ void gather_pad_c2c_kernel(int64_t lines, int64_t n_src, int64_t n_dst,
                                      const C* __restrict__ src, C* __restrict__ dst) {
    const int64_t line = blockIdx.x;
    if (line >= lines) return;
    const C* s = src + line * n_src;
    C* d = dst + line * n_dst;
    for (int64_t i = threadIdx.x; i < n_dst; i += blockDim.x) {
        d[i] = (i < n_src) ? s[i] : C();
    }
}

template <typename T>
__global__ void copy_r2r_kernel(int64_t total, const T* __restrict__ src, T* __restrict__ dst) {
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) dst[i] = src[i];
}

template <typename C>
__global__ void copy_c2c_kernel(int64_t total, const C* __restrict__ src, C* __restrict__ dst) {
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) dst[i] = src[i];
}

template <typename T>
__global__ void scale_r_kernel(T* data, int64_t total, T scale) {
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) data[i] *= scale;
}

template <typename C, typename S>
__global__ void scale_c_kernel(C* data, int64_t total, S scale) {
    // cufftComplex/cufftDoubleComplex are float2/double2: scale componentwise
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

// Project interleaved complex data to its real part.
__global__ void cplx_real_f64_kernel(const cufftDoubleComplex* __restrict__ src,
                                     double* __restrict__ dst, int64_t total) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i < total) dst[i] = src[i].x;
}

__global__ void cplx_real_f32_kernel(const cufftComplex* __restrict__ src,
                                     float* __restrict__ dst, int64_t total) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i < total) dst[i] = src[i].x;
}

template <typename T>
__global__ void fill_r_kernel(T* data, int64_t n, T value) {
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = value;
}

template <typename T>
__global__ void window_fill_kernel(T* w, int64_t n, int64_t L,
                                   double alpha, double beta, int kind) {
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    constexpr double kPi = 3.141592653589793238463;
    double v;
    if (kind == 2) {  // bartlett
        const double num = 2.0 * static_cast<double>(i);
        if (num < static_cast<double>(L)) {
            v = num / static_cast<double>(L);
        } else if (num > static_cast<double>(L)) {
            v = 2.0 - num / static_cast<double>(L);
        } else {
            v = 1.0;
        }
    } else if (kind == 3) {  // blackman
        const double a = kPi * i / static_cast<double>(L);
        v = 0.42 + 0.5 * std::cos(2 * a) - 0.08 * std::cos(4 * a);
    } else {          // hann (alpha=beta=0.5) / hamming
        v = alpha - beta * std::cos(2.0 * kPi * i / L);
    }
    w[i] = static_cast<T>(v);
}

}  // namespace


// ---------------------------------------------------------------------------
// dtype helpers
// ---------------------------------------------------------------------------

namespace {

inline DType real_dtype_of(DType dt) {
    return dt == DType::ComplexDouble ? DType::Float64 : DType::Float32;
}
inline DType complex_dtype_of(DType dt) {
    return dt == DType::Float64 ? DType::ComplexDouble : DType::ComplexFloat;
}

// ---------------------------------------------------------------------------
// batched cuFFT plan over contiguous lines, normalization factor applied to
// ---------------------------------------------------------------------------

template <bool IsDouble>
Tensor core_c2c_impl(const Tensor& x, int64_t n_eff, fft_norm_mode mode, bool forward) {
    using R = typename CudaTypes<IsDouble>::R;
    using C = typename CudaTypes<IsDouble>::C;
    const LineInfo li = last_dim_lines(sizes_of(x));

    auto stream = getCurrentCUDAStream().stream();
    Tensor packed({li.lines, n_eff}, x.dtype());
    {
        const C* src = static_cast<const C*>(x.data_ptr());
        C* dst = static_cast<C*>(packed.data_ptr());
        if constexpr (IsDouble) {
            gather_pad_c2c_kernel<cufftDoubleComplex><<<li.lines, kThreads, 0, stream>>>(
                li.lines, li.len, n_eff,
                reinterpret_cast<const cufftDoubleComplex*>(src),
                reinterpret_cast<cufftDoubleComplex*>(dst));
        } else {
            gather_pad_c2c_kernel<cufftComplex><<<li.lines, kThreads, 0, stream>>>(
                li.lines, li.len, n_eff,
                reinterpret_cast<const cufftComplex*>(src),
                reinterpret_cast<cufftComplex*>(dst));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // Guard zero-length transforms before creating a plan.
    if (n_eff > 0 && li.lines > 0) {
        cufftHandle plan = acquire_plan(IsDouble ? CUFFT_Z2Z : CUFFT_C2C, n_eff, li.lines);
        if (IsDouble) {
            CUFFT_CHECK(cufftExecZ2Z(plan,
                reinterpret_cast<cufftDoubleComplex*>(packed.data_ptr()),
                reinterpret_cast<cufftDoubleComplex*>(packed.data_ptr()),
                forward ? CUFFT_FORWARD : CUFFT_INVERSE));
        } else {
            CUFFT_CHECK(cufftExecC2C(plan,
                reinterpret_cast<cufftComplex*>(packed.data_ptr()),
                reinterpret_cast<cufftComplex*>(packed.data_ptr()),
                forward ? CUFFT_FORWARD : CUFFT_INVERSE));
        }
    }

    const R fct = norm_factor<R>(mode, n_eff);
    if (fct != R(1)) {
        const int64_t total = li.lines * n_eff;
        if constexpr (IsDouble) {
            scale_c_kernel<cufftDoubleComplex, double><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                reinterpret_cast<cufftDoubleComplex*>(packed.data_ptr()), total, fct);
        } else {
            scale_c_kernel<cufftComplex, float><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                reinterpret_cast<cufftComplex*>(packed.data_ptr()), total, fct);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return packed;
}

// ---------------------------------------------------------------------------
// spectrum (N/2+1 bins), matching infer_ft_real_to_complex_onesided_size.
// ---------------------------------------------------------------------------

template <bool IsDouble>
Tensor core_r2c_impl(const Tensor& x, int64_t n_eff, fft_norm_mode mode) {
    using R = typename CudaTypes<IsDouble>::R;
    using C = typename CudaTypes<IsDouble>::C;
    const LineInfo li = last_dim_lines(sizes_of(x));
    const int64_t n_out = infer_onesided(n_eff);

    auto stream = getCurrentCUDAStream().stream();
    Tensor packed({li.lines, n_eff}, real_dtype_of(x.dtype()));
    Tensor spec({li.lines, n_out}, complex_dtype_of(x.dtype()));
    {
        const R* src = static_cast<const R*>(x.data_ptr());
        R* dst = static_cast<R*>(packed.data_ptr());
        if constexpr (IsDouble) {
            gather_pad_r2r_kernel<double><<<li.lines, kThreads, 0, stream>>>(
                li.lines, li.len, n_eff, src, dst);
        } else {
            gather_pad_r2r_kernel<float><<<li.lines, kThreads, 0, stream>>>(
                li.lines, li.len, n_eff, src, dst);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    if (n_eff > 0 && li.lines > 0) {
        cufftHandle plan = acquire_plan(IsDouble ? CUFFT_D2Z : CUFFT_R2C, n_eff, li.lines);
        if (IsDouble) {
            CUFFT_CHECK(cufftExecD2Z(plan,
                reinterpret_cast<cufftDoubleReal*>(packed.data_ptr()),
                reinterpret_cast<cufftDoubleComplex*>(spec.data_ptr())));
        } else {
            CUFFT_CHECK(cufftExecR2C(plan,
                reinterpret_cast<cufftReal*>(packed.data_ptr()),
                reinterpret_cast<cufftComplex*>(spec.data_ptr())));
        }
    }

    const R fct = norm_factor<R>(mode, n_eff);
    if (fct != R(1)) {
        const int64_t total = li.lines * n_out;
        if constexpr (IsDouble) {
            scale_c_kernel<cufftDoubleComplex, double><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                reinterpret_cast<cufftDoubleComplex*>(spec.data_ptr()), total, fct);
        } else {
            scale_c_kernel<cufftComplex, float><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                reinterpret_cast<cufftComplex*>(spec.data_ptr()), total, fct);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return spec;
}

// ---------------------------------------------------------------------------
// (N/2+1 bins); strided prefix-copy then batched C2R.
// ---------------------------------------------------------------------------

template <typename C>
__global__ void copy_strided_c2c_kernel(int64_t lines, int64_t cols, int64_t src_stride,
                                        const C* __restrict__ src, C* __restrict__ dst) {
    const int64_t line = blockIdx.x;
    if (line >= lines) return;
    const C* s = src + line * src_stride;
    C* d = dst + line * cols;
    for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) d[i] = s[i];
}

template <bool IsDouble>
Tensor core_c2r_impl(const Tensor& x, int64_t n_eff, fft_norm_mode mode) {
    using R = typename CudaTypes<IsDouble>::R;
    using C = typename CudaTypes<IsDouble>::C;
    const LineInfo li = last_dim_lines(sizes_of(x));
    const int64_t bins_in = x.size(-1);
    const int64_t bins_needed = n_eff / 2 + 1;
    TP_CHECK(bins_in >= bins_needed, "irfft: not enough frequency bins");

    auto stream = getCurrentCUDAStream().stream();
    Tensor cols({li.lines, bins_needed}, x.dtype());
    {
        if constexpr (IsDouble) {
            copy_strided_c2c_kernel<cufftDoubleComplex><<<li.lines, kThreads, 0, stream>>>(
                li.lines, bins_needed, bins_in,
                reinterpret_cast<const cufftDoubleComplex*>(x.data_ptr()),
                reinterpret_cast<cufftDoubleComplex*>(cols.data_ptr()));
        } else {
            copy_strided_c2c_kernel<cufftComplex><<<li.lines, kThreads, 0, stream>>>(
                li.lines, bins_needed, bins_in,
                reinterpret_cast<const cufftComplex*>(x.data_ptr()),
                reinterpret_cast<cufftComplex*>(cols.data_ptr()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    Tensor out({li.lines, n_eff}, real_dtype_of(x.dtype()));
    if (n_eff > 0 && li.lines > 0) {
        cufftHandle plan = acquire_plan(IsDouble ? CUFFT_Z2D : CUFFT_C2R, n_eff, li.lines);
        if (IsDouble) {
            CUFFT_CHECK(cufftExecZ2D(plan,
                reinterpret_cast<cufftDoubleComplex*>(cols.data_ptr()),
                reinterpret_cast<cufftDoubleReal*>(out.data_ptr())));
        } else {
            CUFFT_CHECK(cufftExecC2R(plan,
                reinterpret_cast<cufftComplex*>(cols.data_ptr()),
                reinterpret_cast<cufftReal*>(out.data_ptr())));
        }
    }

    const R fct = norm_factor<R>(mode, n_eff);
    if (fct != R(1)) {
        const int64_t total = li.lines * n_eff;
        if constexpr (IsDouble) {
            scale_r_kernel<double><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                reinterpret_cast<double*>(out.data_ptr()), total, fct);
        } else {
            scale_r_kernel<float><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                reinterpret_cast<float*>(out.data_ptr()), total, fct);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// native/SpectralOps.cpp fft_c2c / fft_r2c / fft_c2r (:215-330):
//   n defaults to size(dim); resize_fft_input slices from 0 or zero-pads;
//   norm_from_string picks the output factor per direction.
// ---------------------------------------------------------------------------

namespace {

// Shared: make contiguous, move transform dim last (permute+contiguous via
// the dispatcher), return inverse permutation for the final layout fix-up.
std::pair<Tensor, std::vector<int64_t>> prepare_lastdim(const Tensor& self, int64_t dim) {
    Tensor x = self.contiguous();
    return move_dim_last(x, dim);
}

Tensor finish_layout(Tensor&& t, const std::vector<int64_t>& inv_perm) {
    if (inv_perm.empty()) return std::move(t);
    return t.permute(inv_perm).contiguous();
}

}  // namespace

namespace {

// zero-imaginary complex copy (SpectralOps.cpp fft_r2c "fft"/"ifft" path).
template <bool IsDouble>
__global__ void real_to_cplx_kernel(
    int64_t n, const typename CudaTypes<IsDouble>::R* __restrict__ src,
    typename CudaTypes<IsDouble>::C* __restrict__ dst) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    dst[i].x = src[i];
    dst[i].y = typename CudaTypes<IsDouble>::R(0);
}

template <bool IsDouble>
Tensor promote_real_for_c2c_cuda(const Tensor& self) {
    TP_CHECK(self.dtype() == DType::Float32 || self.dtype() == DType::Float64,
             "Unsupported input dtype for spectral op");
    Tensor x = self.contiguous();
    Tensor out(sizes_of(x), complex_dtype_of(x.dtype()));
    auto stream = getCurrentCUDAStream().stream();
    const int64_t n = x.numel();
    if constexpr (IsDouble) {
        real_to_cplx_kernel<true><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            n, static_cast<const double*>(x.data_ptr()),
            reinterpret_cast<cufftDoubleComplex*>(out.data_ptr()));
    } else {
        real_to_cplx_kernel<false><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            n, static_cast<const float*>(x.data_ptr()),
            reinterpret_cast<cufftComplex*>(out.data_ptr()));
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// Adjoint of the real->complex materialization: take the real part.
template <bool IsDouble>
__global__ void cplx_real_part_kernel(
    int64_t n, const typename CudaTypes<IsDouble>::C* __restrict__ src,
    typename CudaTypes<IsDouble>::R* __restrict__ dst) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i].x;
}

template <bool IsDouble>
Tensor extract_real_part_cuda(const Tensor& z) {
    Tensor zc = z.contiguous();
    Tensor out(sizes_of(zc), real_dtype_of(zc.dtype()));
    auto stream = getCurrentCUDAStream().stream();
    const int64_t n = zc.numel();
    if constexpr (IsDouble) {
        cplx_real_part_kernel<true><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            n, reinterpret_cast<const cufftDoubleComplex*>(zc.data_ptr()),
            static_cast<double*>(out.data_ptr()));
    } else {
        cplx_real_part_kernel<false><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            n, reinterpret_cast<const cufftComplex*>(zc.data_ptr()),
            static_cast<float*>(out.data_ptr()));
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

}  // namespace

Tensor fft_fft_cuda(const Tensor& self, int64_t n, int64_t dim, std::string norm) {
    const bool real_in = !is_cplx(self.dtype());
    Tensor inp = real_in
        ? (self.dtype() == DType::Float64 ? promote_real_for_c2c_cuda<true>(self)
                                          : promote_real_for_c2c_cuda<false>(self))
        : self;
    TP_CHECK(inp.dim() >= 1, "fft expects at least 1 dimension");
    dim = wrap_dim(dim, inp.dim());
    auto [x, inv] = prepare_lastdim(inp, dim);
    const int64_t N = x.size(-1);
    const int64_t n_eff = n > 0 ? n : N;
    TP_CHECK(n_eff >= 1, "Invalid number of data points specified");
    if (n > 0 && n != N) x = resize_last_dim<float>(x, n_eff);  // elem-size agnostic copy
    const auto mode = norm_from_string(norm, true);
    Tensor out = inp.dtype() == DType::ComplexDouble
        ? core_c2c_impl<true>(x, n_eff, mode, true)
        : core_c2c_impl<false>(x, n_eff, mode, true);
    return finish_layout(std::move(out), inv);
}

Tensor fft_ifft_cuda(const Tensor& self, int64_t n, int64_t dim, std::string norm) {
    const bool real_in = !is_cplx(self.dtype());
    Tensor inp = real_in
        ? (self.dtype() == DType::Float64 ? promote_real_for_c2c_cuda<true>(self)
                                          : promote_real_for_c2c_cuda<false>(self))
        : self;
    TP_CHECK(inp.dim() >= 1, "ifft expects at least 1 dimension");
    dim = wrap_dim(dim, inp.dim());
    auto [x, inv] = prepare_lastdim(inp, dim);
    const int64_t N = x.size(-1);
    const int64_t n_eff = n > 0 ? n : N;
    TP_CHECK(n_eff >= 1, "Invalid number of data points specified");
    if (n > 0 && n != N) x = resize_last_dim<float>(x, n_eff);
    const auto mode = norm_from_string(norm, false);
    Tensor out = inp.dtype() == DType::ComplexDouble
        ? core_c2c_impl<true>(x, n_eff, mode, false)
        : core_c2c_impl<false>(x, n_eff, mode, false);
    return finish_layout(std::move(out), inv);
}

Tensor fft_rfft_cuda(const Tensor& self, int64_t n, int64_t dim, std::string norm) {
    TP_CHECK(!is_cplx(self.dtype()), "fft.rfft expects a real input");
    TP_CHECK(self.dim() >= 1, "rfft expects at least 1 dimension");
    dim = wrap_dim(dim, self.dim());
    auto [x, inv] = prepare_lastdim(self, dim);
    const int64_t N = x.size(-1);
    const int64_t n_eff = n > 0 ? n : N;
    TP_CHECK(n_eff >= 1, "Invalid number of data points specified");
    if (n > 0 && n != N) x = resize_last_dim<float>(x, n_eff);
    const auto mode = norm_from_string(norm, true);
    Tensor out = self.dtype() == DType::Float64
        ? core_r2c_impl<true>(x, n_eff, mode)
        : core_r2c_impl<false>(x, n_eff, mode);
    return finish_layout(std::move(out), inv);
}

Tensor fft_irfft_cuda(const Tensor& self, int64_t n, int64_t dim, std::string norm) {
    TP_CHECK(is_cplx(self.dtype()), "fft.irfft expects a complex input");
    TP_CHECK(self.dim() >= 1, "irfft expects at least 1 dimension");
    dim = wrap_dim(dim, self.dim());
    auto [x, inv] = prepare_lastdim(self, dim);
    const int64_t F = x.size(-1);
    const int64_t n_eff = n > 0 ? n : 2 * (F - 1);
    TP_CHECK(n_eff >= 1, "Invalid number of data points specified");
    const auto mode = norm_from_string(norm, false);
    Tensor out = self.dtype() == DType::ComplexDouble
        ? core_c2r_impl<true>(x, n_eff, mode)
        : core_c2r_impl<false>(x, n_eff, mode);
    return finish_layout(std::move(out), inv);
}

// ---------------------------------------------------------------------------
// Backward helpers — adjoint = same internal transform with flipped direction
// and identical normalization enum, then resize back to the primal support.
// Formulas mirror upstream tools/autograd/derivatives.yaml:2963-2974:
//   _fft_r2c: fft_r2c_backward(grad, dim, normalization, onesided, size)
//   _fft_c2r: fft_c2r_backward(grad, dim, normalization)
//   _fft_c2c: _fft_c2c(grad, dim, normalization, !forward)
// ---------------------------------------------------------------------------

Tensor fft_fft_backward_cuda(const Tensor& grad, const Tensor& self, int64_t dim, std::string norm) {
    dim = wrap_dim(dim, self.dim());
    const bool real_primal = !is_cplx(self.dtype());
    auto [g, inv] = prepare_lastdim(grad, dim);
    const int64_t input_len = self.size(dim);
    const auto mode = norm_from_string(norm, true);  // forward was fft
    Tensor out = g.dtype() == DType::ComplexDouble
        ? core_c2c_impl<true>(g, g.size(-1), mode, /*forward=*/false)
        : core_c2c_impl<false>(g, g.size(-1), mode, false);
    out = resize_last_dim<float>(out, input_len);
    out = finish_layout(std::move(out), inv);
    // adjoint of the real->complex materialization is taking the real part
    return real_primal
        ? (out.dtype() == DType::ComplexDouble ? extract_real_part_cuda<true>(out)
                                               : extract_real_part_cuda<false>(out))
        : std::move(out);
}

Tensor fft_ifft_backward_cuda(const Tensor& grad, const Tensor& self, int64_t dim, std::string norm) {
    dim = wrap_dim(dim, self.dim());
    const bool real_primal = !is_cplx(self.dtype());
    auto [g, inv] = prepare_lastdim(grad, dim);
    const int64_t input_len = self.size(dim);
    const auto mode = norm_from_string(norm, false);  // forward was ifft
    Tensor out = g.dtype() == DType::ComplexDouble
        ? core_c2c_impl<true>(g, g.size(-1), mode, /*forward=*/true)
        : core_c2c_impl<false>(g, g.size(-1), mode, true);
    out = resize_last_dim<float>(out, input_len);
    out = finish_layout(std::move(out), inv);
    return real_primal
        ? (out.dtype() == DType::ComplexDouble ? extract_real_part_cuda<true>(out)
                                               : extract_real_part_cuda<false>(out))
        : std::move(out);
}

// onesided r2c == [zero-fill imag, c2c fwd, drop half]; backward ==
// [zero-fill twosided spectrum, INVERSE c2c with the forward's normalization,
// take real part].
namespace {
template <bool IsDouble>
Tensor rfft_backward_core_cuda(const Tensor& g, int64_t input_len, fft_norm_mode mode) {
    const int64_t bins = g.size(-1);
    std::vector<int64_t> sizes = sizes_of(g);
    sizes.back() = input_len;
    Tensor full = Tensor::zeros(sizes, g.dtype(), g.device());
    if (bins < input_len) full.slice(-1, 0, bins).copy_(g);
    else full.copy_(g);
    Tensor t = g.dtype() == DType::ComplexDouble
        ? core_c2c_impl<true>(full, input_len, mode, /*forward=*/false)
        : core_c2c_impl<false>(full, input_len, mode, /*forward=*/false);
    return extract_real_part_cuda<IsDouble>(t);
}
}  // namespace

Tensor fft_rfft_backward_cuda(const Tensor& grad, const Tensor& self, int64_t dim, std::string norm) {
    dim = wrap_dim(dim, self.dim());
    auto [g, inv] = prepare_lastdim(grad, dim);
    const int64_t input_len = self.size(dim);
    const auto mode = norm_from_string(norm, true);
    Tensor out = g.dtype() == DType::ComplexDouble
        ? rfft_backward_core_cuda<true>(g, input_len, mode)
        : rfft_backward_core_cuda<false>(g, input_len, mode);
    return finish_layout(std::move(out), inv);
}

// R2C of the real gradient with the primal normalization, then double the bins
// whose conjugate mirror fell outside the onesided range.
namespace {
template <bool IsDouble>
Tensor irfft_backward_core(const Tensor& g, int64_t freq_bins, fft_norm_mode mode) {
    const int64_t M = g.size(-1);
    Tensor spec = core_r2c_impl<IsDouble>(g, M, mode);
    const int64_t got_bins = spec.size(-1);
    const int64_t double_length = M - got_bins;
    if (double_length > 0) {
        Tensor scaled = spec.slice(-1, 1, 1 + double_length).mul(Scalar(2.0));
        spec.slice(-1, 1, 1 + double_length).copy_(scaled);
    }
    if (got_bins != freq_bins) spec = resize_last_dim<float>(spec, freq_bins);
    return spec;
}
}  // namespace

Tensor fft_irfft_backward_cuda(const Tensor& grad, const Tensor& self, int64_t dim, std::string norm) {
    dim = wrap_dim(dim, self.dim());
    auto [g, inv] = prepare_lastdim(grad, dim);
    const int64_t freq_bins = self.size(dim);
    const auto mode = norm_from_string(norm, false);
    Tensor out = g.dtype() == DType::Float64
        ? irfft_backward_core<true>(g, freq_bins, mode)
        : irfft_backward_core<false>(g, freq_bins, mode);
    return finish_layout(std::move(out), inv);
}

// ---------------------------------------------------------------------------
// (:1879-2010): bartlett/blackman/hamming/hann computed over the periodic
// length L = window_length + (periodic ? 1 : 0), hann = hamming(0.5, 0.5).
// ---------------------------------------------------------------------------

namespace {
Tensor window_cuda(int64_t out_len, int64_t formula_len, std::optional<DType> dtype_opt,
                   const char* name, double alpha, double beta, int kind) {
    // kind: 0=hann 1=hamming 2=bartlett 3=blackman
    if (out_len < 0) TP_THROW(ValueError, name, ": window_length must be non-negative");
    DType dt = dtype_opt.value_or(DType::Float32);
    if (dt == DType::Undefined) dt = DType::Float32;
    if (dt != DType::Float32 && dt != DType::Float64)
        TP_THROW(NotImplementedError, name, ": only float32/float64 windows are supported");
    Tensor w({std::max<int64_t>(out_len, 0)}, dt);
    if (out_len == 0) return w;
    auto stream = getCurrentCUDAStream().stream();
    const int64_t total = out_len;
    if (dt == DType::Float64) {
        window_fill_kernel<double><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            static_cast<double*>(w.data_ptr()), total, formula_len, alpha, beta, kind);
    } else {
        window_fill_kernel<float><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            static_cast<float*>(w.data_ptr()), total, formula_len, alpha, beta, kind);
    }
    CUDA_CHECK(cudaGetLastError());
    return w;
}
}  // namespace

inline int64_t window_denominator_cuda(int64_t window_length, bool periodic) {
    return window_length - (periodic ? 0 : 1);
}

Tensor hann_window_cuda(int64_t window_length, bool periodic, std::optional<DType> dtype) {
    const int64_t L = window_denominator_cuda(window_length, periodic);
    return window_cuda(window_length, L, dtype, "hann_window", 0.5, 0.5, 0);
}

Tensor hamming_window_cuda(int64_t window_length, bool periodic, double alpha, double beta, std::optional<DType> dtype) {
    const int64_t L = window_denominator_cuda(window_length, periodic);
    return window_cuda(window_length, L, dtype, "hamming_window", alpha, beta, 1);
}

Tensor bartlett_window_cuda(int64_t window_length, bool periodic, std::optional<DType> dtype) {
    const int64_t L = window_denominator_cuda(window_length, periodic);
    return window_cuda(window_length, L, dtype, "bartlett_window", 0.0, 0.0, 2);
}

Tensor blackman_window_cuda(int64_t window_length, bool periodic, std::optional<DType> dtype) {
    const int64_t L = window_denominator_cuda(window_length, periodic);
    return window_cuda(window_length, L, dtype, "blackman_window", 0.0, 0.0, 3);
}

// ---------------------------------------------------------------------------
//   stft (:940-1030): center pad -> time2col (as_strided) -> window mul ->
//                     batched rfft/c2c with by_root_n normalization ->
//                     transpose to (..., freq, frames)
//   istft (:1046-1250): c2c/c2r with by_n (or by_root_n when normalized) ->
//                     window mul -> overlap-add (unfold_backward) ->
//                     envelope division -> crop / length pad
//   stft_backward: adjoint — conj-symmetry fill + unscaled inverse + real
//                     projection (== C2R) scaled by the primal forward
//                     factor, windowed, overlap-added, padding undone.
// ---------------------------------------------------------------------------

namespace {

template <typename T>
__global__ void pad_time_axis_kernel(int64_t batch, int64_t len, int64_t pad,
                                     bool reflect,
                                     const T* __restrict__ src, T* __restrict__ dst) {
    const int64_t b = blockIdx.x;
    if (b >= batch) return;
    const int64_t out_len = len + 2 * pad;
    const T* in_row = src + b * len;
    T* out_row = dst + b * out_len;
    const int64_t period = (2 * len - 2) > 1 ? (2 * len - 2) : 1;
    for (int64_t i = threadIdx.x; i < pad; i += blockDim.x) {
        if (reflect) {
            int64_t idx = (i + 1) % period;
            if (idx >= len) idx = 2 * len - 2 - idx;
            out_row[pad - 1 - i] = in_row[idx];
        } else {
            out_row[pad - 1 - i] = T(0);
        }
    }
    for (int64_t i = threadIdx.x; i < len; i += blockDim.x) out_row[pad + i] = in_row[i];
    for (int64_t i = threadIdx.x; i < pad; i += blockDim.x) {
        if (reflect) {
            int64_t idx = len - 2 - i;
            idx = (-idx) % period;
            if (idx < 0) idx += period;
            if (idx >= len) idx = 2 * len - 2 - idx;
            out_row[pad + len + i] = in_row[idx];
        } else {
            out_row[pad + len + i] = T(0);
        }
    }
}

// {stride0, hop*stride1, stride1}) then mul(window_).
template <typename T>
__global__ void build_frames_kernel(int64_t batch, int64_t plen, int64_t n_fft,
                                    int64_t hop, int64_t n_frames, const T* __restrict__ win,
                                    const T* __restrict__ sig, T* __restrict__ frames) {
    const int64_t line = blockIdx.x;  // b * n_frames + t
    if (line >= batch * n_frames) return;
    const int64_t b = line / n_frames;
    const int64_t t = line % n_frames;
    const T* row = sig + b * plen + t * hop;
    T* dst = frames + line * n_fft;
    for (int64_t k = threadIdx.x; k < n_fft; k += blockDim.x) {
        dst[k] = row[k] * win[k];
    }
}

template <typename T, typename C>
__global__ void transpose_spec_kernel(int64_t batch, int64_t n_freq, int64_t n_frames,
                                      const C* __restrict__ packed, C* __restrict__ out) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * n_freq * n_frames) return;
    const int64_t t = idx % n_frames;
    const int64_t bf = idx / n_frames;
    const int64_t f = bf % n_freq;
    const int64_t b = bf / n_freq;
    out[(b * n_freq + f) * n_frames + t] = packed[(b * n_frames + t) * n_freq + f];
}

template <bool IsDouble>
Tensor stft_cuda_impl(const Tensor& work, int64_t n_fft, int64_t hop, int64_t win,
                      const std::optional<Tensor>& window, bool normalized, bool onesided,
                      bool was_1d) {
    using R = typename CudaTypes<IsDouble>::R;
    using C = typename CudaTypes<IsDouble>::C;
    const std::vector<int64_t> wsizes = sizes_of(work);  // (batch, plen)
    const int64_t batch = wsizes[0];
    const int64_t plen = wsizes[1];
    const int64_t n_frames = 1 + (plen - n_fft) / hop;
    const int64_t n_freq = infer_onesided(n_fft);

    Tensor win_t = Tensor::zeros({n_fft}, real_dtype_of(work.dtype()), work.device());
    {
        auto stream = getCurrentCUDAStream().stream();
        if (window.has_value()) {
            Tensor wv = window->contiguous();
            TP_CHECK(wv.dim() == 1 && wv.size(0) == win, "stft: window must be 1D of size win_length");
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<R*>(win_t.data_ptr()) + (n_fft - win) / 2,
                wv.data_ptr(), sizeof(R) * win, cudaMemcpyDeviceToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemsetAsync(win_t.data_ptr(), 0, sizeof(R) * n_fft, stream));
            if constexpr (IsDouble) {
                fill_r_kernel<double><<<(n_fft + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    reinterpret_cast<double*>(win_t.data_ptr()) + (n_fft - win) / 2, win, 1.0);
            } else {
                fill_r_kernel<float><<<(n_fft + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    reinterpret_cast<float*>(win_t.data_ptr()) + (n_fft - win) / 2, win, 1.0f);
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // time2col + window multiply into a packed real buffer
    Tensor frames({batch * n_frames, n_fft}, real_dtype_of(work.dtype()));
    {
        auto stream = getCurrentCUDAStream().stream();
        if constexpr (IsDouble) {
            build_frames_kernel<double><<<batch * n_frames, kThreads, 0, stream>>>(
                batch, plen, n_fft, hop, n_frames,
                static_cast<const double*>(win_t.data_ptr()),
                static_cast<const double*>(work.data_ptr()),
                static_cast<double*>(frames.data_ptr()));
        } else {
            build_frames_kernel<float><<<batch * n_frames, kThreads, 0, stream>>>(
                batch, plen, n_fft, hop, n_frames,
                static_cast<const float*>(win_t.data_ptr()),
                static_cast<const float*>(work.data_ptr()),
                static_cast<float*>(frames.data_ptr()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    const auto mode = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::none;
    Tensor spec = core_r2c_impl<IsDouble>(frames, n_fft, mode);
    TP_CHECK(spec.size(-1) == n_freq, "stft: unexpected spectrum width");

    // transpose into output layout (batch, freq, frames)
    Tensor out_c({batch, n_freq, n_frames}, complex_dtype_of(work.dtype()));
    {
        auto stream = getCurrentCUDAStream().stream();
        const int64_t total = batch * n_freq * n_frames;
        if constexpr (IsDouble) {
            transpose_spec_kernel<double, cufftDoubleComplex><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                batch, n_freq, n_frames,
                reinterpret_cast<const cufftDoubleComplex*>(spec.data_ptr()),
                reinterpret_cast<cufftDoubleComplex*>(out_c.data_ptr()));
        } else {
            transpose_spec_kernel<float, cufftComplex><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                batch, n_freq, n_frames,
                reinterpret_cast<const cufftComplex*>(spec.data_ptr()),
                reinterpret_cast<cufftComplex*>(out_c.data_ptr()));
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return out_c;
}

Tensor stft_cuda(const Tensor& self, int64_t n_fft, std::optional<int64_t> hop_length,
                 std::optional<int64_t> win_length, const std::optional<Tensor>& window,
                 bool center, std::string pad_mode, bool normalized, bool onesided,
                 bool return_complex) {
    TP_CHECK(!is_cplx(self.dtype()), "stft: expected a real floating point input");
    TP_CHECK(self.dim() >= 1 && self.dim() <= 2, "stft: expected 1D or 2D input");
    const int64_t hop = hop_length.value_or(n_fft >> 2);
    const int64_t win = win_length.value_or(n_fft);
    TP_CHECK(hop > 0, "stft: expected hop_length > 0");
    TP_CHECK(win > 0 && win <= n_fft, "stft: expected 0 < win_length <= n_fft");

    Tensor x = self.contiguous();
    const bool was_1d = x.dim() == 1;
    if (was_1d) x = x.unsqueeze(0);
    if (center) {
        const int64_t pad = n_fft / 2;
        Tensor padded(std::vector<int64_t>{x.size(0), x.size(1) + 2 * pad}, x.dtype());
        auto stream = getCurrentCUDAStream().stream();
        if (x.dtype() == DType::Float64) {
            pad_time_axis_kernel<double><<<x.size(0), kThreads, 0, stream>>>(
                x.size(0), x.size(1), pad, pad_mode == "reflect",
                static_cast<const double*>(x.data_ptr()),
                static_cast<double*>(padded.data_ptr()));
        } else {
            pad_time_axis_kernel<float><<<x.size(0), kThreads, 0, stream>>>(
                x.size(0), x.size(1), pad, pad_mode == "reflect",
                static_cast<const float*>(x.data_ptr()),
                static_cast<float*>(padded.data_ptr()));
        }
        CUDA_CHECK(cudaGetLastError());
        x = padded;
    }
    TP_CHECK(n_fft > 0 && n_fft <= x.size(1),
             "stft: expected 0 < n_fft <= signal length after padding");

    return self.dtype() == DType::Float64
        ? stft_cuda_impl<true>(x, n_fft, hop, win, window, normalized, onesided, was_1d)
        : stft_cuda_impl<false>(x, n_fft, hop, win, window, normalized, onesided, was_1d);
}

// t covering p of frame[t][p - t*hop] * win[p - t*hop]; envelope likewise
// with win^2. One thread per output sample, sequential deterministic adds.
template <typename T>
__global__ void ola_kernel(int64_t batch, int64_t frames, int64_t n_fft, int64_t hop,
                           int64_t expected_len,
                           const T* __restrict__ tf, const T* __restrict__ win,
                           T* __restrict__ y, T* __restrict__ env) {
    const int64_t b = blockIdx.x;
    const int64_t p = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= batch || p >= expected_len) return;
    const int64_t t_lo = (p >= n_fft) ? (p - n_fft + hop) / hop : 0;
    int64_t t_hi = p / hop;
    if (t_hi > frames - 1) t_hi = frames - 1;
    T acc_y = T(0), acc_e = T(0);
    for (int64_t t = t_lo; t <= t_hi; ++t) {
        const int64_t off = p - t * hop;
        const T w = win[off];
        acc_y += tf[(b * frames + t) * n_fft + off] * w;
        acc_e += w * w;
    }
    y[b * expected_len + p] = acc_y;
    env[b * expected_len + p] = acc_e;
}

// istft gather columns: input (batch, freq, frames) -> (batch*frames, bins)
template <typename C>
__global__ void gather_cols_kernel(int64_t batch, int64_t fft_size, int64_t frames,
                                   int64_t bins,
                                   const C* __restrict__ in, C* __restrict__ cols) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * frames * bins) return;
    const int64_t k = idx % bins;
    const int64_t bt = idx / bins;
    const int64_t t = bt % frames;
    const int64_t b = bt / frames;
    cols[bt * bins + k] = in[(b * fft_size + k) * frames + t];
}

template <typename T>
__global__ void istft_finalize_kernel(int64_t batch, int64_t expected_len,
                                      int64_t start, int64_t out_len,
                                      const T* __restrict__ y, const T* __restrict__ env,
                                      T* __restrict__ out) {
    const int64_t b = blockIdx.x;
    const int64_t i = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= batch || i >= out_len) return;
    const int64_t src = start + i;
    out[b * out_len + i] = y[b * expected_len + src] / env[b * expected_len + src];
}

template <bool IsDouble>
Tensor istft_cuda_impl(const Tensor& input, int64_t n_fft, int64_t hop, int64_t win,
                       const std::optional<Tensor>& window, bool center, bool normalized,
                       std::optional<int64_t> length) {
    using R = typename CudaTypes<IsDouble>::R;
    using C = typename CudaTypes<IsDouble>::C;
    std::vector<int64_t> isizes = sizes_of(input);
    // 2D (freq, frames) -> (len,) or 3D (batch, freq, frames) -> (B, len).
    TP_CHECK(isizes.size() == 2 || isizes.size() == 3,
             "istft: expected a complex tensor with 2 or 3 dimensions");
    const int64_t frames_dim_pos = isizes.size() - 1;
    const int64_t frames = isizes[frames_dim_pos];
    const int64_t fft_size = isizes[frames_dim_pos - 1];
    const bool was_2d = isizes.size() == 2;
    const int64_t batch = was_2d ? 1 : isizes[0];
    const int64_t expected_len = n_fft + hop * (frames - 1);

    Tensor win_t = Tensor::zeros({n_fft}, real_dtype_of(input.dtype()), input.device());
    {
        auto stream = getCurrentCUDAStream().stream();
        R* wp = static_cast<R*>(win_t.data_ptr());
        const int64_t left = (n_fft - win) / 2;
        if (window.has_value()) {
            Tensor wv = window->contiguous();
            TP_CHECK(wv.dim() == 1 && wv.size(0) == win,
                     "istft: Invalid window shape; window has to be 1D and of length win_length");
            CUDA_CHECK(cudaMemcpyAsync(wp + left, wv.data_ptr(),
                                       sizeof(R) * win, cudaMemcpyDeviceToDevice, stream));
        } else {
            fill_r_kernel<R><<<(win + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                wp + left, win, R(1));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // norm = normalized ? by_root_n : by_n, SpectralOps.cpp:1160)
    const int64_t bins = n_fft / 2 + 1;
    Tensor cols({batch * frames, bins}, input.dtype());
    {
        auto stream = getCurrentCUDAStream().stream();
        if constexpr (IsDouble) {
            gather_cols_kernel<cufftDoubleComplex><<<(batch * frames * bins + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                batch, fft_size, frames, bins,
                reinterpret_cast<const cufftDoubleComplex*>(input.contiguous().data_ptr()),
                reinterpret_cast<cufftDoubleComplex*>(cols.data_ptr()));
        } else {
            gather_cols_kernel<cufftComplex><<<(batch * frames * bins + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                batch, fft_size, frames, bins,
                reinterpret_cast<const cufftComplex*>(input.contiguous().data_ptr()),
                reinterpret_cast<cufftComplex*>(cols.data_ptr()));
        }
        CUDA_CHECK(cudaGetLastError());
    }
    const auto mode = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::by_n;
    Tensor tf = core_c2r_impl<IsDouble>(cols, n_fft, mode);

    // overlap-add + envelope (unfold_backward step)
    Tensor y({batch * expected_len}, real_dtype_of(input.dtype()));
    Tensor env({batch * expected_len}, real_dtype_of(input.dtype()));
    {
        auto stream = getCurrentCUDAStream().stream();
        dim3 grid(batch, (expected_len + kThreads - 1) / kThreads);
        if constexpr (IsDouble) {
            ola_kernel<double><<<grid, kThreads, 0, stream>>>(
                batch, frames, n_fft, hop, expected_len,
                static_cast<const double*>(tf.data_ptr()),
                static_cast<const double*>(win_t.data_ptr()),
                static_cast<double*>(y.data_ptr()),
                static_cast<double*>(env.data_ptr()));
        } else {
            ola_kernel<float><<<grid, kThreads, 0, stream>>>(
                batch, frames, n_fft, hop, expected_len,
                static_cast<const float*>(tf.data_ptr()),
                static_cast<const float*>(win_t.data_ptr()),
                static_cast<float*>(y.data_ptr()),
                static_cast<float*>(env.data_ptr()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t start = center ? n_fft / 2 : 0;
    int64_t end;
    if (length.has_value()) end = start + *length;
    else if (center) end = expected_len - n_fft / 2;
    else end = expected_len;
    end = std::min(end, expected_len);
    TP_CHECK(end > start, "istft: requested output length is too small");
    const int64_t out_len = end - start;

    std::vector<int64_t> out_sizes =
        was_2d ? std::vector<int64_t>{out_len} : std::vector<int64_t>{batch, out_len};
    Tensor out(out_sizes, real_dtype_of(input.dtype()));
    {
        auto stream = getCurrentCUDAStream().stream();
        dim3 grid(batch, (out_len + kThreads - 1) / kThreads);
        if constexpr (IsDouble) {
            istft_finalize_kernel<double><<<grid, kThreads, 0, stream>>>(
                batch, expected_len, start, out_len,
                static_cast<const double*>(y.data_ptr()),
                static_cast<const double*>(env.data_ptr()),
                static_cast<double*>(out.data_ptr()));
        } else {
            istft_finalize_kernel<float><<<grid, kThreads, 0, stream>>>(
                batch, expected_len, start, out_len,
                static_cast<const float*>(y.data_ptr()),
                static_cast<const float*>(env.data_ptr()),
                static_cast<float*>(out.data_ptr()));
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return out;
}

Tensor istft_cuda(const Tensor& input, int64_t n_fft, std::optional<int64_t> hop_length,
                  std::optional<int64_t> win_length, const std::optional<Tensor>& window,
                  bool center, bool normalized, bool onesided, std::optional<int64_t> length,
                  bool return_complex) {
    TP_CHECK(is_cplx(input.dtype()),
             "istft requires a complex input matching stft(return_complex=True)");
    TP_CHECK(!return_complex, "istft: complex output path not supported");
    const int64_t fft_size = input.size(-2);
    if (onesided) {
        TP_CHECK(fft_size == n_fft / 2 + 1,
                 "istft: frequency dim must equal n_fft/2+1 when onesided");
    } else {
        TP_CHECK(fft_size == n_fft, "istft: frequency dim must equal n_fft when onesided=False");
    }
    const int64_t hop = hop_length.value_or(n_fft >> 2);
    const int64_t win = win_length.value_or(n_fft);
    TP_CHECK(hop > 0 && hop <= win, "istft: expected 0 < hop_length <= win_length");
    TP_CHECK(win > 0 && win <= n_fft, "istft: expected 0 < win_length <= n_fft");

    return input.dtype() == DType::ComplexDouble
        ? istft_cuda_impl<true>(input, n_fft, hop, win, window, center, normalized, length)
        : istft_cuda_impl<false>(input, n_fft, hop, win, window, center, normalized, length);
}

// ---------------------------------------------------------------------------
// conjugate-symmetry fill + unscaled inverse + real projection (see
// [Fourier Transform Conjugate Symmetry] in SpectralOpsUtils.h), so the
// backward reduces to: gather grad columns -> C2R scaled by the primal
// forward factor -> window multiply -> overlap-add scatter -> unpad.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void ola_scatter_kernel(int64_t batch, int64_t frames, int64_t n_fft,
                                   int64_t hop, int64_t padded_len,
                                   const T* __restrict__ tf, const T* __restrict__ win,
                                   T* __restrict__ xg) {
    const int64_t line = blockIdx.x;
    if (line >= batch * frames) return;
    const int64_t b = line / frames;
    const int64_t t = line % frames;
    const T* fr = tf + line * n_fft;
    T* row = xg + b * padded_len + t * hop;
    for (int64_t k = threadIdx.x; k < n_fft; k += blockDim.x) {
        row[k] += fr[k] * win[k];
    }
}

// Adjoint of pad_time_axis_kernel: crop (constant) / mirror-scatter (reflect).
template <typename T>
__global__ void unpad_gather_kernel(int64_t batch, int64_t padded_len, int64_t pad,
                                    bool reflect, const T* __restrict__ padded,
                                    T* __restrict__ out) {
    const int64_t b = blockIdx.x;
    if (b >= batch) return;
    const int64_t len = padded_len - 2 * pad;
    const T* prow = padded + b * padded_len;
    T* orow = out + b * len;
    for (int64_t j = threadIdx.x; j < len; j += blockDim.x) {
        orow[j] = prow[pad + j];
    }
    if (!reflect) return;
    const int64_t period = (2 * len - 2) > 1 ? (2 * len - 2) : 1;
    for (int64_t i = threadIdx.x; i < pad; i += blockDim.x) {
        int64_t idx = (i + 1) % period;
        if (idx >= len) idx = 2 * len - 2 - idx;
        atomicAdd(&orow[idx], prow[pad - 1 - i]);
        int64_t idx2 = len - 2 - i;
        idx2 = (-idx2) % period;
        if (idx2 < 0) idx2 += period;
        if (idx2 >= len) idx2 = 2 * len - 2 - idx2;
        atomicAdd(&orow[idx2], prow[pad + len + i]);
    }
}

template <bool IsDouble>
Tensor stft_backward_cuda_impl(const Tensor& grad_output, const Tensor& self, int64_t n_fft,
                               int64_t hop, int64_t win_length,
                               const std::optional<Tensor>& window, bool center,
                               bool normalized, bool onesided, const std::string& pad_mode) {
    using R = typename CudaTypes<IsDouble>::R;
    using C = typename CudaTypes<IsDouble>::C;
    std::vector<int64_t> gsizes = sizes_of(grad_output);
    const int64_t n_freq = infer_onesided(n_fft);
    const int64_t frames = gsizes.back();
    const int64_t gfreq = gsizes[gsizes.size() - 2];
    TP_CHECK(gfreq == n_freq, "stft_backward: frequency dim mismatch");
    const bool was_1d = gsizes.size() == 2;
    const int64_t batch = was_1d ? 1 : gsizes[0];

    // adjoint scale == primal forward factor
    const auto mode = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::none;

    // window center-padded to n_fft on device (same as forward builds it)
    Tensor win_t = Tensor::zeros({n_fft}, self.dtype(), self.device());
    {
        auto stream = getCurrentCUDAStream().stream();
        R* wp = static_cast<R*>(win_t.data_ptr());
        const int64_t left = (n_fft - win_length) / 2;
        if (window.has_value()) {
            Tensor wv = window->contiguous();
            CUDA_CHECK(cudaMemcpyAsync(wp + left, wv.data_ptr(),
                                       sizeof(R) * win_length,
                                       cudaMemcpyDeviceToDevice, stream));
        } else {
            if constexpr (IsDouble) {
                fill_r_kernel<double><<<(win_length + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    reinterpret_cast<double*>(wp) + left, win_length, 1.0);
            } else {
                fill_r_kernel<float><<<(win_length + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    reinterpret_cast<float*>(wp) + left, win_length, 1.0f);
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // gather grad columns into packed (batch*frames, bins): reuse the spec
    // transpose kernel (grad layout (batch, freq, frames) -> (frames, freq))
    Tensor cols(std::vector<int64_t>{batch * frames, n_freq}, complex_dtype_of(self.dtype()));
    {
        auto stream = getCurrentCUDAStream().stream();
        if constexpr (IsDouble) {
            transpose_spec_kernel<double, cufftDoubleComplex><<<(batch * n_freq * frames + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                batch, n_freq, frames,
                reinterpret_cast<const cufftDoubleComplex*>(grad_output.contiguous().data_ptr()),
                reinterpret_cast<cufftDoubleComplex*>(cols.data_ptr()));
        } else {
            transpose_spec_kernel<float, cufftComplex><<<(batch * n_freq * frames + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                batch, n_freq, frames,
                reinterpret_cast<const cufftComplex*>(grad_output.contiguous().data_ptr()),
                reinterpret_cast<cufftComplex*>(cols.data_ptr()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // twosided spectrum from the onesided grad, run the INVERSE c2c carrying
    // the forward's normalization, then project to the real part.
    Tensor full = Tensor::zeros(std::vector<int64_t>{batch * frames, n_fft},
                                complex_dtype_of(self.dtype()), self.device());
    if (n_freq < n_fft) full.slice(-1, 0, n_freq).copy_(cols);
    else full.copy_(cols);
    Tensor ctime = core_c2c_impl<IsDouble>(full, n_fft, mode, /*forward=*/false);
    Tensor tf = Tensor::empty(std::vector<int64_t>{batch * frames, n_fft},
                              self.dtype(), self.device());
    {
        auto stream = getCurrentCUDAStream().stream();
        const int64_t total = ctime.numel();
        if constexpr (IsDouble) {
            cplx_real_f64_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                reinterpret_cast<const cufftDoubleComplex*>(ctime.data_ptr()),
                static_cast<double*>(tf.data_ptr()), total);
        } else {
            cplx_real_f32_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                reinterpret_cast<const cufftComplex*>(ctime.data_ptr()),
                static_cast<float*>(tf.data_ptr()), total);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    const int64_t orig_len = self.size(-1);
    const int64_t padded_len = orig_len + (center ? (n_fft / 2) * 2 : 0);
    Tensor xg = Tensor::zeros(std::vector<int64_t>{batch * padded_len}, self.dtype(),
                              self.device());
    {
        auto stream = getCurrentCUDAStream().stream();
        if constexpr (IsDouble) {
            ola_scatter_kernel<double><<<batch * frames, kThreads, 0, stream>>>(
                batch, frames, n_fft, hop, padded_len,
                static_cast<const double*>(tf.data_ptr()),
                static_cast<const double*>(win_t.data_ptr()),
                static_cast<double*>(xg.data_ptr()));
        } else {
            ola_scatter_kernel<float><<<batch * frames, kThreads, 0, stream>>>(
                batch, frames, n_fft, hop, padded_len,
                static_cast<const float*>(tf.data_ptr()),
                static_cast<const float*>(win_t.data_ptr()),
                static_cast<float*>(xg.data_ptr()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<int64_t> out_sizes = sizes_of(self);
    Tensor out(out_sizes, self.dtype());
    {
        auto stream = getCurrentCUDAStream().stream();
        if (!center) {
            CUDA_CHECK(cudaMemcpyAsync(out.data_ptr(), xg.data_ptr(),
                                       sizeof(R) * orig_len,
                                       cudaMemcpyDeviceToDevice, stream));
        } else if (self.dtype() == DType::Float64) {
            unpad_gather_kernel<double><<<batch, kThreads, 0, stream>>>(
                batch, padded_len, n_fft / 2, pad_mode == "reflect",
                static_cast<const double*>(xg.data_ptr()),
                static_cast<double*>(out.data_ptr()));
        } else {
            unpad_gather_kernel<float><<<batch, kThreads, 0, stream>>>(
                batch, padded_len, n_fft / 2, pad_mode == "reflect",
                static_cast<const float*>(xg.data_ptr()),
                static_cast<float*>(out.data_ptr()));
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return out;
}

Tensor stft_backward_cuda(const Tensor& grad_output, const Tensor& self, int64_t n_fft,
                          std::optional<int64_t> hop_length, std::optional<int64_t> win_length,
                          const std::optional<Tensor>& window, bool center,
                          std::string pad_mode, bool normalized, bool onesided) {
    TP_CHECK(!is_cplx(self.dtype()), "stft_backward: expected real input");
    TP_CHECK(!center || pad_mode == "constant" || pad_mode == "reflect",
             "stft_backward: unsupported pad_mode (use constant|reflect)");
    const int64_t hop_r = hop_length.value_or(n_fft >> 2);
    const int64_t win_r = win_length.value_or(n_fft);
    Tensor x = self.contiguous();
    return self.dtype() == DType::Float64
        ? stft_backward_cuda_impl<true>(grad_output, x, n_fft, hop_r, win_r, window,
                                        center, normalized, onesided, pad_mode)
        : stft_backward_cuda_impl<false>(grad_output, x, n_fft, hop_r, win_r, window,
                                         center, normalized, onesided, pad_mode);
}

}  // namespace

// CUDA registration for the spectral kernel family.
TENSORPLAY_LIBRARY_IMPL(CUDA, SpectralKernels) {
    m.impl("fft_fft", fft_fft_cuda);
    m.impl("fft_ifft", fft_ifft_cuda);
    m.impl("fft_rfft", fft_rfft_cuda);
    m.impl("fft_irfft", fft_irfft_cuda);
    m.impl("fft_fft_backward", fft_fft_backward_cuda);
    m.impl("fft_ifft_backward", fft_ifft_backward_cuda);
    m.impl("fft_rfft_backward", fft_rfft_backward_cuda);
    m.impl("fft_irfft_backward", fft_irfft_backward_cuda);
    m.impl("hann_window", hann_window_cuda);
    m.impl("hamming_window", hamming_window_cuda);
    m.impl("bartlett_window", bartlett_window_cuda);
    m.impl("blackman_window", blackman_window_cuda);
    m.impl("stft", stft_cuda);
    m.impl("istft", istft_cuda);
    m.impl("stft_backward", stft_backward_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
