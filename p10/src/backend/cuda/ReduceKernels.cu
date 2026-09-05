// Reduction kernels - CUDA.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <utility>
#include <type_traits>

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

namespace {

constexpr int kThreads = 256;

inline dim3 make_grid(int64_t work) {
    return dim3(static_cast<unsigned>((work + kThreads - 1) / kThreads));
}

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    // Dimension wrapping reports the original (unwrapped) value on error.
    const int64_t min = -ndim;
    const int64_t max = ndim - 1;
    if (dim < min || dim > max) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 min, ", ", max, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

inline void outer_inner(const std::vector<int64_t>& shape, int64_t dim,
                        int64_t& outer, int64_t& inner) {
    outer = 1; inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) inner *= shape[i];
}

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

// ---------------------------------------------------------------------------
// Reduction slice kernels (one thread per output slice)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void slice_max_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                 const T* in, double* out) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        double acc = -std::numeric_limits<double>::infinity();
        for (int64_t j = 0; j < d_size; ++j) {
            double v = static_cast<double>(sp[j * inner]);
            acc = (v != v || v > acc) ? v : acc;  // NaN propagates
        }
        out[si] = acc;
    }
}

template <typename T>
__global__ void slice_min_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                 const T* in, double* out) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        double acc = std::numeric_limits<double>::infinity();
        for (int64_t j = 0; j < d_size; ++j) {
            double v = static_cast<double>(sp[j * inner]);
            acc = (v != v || v < acc) ? v : acc;
        }
        out[si] = acc;
    }
}

template <typename T>
__global__ void slice_nansum_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                    const T* in, double* out) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        double acc = 0;
        for (int64_t j = 0; j < d_size; ++j) {
            double v = static_cast<double>(sp[j * inner]);
            if (v == v) acc += v;
        }
        out[si] = acc;
    }
}

template <typename T>
__global__ void slice_count_nonzero_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                           const T* in, double* out) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        int64_t c = 0;
        for (int64_t j = 0; j < d_size; ++j)
            if (sp[j * inner] != static_cast<T>(0)) ++c;
        out[si] = static_cast<double>(c);
    }
}

template <typename T>
__global__ void slice_logsumexp_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                       const T* in, double* out) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        double m = -std::numeric_limits<double>::infinity();
        bool has_nan = false;
        for (int64_t j = 0; j < d_size; ++j) {
            double v = static_cast<double>(sp[j * inner]);
            if (v != v) { has_nan = true; break; }
            if (v > m) m = v;
        }
        if (has_nan) { out[si] = ::nan(""); continue; }
        if (m == -std::numeric_limits<double>::infinity()) { out[si] = m; continue; }
        double s2 = 0;
        for (int64_t j = 0; j < d_size; ++j)
            s2 += ::exp(static_cast<double>(sp[j * inner]) - m);
        out[si] = m + ::log(s2);
    }
}

template <typename T>
__global__ void slice_mean_f64_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                      const T* in, double* out, bool squares) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        double s2 = 0;
        for (int64_t j = 0; j < d_size; ++j) {
            double v = static_cast<double>(sp[j * inner]);
            s2 += squares ? v * v : v;
        }
        out[si] = s2 / static_cast<double>(d_size);
    }
}

template <typename T>
__global__ void cummaxmin_scan_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                      const T* in, T* vals, int64_t* idxs, bool is_max) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* s2p = in + o * d_size * inner + in2;
        T* vp = vals + o * d_size * inner + in2;
        int64_t* ip = idxs + o * d_size * inner + in2;
        T best = s2p[0];
        int64_t bi = 0;
        vp[0] = best;
        ip[0] = 0;
        for (int64_t j = 1; j < d_size; ++j) {
            double cur = static_cast<double>(s2p[j * inner]);
            double b = static_cast<double>(best);
            if ((is_max && cur > b) || (!is_max && cur < b)) { best = s2p[j * inner]; bi = j; }
            vp[j * inner] = best;
            ip[j * inner] = bi;
        }
    }
}

template <typename T>
__global__ void renorm_slice_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                    const T* in, T* out, double p, double maxnorm) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        T* dp = out + o * d_size * inner + in2;
        double norm = 0;
        if (p == std::numeric_limits<double>::infinity()) {
            for (int64_t j = 0; j < d_size; ++j)
                norm = ::fmax(norm, ::fabs(static_cast<double>(sp[j * inner])));
        } else {
            double s2 = 0;
            for (int64_t j = 0; j < d_size; ++j)
                s2 += ::pow(::fabs(static_cast<double>(sp[j * inner])), p);
            norm = ::pow(s2, 1.0 / p);
        }
        double factor = norm > maxnorm ? maxnorm / norm : 1.0;
        for (int64_t j = 0; j < d_size; ++j)
            dp[j * inner] = static_cast<T>(static_cast<double>(sp[j * inner]) * factor);
    }
}

// Multi-dim iterative reduction driver.
// which: 0=max, 1=min, 2=nansum, 3=count_nonzero
Tensor reduce_iterative(const Tensor& self, std::vector<int64_t> dims, bool keepdim,
                        int which, DType out_dtype_override = DType::Undefined) {
    Tensor cur = self.contiguous();
    int64_t nd = cur.dim();
    std::vector<bool> reduced(nd, false);
    for (auto& d : dims) reduced[wrap_dim(d, nd)] = true;
    auto stream = getCurrentCUDAStream().stream();
    // With keepdim=false each pass erases the reduced axis, so later original
    // dims shift down; track the current position of original dim `dim`.
    int64_t shift = 0;
    for (int64_t dim = 0; dim < nd; ++dim) {
        if (!reduced[dim]) continue;
        const int64_t cur_dim = dim - shift;
        int64_t d_size = cur.size(cur_dim);
        int64_t outer = 1, inner = 1;
        outer_inner(shape_of(cur), cur_dim, outer, inner);
        int64_t slices = outer * inner;
        Tensor accs = Tensor::zeros({slices}, DType::Float64,
                                    self.device());
        if (slices > 0 && d_size > 0) {
            dim3 grid = make_grid(slices), block(kThreads);
#define TP_RI(ctype, name_) \
    case DType::name_: \
        if (which == 0) \
            slice_max_kernel<ctype><<<grid, block, 0, stream>>>( \
                slices, d_size, inner, cur.data_ptr<ctype>(), accs.data_ptr<double>()); \
        else if (which == 1) \
            slice_min_kernel<ctype><<<grid, block, 0, stream>>>( \
                slices, d_size, inner, cur.data_ptr<ctype>(), accs.data_ptr<double>()); \
        else if (which == 2) \
            slice_nansum_kernel<ctype><<<grid, block, 0, stream>>>( \
                slices, d_size, inner, cur.data_ptr<ctype>(), accs.data_ptr<double>()); \
        else \
            slice_count_nonzero_kernel<ctype><<<grid, block, 0, stream>>>( \
                slices, d_size, inner, cur.data_ptr<ctype>(), accs.data_ptr<double>()); \
        break;
            switch (cur.dtype()) {
                TENSORPLAY_FORALL_SCALAR_TYPES(TP_RI)
                default: TP_THROW(TypeError, "reduce: unsupported dtype");
            }
#undef TP_RI
            CUDA_CHECK(cudaGetLastError());
        }
        std::vector<int64_t> ns = shape_of(cur);
        if (keepdim) {
            ns[cur_dim] = 1;
        } else {
            ns.erase(ns.begin() + cur_dim);
            ++shift;
        }
        cur = accs.reshape(ns);
    }
    DType final_dt = out_dtype_override == DType::Undefined ? self.dtype() : out_dtype_override;
    return cur.to(final_dt);
}



// ===========================================================================
// Reduction entry points
// ===========================================================================

// zero_numel_check_dims): reducing an empty tensor is only valid along an
// explicitly given non-empty dim; a full reduction has no identity.
static void zero_numel_check_dims(const Tensor& self, const std::vector<int64_t>& dims,
                                  const char* fn_name) {
    if (dims.empty()) {
        TP_THROW(RuntimeError, fn_name,
                 ": Expected reduction dim to be specified for input.numel() == 0. "
                 "Specify the reduction dim with the 'dim' argument.");
    }
    const int64_t nd = self.dim();
    for (int64_t d : dims) {
        if (d < 0) d += nd;
        TP_CHECK_INDEX(self.size(d) != 0, fn_name,
                       ": Expected reduction dim ", d, " to have non-zero size.");
    }
}

Tensor amax_cuda2(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    if (self.numel() == 0) zero_numel_check_dims(self, dim, "amax()");
    return reduce_iterative(self, dim.empty()
                                       ? [&]{ std::vector<int64_t> a;
                                              for (int64_t i = 0; i < self.dim(); ++i) a.push_back(i);
                                              return a; }()
                                       : dim,
                            keepdim, 0);
}
Tensor amin_cuda2(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    if (self.numel() == 0) zero_numel_check_dims(self, dim, "amin()");
    return reduce_iterative(self, dim.empty()
                                       ? [&]{ std::vector<int64_t> a;
                                              for (int64_t i = 0; i < self.dim(); ++i) a.push_back(i);
                                              return a; }()
                                       : dim,
                            keepdim, 1);
}
std::tuple<Tensor, Tensor> aminmax_cuda(const Tensor& self, std::vector<int64_t> dim,
                                        bool keepdim) {
    if (self.numel() == 0) {
        if (dim.empty()) {
            TP_THROW(RuntimeError, "aminmax(): cannot compute aminmax over an empty dimension as "
                     "the operation has no identity.");
        }
        zero_numel_check_dims(self, dim, "aminmax");
    }
    return {amin_cuda2(self, dim, keepdim), amax_cuda2(self, dim, keepdim)};
}

std::tuple<Tensor, Tensor> aminmax_all_cuda(const Tensor& self) {
    return aminmax_cuda(self, {}, false);
}

std::tuple<Tensor, Tensor> aminmax_dim_cuda(const Tensor& self, int64_t dim,
                                            bool keepdim) {
    return aminmax_cuda(self, {dim}, keepdim);
}
Tensor logsumexp_cuda2(const Tensor& self, int64_t dim, bool keepdim) {
    if (!isFloatingType(self.dtype()))
        TP_THROW(RuntimeError, "logsumexp(): Expected floating point type");
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous().to(DType::Float64);
    int64_t d_size = sc.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(sc), dim, outer, inner);
    int64_t slices = outer * inner;
    Tensor accs = Tensor::zeros({slices}, DType::Float64, self.device());
    if (slices > 0) {
        // The kernel handles d_size==0 itself (writes -inf per slice, matching
        auto stream = getCurrentCUDAStream().stream();
        dim3 grid = make_grid(slices), block(kThreads);
        slice_logsumexp_kernel<double><<<grid, block, 0, stream>>>(
            slices, d_size, inner, sc.data_ptr<double>(), accs.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    std::vector<int64_t> ns = shape_of(sc);
    ns[dim] = keepdim ? 1 : 0;
    if (!keepdim) ns.erase(ns.begin() + dim);
    DType out_dt = self.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    return accs.reshape(ns).to(out_dt);
}
Tensor nansum_cuda2(const Tensor& self, const std::vector<int64_t>& dim_in, bool keepdim) {
    DType out_dt = isFloatingType(self.dtype()) ? self.dtype() : DType::Int64;
    std::vector<int64_t> dim = dim_in;
    if (dim.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) dim.push_back(i);
    }
    return reduce_iterative(self, dim, keepdim, 2, out_dt);
}
Tensor count_nonzero_cuda2(const Tensor& self, const std::vector<int64_t>& dim) {
    return reduce_iterative(self, dim, false, 3, DType::Int64);
}

std::tuple<Tensor, Tensor> cummax_cuda(const Tensor& self, int64_t dim) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    Tensor vals = Tensor::empty(shape_of(sc), sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(shape_of(sc), DType::Int64, sc.device());
    int64_t d_size = sc.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(sc), dim, outer, inner);
    int64_t slices = outer * inner;
    if (slices > 0 && d_size > 0) {
        auto stream = getCurrentCUDAStream().stream();
        dim3 grid = make_grid(slices), block(kThreads);
#define TP_CM(ctype, name_) \
    case DType::name_: \
        cummaxmin_scan_kernel<ctype><<<grid, block, 0, stream>>>( \
            slices, d_size, inner, sc.data_ptr<ctype>(), vals.data_ptr<ctype>(), \
            idxs.data_ptr<int64_t>(), true); \
        break;
        switch (sc.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_CM)
            default: TP_THROW(TypeError, "cummax: unsupported dtype");
        }
#undef TP_CM
        CUDA_CHECK(cudaGetLastError());
    }
    return {vals, idxs};
}
std::tuple<Tensor, Tensor> cummin_cuda(const Tensor& self, int64_t dim) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    Tensor vals = Tensor::empty(shape_of(sc), sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(shape_of(sc), DType::Int64, sc.device());
    int64_t d_size = sc.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(sc), dim, outer, inner);
    int64_t slices = outer * inner;
    if (slices > 0 && d_size > 0) {
        auto stream = getCurrentCUDAStream().stream();
        dim3 grid = make_grid(slices), block(kThreads);
#define TP_CMIN(ctype, name_) \
    case DType::name_: \
        cummaxmin_scan_kernel<ctype><<<grid, block, 0, stream>>>( \
            slices, d_size, inner, sc.data_ptr<ctype>(), vals.data_ptr<ctype>(), \
            idxs.data_ptr<int64_t>(), false); \
        break;
        switch (sc.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_CMIN)
            default: TP_THROW(TypeError, "cummin: unsupported dtype");
        }
#undef TP_CMIN
        CUDA_CHECK(cudaGetLastError());
    }
    return {vals, idxs};
}

std::tuple<Tensor, Tensor> var_mean_cuda(const Tensor& self, std::vector<int64_t> dim,
                                         bool unbiased, bool keepdim) {
    // var = E[x^2] - E[x]^2 corrected by n/(n-ddof); Float64 accumulation.
    Tensor xf = self.to(DType::Float64).contiguous();
    Tensor x2f = xf.mul(xf).contiguous();
    Tensor mean = xf, msq = x2f;
    std::vector<int64_t> dims = dim;
    bool any = false;
    for (auto& d : dims) { d = wrap_dim(d, self.dim()); any = true; }
    if (!any) { dims.clear(); for (int64_t i = 0; i < self.dim(); ++i) dims.push_back(i); }
    auto stream = getCurrentCUDAStream().stream();
    // Ascend over the original dims; with keepdim=false each pass erases the
    // reduced axis, so later dims shift down by the number already removed.
    std::vector<bool> red(static_cast<size_t>(self.dim()), false);
    for (int64_t d2 : dims) red[static_cast<size_t>(d2)] = true;
    int64_t shift = 0;
    for (int64_t dd = 0; dd < self.dim(); ++dd) {
        if (!red[static_cast<size_t>(dd)]) continue;
        const int64_t cur_d = dd - shift;
        int64_t dsz = mean.size(cur_d);
        int64_t outer = 1, inner = 1;
        outer_inner(shape_of(mean), cur_d, outer, inner);
        int64_t slices = outer * inner;
        Tensor m1 = Tensor::zeros({slices}, DType::Float64, self.device());
        Tensor m2 = Tensor::zeros(shape_of(m1), DType::Float64, self.device());
        if (slices > 0) {
            // The kernel divides by dsz, so dsz==0 slices become NaN, matching
            dim3 grid = make_grid(slices), block(kThreads);
            slice_mean_f64_kernel<double><<<grid, block, 0, stream>>>(
                slices, dsz, inner, mean.data_ptr<double>(), m1.data_ptr<double>(), false);
            slice_mean_f64_kernel<double><<<grid, block, 0, stream>>>(
                slices, dsz, inner, msq.data_ptr<double>(), m2.data_ptr<double>(), true);
            CUDA_CHECK(cudaGetLastError());
        }
        std::vector<int64_t> ns = shape_of(mean);
        if (keepdim) {
            ns[cur_d] = 1;
        } else {
            ns.erase(ns.begin() + cur_d);
            ++shift;
        }
        mean = m1.reshape(ns);
        msq = m2.reshape(ns);
    }
    int64_t n_red = 1;
    for (int64_t d2 : dims) n_red *= self.size(d2);
    double ddof = (unbiased && n_red > 1) ? 1.0 : 0.0;
    Tensor var = msq.sub(mean.mul(mean));
    Tensor corr = var.mul(Tensor::full({}, Scalar(n_red / (n_red - ddof)),
                                       DType::Float64, self.device()));
    DType out_dt = self.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    return {corr.to(out_dt), mean.to(out_dt)};
}
std::tuple<Tensor, Tensor> std_mean_cuda(const Tensor& self, std::vector<int64_t> dim,
                                         bool unbiased, bool keepdim) {
    auto vm = var_mean_cuda(self, dim, unbiased, keepdim);
    return {std::get<0>(vm).sqrt(), std::get<1>(vm)};
}

Tensor nanmedian_cuda(const Tensor& self) {
    // Host-staged reference implementation (rare op).
    Tensor host = self.to(DType::Float64).contiguous().to(Device(DeviceType::CPU));
    std::vector<double> vals;
    const double* pp = host.data_ptr<double>();
    for (int64_t i = 0; i < host.numel(); ++i)
        if (!(pp[i] != pp[i])) vals.push_back(pp[i]);
    DType out_dt = isFloatingType(self.dtype()) ? self.dtype() : DType::Int64;
    double med = 0;
    if (!vals.empty()) {
        std::sort(vals.begin(), vals.end());
        med = vals[(vals.size() - 1) / 2];
    }
    return Tensor::zeros({}, out_dt, self.device()).fill_(Scalar(med));
}

std::tuple<Tensor, Tensor> mode_cuda(const Tensor& self, int64_t dim, bool keepdim) {
    // Host-staged reference implementation (rare op).
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor hv = self.contiguous().to(Device(DeviceType::CPU));
    Tensor hi = Tensor::zeros(shape_of(hv), DType::Int64, Device(DeviceType::CPU));
    int64_t d_size = hv.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(hv), dim, outer, inner);
#define TP_MODEH(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = hv.data_ptr<ctype>(); \
        ctype* vp = hv.data_ptr<ctype>(); \
        int64_t* ip = hi.data_ptr<int64_t>(); \
        for (int64_t si = 0; si < outer * inner; ++si) { \
            int64_t o = si / inner, in2 = si % inner; \
            std::vector<std::pair<ctype, int64_t>> buf(d_size); \
            for (int64_t j = 0; j < d_size; ++j) buf[j] = {sp[(o*d_size+j)*inner+in2], j}; \
            std::sort(buf.begin(), buf.end(), [](const std::pair<ctype,int64_t>& a2, \
                                                 const std::pair<ctype,int64_t>& b2){ \
                if (!(a2.first<b2.first) && !(b2.first<a2.first)) return a2.second<b2.second; \
                return a2.first<b2.first; }); \
            ctype bv = buf[0].first; int64_t bc = 0, bi2 = buf[0].second, run = 0; \
            for (int64_t j = 0; j < d_size; ++j) { \
                bool same = j > 0 && !(buf[j].first<buf[j-1].first) && !(buf[j-1].first<buf[j].first); \
                run = same ? run + 1 : 1; \
                if (run > bc) { bc = run; bv = buf[j].first; bi2 = buf[j].second; } } \
            vp[si] = bv; ip[si] = bi2; \
        } \
        break; }
    switch (hv.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MODEH)
        default: TP_THROW(TypeError, "mode: unsupported dtype");
    }
#undef TP_MODEH
    std::vector<int64_t> out_shape = shape_of(hv);
    out_shape[dim] = keepdim ? 1 : 0;
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    return {hv.reshape(out_shape).to(self.device()),
            hi.reshape(out_shape).to(self.device())};
}

std::tuple<Tensor, Tensor> kthvalue_cuda(const Tensor& self, int64_t k, int64_t dim,
                                         bool keepdim) {
    // Host-staged reference implementation (rare op).
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    int64_t d_size = self.size(dim);
    if (k < 1 || k > d_size)
        TP_THROW(RuntimeError, "kthvalue(): selected number k out of range for dim ", dim);
    Tensor host = self.contiguous().to(Device(DeviceType::CPU));
    Tensor vals = Tensor::empty(shape_of(host), host.dtype(), Device(DeviceType::CPU));
    Tensor idxs = Tensor::empty(shape_of(host), DType::Int64, Device(DeviceType::CPU));
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(host), dim, outer, inner);
#define TP_KTHH(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = host.data_ptr<ctype>(); \
        ctype* vp = vals.data_ptr<ctype>(); \
        int64_t* ip = idxs.data_ptr<int64_t>(); \
        for (int64_t si = 0; si < outer * inner; ++si) { \
            int64_t o = si/inner, in2 = si%inner; \
            std::vector<std::pair<ctype, int64_t>> buf(d_size); \
            for (int64_t j = 0; j < d_size; ++j) buf[j] = {sp[(o*d_size+j)*inner+in2], j}; \
            std::stable_sort(buf.begin(), buf.end(), [](const std::pair<ctype,int64_t>& a2, \
                                                        const std::pair<ctype,int64_t>& b2){ \
                return a2.first<b2.first; }); \
            vp[si] = buf[k-1].first; ip[si] = buf[k-1].second; \
        } \
        break; }
    switch (host.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_KTHH)
        default: TP_THROW(TypeError, "kthvalue: unsupported dtype");
    }
#undef TP_KTHH
    std::vector<int64_t> out_shape = shape_of(host);
    out_shape[dim] = keepdim ? 1 : 0;
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    return {vals.reshape(out_shape).to(self.device()),
            idxs.reshape(out_shape).to(self.device())};
}

Tensor renorm_cuda(const Tensor& self, Scalar p, int64_t dim, Scalar maxnorm) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(sc), sc.dtype(), sc.device());
    int64_t d_size = sc.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(sc), dim, outer, inner);
    int64_t slices = outer * inner;
    if (slices > 0 && d_size > 0) {
        auto stream = getCurrentCUDAStream().stream();
        dim3 grid = make_grid(slices), block(kThreads);
        double pd = p.toDouble(), mn = maxnorm.toDouble();
#define TP_REN(ctype, name_) \
    case DType::name_: \
        renorm_slice_kernel<ctype><<<grid, block, 0, stream>>>( \
            slices, d_size, inner, sc.data_ptr<ctype>(), out.data_ptr<ctype>(), pd, mn); \
        break;
        switch (sc.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_REN)
            default: TP_THROW(TypeError, "renorm: unsupported dtype");
        }
#undef TP_REN
        CUDA_CHECK(cudaGetLastError());
    }
    return out;
}

std::tuple<Tensor, Tensor> interop_kthvalue_values_cuda(const Tensor& self, int64_t k, int64_t dim, bool keepdim,
              Tensor& values, Tensor& indices) {
        std::tie(values, indices) = kthvalue_cuda(self, k, dim, keepdim);
        return {values, indices};

}

TENSORPLAY_LIBRARY_IMPL(CUDA, ReduceKernels) {
    m.impl("amax", amax_cuda2);
    m.impl("amin", amin_cuda2);
    m.impl("aminmax", aminmax_cuda);
    m.impl("_aminmax", aminmax_all_cuda);
    m.impl("_aminmax.dim", aminmax_dim_cuda);
    m.impl("logsumexp", logsumexp_cuda2);
    m.impl("nansum", nansum_cuda2);
    m.impl("nanmedian", nanmedian_cuda);
    m.impl("count_nonzero", count_nonzero_cuda2);
    m.impl("count_nonzero.dim_IntList", count_nonzero_cuda2);
    m.impl("cummax", cummax_cuda);
    m.impl("cummin", cummin_cuda);
    m.impl("var_mean", var_mean_cuda);
    m.impl("std_mean", std_mean_cuda);
    m.impl("mode", mode_cuda);
    m.impl("kthvalue", kthvalue_cuda);
    m.impl("kthvalue.values", interop_kthvalue_values_cuda);
    m.impl("renorm", renorm_cuda);
}

} // namespace cuda
} // namespace tensorplay
