// Core operators - CUDA kernels.
//
// grid-stride loops; single-dim reductions assign one thread per output
// slice and walk the reduced dimension sequentially. Rare complex ops
// (mode/kthvalue/nanmedian/dist-special-p) are host-staged reference paths.
#include "Tensor.h"
#include "CUDAComplex.cuh"
#include <thrust/complex.h>
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"
#include "SpecialMath.h"

#include <cuda_runtime.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <tuple>
#include <utility>
#include <type_traits>
#include <optional>
#include <string>

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

// Canonical division from the arithmetic unit.  The complex branch of
// true_divide reuses it instead of restating the promotion table.
Tensor div_kernel(const Tensor& self, const Tensor& other);

namespace {

// Weak scalar participation: a scalar only promotes the tensor dtype when it
// carries a floating type of its own.
inline DType scalar_promote(DType t, const Scalar& s) {
    if (!isFloatingType(s.dtype())) return t;
    if (isFloatingType(t)) return t;
    return DType::Float32;
}

constexpr int kThreads = 256;

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(RuntimeError, "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim - ndim, ")");
    }
    return dim;
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
// Generic elementwise device kernels
// ---------------------------------------------------------------------------

template <typename T, typename Op>
__global__ void ew_binary_kernel(int64_t n, const T* a, const T* b, T* out, Op op) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = op(a[i], b[i]);
}

template <typename T, typename Pred>
__global__ void ew_bool_binary_kernel(int64_t n, const T* a, const T* b, bool* out, Pred pred) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = pred(a[i], b[i]);
}

template <typename T, typename Pred>
__global__ void ew_bool_unary_kernel(int64_t n, const T* a, bool* out, Pred pred) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = pred(a[i]);
}

template <typename T>
__device__ bool logical_truth_cuda(const T& value) {
    return static_cast<bool>(value);
}

template <typename T>
__device__ bool logical_truth_cuda(const thrust::complex<T>& value) {
    return value.real() != T(0) || value.imag() != T(0);
}

template <typename T, typename Pred>
__global__ void ew_logical_binary_kernel(int64_t n, const T* a, const T* b,
                                         bool* out, Pred pred) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        out[i] = pred(logical_truth_cuda(a[i]), logical_truth_cuda(b[i]));
    }
}

template <typename T, typename Pred>
__global__ void ew_logical_unary_kernel(int64_t n, const T* a, bool* out,
                                        Pred pred) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = pred(logical_truth_cuda(a[i]));
}

template <typename T, typename F>
__global__ void ew_unary_kernel(int64_t n, const T* a, T* out, F f) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = f(a[i]);
}

template <typename F>  // double(double,double)
__global__ void fm_binary_f64_kernel(int64_t n, const double* a, const double* b, double* out, F f) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = f(a[i], b[i]);
}

template <typename F>  // double(double,double)
__global__ void fm_binary_f32_kernel(int64_t n, const float* a, const float* b, float* out, F f) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = static_cast<float>(f(static_cast<double>(a[i]),
                                                             static_cast<double>(b[i])));
}

template <typename F>  // double(double)
__global__ void fm_unary_f64_kernel(int64_t n, const double* a, double* out, F f) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = f(a[i]);
}

template <typename F>  // double(double)
__global__ void fm_unary_f32_kernel(int64_t n, const float* a, float* out, F f) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = static_cast<float>(f(static_cast<double>(a[i])));
}

void launch_ew(dim3& grid, dim3& block, int64_t n) {
    block = dim3(kThreads);
    grid = dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads));
}

// Binary on common promoted dtype.
template <typename Op>
Tensor binary_same_cuda(const Tensor& a_in, const Tensor& b_in, Op op, const char* name) {
    std::vector<int64_t> out_shape = broadcast_shapes(shape_of(a_in), shape_of(b_in));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    Tensor ac = a_in.to(dt).expand(out_shape).contiguous();
    Tensor bc = b_in.to(dt).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, dt, a_in.device());
    int64_t n = out.numel();
    if (n == 0) return out;
    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
#define TP_BIN(ctype, name_) \
    case DType::name_: \
        ew_binary_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, ac.data_ptr<ctype>(), bc.data_ptr<ctype>(), out.data_ptr<ctype>(), op); \
        break;
    switch (dt) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BIN)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BIN
    CUDA_CHECK(cudaGetLastError());
    return out;
}

template <typename Pred>
Tensor binary_bool_cuda(const Tensor& a_in, const Tensor& b_in, Pred pred, const char* name) {
    std::vector<int64_t> out_shape = broadcast_shapes(shape_of(a_in), shape_of(b_in));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    Tensor ac = a_in.to(dt).expand(out_shape).contiguous();
    Tensor bc = b_in.to(dt).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, DType::Bool, a_in.device());
    int64_t n = out.numel();
    if (n == 0) return out;
    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
#define TP_BBIN(ctype, name_) \
    case DType::name_: \
        ew_bool_binary_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, ac.data_ptr<ctype>(), bc.data_ptr<ctype>(), out.data_ptr<bool>(), pred); \
        break;
    switch (dt) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BBIN)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BBIN
    CUDA_CHECK(cudaGetLastError());
    return out;
}

template <typename Pred>
Tensor bool_unary_cuda(const Tensor& self, Pred pred, const char* name) {
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(self), DType::Bool, self.device());
    int64_t n = self.numel();
    if (n == 0) return out;
    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
#define TP_BU(ctype, name_) \
    case DType::name_: \
        ew_bool_unary_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, sc.data_ptr<ctype>(), out.data_ptr<bool>(), pred); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BU)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BU
    CUDA_CHECK(cudaGetLastError());
    return out;
}
template <typename Pred>
Tensor logical_binary_cuda(const Tensor& a_in, const Tensor& b_in, Pred pred,
                          const char* name) {
    std::vector<int64_t> out_shape = broadcast_shapes(shape_of(a_in), shape_of(b_in));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    Tensor ac = a_in.to(dt).expand(out_shape).contiguous();
    Tensor bc = b_in.to(dt).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, DType::Bool, a_in.device());
    int64_t n = out.numel();
    if (n == 0) return out;
    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
#define TP_LOGICAL_BIN(ctype, name_) \
    case DType::name_: \
        ew_logical_binary_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, ac.data_ptr<ctype>(), bc.data_ptr<ctype>(), out.data_ptr<bool>(), pred); \
        break;
    switch (dt) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_LOGICAL_BIN)
        case DType::ComplexFloat:
            ew_logical_binary_kernel<thrust::complex<float>><<<grid, block, 0, stream>>>(
                n, static_cast<const thrust::complex<float>*>(ac.data_ptr()),
                static_cast<const thrust::complex<float>*>(bc.data_ptr()),
                out.data_ptr<bool>(), pred);
            break;
        case DType::ComplexDouble:
            ew_logical_binary_kernel<thrust::complex<double>><<<grid, block, 0, stream>>>(
                n, static_cast<const thrust::complex<double>*>(ac.data_ptr()),
                static_cast<const thrust::complex<double>*>(bc.data_ptr()),
                out.data_ptr<bool>(), pred);
            break;
        case DType::ComplexHalf:
        case DType::BComplex32:
            TP_THROW(NotImplementedError, name,
                     ": reduced complex types are not supported on CUDA");
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_LOGICAL_BIN
    CUDA_CHECK(cudaGetLastError());
    return out;
}

template <typename Pred>
Tensor logical_unary_cuda(const Tensor& self, Pred pred, const char* name) {
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(self), DType::Bool, self.device());
    int64_t n = self.numel();
    if (n == 0) return out;
    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
#define TP_LOGICAL_UNARY(ctype, name_) \
    case DType::name_: \
        ew_logical_unary_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, sc.data_ptr<ctype>(), out.data_ptr<bool>(), pred); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_LOGICAL_UNARY)
        case DType::ComplexFloat:
            ew_logical_unary_kernel<thrust::complex<float>><<<grid, block, 0, stream>>>(
                n, static_cast<const thrust::complex<float>*>(sc.data_ptr()),
                out.data_ptr<bool>(), pred);
            break;
        case DType::ComplexDouble:
            ew_logical_unary_kernel<thrust::complex<double>><<<grid, block, 0, stream>>>(
                n, static_cast<const thrust::complex<double>*>(sc.data_ptr()),
                out.data_ptr<bool>(), pred);
            break;
        case DType::ComplexHalf:
        case DType::BComplex32:
            TP_THROW(NotImplementedError, name,
                     ": reduced complex types are not supported on CUDA");
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_LOGICAL_UNARY
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// Dtype-preserving unary.
template <typename F>
Tensor dtype_unary_cuda(const Tensor& self, F f, const char* name) {
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    int64_t n = self.numel();
    if (n == 0) return out;
    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
#define TP_DU(ctype, name_) \
    case DType::name_: \
        ew_unary_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, sc.data_ptr<ctype>(), out.data_ptr<ctype>(), f); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_DU)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_DU
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// Math unary: f: double->double. Integral->Float32; Half/BF16 compute in
// float and keep dtype; Float32/Float64 preserved.
template <typename F>
Tensor float_math_cuda(const Tensor& self, F f, const char* name) {
    DType in = self.dtype();
    DType out_dt = isFloatingType(in) ? in : DType::Float32;
    DType compute_dt = (in == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor w = (in == compute_dt) ? self.contiguous() : self.to(compute_dt).contiguous();
    Tensor t = Tensor::empty(shape_of(w), compute_dt, w.device());
    int64_t n = w.numel();
    if (n > 0) {
        dim3 grid, block;
        launch_ew(grid, block, n);
        auto stream = getCurrentCUDAStream().stream();
        if (compute_dt == DType::Float64) {
            fm_unary_f64_kernel<<<grid, block, 0, stream>>>(n, w.data_ptr<double>(),
                                                            t.data_ptr<double>(), f);
        } else {
            fm_unary_f32_kernel<<<grid, block, 0, stream>>>(n, w.data_ptr<float>(),
                                                            t.data_ptr<float>(), f);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return (out_dt == compute_dt) ? t : t.to(out_dt);
}

// Math binary with floating promotion.
template <typename F>
Tensor binary_float_cuda(const Tensor& a_in, const Tensor& b_in, F f, const char* name) {
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    if (!isFloatingType(dt)) dt = DType::Float32;
    // Reduced-width inputs are evaluated in Float32 and narrowed once at the
    // end; the launches below only ever address float or double buffers.
    DType compute_dt = (dt == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor ac = a_in.to(compute_dt)
                    .expand(broadcast_shapes(shape_of(a_in), shape_of(b_in)))
                    .contiguous();
    Tensor bc = b_in.to(compute_dt).expand(shape_of(ac)).contiguous();
    Tensor out = Tensor::empty(shape_of(ac), compute_dt, a_in.device());
    int64_t n = out.numel();
    if (n == 0) return (dt == compute_dt) ? out : out.to(dt);
    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
    if (compute_dt == DType::Float64) {
        fm_binary_f64_kernel<<<grid, block, 0, stream>>>(
            n, ac.data_ptr<double>(), bc.data_ptr<double>(), out.data_ptr<double>(), f);
    } else {
        fm_binary_f32_kernel<<<grid, block, 0, stream>>>(
            n, ac.data_ptr<float>(), bc.data_ptr<float>(), out.data_ptr<float>(), f);
    }
    CUDA_CHECK(cudaGetLastError());
    return (dt == compute_dt) ? out : out.to(dt);
}

// ---------------------------------------------------------------------------
// Single-dim slice reduction: one thread per output slice, sequential along
// the reduced dimension. Accumulates in double.
// ---------------------------------------------------------------------------

template <typename T, class Step>
__global__ void slice_reduce_f64_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                        const T* in, double* out, double init, Step step) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        double acc = init;
        for (int64_t j = 0; j < d_size; ++j) acc = step(acc, static_cast<double>(sp[j * inner]));
        out[si] = acc;
    }
}

} // anonymous namespace

// ===========================================================================
// Arithmetic
// ===========================================================================

Tensor rsub_scalar_cuda(const Tensor& self, Scalar other, Scalar alpha) {
    DType dt = isFloatingType(other.dtype())
                   ? (isFloatingType(self.dtype()) ? self.dtype() : DType::Float32)
                   : self.dtype();
    Tensor sc = self.to(dt).contiguous();
    Tensor full = Tensor::full({}, other, dt, self.device())
                      .expand(shape_of(sc)).contiguous();
    double al = alpha.toDouble();
    // other - alpha * self: alpha scales the subtrahend, which is self here.
    return binary_same_cuda(sc, full,
                            [al] __device__ (auto s, auto o) {
                                using T = decltype(o);
                                return static_cast<T>(o - al * s);
                            },
                            "rsub");
}

Tensor rsub_tensor_cuda(const Tensor& self, const Tensor& other, Scalar alpha) {
    double al = alpha.toDouble();
    return binary_same_cuda(self, other,
                            [al] __device__ (auto s, auto o) {
                                using T = decltype(o);
                                return static_cast<T>(o - al * s);
                            },
                            "rsub");
}

Tensor true_divide_tensor_cuda(const Tensor& self, const Tensor& other) {
    // The float loop only addresses real buffers, so a complex operand takes
    // the canonical division path instead.
    if (isComplexType(self.dtype()) || isComplexType(other.dtype())) {
        return div_kernel(self, other);
    }
    return binary_float_cuda(self, other,
                             [] __device__ (double x, double y) { return x / y; }, "true_divide");
}
Tensor true_divide_scalar_cuda(const Tensor& self, Scalar other) {
    // A Float32 stand-in would widen Half/BFloat16 inputs.
    const DType dt = scalar_promote(self.dtype(), other);
    return true_divide_tensor_cuda(self, Tensor::full({}, other, dt, self.device()));
}
Tensor divide_tensor_cuda(const Tensor& self, const Tensor& other) {
    return true_divide_tensor_cuda(self, other);
}
Tensor divide_scalar_cuda(const Tensor& self, Scalar other) {
    return true_divide_scalar_cuda(self, other);
}

Tensor remainder_tensor_cuda(const Tensor& self, const Tensor& other) {
    return binary_same_cuda(self, other,
                            [] __device__ (auto x, auto y) -> decltype(x) {
                                using T = decltype(x);
                                T r;
                                if constexpr (std::is_integral_v<T>)
                                    r = static_cast<T>(x % y);
                                else  // Half/BFloat16/float/double via fmod
                                    r = static_cast<T>(::fmod(static_cast<double>(x), static_cast<double>(y)));
                                if (r != T(0) && ((r < static_cast<T>(0)) != (y < static_cast<T>(0)))) r = static_cast<T>(r + y);
                                return r;
                            },
                            "remainder");
}
Tensor remainder_scalar_cuda(const Tensor& self, Scalar other) {
    // Forcing the scalar into self's dtype would truncate a float divisor
    // against an integral tensor; the pair promotes first.
    const DType dt = scalar_promote(self.dtype(), other);
    return remainder_tensor_cuda(self.to(dt), Tensor::full({}, other, dt, self.device()));
}
Tensor remainder_scalar_tensor_cuda(Scalar self, const Tensor& other) {
    const DType dt = scalar_promote(other.dtype(), self);
    return remainder_tensor_cuda(Tensor::full({}, self, dt, other.device()), other.to(dt));
}
Tensor fmod_tensor_cuda(const Tensor& self, const Tensor& other) {
    return binary_same_cuda(self, other,
                            [] __device__ (auto x, auto y) -> decltype(x) {
                                if constexpr (std::is_integral_v<decltype(x)>)
                                    return static_cast<decltype(x)>(x % y);
                                else
                                    return static_cast<decltype(x)>(::fmod(static_cast<double>(x), static_cast<double>(y)));
                            },
                            "fmod");
}
Tensor fmod_scalar_cuda(const Tensor& self, Scalar other) {
    const DType dt = scalar_promote(self.dtype(), other);
    return fmod_tensor_cuda(self.to(dt), Tensor::full({}, other, dt, self.device()));
}
Tensor subtract_tensor_cuda(const Tensor& self, const Tensor& other, Scalar alpha) {
    // alpha == 1 is by far the common call and keeps the unscaled loop.
    if (!alpha.isComplex() && alpha.toDouble() == 1.0) {
        return binary_same_cuda(self, other,
                                [] __device__ (auto x, auto y) { return x - y; }, "subtract");
    }
    const double al = alpha.toDouble();
    return binary_same_cuda(self, other,
                            [al] __device__ (auto x, auto y) {
                                using T = decltype(x);
                                return static_cast<T>(x - y * al);
                            }, "subtract");
}
Tensor subtract_scalar_cuda(const Tensor& self, Scalar other, Scalar alpha) {
    const DType dt = scalar_promote(self.dtype(), other);
    return subtract_tensor_cuda(self.to(dt),
                                Tensor::full({}, other, dt, self.device()), alpha);
}
Tensor multiply_tensor_cuda(const Tensor& self, const Tensor& other) {
    return binary_same_cuda(self, other,
                            [] __device__ (auto x, auto y) { return x * y; }, "multiply");
}
Tensor multiply_scalar_cuda(const Tensor& self, Scalar other) {
    double ov = other.toDouble();
    return dtype_unary_cuda(self,
                            [ov] __device__ (auto x) {
                                using T = decltype(x);
                                return static_cast<T>(static_cast<double>(x) * ov);
                            },
                            "multiply");
}

// ---------------------------------------------------------------------------
// Division with an explicit rounding mode
// ---------------------------------------------------------------------------
namespace {

enum class DivRounding { kTrue, kTrunc, kFloor };

DivRounding parse_div_rounding(const std::optional<std::string>& mode) {
    if (!mode.has_value()) return DivRounding::kTrue;
    if (*mode == "trunc") return DivRounding::kTrunc;
    if (*mode == "floor") return DivRounding::kFloor;
    TP_THROW(RuntimeError,
             std::string("div expected rounding_mode to be one of None, 'trunc' "
                         "or 'floor' but found '") + *mode + "'");
}

Tensor div_rounded_core(const Tensor& a, const Tensor& b, DivRounding rounding) {
    if (rounding == DivRounding::kTrue) return true_divide_tensor_cuda(a, b);
    // Rounded division stays in the input dtype: an integral pair must come
    // back integral, which the float promotion of true division loses.
    const bool floor_mode = (rounding == DivRounding::kFloor);
    return binary_same_cuda(a, b, [floor_mode] __device__ (auto x, auto y) -> decltype(x) {
        using T = decltype(x);
        if constexpr (std::is_integral_v<T>) {
            if (y == T(0)) return T(0);
            T q = static_cast<T>(x / y);
            if (floor_mode) {
                // The quotient truncates toward zero, so a remainder whose
                // sign disagrees with the divisor sits one step above the
                // floor.
                T r = static_cast<T>(x - q * y);
                if (r != T(0) && ((r < T(0)) != (y < T(0)))) q = static_cast<T>(q - T(1));
            }
            return q;
        } else {
            // Half/BFloat16 round through Float32, the width their arithmetic
            // is defined at; float and double keep their own.
            using C = std::conditional_t<std::is_same_v<T, double>, double, float>;
            const C q = static_cast<C>(x) / static_cast<C>(y);
            return static_cast<T>(floor_mode ? ::floor(q) : ::trunc(q));
        }
    }, "div");
}

Tensor div_rounded_scalar(const Tensor& self, Scalar other, DivRounding rounding) {
    if (rounding == DivRounding::kTrue) return true_divide_scalar_cuda(self, other);
    const DType dt = scalar_promote(self.dtype(), other);
    return div_rounded_core(self.to(dt), Tensor::full({}, other, dt, self.device()),
                            rounding);
}

}  // namespace

Tensor div_mode_tensor_cuda(const Tensor& self, const Tensor& other,
                            std::optional<std::string> rounding_mode) {
    return div_rounded_core(self, other, parse_div_rounding(rounding_mode));
}
Tensor div_mode_scalar_cuda(const Tensor& self, Scalar other,
                            std::optional<std::string> rounding_mode) {
    return div_rounded_scalar(self, other, parse_div_rounding(rounding_mode));
}
Tensor floor_divide_cuda(const Tensor& self, const Tensor& other) {
    return div_rounded_core(self, other, DivRounding::kFloor);
}
Tensor floor_divide_scalar_cuda(const Tensor& self, Scalar other) {
    return div_rounded_scalar(self, other, DivRounding::kFloor);
}

Tensor negative_cuda(const Tensor& self) {
    return dtype_unary_cuda(self,
                            [] __device__ (auto x) { return static_cast<decltype(x)>(-x); },
                            "negative");
}
Tensor positive_cuda(const Tensor& self) { return self.clone(); }

// ===========================================================================
// Comparisons / logic
// ===========================================================================

Tensor greater_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, [] __device__ (auto x, auto y) { return x > y; }, "greater");
}
Tensor greater_equal_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, [] __device__ (auto x, auto y) { return x >= y; }, "greater_equal");
}
Tensor less_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, [] __device__ (auto x, auto y) { return x < y; }, "less");
}
Tensor less_equal_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, [] __device__ (auto x, auto y) { return x <= y; }, "less_equal");
}
Tensor not_equal_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, [] __device__ (auto x, auto y) { return x != y; }, "not_equal");
}
Tensor signbit_cuda(const Tensor& self) {
    return bool_unary_cuda(self, [] __device__ (auto x) -> bool {
        return static_cast<double>(x) < 0.0 ||
               (static_cast<double>(x) == 0.0 && 1.0 / static_cast<double>(x) < 0.0);
    }, "signbit");
}
Tensor logical_not_cuda(const Tensor& self) {
    return logical_unary_cuda(self, [] __device__ (bool x) -> bool { return !x; },
                              "logical_not");
}
Tensor logical_and_cuda(const Tensor& a, const Tensor& b) {
    return logical_binary_cuda(a, b, [] __device__ (bool x, bool y) -> bool {
        return x && y;
    }, "logical_and");
}
Tensor logical_or_cuda(const Tensor& a, const Tensor& b) {
    return logical_binary_cuda(a, b, [] __device__ (bool x, bool y) -> bool {
        return x || y;
    }, "logical_or");
}
Tensor logical_xor_cuda(const Tensor& a, const Tensor& b) {
    return logical_binary_cuda(a, b, [] __device__ (bool x, bool y) -> bool {
        return x != y;
    }, "logical_xor");
}
Tensor isfinite_cuda(const Tensor& self) {
    return bool_unary_cuda(self, [] __device__ (auto x) -> bool {
        double d = static_cast<double>(x);
        return d == d && d != std::numeric_limits<double>::infinity() &&
               d != -std::numeric_limits<double>::infinity();
    }, "isfinite");
}
Tensor isinf_cuda(const Tensor& self) {
    return bool_unary_cuda(self, [] __device__ (auto x) -> bool {
        double d = static_cast<double>(x);
        return d == std::numeric_limits<double>::infinity() ||
               d == -std::numeric_limits<double>::infinity();
    }, "isinf");
}
Tensor isnan_cuda(const Tensor& self) {
    return bool_unary_cuda(self, [] __device__ (auto x) -> bool {
        double d = static_cast<double>(x);
        return d != d;
    }, "isnan");
}
Tensor isneginf_cuda(const Tensor& self) {
    return bool_unary_cuda(self, [] __device__ (auto x) -> bool {
        return static_cast<double>(x) == -std::numeric_limits<double>::infinity();
    }, "isneginf");
}
Tensor isposinf_cuda(const Tensor& self) {
    return bool_unary_cuda(self, [] __device__ (auto x) -> bool {
        return static_cast<double>(x) == std::numeric_limits<double>::infinity();
    }, "isposinf");
}

// ===========================================================================
// Math functions
// ===========================================================================

Tensor reciprocal_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        if (self.dtype() != DType::ComplexFloat &&
            self.dtype() != DType::ComplexDouble)
            TP_THROW(NotImplementedError,
                     "CUDA reciprocal: half complexes not supported");
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), self.dtype(),
            self.device());
        const int64_t n = self.numel();
        auto stream = getCurrentCUDAStream().stream();
        Tensor sc = self.contiguous();
        if (self.dtype() == DType::ComplexFloat)
            cuda::cplx::launch_unary<float>(
                n, sc.data_ptr(), result.data_ptr(),
                cuda::cplx::RecipOp{}, stream);
        else
            cuda::cplx::launch_unary<double>(
                n, sc.data_ptr(), result.data_ptr(),
                cuda::cplx::RecipOp{}, stream);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    return float_math_cuda(self, [] __device__ (double x) { return 1.0 / x; }, "reciprocal");
}
Tensor sgn_cuda(const Tensor& self) {
    return dtype_unary_cuda(self,
                            [] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                double d = static_cast<double>(x);
                                if (d != d) return static_cast<T>(x);
                                if (d > 0) return static_cast<T>(1);
                                if (d < 0) return static_cast<T>(-1);
                                return static_cast<T>(0);
                            },
                            "sgn");
}
Tensor exp2_cuda(const Tensor& self) {
    return float_math_cuda(self, [] __device__ (double x) { return ::exp2(x); }, "exp2");
}
Tensor sinc_cuda(const Tensor& self) {
    return float_math_cuda(self, [] __device__ (double x) {
        double px = M_PI * x;
        return ::fabs(px) < 1e-30 ? 1.0 : ::sin(px) / px;
    }, "sinc");
}
Tensor deg2rad_cuda(const Tensor& self) {
    return float_math_cuda(self, [] __device__ (double x) { return x * (M_PI / 180.0); }, "deg2rad");
}
Tensor rad2deg_cuda(const Tensor& self) {
    return float_math_cuda(self, [] __device__ (double x) { return x * (180.0 / M_PI); }, "rad2deg");
}
Tensor fix_cuda(const Tensor& self) {
    return dtype_unary_cuda(self,
                            [] __device__ (auto x) -> decltype(x) {
                                if constexpr (std::is_floating_point_v<decltype(x)>)
                                    return static_cast<decltype(x)>(::trunc(static_cast<double>(x)));
                                else
                                    return x;
                            },
                            "fix");
}
Tensor erfinv_cuda(const Tensor& self) {
    // CUDA has no native erfinv; use the Cephes calc_erfinv from SpecialMath.h
    // host-only ::erfinv — linking that from device code leaves an undefined
    // symbol in libp10.so.
    return float_math_cuda(self, [] __device__ (double x) { return tensorplay::special_math::calc_erfinv(x); }, "erfinv");
}
Tensor logit_cuda(const Tensor& self, std::optional<Scalar> eps) {
    double e = eps.has_value() ? eps->toDouble() : -1.0;
    return float_math_cuda(self,
                           [e] __device__ (double p) {
                               if (e >= 0) p = ::fmin(::fmax(p, e), 1.0 - e);
                               return ::log(p / (1.0 - p));
                           },
                           "logit");
}
Tensor digamma_cuda(const Tensor& self) {
    return float_math_cuda(self, [] __device__ (double v) {
        if (v <= 0 && v == ::floor(v)) return ::nan("");
        double r = 0;
        while (v < 6.0) { r -= 1.0 / v; v += 1.0; }
        double inv = 1.0 / v, inv2 = inv * inv;
        r += ::log(v) - 0.5 * inv
             - inv2 * (1.0/12.0 - inv2 * (1.0/120.0 - inv2 * (1.0/252.0 - inv2 * (1.0/240.0 - inv2 / 132.0))));
        return r;
    }, "digamma");
}
Tensor i0_cuda(const Tensor& self) {
    // Chebyshev expansion, valid over the whole range; see i0_cpu.
    return float_math_cuda(self, [] __device__ (double v) {
        return tensorplay::special_math::modified_bessel_i0_forward(v);
    }, "i0");
}
Tensor nan_to_num_cuda(const Tensor& self, Scalar nan,
                       std::optional<Scalar> posinf, std::optional<Scalar> neginf) {
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    int64_t n = self.numel();
    if (n == 0) return out;
    double nan_v = nan.toDouble();
    bool has_pos = posinf.has_value(), has_neg = neginf.has_value();
    double pos_v = has_pos ? posinf->toDouble() : std::numeric_limits<double>::infinity();
    double neg_v = has_neg ? neginf->toDouble() : -std::numeric_limits<double>::infinity();
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid, block;
    launch_ew(grid, block, n);
    // Per-dtype kernel with the replacement rules baked in.
#define TP_NTN(ctype, name_) \
    case DType::name_: { \
        ctype pv = has_pos ? static_cast<ctype>(pos_v) \
                           : (std::numeric_limits<ctype>::has_infinity \
                                  ? std::numeric_limits<ctype>::infinity() \
                                  : std::numeric_limits<ctype>::max()); \
        ctype nv = has_neg ? static_cast<ctype>(neg_v) \
                           : (std::numeric_limits<ctype>::has_infinity \
                                  ? -std::numeric_limits<ctype>::infinity() \
                                  : std::numeric_limits<ctype>::lowest()); \
        ctype na = static_cast<ctype>(nan_v); \
        ew_unary_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, sc.data_ptr<ctype>(), out.data_ptr<ctype>(), \
            [na, pv, nv] __device__ (ctype v) { \
                if constexpr (std::is_floating_point_v<ctype>) { \
                    if (v != v) return na; \
                    if (v == std::numeric_limits<ctype>::infinity()) return pv; \
                    if (v == -std::numeric_limits<ctype>::infinity()) return nv; \
                } \
                return v; \
            }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_NTN)
        default: TP_THROW(TypeError, "nan_to_num: unsupported dtype");
    }
#undef TP_NTN
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor xlogy_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, [] __device__ (double x, double y) {
        return tensorplay::special_math::calc_xlogy(x, y);
    }, "xlogy");
}
Tensor logaddexp_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, [] __device__ (double x, double y) {
        double m = ::fmax(x, y);
        if (m == -std::numeric_limits<double>::infinity() || m != m) return m;
        return m + ::log1p(::exp(-::fabs(x - y)));
    }, "logaddexp");
}
Tensor logaddexp2_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, [] __device__ (double x, double y) {
        double m = ::fmax(x, y);
        if (m == -std::numeric_limits<double>::infinity() || m != m) return m;
        return m + ::log1p(::exp2(-::fabs(x - y))) / M_LN2;
    }, "logaddexp2");
}
Tensor copysign_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, [] __device__ (double x, double y) {
        return ::copysign(x, y);
    }, "copysign");
}
Tensor copysign_scalar_cuda(const Tensor& self, Scalar other) {
    // The sign comes from the scalar alone, so the divisor width never
    // participates in promotion -- Float32 carries every sign bit exactly.
    return copysign_cuda(self, Tensor::full({}, other, DType::Float32, self.device()));
}
Tensor hypot_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, [] __device__ (double x, double y) {
        return ::hypot(x, y);
    }, "hypot");
}
Tensor nextafter_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, [] __device__ (double x, double y) {
        return ::nextafter(x, y);
    }, "nextafter");
}
Tensor gcd_cuda(const Tensor& a, const Tensor& b) {
    DType dt = promoteTypes(a.dtype(), b.dtype());
    if (isFloatingType(dt)) TP_THROW(TypeError, "gcd only supports integral tensors");
    return binary_same_cuda(a, b,
                            [] __device__ (auto x, auto y) -> decltype(x) {
                                using T = decltype(x);
                                long long ux = static_cast<long long>(x < static_cast<T>(0) ? -x : x);
                                long long uy = static_cast<long long>(y < static_cast<T>(0) ? -y : y);
                                while (uy) { long long t = ux % uy; ux = uy; uy = t; }
                                return static_cast<T>(ux);
                            },
                            "gcd");
}
Tensor lcm_cuda(const Tensor& a, const Tensor& b) {
    DType dt = promoteTypes(a.dtype(), b.dtype());
    if (isFloatingType(dt)) TP_THROW(TypeError, "lcm only supports integral tensors");
    return binary_same_cuda(a, b,
                            [] __device__ (auto x, auto y) -> decltype(x) {
                                using T = decltype(x);
                                long long ux = static_cast<long long>(x < static_cast<T>(0) ? -x : x);
                                long long uy = static_cast<long long>(y < static_cast<T>(0) ? -y : y);
                                long long g = ux, t2 = uy;
                                while (t2) { long long t3 = g % t2; g = t2; t2 = t3; }
                                if (g == 0) return static_cast<T>(0);
                                return static_cast<T>(ux / g * uy);
                            },
                            "lcm");
}
Tensor heaviside_cuda(const Tensor& a, const Tensor& values) {
    return binary_same_cuda(a, values,
                            [] __device__ (auto x, auto v) -> decltype(x) {
                                using T = decltype(x);
                                double xd = static_cast<double>(x);
                                if (xd < 0.0) return static_cast<T>(0);
                                if (xd == 0.0) return static_cast<T>(v);
                                return static_cast<T>(1);
                            },
                            "heaviside");
}

// ===========================================================================
// Clamp family
// ===========================================================================

Tensor clamp_min_scalar_cuda(const Tensor& self, Scalar min) {
    double lo = min.toDouble();
    return dtype_unary_cuda(self,
                            [lo] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                return static_cast<double>(x) < lo ? static_cast<T>(lo)
                                                                   : static_cast<T>(x);
                            },
                            "clamp_min");
}
Tensor clamp_max_scalar_cuda(const Tensor& self, Scalar max) {
    double hi = max.toDouble();
    return dtype_unary_cuda(self,
                            [hi] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                return static_cast<double>(x) > hi ? static_cast<T>(hi)
                                                                   : static_cast<T>(x);
                            },
                            "clamp_max");
}
Tensor clamp_min_tensor_cuda(const Tensor& self, const Tensor& min) {
    return binary_same_cuda(self, min,
                            [] __device__ (auto x, auto m) -> decltype(x) {
                                using T = decltype(x);
                                return static_cast<double>(m) > static_cast<double>(x)
                                           ? static_cast<T>(m) : static_cast<T>(x);
                            },
                            "clamp_min");
}
Tensor clamp_max_tensor_cuda(const Tensor& self, const Tensor& max) {
    return binary_same_cuda(self, max,
                            [] __device__ (auto x, auto m) -> decltype(x) {
                                using T = decltype(x);
                                return static_cast<double>(m) < static_cast<double>(x)
                                           ? static_cast<T>(m) : static_cast<T>(x);
                            },
                            "clamp_max");
}
Tensor clip_cuda(const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    if (min.has_value() && max.has_value()) {
        Tensor r = clamp_min_scalar_cuda(self, *min);
        return clamp_max_scalar_cuda(r, *max);
    }
    if (min.has_value()) return clamp_min_scalar_cuda(self, *min);
    if (max.has_value()) return clamp_max_scalar_cuda(self, *max);
    return self.clone();
}
Tensor& clamp__cuda(Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    Tensor r = clip_cuda(self, std::move(min), std::move(max));
    self.copy_(r);
    return self;
}

// Keep them separate from clamp_ (which accepts two optional bounds): these
// are the unsuffixed dispatcher names used by the generated native schemas.
Tensor& clamp_min__scalar_cuda(Tensor& self, Scalar min) {
    self.copy_(clamp_min_scalar_cuda(self, min));
    return self;
}
Tensor& clamp_max__scalar_cuda(Tensor& self, Scalar max) {
    self.copy_(clamp_max_scalar_cuda(self, max));
    return self;
}

// ===========================================================================
// Activations
// ===========================================================================

Tensor selu_cuda(const Tensor& self) {
    constexpr double kAlpha = 1.6732632423543772848170429916717;
    constexpr double kScale = 1.0507009873554804934193349852946;
    return dtype_unary_cuda(self,
                            [=] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                double v = static_cast<double>(x);
                                return static_cast<T>(v > 0 ? kScale * v
                                                            : kScale * kAlpha * (::exp(v) - 1.0));
                            },
                            "selu");
}
Tensor celu_cuda(const Tensor& self, Scalar alpha) {
    double a = alpha.toDouble();
    return dtype_unary_cuda(self,
                            [a] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                double v = static_cast<double>(x);
                                return static_cast<T>(v > 0 ? v : a * (::exp(v / a) - 1.0));
                            },
                            "celu");
}
Tensor hardshrink_cuda(const Tensor& self, Scalar lambd) {
    double l = lambd.toDouble();
    // lambd.to<scalar_t>(), so float32 boundary values compare exactly.
    return dtype_unary_cuda(self,
                            [l] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                const double lt = static_cast<double>(static_cast<T>(l));
                                double v = static_cast<double>(x);
                                return (v >= -lt && v <= lt) ? static_cast<T>(0) : x;
                            },
                            "hardshrink");
}
Tensor softshrink_cuda(const Tensor& self, Scalar lambd) {
    double l = lambd.toDouble();
    // (a < -l ? a+l : 0)); the v*0 middle branch keeps NaN propagating.
    return dtype_unary_cuda(self,
                            [l] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                const double lt = static_cast<double>(static_cast<T>(l));
                                double v = static_cast<double>(x);
                                if (v > lt) return static_cast<T>(v - lt);
                                if (v < -lt) return static_cast<T>(v + lt);
                                return static_cast<T>(v * 0.0);
                            },
                            "softshrink");
}
// hard/soft): grad passes through where self is outside the inclusive
// [-lambd, lambd] band.
Tensor hardshrink_backward_cuda(const Tensor& grad_out, const Tensor& self, Scalar lambd) {
    double l = lambd.toDouble();
    return binary_same_cuda(grad_out, self,
                            [l] __device__ (auto g, auto s) -> decltype(g) {
                                using T = decltype(g);
                                const double lt = static_cast<double>(static_cast<T>(l));
                                double v = static_cast<double>(s);
                                return (v >= -lt && v <= lt) ? static_cast<T>(0) : g;
                            },
                            "hardshrink_backward");
}
Tensor softshrink_backward_cuda(const Tensor& grad_output, const Tensor& self, Scalar lambd) {
    double l = lambd.toDouble();
    return binary_same_cuda(grad_output, self,
                            [l] __device__ (auto g, auto s) -> decltype(g) {
                                using T = decltype(g);
                                const double lt = static_cast<double>(static_cast<T>(l));
                                double v = static_cast<double>(s);
                                return (v >= -lt && v <= lt) ? static_cast<T>(0) : g;
                            },
                            "softshrink_backward");
}
Tensor sigmoid_backward_cuda(const Tensor& grad_output, const Tensor& output) {
    return binary_same_cuda(grad_output, output,
                            [] __device__ (auto g, auto o) -> decltype(g) {
                                using T = decltype(o);
                                return g * o * (static_cast<T>(1) - o);
                            },
                            "sigmoid_backward");
}
Tensor tanh_backward_cuda(const Tensor& grad_output, const Tensor& output) {
    return binary_same_cuda(grad_output, output,
                            [] __device__ (auto g, auto o) -> decltype(g) {
                                using T = decltype(o);
                                return g * (static_cast<T>(1) - o * o);
                            },
                            "tanh_backward");
}
// eps (eps<0) the gradient is dy/(x(1-x)) inside [0,1] and NaN outside; with
// eps>=0 values outside [eps, 1-eps] (compared in the element dtype) are
// masked to zero. Exact 0/1 fall through to the division (dy/0 -> inf).
Tensor logit_backward_cuda(const Tensor& grad_output, const Tensor& self, std::optional<Scalar> eps) {
    double e = eps.has_value() ? eps->toDouble() : -1.0;
    return binary_same_cuda(grad_output, self,
                            [e] __device__ (auto g, auto s) -> decltype(g) {
                                using T = decltype(s);
                                const T zero = static_cast<T>(0);
                                const T one = static_cast<T>(1);
                                if (e < 0) {
                                    if (s < zero || s > one) return std::numeric_limits<T>::quiet_NaN();
                                    return g / (s * (one - s));
                                }
                                const T lo = static_cast<T>(e);
                                const T hi = one - lo;
                                if (s < lo || s > hi) return zero;
                                return g / (s * (one - s));
                            },
                            "logit_backward");
}
Tensor threshold_cuda(const Tensor& self, Scalar threshold, Scalar value) {
    double t = threshold.toDouble(), val = value.toDouble();
    return dtype_unary_cuda(self,
                            [t, val] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                return static_cast<double>(x) <= t ? static_cast<T>(val)
                                                                   : static_cast<T>(x);
                            },
                            "threshold");
}
namespace {

// Arithmetic domain of the PReLU evaluation: single precision stays single
// precision, half formats widen to float so the slope product rounds once,
// and the integral element types follow the same double-precision path the
// host kernels use.
template <typename T> struct PReluMath { using type = double; };
template <> struct PReluMath<float> { using type = float; };
template <> struct PReluMath<double> { using type = double; };
template <> struct PReluMath<Half> { using type = float; };
template <> struct PReluMath<BFloat16> { using type = float; };

// One slope per channel: with `inner` the number of elements that follow the
// channel axis and `C` the channel count, a linear position i reads slope
// (i / inner) % C.  C == 1 is the shared-slope form and skips the division.
template <typename scalar_t, typename math_t>
__global__ void prelu_channel_kernel(int64_t n, int64_t inner, int64_t C,
                                     const scalar_t* __restrict__ input,
                                     const scalar_t* __restrict__ weight,
                                     scalar_t* __restrict__ output) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += step) {
        const math_t x = static_cast<math_t>(input[i]);
        const int64_t c = (C == 1) ? 0 : (i / inner) % C;
        const math_t w = static_cast<math_t>(weight[c]);
        output[i] = static_cast<scalar_t>(x > math_t(0) ? x : w * x);
    }
}

// Both gradients in one pass:
//     grad_input  = x > 0 ? g : w * g
//     grad_weight = x > 0 ? 0 : x * g
// grad_weight keeps the input geometry; folding it onto the weight shape is
// the caller's job.
template <typename scalar_t, typename math_t>
__global__ void prelu_channel_backward_kernel(
        int64_t n, int64_t inner, int64_t C,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ weight,
        const scalar_t* __restrict__ grad_output,
        scalar_t* __restrict__ grad_input,
        scalar_t* __restrict__ grad_weight) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += step) {
        const math_t x = static_cast<math_t>(input[i]);
        const math_t g = static_cast<math_t>(grad_output[i]);
        const int64_t c = (C == 1) ? 0 : (i / inner) % C;
        const math_t w = static_cast<math_t>(weight[c]);
        const bool positive = x > math_t(0);
        grad_input[i] = static_cast<scalar_t>(positive ? g : w * g);
        grad_weight[i] = static_cast<scalar_t>(positive ? math_t(0) : x * g);
    }
}

// Channel count and trailing-element count for the slope lookup: with
// `inner` the number of elements that follow the channel axis, a linear
// position i reads slope (i / inner) % C.
struct PReluGeometry {
    int64_t C = 1;
    int64_t inner = 1;
};

// Resolves the weight of the broadcast overload into a contiguous slope
// buffer plus its lookup geometry.  A weight that varies along a single axis
// (the usual per-channel parameter) is read in place; any other broadcast is
// materialized at the input geometry and indexed elementwise, which keeps the
// whole evaluation on device.
Tensor prelu_prepare_broadcast_weight(const Tensor& self, const Tensor& weight,
                                      PReluGeometry* geometry) {
    *geometry = PReluGeometry{};
    Tensor slope =
        weight.dtype() == self.dtype() ? weight : weight.to(self.dtype());
    if (slope.numel() == 1) return slope.contiguous();

    const int64_t self_rank = self.dim();
    const int64_t weight_rank = slope.dim();
    if (weight_rank <= self_rank) {
        const int64_t offset = self_rank - weight_rank;
        int64_t channel = -1;
        bool single_axis = true;
        for (int64_t d = 0; d < weight_rank; ++d) {
            if (slope.size(d) == 1) continue;
            if (channel >= 0 || self.size(d + offset) != slope.size(d)) {
                single_axis = false;
                break;
            }
            channel = d + offset;
        }
        if (single_axis && channel >= 0) {
            geometry->C = self.size(channel);
            int64_t inner = 1;
            for (int64_t k = channel + 1; k < self_rank; ++k) inner *= self.size(k);
            geometry->inner = inner;
            return slope.contiguous();
        }
    }

    Tensor expanded = slope.expand(shape_of(self)).contiguous();
    geometry->C = expanded.numel();
    geometry->inner = 1;
    return expanded;
}

// Slope geometry of the public overload, whose weight is a scalar or a plain
// 1-D vector: the channel axis is dim 1 once the input has a batch dimension
// and dim 0 otherwise.
PReluGeometry prelu_vector_geometry(const Tensor& self, const Tensor& weight) {
    PReluGeometry geometry;
    const int64_t C = weight.numel();
    if (C == 1) return geometry;
    TP_CHECK(self.dim() > 0, "prelu: a per-channel weight needs a shaped input");
    const int64_t channel_dim = self.dim() >= 2 ? 1 : 0;
    TP_CHECK(self.size(channel_dim) == C,
             "prelu: weight numel does not match the input channel count");
    geometry.C = C;
    int64_t inner = 1;
    for (int64_t k = channel_dim + 1; k < self.dim(); ++k) inner *= self.size(k);
    geometry.inner = inner;
    return geometry;
}

Tensor prelu_forward_common(const Tensor& self, const Tensor& slope,
                            const PReluGeometry& geometry) {
    const Tensor input = self.contiguous();
    Tensor output = Tensor::empty(shape_of(input), self.dtype(), self.device());
    const int64_t n = input.numel();
    if (n == 0) return output;

    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
#define TP_PRELU_FWD(ctype, name)                                              \
    case DType::name:                                                          \
        prelu_channel_kernel<ctype, typename PReluMath<ctype>::type>           \
            <<<grid, block, 0, stream>>>(n, geometry.inner, geometry.C,        \
                                         input.data_ptr<ctype>(),              \
                                         slope.data_ptr<ctype>(),              \
                                         output.data_ptr<ctype>());            \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_PRELU_FWD)
        default: TP_THROW(TypeError, "prelu: unsupported dtype");
    }
#undef TP_PRELU_FWD
    CUDA_CHECK(cudaGetLastError());
    return output;
}

}  // anonymous namespace

// The public overload accepts either spelling of the slope: a scalar or a
// plain per-channel vector, or a weight the caller has already shaped to
// broadcast over the input.  Both name the same channel axis, so they answer
// the same values.
Tensor prelu_cuda(const Tensor& self, const Tensor& weight) {
    if (weight.dim() > 1) {
        PReluGeometry geometry;
        const Tensor slope =
            prelu_prepare_broadcast_weight(self, weight, &geometry);
        return prelu_forward_common(self, slope, geometry);
    }
    const Tensor slope = weight.dtype() == self.dtype()
                             ? weight.contiguous()
                             : weight.to(self.dtype()).contiguous();
    return prelu_forward_common(self, slope,
                                prelu_vector_geometry(self, weight));
}

Tensor _prelu_kernel_cuda(const Tensor& self, const Tensor& weight) {
    PReluGeometry geometry;
    const Tensor slope =
        prelu_prepare_broadcast_weight(self, weight, &geometry);
    return prelu_forward_common(self, slope, geometry);
}

std::tuple<Tensor, Tensor> _prelu_kernel_backward_cuda(const Tensor& grad_output,
                                                       const Tensor& self,
                                                       const Tensor& weight) {
    TP_CHECK(self.dtype() == weight.dtype() &&
                 self.dtype() == grad_output.dtype(),
             "_prelu_kernel_backward: input, weight and grad_output must share "
             "one dtype");
    PReluGeometry geometry;
    const Tensor slope =
        prelu_prepare_broadcast_weight(self, weight, &geometry);
    const Tensor input = self.contiguous();
    const Tensor grad = grad_output.contiguous();
    TP_CHECK(grad.numel() == input.numel(),
             "_prelu_kernel_backward: grad_output must match the input shape");

    Tensor grad_input =
        Tensor::empty(shape_of(input), self.dtype(), self.device());
    Tensor grad_weight =
        Tensor::empty(shape_of(input), self.dtype(), self.device());
    const int64_t n = input.numel();
    if (n == 0) return {grad_input, grad_weight};

    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
#define TP_PRELU_BWD(ctype)                                                    \
    prelu_channel_backward_kernel<ctype, typename PReluMath<ctype>::type>      \
        <<<grid, block, 0, stream>>>(n, geometry.inner, geometry.C,            \
                                     input.data_ptr<ctype>(),                  \
                                     slope.data_ptr<ctype>(),                  \
                                     grad.data_ptr<ctype>(),                   \
                                     grad_input.data_ptr<ctype>(),             \
                                     grad_weight.data_ptr<ctype>())
    switch (self.dtype()) {
        case DType::Float32: TP_PRELU_BWD(float); break;
        case DType::Float64: TP_PRELU_BWD(double); break;
        case DType::Float16: TP_PRELU_BWD(Half); break;
        case DType::BFloat16: TP_PRELU_BWD(BFloat16); break;
        default:
            TP_THROW(TypeError, "_prelu_kernel_backward: unsupported dtype");
    }
#undef TP_PRELU_BWD
    CUDA_CHECK(cudaGetLastError());
    return {grad_input, grad_weight};
}

// ===========================================================================
// ===========================================================================

// Cross-TU kernels reused by the composites below.
Tensor nansum_cuda2(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim);
Tensor sum_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, DType dtype);
Tensor isnan_cuda(const Tensor& self);

namespace {

__global__ void nanmean_zero_mask_kernel(int64_t n, const float* count, float* data) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        if (count[i] == 0.0f) data[i] = std::numeric_limits<float>::quiet_NaN();
    }
}

} // anonymous namespace

Tensor nanmean_cuda(const Tensor& self, std::optional<int64_t> dim_opt, bool keepdim,
                    std::optional<DType> dtype) {
    DType acc_dt = dtype.value_or(DType::Undefined);
    if (!isFloatingType(self.dtype()) && !isComplexType(self.dtype())) {
        TP_THROW(TypeError,
                 "nanmean(): expected input to have floating point or complex dtype but got ",
                 toString(self.dtype()));
    }
    if (acc_dt != DType::Undefined && !isFloatingType(acc_dt) &&
        !isComplexType(acc_dt)) {
        TP_THROW(TypeError,
                 "nanmean(): could not infer output dtype. Optional dtype must be either a floating point or complex dtype. Got: ",
                 toString(acc_dt));
    }
    Tensor x = self;
    if (acc_dt != DType::Undefined && x.dtype() != acc_dt) {
        x = x.to(acc_dt);
    } else if (isReducedFloatingType(x.dtype()) && acc_dt == DType::Undefined) {
        x = x.to(DType::Float32);
    }
    std::vector<int64_t> dims;
    if (dim_opt.has_value()) dims.push_back(*dim_opt);
    else {
        // global reduction over every dimension
        for (int64_t i = 0; i < x.dim(); ++i) dims.push_back(i);
    }
    Tensor total = nansum_cuda2(x, dims, keepdim);
    Tensor valid = isnan_cuda(x).logical_not();
    Tensor count = sum_dim_kernel(valid.to(DType::Float32), dims, keepdim, DType::Float32);
    Tensor quot = total.div(count);
    Tensor zero = count.eq(Scalar(0.0f));
    Tensor result = quot.masked_fill(zero, Scalar(std::numeric_limits<double>::quiet_NaN()));
    return result.to(acc_dt != DType::Undefined ? acc_dt : total.dtype());
}

// The bitwise family lives in BitwiseKernels.cu.

TENSORPLAY_LIBRARY_IMPL(CUDA, OpsKernels) {
    m.impl("rsub.Scalar", rsub_scalar_cuda);
    m.impl("rsub.Tensor", rsub_tensor_cuda);
    m.impl("true_divide.Tensor", true_divide_tensor_cuda);
    m.impl("true_divide.Scalar", true_divide_scalar_cuda);
    m.impl("divide.Tensor", divide_tensor_cuda);
    m.impl("divide.Scalar", divide_scalar_cuda);
    m.impl("remainder.Tensor", remainder_tensor_cuda);
    m.impl("remainder.Scalar", remainder_scalar_cuda);
    m.impl("fmod.Tensor", fmod_tensor_cuda);
    m.impl("fmod.Scalar", fmod_scalar_cuda);
    m.impl("subtract.Tensor", subtract_tensor_cuda);
    m.impl("subtract.Scalar", subtract_scalar_cuda);
    m.impl("multiply.Tensor", multiply_tensor_cuda);
    m.impl("multiply.Scalar", multiply_scalar_cuda);
    m.impl("remainder.Scalar_Tensor", remainder_scalar_tensor_cuda);
    m.impl("div.Tensor_mode", div_mode_tensor_cuda);
    m.impl("div.Scalar_mode", div_mode_scalar_cuda);
    m.impl("divide.Tensor_mode", div_mode_tensor_cuda);
    m.impl("divide.Scalar_mode", div_mode_scalar_cuda);
    m.impl("floor_divide", floor_divide_cuda);
    m.impl("floor_divide.Scalar", floor_divide_scalar_cuda);
    m.impl("negative", negative_cuda);
    m.impl("positive", positive_cuda);
    m.impl("greater", greater_cuda);
    m.impl("greater_equal", greater_equal_cuda);
    m.impl("less", less_cuda);
    m.impl("less_equal", less_equal_cuda);
    m.impl("not_equal", not_equal_cuda);
    m.impl("signbit", signbit_cuda);
    m.impl("logical_not", logical_not_cuda);
    m.impl("logical_and", logical_and_cuda);
    m.impl("logical_or", logical_or_cuda);
    m.impl("logical_xor", logical_xor_cuda);
    m.impl("isfinite", isfinite_cuda);
    m.impl("isinf", isinf_cuda);
    m.impl("isnan", isnan_cuda);
    m.impl("isneginf", isneginf_cuda);
    m.impl("isposinf", isposinf_cuda);
    m.impl("reciprocal", reciprocal_cuda);
    m.impl("sgn", sgn_cuda);
    m.impl("exp2", exp2_cuda);
    m.impl("sinc", sinc_cuda);
    m.impl("deg2rad", deg2rad_cuda);
    m.impl("rad2deg", rad2deg_cuda);
    m.impl("fix", fix_cuda);
    m.impl("erfinv", erfinv_cuda);
    m.impl("logit", logit_cuda);
    m.impl("digamma", digamma_cuda);
    m.impl("i0", i0_cuda);
    m.impl("nan_to_num", nan_to_num_cuda);
    m.impl("xlogy", xlogy_cuda);
    m.impl("logaddexp", logaddexp_cuda);
    m.impl("logaddexp2", logaddexp2_cuda);
    m.impl("copysign.Tensor", copysign_cuda);
    m.impl("copysign.Scalar", copysign_scalar_cuda);
    m.impl("hypot", hypot_cuda);
    m.impl("nextafter", nextafter_cuda);
    m.impl("gcd", gcd_cuda);
    m.impl("lcm", lcm_cuda);
    m.impl("heaviside", heaviside_cuda);
    m.impl("clamp_", clamp__cuda);
    m.impl("clamp_min", clamp_min_scalar_cuda);
    m.impl("clamp_max", clamp_max_scalar_cuda);
    m.impl("clamp_min_", clamp_min__scalar_cuda);
    m.impl("clamp_max_", clamp_max__scalar_cuda);
    m.impl("clamp_min.Scalar", clamp_min_scalar_cuda);
    m.impl("clamp_max.Scalar", clamp_max_scalar_cuda);
    m.impl("clamp_min.Tensor", clamp_min_tensor_cuda);
    m.impl("clamp_max.Tensor", clamp_max_tensor_cuda);
    m.impl("clip", clip_cuda);
    m.impl("selu", selu_cuda);
    m.impl("celu", celu_cuda);
    m.impl("hardshrink", hardshrink_cuda);
    m.impl("hardshrink_backward", hardshrink_backward_cuda);
    m.impl("softshrink", softshrink_cuda);
    m.impl("softshrink_backward", softshrink_backward_cuda);
    m.impl("sigmoid_backward", sigmoid_backward_cuda);
    m.impl("tanh_backward", tanh_backward_cuda);
    m.impl("logit_backward", logit_backward_cuda);
    m.impl("threshold", threshold_cuda);
    m.impl("prelu", prelu_cuda);
    m.impl("_prelu_kernel", _prelu_kernel_cuda);
    m.impl("_prelu_kernel_backward", _prelu_kernel_backward_cuda);
    m.impl("nanmean", nanmean_cuda);
}

} // namespace cuda
} // namespace tensorplay
