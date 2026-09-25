#pragma once

// Pointwise plumbing shared by the operator families: the functor table for
// the math kernels, the launch helper and the grid-stride drivers that back
// the binary, comparison, unary-math and clamp families.  The helpers keep
// internal linkage, matching the file-local scope they had while the families
// shared one translation unit.

#include "Tensor.h"
#include "Complex.h"
#include "CUDAComplex.cuh"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"
#include "CUDALoops.cuh"
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

// Canonical division and subtraction from the arithmetic unit; the
// pointwise families below call them instead of restating the promotion
// tables.
Tensor div_kernel(const Tensor& self, const Tensor& other);
Tensor sub_kernel(const Tensor& self, const Tensor& other,
                   const Scalar& alpha);

namespace {

template <typename Derived>
struct GenFnBase {
    template <typename... Args>
    __host__ __device__ auto operator()(Args... args) const {
        return static_cast<const Derived*>(this)->run(args...);
    }
};

struct HFn1 : GenFnBase<HFn1> {
    __host__ __device__ auto run(double x, double y) const {
 return x / y; 
    }
};

struct HFn2 : GenFnBase<HFn2> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto y) const -> decltype(x) {
                                using T = decltype(x);
                                T r;
                                if constexpr (std::is_integral_v<T>)
                                    r = static_cast<T>(x % y);
                                else  // Half/BFloat16/float/double via fmod
                                    r = static_cast<T>(::fmod(static_cast<double>(x), static_cast<double>(y)));
                                if (r != T(0) && ((r < static_cast<T>(0)) != (y < static_cast<T>(0)))) r = static_cast<T>(r + y);
                                return r;
                            
    }
};

struct HFn3 : GenFnBase<HFn3> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto y) const -> decltype(x) {
                                if constexpr (std::is_integral_v<decltype(x)>)
                                    return static_cast<decltype(x)>(x % y);
                                else
                                    return static_cast<decltype(x)>(::fmod(static_cast<double>(x), static_cast<double>(y)));
                            
    }
};

struct HFn4 : GenFnBase<HFn4> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto y) const -> decltype(x) {
 return x - y; 
    }
};

struct HFn5 : GenFnBase<HFn5> {
    double al;
    __host__ __device__ HFn5(double a_al) : al(a_al) {}
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto y) const {
                                using T = decltype(x);
                                return static_cast<T>(x - y * al);
                            
    }
};

struct HFn6 : GenFnBase<HFn6> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto y) const {
 return x * y; 
    }
};

struct HFn7 : GenFnBase<HFn7> {
    double ov;
    __host__ __device__ HFn7(double a_ov) : ov(a_ov) {}
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const -> decltype(x) {
                                using T = decltype(x);
                                return static_cast<T>(static_cast<double>(x) * ov);
                            
    }
};

struct HFn8 : GenFnBase<HFn8> {
    bool floor_mode;
    __host__ __device__ HFn8(bool a_floor_mode) : floor_mode(a_floor_mode) {}
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto y) const {
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
    
    }
};

struct HFn9 : GenFnBase<HFn9> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const -> decltype(x) {
 return static_cast<decltype(x)>(-x); 
    }
};

struct HFn10 : GenFnBase<HFn10> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto y) const {
 return x > y; 
    }
};

struct HFn11 : GenFnBase<HFn11> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto y) const {
 return x >= y; 
    }
};

struct HFn12 : GenFnBase<HFn12> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto y) const {
 return x < y; 
    }
};

struct HFn13 : GenFnBase<HFn13> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto y) const {
 return x <= y; 
    }
};

struct HFn14 : GenFnBase<HFn14> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto y) const {
 return x != y; 
    }
};

struct HFn15 : GenFnBase<HFn15> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const -> decltype(x) {
        return static_cast<double>(x) < 0.0 ||
               (static_cast<double>(x) == 0.0 && 1.0 / static_cast<double>(x) < 0.0);
    
    }
};

struct HFn16 : GenFnBase<HFn16> {
    __host__ __device__ auto run(bool x) const {
 return !x; 
    }
};

struct HFn17 : GenFnBase<HFn17> {
    __host__ __device__ auto run(bool x, bool y) const {
        return x && y;
    
    }
};

struct HFn18 : GenFnBase<HFn18> {
    __host__ __device__ auto run(bool x, bool y) const {
        return x || y;
    
    }
};

struct HFn19 : GenFnBase<HFn19> {
    __host__ __device__ auto run(bool x, bool y) const {
        return x != y;
    
    }
};

struct HFn20 : GenFnBase<HFn20> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const -> decltype(x) {
        double d = static_cast<double>(x);
        return d == d && d != std::numeric_limits<double>::infinity() &&
               d != -std::numeric_limits<double>::infinity();
    
    }
};

struct HFn21 : GenFnBase<HFn21> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const -> decltype(x) {
        double d = static_cast<double>(x);
        return d == std::numeric_limits<double>::infinity() ||
               d == -std::numeric_limits<double>::infinity();
    
    }
};

struct HFn22 : GenFnBase<HFn22> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const -> decltype(x) {
        double d = static_cast<double>(x);
        return d != d;
    
    }
};

struct HFn23 : GenFnBase<HFn23> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const -> decltype(x) {
        return static_cast<double>(x) == -std::numeric_limits<double>::infinity();
    
    }
};

struct HFn24 : GenFnBase<HFn24> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const -> decltype(x) {
        return static_cast<double>(x) == std::numeric_limits<double>::infinity();
    
    }
};

struct HFn25 : GenFnBase<HFn25> {
    __host__ __device__ auto run(double x) const {
 return 1.0 / x; 
    }
};

struct HFn26 : GenFnBase<HFn26> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const {
                                using T = decltype(x);
                                double d = static_cast<double>(x);
                                if (d != d) return static_cast<T>(x);
                                if (d > 0) return static_cast<T>(1);
                                if (d < 0) return static_cast<T>(-1);
                                return static_cast<T>(0);
                            
    }
};

struct HFn27 : GenFnBase<HFn27> {
    __host__ __device__ auto run(double x) const {
 return ::exp2(x); 
    }
};

struct HFn28 : GenFnBase<HFn28> {
    __host__ __device__ auto run(double x) const {
        double px = M_PI * x;
        return ::fabs(px) < 1e-30 ? 1.0 : ::sin(px) / px;
    
    }
};

struct HFn29 : GenFnBase<HFn29> {
    __host__ __device__ auto run(double x) const {
 return x * (M_PI / 180.0); 
    }
};

struct HFn30 : GenFnBase<HFn30> {
    __host__ __device__ auto run(double x) const {
 return x * (180.0 / M_PI); 
    }
};

struct HFn31 : GenFnBase<HFn31> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const {
                                if constexpr (std::is_floating_point_v<decltype(x)>)
                                    return static_cast<decltype(x)>(::trunc(static_cast<double>(x)));
                                else
                                    return x;
                            
    }
};

struct HFn32 : GenFnBase<HFn32> {
    __host__ __device__ auto run(double x) const {
 return tensorplay::special_math::calc_erfinv(x); 
    }
};

struct HFn33 : GenFnBase<HFn33> {
    double e;
    __host__ __device__ HFn33(double a_e) : e(a_e) {}
    __host__ __device__ auto run(double p) const {
                               if (e >= 0) p = ::fmin(::fmax(p, e), 1.0 - e);
                               return ::log(p / (1.0 - p));
                           
    }
};

struct HFn34 : GenFnBase<HFn34> {
    __host__ __device__ auto run(double v) const {
        if (v <= 0 && v == ::floor(v)) return ::nan("");
        double r = 0;
        while (v < 6.0) { r -= 1.0 / v; v += 1.0; }
        double inv = 1.0 / v, inv2 = inv * inv;
        r += ::log(v) - 0.5 * inv
             - inv2 * (1.0/12.0 - inv2 * (1.0/120.0 - inv2 * (1.0/252.0 - inv2 * (1.0/240.0 - inv2 / 132.0))));
        return r;
    
    }
};

struct HFn35 : GenFnBase<HFn35> {
    __host__ __device__ auto run(double v) const {
        return tensorplay::special_math::modified_bessel_i0_forward(v);
    
    }
};

struct HFn39 : GenFnBase<HFn39> {
    __host__ __device__ auto run(double x, double y) const {
        return tensorplay::special_math::calc_xlogy(x, y);
    
    }
};

struct HFn40 : GenFnBase<HFn40> {
    __host__ __device__ auto run(double x, double y) const {
        double m = ::fmax(x, y);
        if (m == -std::numeric_limits<double>::infinity() || m != m) return m;
        return m + ::log1p(::exp(-::fabs(x - y)));
    
    }
};

struct HFn41 : GenFnBase<HFn41> {
    __host__ __device__ auto run(double x, double y) const {
        double m = ::fmax(x, y);
        if (m == -std::numeric_limits<double>::infinity() || m != m) return m;
        return m + ::log1p(::exp2(-::fabs(x - y))) / M_LN2;
    
    }
};

struct HFn42 : GenFnBase<HFn42> {
    __host__ __device__ auto run(double x, double y) const {
        return ::copysign(x, y);
    
    }
};

struct HFn43 : GenFnBase<HFn43> {
    __host__ __device__ auto run(double x, double y) const {
        return ::hypot(x, y);
    
    }
};

struct HFn44 : GenFnBase<HFn44> {
    __host__ __device__ auto run(double x, double y) const {
        return ::nextafter(x, y);
    
    }
};

struct HFn45 : GenFnBase<HFn45> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto y) const {
                                using T = decltype(x);
                                long long ux = static_cast<long long>(x < static_cast<T>(0) ? -x : x);
                                long long uy = static_cast<long long>(y < static_cast<T>(0) ? -y : y);
                                while (uy) { long long t = ux % uy; ux = uy; uy = t; }
                                return static_cast<T>(ux);
                            
    }
};

struct HFn46 : GenFnBase<HFn46> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto y) const {
                                using T = decltype(x);
                                long long ux = static_cast<long long>(x < static_cast<T>(0) ? -x : x);
                                long long uy = static_cast<long long>(y < static_cast<T>(0) ? -y : y);
                                long long g = ux, t2 = uy;
                                while (t2) { long long t3 = g % t2; g = t2; t2 = t3; }
                                if (g == 0) return static_cast<T>(0);
                                return static_cast<T>(ux / g * uy);
                            
    }
};

struct HFn47 : GenFnBase<HFn47> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto v) const {
                                using T = decltype(x);
                                double xd = static_cast<double>(x);
                                if (xd < 0.0) return static_cast<T>(0);
                                if (xd == 0.0) return static_cast<T>(v);
                                return static_cast<T>(1);
                            
    }
};

struct HFn48 : GenFnBase<HFn48> {
    double lo;
    __host__ __device__ HFn48(double a_lo) : lo(a_lo) {}
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const {
                                using T = decltype(x);
                                return static_cast<double>(x) < lo ? static_cast<T>(lo)
                                                                   : static_cast<T>(x);
                            
    }
};

struct HFn49 : GenFnBase<HFn49> {
    double hi;
    __host__ __device__ HFn49(double a_hi) : hi(a_hi) {}
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const {
                                using T = decltype(x);
                                return static_cast<double>(x) > hi ? static_cast<T>(hi)
                                                                   : static_cast<T>(x);
                            
    }
};

struct HFn50 : GenFnBase<HFn50> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto m) const {
                                using T = decltype(x);
                                return static_cast<double>(m) > static_cast<double>(x)
                                           ? static_cast<T>(m) : static_cast<T>(x);
                            
    }
};

struct HFn51 : GenFnBase<HFn51> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto x, auto m) const {
                                using T = decltype(x);
                                return static_cast<double>(m) < static_cast<double>(x)
                                           ? static_cast<T>(m) : static_cast<T>(x);
                            
    }
};

struct HFn52 : GenFnBase<HFn52> {
    double kAlpha;
    double kScale;
    __host__ __device__ HFn52(double a_kAlpha, double a_kScale) : kAlpha(a_kAlpha), kScale(a_kScale) {}
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const {
                                using T = decltype(x);
                                double v = static_cast<double>(x);
                                return static_cast<T>(v > 0 ? kScale * v
                                                            : kScale * kAlpha * (::exp(v) - 1.0));
    }
};

struct HFn53 : GenFnBase<HFn53> {
    double a;
    __host__ __device__ HFn53(double a_a) : a(a_a) {}
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const {
                                using T = decltype(x);
                                double v = static_cast<double>(x);
                                return static_cast<T>(v > 0 ? v : a * (::exp(v / a) - 1.0));
                            
    }
};

struct HFn54 : GenFnBase<HFn54> {
    double l;
    __host__ __device__ HFn54(double a_l) : l(a_l) {}
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const {
                                using T = decltype(x);
                                const double lt = static_cast<double>(static_cast<T>(l));
                                double v = static_cast<double>(x);
                                return (v >= -lt && v <= lt) ? static_cast<T>(0) : x;
                            
    }
};

struct HFn55 : GenFnBase<HFn55> {
    double l;
    __host__ __device__ HFn55(double a_l) : l(a_l) {}
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const {
                                using T = decltype(x);
                                const double lt = static_cast<double>(static_cast<T>(l));
                                double v = static_cast<double>(x);
                                if (v > lt) return static_cast<T>(v - lt);
                                if (v < -lt) return static_cast<T>(v + lt);
                                return static_cast<T>(v * 0.0);
                            
    }
};

struct HFn56 : GenFnBase<HFn56> {
    double l;
    __host__ __device__ HFn56(double a_l) : l(a_l) {}
    template <typename... TP_A>
    __host__ __device__ auto run(auto g, auto s) const {
                                using T = decltype(g);
                                const double lt = static_cast<double>(static_cast<T>(l));
                                double v = static_cast<double>(s);
                                return (v >= -lt && v <= lt) ? static_cast<T>(0) : g;
                            
    }
};

struct HFn57 : GenFnBase<HFn57> {
    double l;
    __host__ __device__ HFn57(double a_l) : l(a_l) {}
    template <typename... TP_A>
    __host__ __device__ auto run(auto g, auto s) const {
                                using T = decltype(g);
                                const double lt = static_cast<double>(static_cast<T>(l));
                                double v = static_cast<double>(s);
                                return (v >= -lt && v <= lt) ? static_cast<T>(0) : g;
                            
    }
};

struct HFn58 : GenFnBase<HFn58> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto g, auto o) const {
                                using T = decltype(o);
                                return g * o * (static_cast<T>(1) - o);
                            
    }
};

struct HFn59 : GenFnBase<HFn59> {
    template <typename... TP_A>
    __host__ __device__ auto run(auto g, auto o) const {
                                using T = decltype(o);
                                return g * (static_cast<T>(1) - o * o);
                            
    }
};

struct HFn60 : GenFnBase<HFn60> {
    double e;
    __host__ __device__ HFn60(double a_e) : e(a_e) {}
    template <typename... TP_A>
    __host__ __device__ auto run(auto g, auto s) const -> decltype(s) {
                                using T = decltype(s);
                                const T zero = static_cast<T>(0);
                                const T one = static_cast<T>(1);
                                if (e < 0) {
                                    if (s < zero || s > one) return std::numeric_limits<T>::quiet_NaN();
                                    return static_cast<T>(g / (s * (one - s)));
                                }
                                const T lo = static_cast<T>(e);
                                const T hi = one - lo;
                                if (s < lo || s > hi) return zero;
                                return static_cast<T>(g / (s * (one - s)));
    }
};

struct HFn61 : GenFnBase<HFn61> {
    double t;
    double val;
    __host__ __device__ HFn61(double a_t, double a_val) : t(a_t), val(a_val) {}
    template <typename... TP_A>
    __host__ __device__ auto run(auto x) const {
                                using T = decltype(x);
                                return static_cast<double>(x) <= t ? static_cast<T>(val)
                                                                   : static_cast<T>(x);
                            
    }
};


#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

// Canonical division from the arithmetic unit.  The complex branch of
// true_divide reuses it instead of restating the promotion table.




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

template <typename T>
__host__ __device__ bool logical_truth_cuda(const T& value) {
    return static_cast<bool>(value != T(0));
}

template <typename T>
__host__ __device__ bool logical_truth_cuda(const tensorplay::complex<T>& value) {
    return value.real() != T(0) || value.imag() != T(0);
}

template <typename T>
__host__ __device__ T nan_to_num_replace_cuda(
        T value, T nan_replacement, T posinf_replacement,
        T neginf_replacement) {
    return value != value
        ? nan_replacement
        : (value == std::numeric_limits<T>::infinity()
            ? posinf_replacement
            : (value == -std::numeric_limits<T>::infinity()
                ? neginf_replacement
                : value));
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
    Tensor ac = a_in.to(dt).expand(out_shape);
    Tensor bc = b_in.to(dt).expand(out_shape);
    Tensor out = Tensor::empty(out_shape, dt, a_in.device());
    if (out.numel() == 0) return out;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(out)
        .add_const_input(ac)
        .add_const_input(bc)
        .build();
#define TP_BIN(ctype, name_) \
    case DType::name_: \
        gpu_kernel(iter, [op] __host__ __device__(ctype lhs, ctype rhs) -> ctype { \
            return op(lhs, rhs); \
        }); \
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
    Tensor ac = a_in.to(dt).expand(out_shape);
    Tensor bc = b_in.to(dt).expand(out_shape);
    Tensor out = Tensor::empty(out_shape, DType::Bool, a_in.device());
    if (out.numel() == 0) return out;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(out)
        .add_const_input(ac)
        .add_const_input(bc)
        .build();
#define TP_BBIN(ctype, name_) \
    case DType::name_: \
        gpu_kernel(iter, [pred] __host__ __device__(ctype lhs, ctype rhs) -> bool { \
            return pred(lhs, rhs); \
        }); \
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
    Tensor out = Tensor::empty(shape_of(self), DType::Bool, self.device());
    if (self.numel() == 0) return out;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(out)
        .add_const_input(self)
        .build();
#define TP_BU(ctype, name_) \
    case DType::name_: \
        gpu_kernel(iter, [pred] __host__ __device__(ctype value) -> bool { \
            return pred(value); \
        }); \
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
    Tensor ac = a_in.to(dt).expand(out_shape);
    Tensor bc = b_in.to(dt).expand(out_shape);
    Tensor out = Tensor::empty(out_shape, DType::Bool, a_in.device());
    if (out.numel() == 0) return out;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(out)
        .add_const_input(ac)
        .add_const_input(bc)
        .build();
#define TP_LOGICAL_BIN(ctype, name_) \
    case DType::name_: \
        gpu_kernel(iter, [pred] __host__ __device__(ctype lhs, ctype rhs) -> bool { \
            return pred(logical_truth_cuda(lhs), logical_truth_cuda(rhs)); \
        }); \
        break;
    switch (dt) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_LOGICAL_BIN)
        case DType::ComplexFloat:
            gpu_kernel(iter, [pred] __host__ __device__(tensorplay::complex<float> lhs,
                                                tensorplay::complex<float> rhs) -> bool {
                return pred(logical_truth_cuda(lhs), logical_truth_cuda(rhs));
            });
            break;
        case DType::ComplexDouble:
            gpu_kernel(iter, [pred] __host__ __device__(tensorplay::complex<double> lhs,
                                                tensorplay::complex<double> rhs) -> bool {
                return pred(logical_truth_cuda(lhs), logical_truth_cuda(rhs));
            });
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
    Tensor out = Tensor::empty(shape_of(self), DType::Bool, self.device());
    if (out.numel() == 0) return out;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(out)
        .add_const_input(self)
        .build();
#define TP_LOGICAL_UNARY(ctype, name_) \
    case DType::name_: \
        gpu_kernel(iter, [pred] __host__ __device__(ctype value) -> bool { \
            return pred(logical_truth_cuda(value)); \
        }); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_LOGICAL_UNARY)
        case DType::ComplexFloat:
            gpu_kernel(iter, [pred] __host__ __device__(tensorplay::complex<float> value) -> bool {
                return pred(logical_truth_cuda(value));
            });
            break;
        case DType::ComplexDouble:
            gpu_kernel(iter, [pred] __host__ __device__(tensorplay::complex<double> value) -> bool {
                return pred(logical_truth_cuda(value));
            });
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
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    if (self.numel() == 0) return out;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(out)
        .add_const_input(self)
        .build();
#define TP_DU(ctype, name_) \
    case DType::name_: \
        gpu_kernel(iter, [f] __host__ __device__(ctype value) -> ctype { \
            return f(value); \
        }); \
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
    Tensor w = (in == compute_dt) ? self : self.to(compute_dt);
    Tensor t = Tensor::empty(shape_of(w), compute_dt, w.device());
    if (w.numel() > 0) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(t)
            .add_const_input(w)
            .build();
        if (compute_dt == DType::Float64) {
            gpu_kernel(iter, [f] __host__ __device__(double x) -> double {
                return f(x);
            });
        } else {
            gpu_kernel(iter, [f] __host__ __device__(float x) -> float {
                return static_cast<float>(f(static_cast<double>(x)));
            });
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
    DType compute_dt = (dt == DType::Float64) ? DType::Float64 : DType::Float32;
    const std::vector<int64_t> out_shape = broadcast_shapes(
        shape_of(a_in), shape_of(b_in));
    Tensor ac = a_in.to(compute_dt).expand(out_shape);
    Tensor bc = b_in.to(compute_dt).expand(out_shape);
    Tensor out = Tensor::empty(out_shape, compute_dt, a_in.device());
    if (out.numel() == 0) return (dt == compute_dt) ? out : out.to(dt);
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(out)
        .add_const_input(ac)
        .add_const_input(bc)
        .build();
    if (compute_dt == DType::Float64) {
        gpu_kernel(iter, [f] __host__ __device__(double x, double y) -> double {
            return f(x, y);
        });
    } else {
        gpu_kernel(iter, [f] __host__ __device__(float x, float y) -> float {
            return static_cast<float>(f(static_cast<double>(x),
                                        static_cast<double>(y)));
        });
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


// ===========================================================================
// Arithmetic
// ===========================================================================




enum class DivRounding { kTrue, kTrunc, kFloor };

DivRounding parse_div_rounding(const std::optional<std::string>& mode) {
    if (!mode.has_value()) return DivRounding::kTrue;
    if (*mode == "trunc") return DivRounding::kTrunc;
    if (*mode == "floor") return DivRounding::kFloor;
    TP_THROW(RuntimeError,
             std::string("div expected rounding_mode to be one of None, 'trunc' "
                         "or 'floor' but found '") + *mode + "'");
}


// Arithmetic domain of the PReLU evaluation: single precision stays single
// precision, while half formats widen to float so the slope product rounds
// once.

template <typename T> struct PReluMath { using type = double; };

template <> struct PReluMath<float> { using type = float; };

template <> struct PReluMath<double> { using type = double; };

template <> struct PReluMath<Half> { using type = float; };

template <> struct PReluMath<BFloat16> { using type = float; };

Tensor nansum_cuda2(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim);

Tensor sum_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, DType dtype);



}  // namespace

}  // namespace cuda
}  // namespace tensorplay
