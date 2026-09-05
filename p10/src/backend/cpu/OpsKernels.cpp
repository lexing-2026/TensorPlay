// Elementwise arithmetic, comparison, math, clamp, and activation CPU kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Utils.h"
#include "TensorIteratorOps.h"
#include "Exception.h"
#include "Parallel.h"
#include "TypePromotion.h"
#include "SpecialMath.h"
#include "cpu/ComplexUnary.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <complex>
#include <limits>
#include <cstring>
#include <utility>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include <type_traits>
#include <optional>
#include <string>

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

namespace {

inline DType scalar_promote(DType t, const Scalar& s) {
    // Weak scalar participation: scalars only promote the tensor dtype when
    // they carry a floating type of their own.
    if (!isFloatingType(s.dtype())) return t;
    if (isFloatingType(t)) return t;
    return DType::Float32;
}

// ---------------------------------------------------------------------------
// Elementwise helpers
// ---------------------------------------------------------------------------

// Broadcast both inputs to a common promoted dtype; op returns that dtype.
// kArith selects the complex-capable TensorIterator applier for pure
// arithmetic functors (rsub/subtract/multiply); ordering/fmod-style callers
// must keep the default because those operations have no complex ordering
// semantics.
template <bool kArith = false, typename Op>
Tensor binary_same_kernel(const Tensor& a_in, const Tensor& b_in, Op op, const char* name) {
    TP_CHECK(a_in.device() == b_in.device(),
             "Expected all tensors to be on the same device, but found ",
             a_in.device().toString(), " and ", b_in.device().toString(),
             "!");
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    // Cast to the promoted dtype but do NOT materialize broadcasts:
    // TensorIterator handles expansion + strided access natively.
    Tensor ac = (a_in.dtype() == dt ? a_in : a_in.to(dt));
    Tensor bc = (b_in.dtype() == dt ? b_in : b_in.to(dt));
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    Tensor out = Tensor::empty(out_shape, dt, a_in.device());
    if constexpr (kArith) {
        ti_apply_arith(out, ac, bc, op);
    } else {
        ti_apply_binary(out, ac, bc, op);
    }
    (void)name;
    return out;
}

// Binary op whose inputs promote to a FLOATING dtype first (ints -> Float32).
template <typename F>  // F: (double,double) -> double
Tensor binary_float_kernel(const Tensor& a_in, const Tensor& b_in, F f, const char* name) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    if (!isFloatingType(dt)) dt = DType::Float32;
    // Reduced-width inputs are evaluated in Float32 and narrowed once at the
    // end; the loops below only ever address float or double buffers.
    DType compute_dt = (dt == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor ac = a_in.to(compute_dt).expand(out_shape).contiguous();
    Tensor bc = b_in.to(compute_dt).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, compute_dt, a_in.device());
    int64_t n = out.numel();
    if (compute_dt == DType::Float64) {
        const double* ap = ac.data_ptr<double>();
        const double* bp = bc.data_ptr<double>();
        double* dp = out.data_ptr<double>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) dp[i] = f(ap[i], bp[i]);
        });
    } else {
        const float* ap = ac.data_ptr<float>();
        const float* bp = bc.data_ptr<float>();
        float* dp = out.data_ptr<float>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) dp[i] = static_cast<float>(f(ap[i], bp[i]));
        });
    }
    return (dt == compute_dt) ? out : out.to(dt);
}

template <typename Pred>
Tensor binary_bool_kernel(const Tensor& a_in, const Tensor& b_in, Pred pred, const char* name) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    Tensor ac = (a_in.dtype() == dt ? a_in : a_in.to(dt)).expand(out_shape).contiguous();
    Tensor bc = (b_in.dtype() == dt ? b_in : b_in.to(dt)).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, DType::Bool, a_in.device());
    int64_t n = out.numel();
#define TP_BBIN_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* ap = ac.data_ptr<ctype>(); \
        const ctype* bp = bc.data_ptr<ctype>(); \
        bool* dp = out.data_ptr<bool>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) dp[i] = pred(ap[i], bp[i]); \
        }); \
        break; \
    }
    switch (dt) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BBIN_CASE)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BBIN_CASE
    return out;
}

template <typename Pred>
Tensor bool_unary_kernel(const Tensor& self, Pred pred, const char* name) {
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), DType::Bool, self.device());
    Tensor sc = self.contiguous();
    int64_t n = self.numel();
#define TP_BU_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        bool* dp = out.data_ptr<bool>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) dp[i] = pred(sp[i]); \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BU_CASE)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BU_CASE
    return out;
}

template <typename T>
inline bool logical_truth_cpu(const T& value) {
    return static_cast<bool>(value);
}

template <typename T>
inline bool logical_truth_cpu(const std::complex<T>& value) {
    return static_cast<bool>(value.real()) || static_cast<bool>(value.imag());
}

template <typename Pred>
Tensor logical_binary_kernel(const Tensor& a_in, const Tensor& b_in, Pred pred,
                             const char* name) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    Tensor ac = (a_in.dtype() == dt ? a_in : a_in.to(dt)).expand(out_shape).contiguous();
    Tensor bc = (b_in.dtype() == dt ? b_in : b_in.to(dt)).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, DType::Bool, a_in.device());
    const int64_t n = out.numel();
#define TP_LOGICAL_BIN_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* ap = ac.data_ptr<ctype>(); \
        const ctype* bp = bc.data_ptr<ctype>(); \
        bool* dp = out.data_ptr<bool>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) \
                dp[i] = pred(logical_truth_cpu(ap[i]), logical_truth_cpu(bp[i])); \
        }); \
        break; \
    }
    switch (dt) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_LOGICAL_BIN_CASE)
#define TP_LOGICAL_BIN_COMPLEX_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* ap = reinterpret_cast<const ctype*>(ac.data_ptr()); \
        const ctype* bp = reinterpret_cast<const ctype*>(bc.data_ptr()); \
        bool* dp = out.data_ptr<bool>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) \
                dp[i] = pred(logical_truth_cpu(ap[i]), logical_truth_cpu(bp[i])); \
        }); \
        break; \
    }
        TP_LOGICAL_BIN_COMPLEX_CASE(std::complex<Half>, ComplexHalf)
        TP_LOGICAL_BIN_COMPLEX_CASE(std::complex<float>, ComplexFloat)
        TP_LOGICAL_BIN_COMPLEX_CASE(std::complex<double>, ComplexDouble)
        TP_LOGICAL_BIN_COMPLEX_CASE(std::complex<BFloat16>, BComplex32)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_LOGICAL_BIN_COMPLEX_CASE
#undef TP_LOGICAL_BIN_CASE
    return out;
}

template <typename Pred>
Tensor logical_unary_kernel(const Tensor& self, Pred pred, const char* name) {
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                               DType::Bool, self.device());
    Tensor sc = self.contiguous();
    const int64_t n = self.numel();
#define TP_LOGICAL_UNARY_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        bool* dp = out.data_ptr<bool>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) \
                dp[i] = pred(logical_truth_cpu(sp[i])); \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_LOGICAL_UNARY_CASE)
#define TP_LOGICAL_UNARY_COMPLEX_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = reinterpret_cast<const ctype*>(sc.data_ptr()); \
        bool* dp = out.data_ptr<bool>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) \
                dp[i] = pred(logical_truth_cpu(sp[i])); \
        }); \
        break; \
    }
        TP_LOGICAL_UNARY_COMPLEX_CASE(std::complex<Half>, ComplexHalf)
        TP_LOGICAL_UNARY_COMPLEX_CASE(std::complex<float>, ComplexFloat)
        TP_LOGICAL_UNARY_COMPLEX_CASE(std::complex<double>, ComplexDouble)
        TP_LOGICAL_UNARY_COMPLEX_CASE(std::complex<BFloat16>, BComplex32)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_LOGICAL_UNARY_COMPLEX_CASE
#undef TP_LOGICAL_UNARY_CASE
    return out;
}

// Dtype-preserving unary (used by sgn, fix, negative, ...).
template <typename F>
Tensor dtype_unary_kernel(const Tensor& self, F f, const char* name) {
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    Tensor sc = self.contiguous();
    int64_t n = self.numel();
#define TP_DU_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) dp[i] = static_cast<ctype>(f(sp[i])); \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_DU_CASE)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_DU_CASE
    return out;
}

// integral inputs yield Float32; Half/BFloat16 compute in float and keep
// their dtype; Float32/Float64 preserved.
template <typename F>
Tensor float_math_kernel(const Tensor& self, F f, const char* name) {
    DType in = self.dtype();
    DType out_dt = isFloatingType(in) ? in : DType::Float32;
    DType compute_dt = (in == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor w = (self.dtype() == compute_dt) ? self.contiguous()
                                            : self.to(compute_dt).contiguous();
    Tensor t = Tensor::empty(static_cast<std::vector<int64_t>>(w.shape()), compute_dt, w.device());
    int64_t n = w.numel();
    if (compute_dt == DType::Float64) {
        const double* sp = w.data_ptr<double>();
        double* dp = t.data_ptr<double>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) dp[i] = f(sp[i]);
        });
    } else {
        const float* sp = w.data_ptr<float>();
        float* dp = t.data_ptr<float>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) dp[i] = static_cast<float>(f(sp[i]));
        });
    }
    return (out_dt == compute_dt) ? t : t.to(out_dt);
}

} // anonymous namespace

// ===========================================================================
// Arithmetic
// ===========================================================================

Tensor rsub_scalar_cpu(const Tensor& self, Scalar other, Scalar alpha) {
    // Reversed subtraction: other - alpha * self, under weak scalar promotion.
    // alpha scales the subtrahend, which is self here, not other.
    DType dt = scalar_promote(self.dtype(), other);
    Tensor sc = self.to(dt).contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()), dt, self.device());
    double o = other.toDouble(), al = alpha.toDouble();
    int64_t n = sc.numel();
#define TP_RS_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        ctype ov = static_cast<ctype>(o), av = static_cast<ctype>(al); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t i = b; i < e; ++i) dp[i] = static_cast<ctype>(ov - av * sp[i]); \
        }); \
        break; \
    }
    switch (dt) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_RS_CASE)
        default: TP_THROW(TypeError, "rsub: unsupported dtype");
    }
#undef TP_RS_CASE
    return out;
}

Tensor rsub_tensor_cpu(const Tensor& self, const Tensor& other, Scalar alpha) {
    // other - alpha * self: the same arithmetic as sub with the operands
    // exchanged, so alpha still scales the subtrahend.
    return binary_same_kernel<true>(self, other,
        [alpha](auto s, auto o) {
            using T = decltype(o);
            if constexpr (is_complex_type_v<T> || std::is_floating_point_v<T>) {
                return o - s * alpha.to<T>();
            } else {
                return static_cast<T>(o - s * alpha.to<double>());
            }
        }, "rsub");
}

static Tensor true_divide_core(const Tensor& a, const Tensor& b) {
    // Integral inputs promote to the default float type.
    // A complex operand keeps its own width instead -- the float loop only
    // addresses real buffers, so it would drop the imaginary halves.
    if (isComplexType(a.dtype()) || isComplexType(b.dtype())) {
        return binary_same_kernel<true>(a, b,
            [](auto x, auto y) { return x / y; }, "true_divide");
    }
    return binary_float_kernel(a, b, [](double x, double y) { return x / y; }, "true_divide");
}
Tensor true_divide_tensor_cpu(const Tensor& self, const Tensor& other) { return true_divide_core(self, other); }
Tensor true_divide_scalar_cpu(const Tensor& self, Scalar other) {
    // A Float32 stand-in would widen Half/BFloat16 inputs; the weak-scalar
    // rule keeps the tensor dtype unless the scalar itself is floating.
    return true_divide_core(
        self, Tensor::full({}, other, scalar_promote(self.dtype(), other), self.device()));
}
Tensor divide_tensor_cpu(const Tensor& self, const Tensor& other) { return true_divide_core(self, other); }
Tensor divide_scalar_cpu(const Tensor& self, Scalar other) { return true_divide_scalar_cpu(self, other); }

Tensor remainder_tensor_cpu(const Tensor& self, const Tensor& other) {
    // Python modulo: sign follows the divisor
    return binary_same_kernel(self, other, [](auto x, auto y) {
        using T = decltype(x);
        if constexpr (std::is_integral_v<T>) {
            auto r = x % y;
            if (r != T(0) && ((r < 0) != (y < 0))) r = static_cast<T>(r + y);
            return static_cast<T>(r);
        } else {
            // Float/Half/BFloat16 route through fmod on double opmath.
            double xf = static_cast<double>(x), yf = static_cast<double>(y);
            double r = std::fmod(xf, yf);
            if (r != 0.0 && ((r < 0.0) != (yf < 0.0))) r += yf;
            return static_cast<T>(r);
        }
    }, "remainder");
}
Tensor remainder_scalar_cpu(const Tensor& self, Scalar other) {
    // Forcing the scalar into self's dtype would truncate a float divisor
    // against an integral tensor; the pair promotes first.
    const DType dt = scalar_promote(self.dtype(), other);
    return remainder_tensor_cpu(self.to(dt), Tensor::full({}, other, dt, self.device()));
}

Tensor remainder_scalar_tensor_cpu(Scalar self, const Tensor& other) {
    const DType dt = scalar_promote(other.dtype(), self);
    return remainder_tensor_cpu(Tensor::full({}, self, dt, other.device()), other.to(dt));
}

Tensor fmod_tensor_cpu(const Tensor& self, const Tensor& other) {
    // C fmod: sign follows the dividend
    return binary_same_kernel(self, other, [](auto x, auto y) -> decltype(x) {
        if constexpr (std::is_integral_v<decltype(x)>)
            return static_cast<decltype(x)>(x % y);
        else
            // Covers float/double/Half/BFloat16 via double opmath.
            return static_cast<decltype(x)>(std::fmod(static_cast<double>(x),
                                                      static_cast<double>(y)));
    }, "fmod");
}
Tensor fmod_scalar_cpu(const Tensor& self, Scalar other) {
    const DType dt = scalar_promote(self.dtype(), other);
    return fmod_tensor_cpu(self.to(dt), Tensor::full({}, other, dt, self.device()));
}

Tensor subtract_tensor_cpu(const Tensor& self, const Tensor& other, Scalar alpha) {
    // alpha == 1 is by far the common call, and the scaled form costs a
    // Scalar conversion per element, so it keeps its own loop.
    if (!alpha.isComplex() && alpha.toDouble() == 1.0) {
        return binary_same_kernel<true>(self, other,
            [](auto x, auto y) { return x - y; }, "subtract");
    }
    return binary_same_kernel<true>(self, other,
        [alpha](auto x, auto y) {
            using T = decltype(x);
            if constexpr (is_complex_type_v<T> || std::is_floating_point_v<T>) {
                return x - y * alpha.to<T>();
            } else {
                return static_cast<T>(x - y * alpha.to<double>());
            }
        }, "subtract");
}
Tensor subtract_scalar_cpu(const Tensor& self, Scalar other, Scalar alpha) {
    DType dt = scalar_promote(self.dtype(), other);
    return subtract_tensor_cpu(self.to(dt),
                               Tensor::full({}, other, dt, self.device()), alpha);
}
Tensor multiply_tensor_cpu(const Tensor& self, const Tensor& other) {
    return binary_same_kernel<true>(self, other, [](auto x, auto y) { return x * y; }, "multiply");
}
Tensor multiply_scalar_cpu(const Tensor& self, Scalar other) {
    if (isComplexType(self.dtype()) || other.isComplex()) {
        DType dt;
        if (isComplexType(self.dtype())) {
            dt = other.isComplex() ? promoteTypes(self.dtype(), other.dtype())
                                   : self.dtype();
        } else {
            // weak-scalar rule: int tensors go to complex64; float tensors
            // widen through their own complex width
            const DType complex_dtype = isFloatingType(self.dtype())
                ? toComplexType(self.dtype())
                : DType::ComplexFloat;
            if (complex_dtype == DType::Undefined) {
                TP_THROW(TypeError, "Cannot promote ", toString(self.dtype()),
                         " with a complex dtype");
            }
            dt = promoteTypes(complex_dtype, other.dtype());
        }
        return complex_unary_op_kernel(self.to(dt), [other](auto x) {
            return x * other.to<std::decay_t<decltype(x)>>();
        });
    }
    return dtype_unary_kernel(self, [other](auto x) {
        return static_cast<decltype(x)>(x * other.to<double>());
    }, "multiply");
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

// The hardware quotient truncates toward zero, so a remainder whose sign
// disagrees with the divisor sits one step above the floor.
template <typename T>
inline T int_floor_div(T x, T y) {
    T q = static_cast<T>(x / y);
    T r = static_cast<T>(x - q * y);
    if (r != T(0) && ((r < T(0)) != (y < T(0)))) q = static_cast<T>(q - T(1));
    return q;
}

Tensor div_rounded_core(const Tensor& a, const Tensor& b, DivRounding rounding) {
    if (rounding == DivRounding::kTrue) return true_divide_core(a, b);
    // Rounded division stays in the input dtype: an integral pair must come
    // back integral, which the float promotion of true division loses.
    const bool floor_mode = (rounding == DivRounding::kFloor);
    return binary_same_kernel(a, b, [floor_mode](auto x, auto y) -> decltype(x) {
        using T = decltype(x);
        if constexpr (std::is_integral_v<T>) {
            if (y == T(0)) TP_THROW(RuntimeError, "ZeroDivisionError");
            return floor_mode ? int_floor_div<T>(x, y) : static_cast<T>(x / y);
        } else {
            // Half/BFloat16 round through Float32, the width their arithmetic
            // is defined at; float and double keep their own.
            using C = std::conditional_t<std::is_same_v<T, double>, double, float>;
            const C q = static_cast<C>(x) / static_cast<C>(y);
            return static_cast<T>(floor_mode ? std::floor(q) : std::trunc(q));
        }
    }, "div");
}

Tensor div_rounded_scalar(const Tensor& self, Scalar other, DivRounding rounding) {
    if (rounding == DivRounding::kTrue) return true_divide_scalar_cpu(self, other);
    const DType dt = scalar_promote(self.dtype(), other);
    return div_rounded_core(self.to(dt), Tensor::full({}, other, dt, self.device()),
                            rounding);
}

}  // namespace

Tensor div_mode_tensor_cpu(const Tensor& self, const Tensor& other,
                           std::optional<std::string> rounding_mode) {
    return div_rounded_core(self, other, parse_div_rounding(rounding_mode));
}
Tensor div_mode_scalar_cpu(const Tensor& self, Scalar other,
                           std::optional<std::string> rounding_mode) {
    return div_rounded_scalar(self, other, parse_div_rounding(rounding_mode));
}
Tensor floor_divide_cpu(const Tensor& self, const Tensor& other) {
    return div_rounded_core(self, other, DivRounding::kFloor);
}
Tensor floor_divide_scalar_cpu(const Tensor& self, Scalar other) {
    return div_rounded_scalar(self, other, DivRounding::kFloor);
}

Tensor negative_cpu(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        return complex_unary_op_kernel(self, [](auto x) { return -x; });
    }
    return dtype_unary_kernel(self, [](auto x) { return static_cast<decltype(x)>(-x); }, "negative");
}
Tensor positive_cpu(const Tensor& self) { return self.clone(); }

// ===========================================================================
// Comparisons / logic
// ===========================================================================

Tensor greater_cpu(const Tensor& a, const Tensor& b) {
    return binary_bool_kernel(a, b, [](auto x, auto y) { return x > y; }, "greater");
}
Tensor greater_equal_cpu(const Tensor& a, const Tensor& b) {
    return binary_bool_kernel(a, b, [](auto x, auto y) { return x >= y; }, "greater_equal");
}
Tensor less_cpu(const Tensor& a, const Tensor& b) {
    return binary_bool_kernel(a, b, [](auto x, auto y) { return x < y; }, "less");
}
Tensor less_equal_cpu(const Tensor& a, const Tensor& b) {
    return binary_bool_kernel(a, b, [](auto x, auto y) { return x <= y; }, "less_equal");
}
Tensor not_equal_cpu(const Tensor& a, const Tensor& b) {
    return binary_bool_kernel(a, b, [](auto x, auto y) { return x != y; }, "not_equal");
}
Tensor signbit_cpu(const Tensor& self) {
    return bool_unary_kernel(self, [](auto x) {
        return static_cast<double>(x) < 0.0;
    }, "signbit");
}
Tensor logical_not_cpu(const Tensor& self) {
    return logical_unary_kernel(self, [](bool x) { return !x; }, "logical_not");
}
Tensor logical_and_cpu(const Tensor& a, const Tensor& b) {
    return logical_binary_kernel(a, b, [](bool x, bool y) { return x && y; }, "logical_and");
}
Tensor logical_or_cpu(const Tensor& a, const Tensor& b) {
    return logical_binary_kernel(a, b, [](bool x, bool y) { return x || y; }, "logical_or");
}
Tensor logical_xor_cpu(const Tensor& a, const Tensor& b) {
    return logical_binary_kernel(a, b, [](bool x, bool y) { return x != y; }, "logical_xor");
}
Tensor isfinite_cpu(const Tensor& self) {
    return bool_unary_kernel(self, [](auto x) {
        using T = decltype(x);
        if constexpr (std::is_floating_point_v<T>)
            return std::isfinite(static_cast<double>(x));
        else return true;
    }, "isfinite");
}
Tensor isinf_cpu(const Tensor& self) {
    // Integral tensors never take infinite values
    return bool_unary_kernel(self, [](auto x) {
        using T = decltype(x);
        if constexpr (std::is_floating_point_v<T>)
            return std::isinf(static_cast<double>(x));
        else return false;
    }, "isinf");
}
Tensor isnan_cpu(const Tensor& self) {
    // A value is not equal to itself only when it is not a number
    return bool_unary_kernel(self, [](auto x) {
        return static_cast<double>(x) != static_cast<double>(x);
    }, "isnan");
}
Tensor isneginf_cpu(const Tensor& self) {
    return bool_unary_kernel(self, [](auto x) {
        return static_cast<double>(x) == -std::numeric_limits<double>::infinity();
    }, "isneginf");
}
Tensor isposinf_cpu(const Tensor& self) {
    return bool_unary_kernel(self, [](auto x) {
        return static_cast<double>(x) == std::numeric_limits<double>::infinity();
    }, "isposinf");
}

// ===========================================================================
// Math functions
// ===========================================================================

Tensor reciprocal_cpu(const Tensor& self) {
    // the old float_math_kernel path silently dropped the imaginary part.
    if (isComplexType(self.dtype())) {
        return complex_unary_op_kernel(self, [](auto z) {
            using T = decltype(z);
            return static_cast<T>(1) / z;
        });
    }
    return float_math_kernel(self, [](double x) { return 1.0 / x; }, "reciprocal");
}
Tensor sgn_cpu(const Tensor& self) {
    return dtype_unary_kernel(self, [](auto x) -> decltype(x) {
        using T = decltype(x);
        double d = static_cast<double>(x);
        if (d != d) return static_cast<T>(x);           // NaN passthrough
        if (d > 0) return static_cast<T>(1);
        if (d < 0) return static_cast<T>(-1);
        return static_cast<T>(0);
    }, "sgn");
}
Tensor exp2_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return std::exp2(x); }, "exp2");
}
Tensor sinc_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) {
        double px = M_PI * x;
        return std::fabs(px) < 1e-30 ? 1.0 : std::sin(px) / px;
    }, "sinc");
}
Tensor deg2rad_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return x * (M_PI / 180.0); }, "deg2rad");
}
Tensor rad2deg_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return x * (180.0 / M_PI); }, "rad2deg");
}
Tensor fix_cpu(const Tensor& self) {
    return dtype_unary_kernel(self, [](auto x) -> decltype(x) {
        if constexpr (std::is_floating_point_v<decltype(x)>) return std::trunc(x);
        else return x;
    }, "fix");
}
Tensor erfinv_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) {
        return special_math::calc_erfinv(x);
    }, "erfinv");
}
Tensor logit_cpu(const Tensor& self, std::optional<Scalar> eps) {
    double e = eps.has_value() ? eps->toDouble() : -1.0;
    return float_math_kernel(self, [e](double p) {
        if (e >= 0) p = std::min(std::max(p, e), 1.0 - e);
        return std::log(p / (1.0 - p));
    }, "logit");
}
Tensor digamma_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double v) {
        if (v <= 0 && v == std::floor(v)) return std::numeric_limits<double>::quiet_NaN();
        double r = 0;
        while (v < 6.0) { r -= 1.0 / v; v += 1.0; }
        double inv = 1.0 / v, inv2 = inv * inv;
        r += std::log(v) - 0.5 * inv
             - inv2 * (1.0/12.0 - inv2 * (1.0/120.0 - inv2 * (1.0/252.0 - inv2 * (1.0/240.0 - inv2 / 132.0))));
        return r;
    }, "digamma");
}
Tensor i0_cpu(const Tensor& self) {
    // Modified Bessel I0.  The Chebyshev expansion holds across the whole
    // range; the ((|x|/2)^k / k!)^2 series it replaces needs more terms than
    // any fixed cap allows once |x| passes ~50.
    return float_math_kernel(self, [](double v) {
        return tensorplay::special_math::modified_bessel_i0_forward(v);
    }, "i0");
}
Tensor nan_to_num_cpu(const Tensor& self, Scalar nan,
                      std::optional<Scalar> posinf, std::optional<Scalar> neginf) {
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    double nan_v = nan.toDouble();
    bool has_pos = posinf.has_value(), has_neg = neginf.has_value();
    double pos_v = has_pos ? posinf->toDouble() : std::numeric_limits<double>::infinity();
    double neg_v = has_neg ? neginf->toDouble() : -std::numeric_limits<double>::infinity();
#define TP_NTN_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        ctype pv, nv; \
        if (has_pos) pv = static_cast<ctype>(pos_v); \
        else pv = std::numeric_limits<ctype>::max(); \
        if (has_neg) nv = static_cast<ctype>(neg_v); \
        else nv = std::numeric_limits<ctype>::lowest(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t i = b; i < e; ++i) { \
                ctype v = sp[i]; \
                if constexpr (std::is_floating_point_v<ctype>) { \
                    double dv = static_cast<double>(v); \
                    if (dv != dv) v = static_cast<ctype>(nan_v); \
                    else if (v == std::numeric_limits<ctype>::infinity()) v = pv; \
                    else if (v == -std::numeric_limits<ctype>::infinity()) v = nv; \
                } \
                dp[i] = v; \
            } \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_NTN_CASE)
        default: TP_THROW(TypeError, "nan_to_num: unsupported dtype");
    }
#undef TP_NTN_CASE
    return out;
}

Tensor xlogy_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, [](double x, double y) {
        return tensorplay::special_math::calc_xlogy(x, y);
    }, "xlogy");
}
Tensor logaddexp_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, [](double x, double y) {
        double m = std::max(x, y);
        if (m == -std::numeric_limits<double>::infinity()) return m;
        if (m != m) return m;  // NaN propagates
        return m + std::log1p(std::exp(-std::fabs(x - y)));
    }, "logaddexp");
}
Tensor logaddexp2_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, [](double x, double y) {
        double m = std::max(x, y);
        if (m == -std::numeric_limits<double>::infinity()) return m;
        if (m != m) return m;
        return m + std::log1p(std::exp2(-std::fabs(x - y))) / M_LN2;
    }, "logaddexp2");
}
Tensor copysign_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, [](double x, double y) {
        return std::copysign(x, y);
    }, "copysign");
}
Tensor copysign_scalar_cpu(const Tensor& self, Scalar other) {
    // The sign comes from the scalar alone, so the divisor width never
    // participates in promotion -- Float32 carries every sign bit exactly.
    return copysign_cpu(self, Tensor::full({}, other, DType::Float32, self.device()));
}
Tensor hypot_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, [](double x, double y) {
        return std::hypot(x, y);
    }, "hypot");
}

Tensor atan2_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, [](double x, double y) {
        return std::atan2(x, y);
    }, "atan2");
}
Tensor nextafter_cpu(const Tensor& a, const Tensor& b) {
    // The step must happen in the element dtype: a double-precision step from
    // a Float32 value rounds back to the original number when narrowed.
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a.shape()),
        static_cast<std::vector<int64_t>>(b.shape()));
    DType dt = promoteTypes(a.dtype(), b.dtype());
    if (!isFloatingType(dt)) dt = DType::Float32;
    Tensor ac = a.to(dt).expand(out_shape).contiguous();
    Tensor bc = b.to(dt).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, dt, a.device());
    int64_t n = out.numel();
    if (dt == DType::Float64) {
        const double* ap = ac.data_ptr<double>();
        const double* bp = bc.data_ptr<double>();
        double* dp = out.data_ptr<double>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) dp[i] = std::nextafter(ap[i], bp[i]);
        });
    } else {
        const float* ap = ac.data_ptr<float>();
        const float* bp = bc.data_ptr<float>();
        float* dp = out.data_ptr<float>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) dp[i] = std::nextafter(ap[i], bp[i]);
        });
    }
    return out;
}
Tensor gcd_cpu(const Tensor& a, const Tensor& b) {
    DType dt = promoteTypes(a.dtype(), b.dtype());
    if (isFloatingType(dt)) TP_THROW(TypeError, "gcd only supports integral tensors");
    return binary_same_kernel(a, b, [](auto x, auto y) -> decltype(x) {
        using T = decltype(x);
        long long ux = static_cast<long long>(x < T(0) ? -x : x);
        long long uy = static_cast<long long>(y < T(0) ? -y : y);
        while (uy) { long long t = ux % uy; ux = uy; uy = t; }
        return static_cast<T>(ux);
    }, "gcd");
}
Tensor lcm_cpu(const Tensor& a, const Tensor& b) {
    DType dt = promoteTypes(a.dtype(), b.dtype());
    if (isFloatingType(dt)) TP_THROW(TypeError, "lcm only supports integral tensors");
    return binary_same_kernel(a, b, [](auto x, auto y) -> decltype(x) {
        using T = decltype(x);
        long long ux = static_cast<long long>(static_cast<float>(x) < 0.0f ? -x : x);
        long long uy = static_cast<long long>(static_cast<float>(y) < 0.0f ? -y : y);
        long long g = ux, t2 = uy;
        while (t2) { long long t3 = g % t2; g = t2; t2 = t3; }
        if (g == 0) return static_cast<T>(0);
        return static_cast<T>(ux / g * uy);
    }, "lcm");
}
Tensor heaviside_cpu(const Tensor& a, const Tensor& values) {
    return binary_same_kernel(a, values, [](auto x, auto v) -> decltype(x) {
        using T = decltype(x);
        double xd = static_cast<double>(x);
        if (xd < 0.0) return static_cast<T>(0);
        if (xd == 0.0) return static_cast<T>(v);
        return static_cast<T>(1);
    }, "heaviside");
}

// ===========================================================================
// Clamp family
// ===========================================================================

namespace clamp_row {
#if defined(__x86_64__)
inline bool avx512_ok() {
    static const bool ok = __builtin_cpu_supports("avx512f") != 0 &&
                           __builtin_cpu_supports("avx512vl") != 0 &&
                           __builtin_cpu_supports("avx512dq") != 0;
    return ok;
}

// NaN propagation: the finite bound is the first source of max/min so a NaN
// lane (second source) flows through untouched, matching the scalar ternary.
__attribute__((target("avx512f")))
inline void f32_512(const float* in, float* out, int64_t n, float lo, float hi) {
    const __m512 vlo = _mm512_set1_ps(lo), vhi = _mm512_set1_ps(hi);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(in + i);
        v = _mm512_min_ps(vhi, _mm512_max_ps(vlo, v));
        _mm512_storeu_ps(out + i, v);
    }
    for (; i < n; ++i) {
        float v = in[i];
        v = v < lo ? lo : v;
        v = v > hi ? hi : v;
        out[i] = v;
    }
}

__attribute__((target("avx512f")))
inline void f64_512(const double* in, double* out, int64_t n, double lo, double hi) {
    const __m512d vlo = _mm512_set1_pd(lo), vhi = _mm512_set1_pd(hi);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_loadu_pd(in + i);
        v = _mm512_min_pd(vhi, _mm512_max_pd(vlo, v));
        _mm512_storeu_pd(out + i, v);
    }
    for (; i < n; ++i) {
        double v = in[i];
        v = v < lo ? lo : v;
        v = v > hi ? hi : v;
        out[i] = v;
    }
}
#endif
}  // namespace clamp_row

namespace {
// Shared contiguous implementation: one streaming pass, no intermediate
// tensor, bounds applied together.  NaN input stays NaN (comparisons are
// false), matching the per-bound ternary kernels below.
template <typename T>
Tensor clamp_range_contig(const Tensor& self, T lo, T hi, bool has_lo, bool has_hi) {
    Tensor self_c = self.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    const T* in = self_c.data_ptr<T>();
    T* out = result.data_ptr<T>();
    const int64_t n = self_c.numel();
    tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            T v = in[i];
            if (has_lo && v < lo) v = lo;
            if (has_hi && v > hi) v = hi;
            out[i] = v;
        }
    });
    return result;
}
}  // namespace

Tensor clamp_min_scalar_cpu(const Tensor& self, Scalar min) {
    if (self.dtype() == DType::Float32 && self.is_contiguous()) {
        Tensor self_c = self.contiguous();
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
        float lo = static_cast<float>(min.toDouble());
        const float* in = self_c.data_ptr<float>();
        float* out = result.data_ptr<float>();
        const int64_t n = self_c.numel();
#if defined(__x86_64__)
        if (clamp_row::avx512_ok()) {
            tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                clamp_row::f32_512(in + b, out + b, e - b, lo,
                                   std::numeric_limits<float>::infinity());
            });
            return result;
        }
#endif
        tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) out[i] = in[i] < lo ? lo : in[i];
        });
        return result;
    }
    double lo = min.toDouble();
    return dtype_unary_kernel(self, [lo](auto x) -> decltype(x) {
        using T = decltype(x);
        return static_cast<double>(x) < lo ? static_cast<T>(lo) : static_cast<T>(x);
    }, "clamp_min");
}
Tensor clamp_max_scalar_cpu(const Tensor& self, Scalar max) {
    if (self.dtype() == DType::Float32 && self.is_contiguous()) {
        Tensor self_c = self.contiguous();
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
        float hi = static_cast<float>(max.toDouble());
        const float* in = self_c.data_ptr<float>();
        float* out = result.data_ptr<float>();
        const int64_t n = self_c.numel();
#if defined(__x86_64__)
        if (clamp_row::avx512_ok()) {
            tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                clamp_row::f32_512(in + b, out + b, e - b,
                                   -std::numeric_limits<float>::infinity(), hi);
            });
            return result;
        }
#endif
        tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) out[i] = in[i] > hi ? hi : in[i];
        });
        return result;
    }
    double hi = max.toDouble();
    return dtype_unary_kernel(self, [hi](auto x) -> decltype(x) {
        using T = decltype(x);
        return static_cast<double>(x) > hi ? static_cast<T>(hi) : static_cast<T>(x);
    }, "clamp_max");
}
Tensor clamp_min_tensor_cpu(const Tensor& self, const Tensor& min) {
    return binary_same_kernel(self, min, [](auto x, auto m) -> decltype(x) {
        using T = decltype(x);
        return static_cast<double>(m) > static_cast<double>(x) ? static_cast<T>(m) : static_cast<T>(x);
    }, "clamp_min");
}
Tensor clamp_max_tensor_cpu(const Tensor& self, const Tensor& max) {
    return binary_same_kernel(self, max, [](auto x, auto m) -> decltype(x) {
        using T = decltype(x);
        return static_cast<double>(m) < static_cast<double>(x) ? static_cast<T>(m) : static_cast<T>(x);
    }, "clamp_max");
}
Tensor clip_cpu(const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    if (min.has_value() && max.has_value()) {
        const double lo = min->toDouble();
        const double hi = max->toDouble();
        if (self.dtype() == DType::Float32 && self.is_contiguous()) {
            Tensor self_c = self.contiguous();
            Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
            const float lo32 = static_cast<float>(lo), hi32 = static_cast<float>(hi);
            const float* in = self_c.data_ptr<float>();
            float* out = result.data_ptr<float>();
            const int64_t n = self_c.numel();
#if defined(__x86_64__)
            if (clamp_row::avx512_ok()) {
                tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                    clamp_row::f32_512(in + b, out + b, e - b, lo32, hi32);
                });
                return result;
            }
#endif
            tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                for (int64_t i = b; i < e; ++i) {
                    float v = in[i];
                    v = v < lo32 ? lo32 : v;
                    v = v > hi32 ? hi32 : v;
                    out[i] = v;
                }
            });
            return result;
        }
        if (self.dtype() == DType::Float64 && self.is_contiguous()) {
            Tensor self_c = self.contiguous();
            Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
            const double* in = self_c.data_ptr<double>();
            double* out = result.data_ptr<double>();
            const int64_t n = self_c.numel();
#if defined(__x86_64__)
            if (clamp_row::avx512_ok()) {
                tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                    clamp_row::f64_512(in + b, out + b, e - b, lo, hi);
                });
                return result;
            }
#endif
            tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                for (int64_t i = b; i < e; ++i) {
                    double v = in[i];
                    v = v < lo ? lo : v;
                    v = v > hi ? hi : v;
                    out[i] = v;
                }
            });
            return result;
        }
        Tensor r = clamp_min_scalar_cpu(self, *min);
        return clamp_max_scalar_cpu(r, *max);
    }
    if (min.has_value()) return clamp_min_scalar_cpu(self, *min);
    if (max.has_value()) return clamp_max_scalar_cpu(self, *max);
    return self.clone();
}
Tensor& clamp__cpu(Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    Tensor r = clip_cpu(self, std::move(min), std::move(max));
    self.copy_(r);
    return self;
}

// ===========================================================================
// Activations
// ===========================================================================

Tensor selu_cpu(const Tensor& self) {
    // selu: scale * max(0,x) + scale * alpha * (exp(min(0,x)) - 1)
    constexpr double kAlpha = 1.6732632423543772848170429916717;
    constexpr double kScale = 1.0507009873554804934193349852946;
    return dtype_unary_kernel(self, [](auto x) -> decltype(x) {
        using T = decltype(x);
        double v = static_cast<double>(x);
        return static_cast<T>(v > 0 ? kScale * v : kScale * kAlpha * (std::exp(v) - 1.0));
    }, "selu");
}
Tensor celu_cpu(const Tensor& self, Scalar alpha) {
    // celu: max(0,x) + alpha * (exp(x/alpha) - 1) on the negative side
    double a = alpha.toDouble();
    return dtype_unary_kernel(self, [a](auto x) -> decltype(x) {
        using T = decltype(x);
        double v = static_cast<double>(x);
        return static_cast<T>(v > 0 ? v : a * (std::exp(v / a) - 1.0));
    }, "celu");
}
Tensor hardshrink_cpu(const Tensor& self, Scalar lambd) {
    double l = lambd.toDouble();
    // lambd.to<scalar_t>(), so float32 boundary values compare exactly.
    return dtype_unary_kernel(self, [l](auto x) -> decltype(x) {
        using T = decltype(x);
        const double lt = static_cast<double>(static_cast<T>(l));
        double v = static_cast<double>(x);
        return (v >= -lt && v <= lt) ? static_cast<T>(0) : x;
    }, "hardshrink");
}
Tensor softshrink_cpu(const Tensor& self, Scalar lambd) {
    double l = lambd.toDouble();
    return dtype_unary_kernel(self, [l](auto x) -> decltype(x) {
        using T = decltype(x);
        const double lt = static_cast<double>(static_cast<T>(l));
        double v = static_cast<double>(x);
        if (v > lt) return static_cast<T>(v - lt);
        if (v < -lt) return static_cast<T>(v + lt);
        return static_cast<T>(v * 0.0);
    }, "softshrink");
}
// passes through where self is outside the inclusive [-lambd, lambd] band.
Tensor hardshrink_backward_cpu(const Tensor& grad_out, const Tensor& self, Scalar lambd) {
    double l = lambd.toDouble();
    return binary_same_kernel(grad_out, self, [l](auto g, auto s) -> decltype(g) {
        using T = decltype(g);
        const double lt = static_cast<double>(static_cast<T>(l));
        double v = static_cast<double>(s);
        return (v >= -lt && v <= lt) ? static_cast<T>(0) : g;
    }, "hardshrink_backward");
}
Tensor softshrink_backward_cpu(const Tensor& grad_output, const Tensor& self, Scalar lambd) {
    double l = lambd.toDouble();
    return binary_same_kernel(grad_output, self, [l](auto g, auto s) -> decltype(g) {
        using T = decltype(g);
        const double lt = static_cast<double>(static_cast<T>(l));
        double v = static_cast<double>(s);
        return (v >= -lt && v <= lt) ? static_cast<T>(0) : g;
    }, "softshrink_backward");
}
// where `output` is the saved forward result of sigmoid.
Tensor sigmoid_backward_cpu(const Tensor& grad_output, const Tensor& output) {
    return binary_same_kernel(grad_output, output, [](auto g, auto o) -> decltype(g) {
        using T = decltype(o);
        return g * o * (static_cast<T>(1) - o);
    }, "sigmoid_backward");
}
// where `output` is the saved forward result of tanh.
Tensor tanh_backward_cpu(const Tensor& grad_output, const Tensor& output) {
    return binary_same_kernel(grad_output, output, [](auto g, auto o) -> decltype(g) {
        using T = decltype(o);
        return g * (static_cast<T>(1) - o * o);
    }, "tanh_backward");
}
// gradient is dy/(x(1-x)) inside [0,1], NaN outside, and dy*inf at exact
// 0/1; with eps>=0 values outside [eps, 1-eps] (compared in scalar_t) are
// masked to zero.
Tensor logit_backward_cpu(const Tensor& grad_output, const Tensor& self, std::optional<Scalar> eps) {
    double e = eps.has_value() ? eps->toDouble() : -1.0;
    return binary_same_kernel(grad_output, self, [e](auto g, auto s) -> decltype(g) {
        using T = decltype(s);
        const T zero = static_cast<T>(0);
        const T one = static_cast<T>(1);
        if (e < 0) {
            if (s < zero || s > one) return std::numeric_limits<T>::quiet_NaN();
            if (s == zero || s == one) return g * std::numeric_limits<T>::infinity();
            return g / (s * (one - s));
        }
        // (float32 1 - 0.2f == 0.8f), so the band check must too.
        const T lo = static_cast<T>(e);
        const T hi = one - lo;
        if (s < lo || s > hi) return zero;
        if (s == zero || s == one) return g * std::numeric_limits<T>::infinity();
        return g / (s * (one - s));
    }, "logit_backward");
}
Tensor threshold_cpu(const Tensor& self, Scalar threshold, Scalar value) {
    double t = threshold.toDouble(), val = value.toDouble();
    return dtype_unary_kernel(self, [t, val](auto x) -> decltype(x) {
        using T = decltype(x);
        return static_cast<double>(x) <= t ? static_cast<T>(val) : static_cast<T>(x);
    }, "threshold");
}
Tensor prelu_cpu(const Tensor& self, const Tensor& weight) {
    // PReLU: weight shared when numel==1, otherwise per-channel
    // (channel = dim 0 for 1-D input, dim 1 for >=2-D input).
    Tensor wc = weight.contiguous();
    if (wc.numel() == 1) {
        double w = wc.data_ptr<double>() ? wc.item().toDouble() : 0.0;
        return dtype_unary_kernel(self, [w](auto x) -> decltype(x) {
            using T = decltype(x);
            double v = static_cast<double>(x);
            return static_cast<T>(v > 0 ? v : w * v);
        }, "prelu");
    }
    int64_t channels = self.dim() >= 1 ? self.size(0) : 1;
    int64_t per_ch = self.numel() / std::max<int64_t>(channels, 1);
    if (self.dim() >= 2) {
        // channel dim is dim 1; outer = dim 0
        int64_t N = self.size(0), C = self.size(1);
        per_ch = self.numel() / std::max<int64_t>(N * C, 1);
        channels = C;
    }
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
#define TP_PRELU_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        const ctype* wp = wc.to(self.dtype()).contiguous().data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t i = b; i < e; ++i) { \
                int64_t ch = 0; \
                if (self.dim() >= 2) ch = (i / per_ch) % channels; \
                else ch = (per_ch > 0) ? (i / per_ch) % channels : 0; \
                double v = static_cast<double>(sp[i]); \
                double w = static_cast<double>(wp[ch]); \
                dp[i] = static_cast<ctype>(v > 0 ? v : w * v); \
            } \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_PRELU_CASE)
        default: TP_THROW(TypeError, "prelu: unsupported dtype");
    }
#undef TP_PRELU_CASE
    return out;
}

// The bitwise family lives in BitwiseKernels.cpp.

TENSORPLAY_LIBRARY_IMPL(CPU, OpsKernels) {
    // Arithmetic
    m.impl("rsub.Scalar", rsub_scalar_cpu);
    m.impl("rsub.Tensor", rsub_tensor_cpu);
    m.impl("true_divide.Tensor", true_divide_tensor_cpu);
    m.impl("true_divide.Scalar", true_divide_scalar_cpu);
    m.impl("divide.Tensor", divide_tensor_cpu);
    m.impl("divide.Scalar", divide_scalar_cpu);
    m.impl("remainder.Tensor", remainder_tensor_cpu);
    m.impl("remainder.Scalar", remainder_scalar_cpu);
    m.impl("fmod.Tensor", fmod_tensor_cpu);
    m.impl("fmod.Scalar", fmod_scalar_cpu);
    m.impl("subtract.Tensor", subtract_tensor_cpu);
    m.impl("subtract.Scalar", subtract_scalar_cpu);
    m.impl("multiply.Tensor", multiply_tensor_cpu);
    m.impl("multiply.Scalar", multiply_scalar_cpu);
    m.impl("remainder.Scalar_Tensor", remainder_scalar_tensor_cpu);
    m.impl("div.Tensor_mode", div_mode_tensor_cpu);
    m.impl("div.Scalar_mode", div_mode_scalar_cpu);
    m.impl("divide.Tensor_mode", div_mode_tensor_cpu);
    m.impl("divide.Scalar_mode", div_mode_scalar_cpu);
    m.impl("floor_divide", floor_divide_cpu);
    m.impl("floor_divide.Scalar", floor_divide_scalar_cpu);
    m.impl("negative", negative_cpu);
    m.impl("positive", positive_cpu);
    // Comparisons / logic
    m.impl("greater", greater_cpu);
    m.impl("greater_equal", greater_equal_cpu);
    m.impl("less", less_cpu);
    m.impl("less_equal", less_equal_cpu);
    m.impl("not_equal", not_equal_cpu);
    m.impl("signbit", signbit_cpu);
    m.impl("logical_not", logical_not_cpu);
    m.impl("logical_and", logical_and_cpu);
    m.impl("logical_or", logical_or_cpu);
    m.impl("logical_xor", logical_xor_cpu);
    m.impl("isfinite", isfinite_cpu);
    m.impl("isinf", isinf_cpu);
    m.impl("isnan", isnan_cpu);
    m.impl("isneginf", isneginf_cpu);
    m.impl("isposinf", isposinf_cpu);
    // Math
    m.impl("reciprocal", reciprocal_cpu);
    m.impl("sgn", sgn_cpu);
    m.impl("exp2", exp2_cpu);
    m.impl("sinc", sinc_cpu);
    m.impl("deg2rad", deg2rad_cpu);
    m.impl("rad2deg", rad2deg_cpu);
    m.impl("fix", fix_cpu);
    m.impl("erfinv", erfinv_cpu);
    m.impl("logit", logit_cpu);
    m.impl("digamma", digamma_cpu);
    m.impl("i0", i0_cpu);
    m.impl("nan_to_num", nan_to_num_cpu);
    m.impl("xlogy", xlogy_cpu);
    m.impl("logaddexp", logaddexp_cpu);
    m.impl("logaddexp2", logaddexp2_cpu);
    m.impl("copysign.Tensor", copysign_cpu);
    m.impl("copysign.Scalar", copysign_scalar_cpu);
    m.impl("hypot", hypot_cpu);
    m.impl("atan2", atan2_cpu);
    m.impl("arctan2", atan2_cpu);
    m.impl("nextafter", nextafter_cpu);
    m.impl("gcd", gcd_cpu);
    m.impl("lcm", lcm_cpu);
    m.impl("heaviside", heaviside_cpu);
    // Clamp family
    m.impl("clamp_", clamp__cpu);
    m.impl("clamp_min.Scalar", clamp_min_scalar_cpu);
    m.impl("clamp_max.Scalar", clamp_max_scalar_cpu);
    m.impl("clamp_min.Tensor", clamp_min_tensor_cpu);
    m.impl("clamp_max.Tensor", clamp_max_tensor_cpu);
    m.impl("clip", clip_cpu);
    // Activations
    m.impl("selu", selu_cpu);
    m.impl("celu", celu_cpu);
    m.impl("hardshrink", hardshrink_cpu);
    m.impl("hardshrink_backward", hardshrink_backward_cpu);
    m.impl("softshrink", softshrink_cpu);
    m.impl("softshrink_backward", softshrink_backward_cpu);
    m.impl("sigmoid_backward", sigmoid_backward_cpu);
    m.impl("tanh_backward", tanh_backward_cpu);
    m.impl("logit_backward", logit_backward_cpu);
    m.impl("threshold", threshold_cpu);
    m.impl("prelu", prelu_cpu);
}

} // namespace cpu
} // namespace tensorplay
