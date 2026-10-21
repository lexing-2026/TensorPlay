#include "QuantTensorOperators.h"
// Elementwise arithmetic, comparison, math, clamp, and activation CPU kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Utils.h"
#include "TensorIteratorOps.h"
#include "OutWrite.h"
#include "Exception.h"
#include "Parallel.h"
#include "TypePromotion.h"
#include "SpecialMath.h"
#include "Complex.h"
#include "cpu/ComplexUnary.h"
#include "cpu/FloatMathLoops.h"
#include "cpu/VecBinaryMath.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
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

// The same map with the destination the caller already holds.  The iterator
// walks the broadcast the allocating form would walk, at the promoted width,
// and stores into that buffer instead of a fresh one; overlap with an operand
// is the iterator's own judgement, so an in-place call stays a single pass.
// A destination that needs a cast or a resize -- or a pair whose answer the
// iterator cannot address at all -- takes the value from the allocating kernel.
template <typename Op, typename Alloc>
Tensor& binary_same_out_kernel(const Tensor& a_in, const Tensor& b_in, Tensor& out,
                               Alloc allocating,
                               Op op) {
    TP_CHECK(a_in.device() == b_in.device(),
             "Expected all tensors to be on the same device, but found ",
             a_in.device().toString(), " and ", b_in.device().toString(),
             "!");
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    const std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    if (!isComplexType(dt) && !isQuantizedType(dt) &&
        destination_takes(out, a_in.device(), dt, out_shape)) {
        Tensor ac = (a_in.dtype() == dt ? a_in : a_in.to(dt));
        Tensor bc = (b_in.dtype() == dt ? b_in : b_in.to(dt));
        ti_apply_binary(out, ac, bc, op);
        return out;
    }
    return write_out(out, allocating(a_in, b_in));
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
Tensor bool_unary_kernel(const Tensor& self, Pred pred, const char* name,
                         const Tensor* destination = nullptr) {
    Tensor out = destination != nullptr
                     ? *destination
                     : Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                     DType::Bool, self.device());
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
inline bool logical_truth_cpu(const complex<T>& value) {
    return static_cast<bool>(value.real()) || static_cast<bool>(value.imag());
}

template <typename Pred>
Tensor logical_binary_kernel(const Tensor& a_in, const Tensor& b_in, Pred pred,
                             const char* name, const Tensor* destination = nullptr) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    Tensor ac = (a_in.dtype() == dt ? a_in : a_in.to(dt)).expand(out_shape).contiguous();
    Tensor bc = (b_in.dtype() == dt ? b_in : b_in.to(dt)).expand(out_shape).contiguous();
    Tensor out = destination != nullptr
                     ? *destination
                     : Tensor::empty(out_shape, DType::Bool, a_in.device());
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
        TP_LOGICAL_BIN_COMPLEX_CASE(complex<Half>, ComplexHalf)
        TP_LOGICAL_BIN_COMPLEX_CASE(complex<float>, ComplexFloat)
        TP_LOGICAL_BIN_COMPLEX_CASE(complex<double>, ComplexDouble)
        TP_LOGICAL_BIN_COMPLEX_CASE(complex<BFloat16>, BComplex32)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_LOGICAL_BIN_COMPLEX_CASE
#undef TP_LOGICAL_BIN_CASE
    return out;
}

template <typename Pred>
Tensor logical_unary_kernel(const Tensor& self, Pred pred, const char* name,
                            const Tensor* destination = nullptr) {
    Tensor out = destination != nullptr
                     ? *destination
                     : Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
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
        TP_LOGICAL_UNARY_COMPLEX_CASE(complex<Half>, ComplexHalf)
        TP_LOGICAL_UNARY_COMPLEX_CASE(complex<float>, ComplexFloat)
        TP_LOGICAL_UNARY_COMPLEX_CASE(complex<double>, ComplexDouble)
        TP_LOGICAL_UNARY_COMPLEX_CASE(complex<BFloat16>, BComplex32)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_LOGICAL_UNARY_COMPLEX_CASE
#undef TP_LOGICAL_UNARY_CASE
    return out;
}

// Dtype-preserving unary (used by sgn, fix, negative, ...).
template <typename F>
Tensor dtype_unary_kernel(const Tensor& self, F f, const char* name,
                          const Tensor* destination = nullptr) {
    Tensor out = destination != nullptr
                     ? *destination
                     : Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                     self.dtype(), self.device());
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

// The same handoff for the loops that answer with one byte per element, and for
// the one that answers in the operand's own width.  A destination takes the pass
// when it is packed, holds the broadcast shape the flags come at in the width the
// loop writes, and is not a second window over the bytes the loop reads.
template <typename Pred>
Tensor& bool_unary_out_kernel(const Tensor& self, Tensor& out,
                              Tensor (*allocating)(const Tensor&), Pred pred,
                              const char* name) {
    if (destination_takes_flat(out, self.device(), DType::Bool,
                               static_cast<std::vector<int64_t>>(self.shape()),
                               self)) {
        bool_unary_kernel(self, pred, name, &out);
        return out;
    }
    return write_out(out, allocating(self));
}

template <typename Pred>
Tensor& logical_unary_out_kernel(const Tensor& self, Tensor& out,
                                 Tensor (*allocating)(const Tensor&), Pred pred,
                                 const char* name) {
    if (destination_takes_flat(out, self.device(), DType::Bool,
                               static_cast<std::vector<int64_t>>(self.shape()),
                               self)) {
        logical_unary_kernel(self, pred, name, &out);
        return out;
    }
    return write_out(out, allocating(self));
}

template <typename Pred>
Tensor& logical_binary_out_kernel(const Tensor& self, const Tensor& other, Tensor& out,
                                 Tensor (*allocating)(const Tensor&, const Tensor&),
                                 Pred pred, const char* name) {
    const std::vector<int64_t> shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    if (destination_takes_flat(out, self.device(), DType::Bool, shape, self, other)) {
        logical_binary_kernel(self, other, pred, name, &out);
        return out;
    }
    return write_out(out, allocating(self, other));
}

// A map that keeps the operand's width: the destination has to be packed, that
// width, and not a second window over the bytes the loop reads.  `allocating` is
// whatever spells the same map in the form that fills a fresh tensor, so a map
// carrying a parameter can hand its own bound value to both spellings.
template <typename F, typename Alloc>
Tensor& dtype_unary_out_kernel(const Tensor& self, Tensor& out,
                               Alloc allocating, F f,
                               const char* name) {
    const DType in = self.dtype();
    if (!isComplexType(in) && !isQuantizedType(in) &&
        destination_takes_flat(out, self.device(), in,
                               static_cast<std::vector<int64_t>>(self.shape()), self)) {
        dtype_unary_kernel(self, f, name, &out);
        return out;
    }
    return write_out(out, allocating(self));
}

struct SgnScalar {
    template <typename T>
    T operator()(T x) const {
        double d = static_cast<double>(x);
        if (d != d) return static_cast<T>(x);           // NaN passthrough
        if (d > 0) return static_cast<T>(1);
        if (d < 0) return static_cast<T>(-1);
        return static_cast<T>(0);
    }
};

struct SignbitPredicate {
    template <typename T>
    bool operator()(T x) const { return static_cast<double>(x) < 0.0; }
};
struct IsNeginfPredicate {
    template <typename T>
    bool operator()(T x) const {
        return static_cast<double>(x) == -std::numeric_limits<double>::infinity();
    }
};
struct IsPosinfPredicate {
    template <typename T>
    bool operator()(T x) const {
        return static_cast<double>(x) == std::numeric_limits<double>::infinity();
    }
};
struct LogicalNotPredicate {
    bool operator()(bool x) const { return !x; }
};
struct LogicalAndPredicate {
    bool operator()(bool x, bool y) const { return x && y; }
};
struct LogicalOrPredicate {
    bool operator()(bool x, bool y) const { return x || y; }
};
struct LogicalXorPredicate {
    bool operator()(bool x, bool y) const { return x != y; }
};

struct ReciprocalScalar {
    double operator()(double x) const { return 1.0 / x; }
};
struct Exp2Scalar {
    double operator()(double x) const { return std::exp2(x); }
};
struct SincScalar {
    double operator()(double x) const {
        double px = M_PI * x;
        return std::fabs(px) < 1e-30 ? 1.0 : std::sin(px) / px;
    }
};
struct Deg2RadScalar {
    double operator()(double x) const { return x * (M_PI / 180.0); }
};
struct Rad2DegScalar {
    double operator()(double x) const { return x * (180.0 / M_PI); }
};
struct ErfinvScalar {
    double operator()(double x) const { return special_math::calc_erfinv(x); }
};
struct I0Scalar {
    double operator()(double x) const {
        return tensorplay::special_math::modified_bessel_i0_forward(x);
    }
};
struct DigammaScalar {
    double operator()(double v) const {
        if (v <= 0 && v == std::floor(v)) return std::numeric_limits<double>::quiet_NaN();
        double r = 0;
        while (v < 6.0) { r -= 1.0 / v; v += 1.0; }
        double inv = 1.0 / v, inv2 = inv * inv;
        r += std::log(v) - 0.5 * inv
             - inv2 * (1.0/12.0 - inv2 * (1.0/120.0 - inv2 * (1.0/252.0 - inv2 * (1.0/240.0 - inv2 / 132.0))));
        return r;
    }
};
struct LogitScalar {
    double e;
    double operator()(double p) const {
        if (e >= 0) p = std::min(std::max(p, e), 1.0 - e);
        return std::log(p / (1.0 - p));
    }
};

} // anonymous namespace

// ===========================================================================
// Arithmetic
// ===========================================================================

Tensor rsub_scalar_cpu(const Tensor& self, const Scalar& other, const Scalar& alpha) {
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

Tensor rsub_tensor_cpu(const Tensor& self, const Tensor& other, const Scalar& alpha) {
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
Tensor true_divide_scalar_cpu(const Tensor& self, const Scalar& other) {
    // A Float32 stand-in would widen Half/BFloat16 inputs; the weak-scalar
    // rule keeps the tensor dtype unless the scalar itself is floating.
    return true_divide_core(
        self, Tensor::full({}, other, scalar_promote(self.dtype(), other), self.device()));
}
Tensor divide_tensor_cpu(const Tensor& self, const Tensor& other) { return true_divide_core(self, other); }
Tensor divide_scalar_cpu(const Tensor& self, const Scalar& other) { return true_divide_scalar_cpu(self, other); }

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
Tensor remainder_scalar_cpu(const Tensor& self, const Scalar& other) {
    // Forcing the scalar into self's dtype would truncate a float divisor
    // against an integral tensor; the pair promotes first.
    const DType dt = scalar_promote(self.dtype(), other);
    return remainder_tensor_cpu(self.to(dt), Tensor::full({}, other, dt, self.device()));
}

Tensor remainder_scalar_tensor_cpu(const Scalar& self, const Tensor& other) {
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
Tensor fmod_scalar_cpu(const Tensor& self, const Scalar& other) {
    const DType dt = scalar_promote(self.dtype(), other);
    return fmod_tensor_cpu(self.to(dt), Tensor::full({}, other, dt, self.device()));
}

Tensor subtract_tensor_cpu(const Tensor& self, const Tensor& other, const Scalar& alpha) {
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
Tensor subtract_scalar_cpu(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    DType dt = scalar_promote(self.dtype(), other);
    return subtract_tensor_cpu(self.to(dt),
                               Tensor::full({}, other, dt, self.device()), alpha);
}
Tensor multiply_tensor_cpu(const Tensor& self, const Tensor& other) {
    return binary_same_kernel<true>(self, other, [](auto x, auto y) { return x * y; }, "multiply");
}
Tensor multiply_scalar_cpu(const Tensor& self, const Scalar& other) {
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

// The quotient rounded once per element, held by both the form that allocates
// its answer and the form that is handed one.
struct DivRoundedPair {
    bool floor_mode;

    template <typename T>
    inline T operator()(T x, T y) const {
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
    }
};

Tensor div_rounded_core(const Tensor& a, const Tensor& b, DivRounding rounding) {
    if (rounding == DivRounding::kTrue) return true_divide_core(a, b);
    // Rounded division stays in the input dtype: an integral pair must come
    // back integral, which the float promotion of true division loses.
    return binary_same_kernel(a, b,
                              DivRoundedPair{rounding == DivRounding::kFloor}, "div");
}

Tensor div_rounded_scalar(const Tensor& self, Scalar other, DivRounding rounding) {
    if (rounding == DivRounding::kTrue) return true_divide_scalar_cpu(self, other);
    const DType dt = scalar_promote(self.dtype(), other);
    return div_rounded_core(self.to(dt), Tensor::full({}, other, dt, self.device()),
                            rounding);
}

}  // namespace

Tensor div_mode_tensor_cpu(const Tensor& self, const Tensor& other,
                           const std::optional<std::string>& rounding_mode) {
    return div_rounded_core(self, other, parse_div_rounding(rounding_mode));
}
Tensor div_mode_scalar_cpu(const Tensor& self, const Scalar& other,
                           const std::optional<std::string>& rounding_mode) {
    return div_rounded_scalar(self, other, parse_div_rounding(rounding_mode));
}
Tensor floor_divide_cpu(const Tensor& self, const Tensor& other) {
    return div_rounded_core(self, other, DivRounding::kFloor);
}
Tensor& floor_divide_out_cpu(const Tensor& self, const Tensor& other, Tensor& out) {
    return binary_same_out_kernel(self, other, out, floor_divide_cpu,
                                  DivRoundedPair{/*floor_mode=*/true});
}
Tensor floor_divide_scalar_cpu(const Tensor& self, const Scalar& other) {
    return div_rounded_scalar(self, other, DivRounding::kFloor);
}

Tensor negative_cpu(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        return complex_unary_op_kernel(self, [](auto x) { return -x; });
    }
    return dtype_unary_kernel(self, [](auto x) { return static_cast<decltype(x)>(-x); }, "negative");
}
// Negation keeps the operand's width, so the loop can write straight into a
// destination held in that width.  Where it cannot -- a complex or quantized
// operand, a cast, a stride the loop would have to follow -- the value comes
// from the allocating kernel and is handed over.
Tensor& negative_out_cpu(const Tensor& self, Tensor& out) {
    return dtype_unary_out_kernel(
        self, out, negative_cpu,
        [](auto x) { return static_cast<decltype(x)>(-x); }, "negative");
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
    return bool_unary_kernel(self, SignbitPredicate{}, "signbit");
}
Tensor& signbit_out_cpu(const Tensor& self, Tensor& out) {
    return bool_unary_out_kernel(self, out, signbit_cpu, SignbitPredicate{}, "signbit");
}
Tensor logical_not_cpu(const Tensor& self) {
    return logical_unary_kernel(self, LogicalNotPredicate{}, "logical_not");
}
Tensor& logical_not_out_cpu(const Tensor& self, Tensor& out) {
    return logical_unary_out_kernel(self, out, logical_not_cpu,
                                    LogicalNotPredicate{}, "logical_not");
}
Tensor logical_and_cpu(const Tensor& a, const Tensor& b) {
    return logical_binary_kernel(a, b, LogicalAndPredicate{}, "logical_and");
}
Tensor& logical_and_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return logical_binary_out_kernel(a, b, out, logical_and_cpu,
                                     LogicalAndPredicate{}, "logical_and");
}
Tensor logical_or_cpu(const Tensor& a, const Tensor& b) {
    return logical_binary_kernel(a, b, LogicalOrPredicate{}, "logical_or");
}
Tensor& logical_or_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return logical_binary_out_kernel(a, b, out, logical_or_cpu,
                                     LogicalOrPredicate{}, "logical_or");
}
Tensor logical_xor_cpu(const Tensor& a, const Tensor& b) {
    return logical_binary_kernel(a, b, LogicalXorPredicate{}, "logical_xor");
}
Tensor& logical_xor_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return logical_binary_out_kernel(a, b, out, logical_xor_cpu,
                                     LogicalXorPredicate{}, "logical_xor");
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
    return bool_unary_kernel(self, IsNeginfPredicate{}, "isneginf");
}
Tensor& isneginf_out_cpu(const Tensor& self, Tensor& out) {
    return bool_unary_out_kernel(self, out, isneginf_cpu, IsNeginfPredicate{}, "isneginf");
}
Tensor isposinf_cpu(const Tensor& self) {
    return bool_unary_kernel(self, IsPosinfPredicate{}, "isposinf");
}
Tensor& isposinf_out_cpu(const Tensor& self, Tensor& out) {
    return bool_unary_out_kernel(self, out, isposinf_cpu, IsPosinfPredicate{}, "isposinf");
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
    return float_math_kernel(self, ReciprocalScalar{}, "reciprocal");
}
Tensor& reciprocal_out_cpu(const Tensor& self, Tensor& out) {
    return float_math_out_kernel(self, out, reciprocal_cpu, ReciprocalScalar{},
                                 "reciprocal");
}
Tensor sgn_cpu(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        // z/|z| for nonzero z, zero at the origin; NaN flows through the
        // division untouched.
        return complex_unary_op_kernel(self, [](auto z) -> decltype(z) {
            using T = decltype(z);
            if (z == T(0, 0)) return T(0, 0);
            return z / std::hypot(z.real(), z.imag());
        });
    }
    return dtype_unary_kernel(self, SgnScalar{}, "sgn");
}
Tensor& sgn_out_cpu(const Tensor& self, Tensor& out) {
    return dtype_unary_out_kernel(self, out, sgn_cpu, SgnScalar{}, "sgn");
}
Tensor exp2_cpu(const Tensor& self) {
    return float_math_kernel(self, Exp2Scalar{}, "exp2");
}
Tensor& exp2_out_cpu(const Tensor& self, Tensor& out) {
    return float_math_out_kernel(self, out, exp2_cpu, Exp2Scalar{}, "exp2");
}
Tensor sinc_cpu(const Tensor& self) {
    return float_math_kernel(self, SincScalar{}, "sinc");
}
Tensor& sinc_out_cpu(const Tensor& self, Tensor& out) {
    return float_math_out_kernel(self, out, sinc_cpu, SincScalar{}, "sinc");
}
Tensor deg2rad_cpu(const Tensor& self) {
    return float_math_kernel(self, Deg2RadScalar{}, "deg2rad");
}
Tensor& deg2rad_out_cpu(const Tensor& self, Tensor& out) {
    return float_math_out_kernel(self, out, deg2rad_cpu, Deg2RadScalar{}, "deg2rad");
}
Tensor rad2deg_cpu(const Tensor& self) {
    return float_math_kernel(self, Rad2DegScalar{}, "rad2deg");
}
Tensor& rad2deg_out_cpu(const Tensor& self, Tensor& out) {
    return float_math_out_kernel(self, out, rad2deg_cpu, Rad2DegScalar{}, "rad2deg");
}
Tensor erfinv_cpu(const Tensor& self) {
    return float_math_kernel(self, ErfinvScalar{}, "erfinv");
}
Tensor& erfinv_out_cpu(const Tensor& self, Tensor& out) {
    return float_math_out_kernel(self, out, erfinv_cpu, ErfinvScalar{}, "erfinv");
}
Tensor logit_cpu(const Tensor& self, const std::optional<Scalar>& eps) {
    double e = eps.has_value() ? eps->toDouble() : -1.0;
    return float_math_kernel(self, LogitScalar{e}, "logit");
}
// The destination form spells eps as the schema does, a width the caller chose;
// the value it holds is what the allocating form would have clamped to.
Tensor& logit_out_cpu(const Tensor& self, std::optional<double> eps, Tensor& out) {
    if (float_math_takes(self, out)) {
        float_math_kernel(self, LogitScalar{eps.has_value() ? *eps : -1.0},
                          "logit", &out);
        return out;
    }
    std::optional<Scalar> width;
    if (eps.has_value()) width = Scalar(*eps);
    return write_out(out, logit_cpu(self, width));
}
Tensor digamma_cpu(const Tensor& self) {
    return float_math_kernel(self, DigammaScalar{}, "digamma");
}
Tensor& digamma_out_cpu(const Tensor& self, Tensor& out) {
    return float_math_out_kernel(self, out, digamma_cpu, DigammaScalar{}, "digamma");
}
Tensor i0_cpu(const Tensor& self) {
    // Modified Bessel I0.  The Chebyshev expansion holds across the whole
    // range; the ((|x|/2)^k / k!)^2 series it replaces needs more terms than
    // any fixed cap allows once |x| passes ~50.
    return float_math_kernel(self, I0Scalar{}, "i0");
}
Tensor& i0_out_cpu(const Tensor& self, Tensor& out) {
    return float_math_out_kernel(self, out, i0_cpu, I0Scalar{}, "i0");
}
Tensor nan_to_num_cpu(const Tensor& self, const Scalar& nan,
                      const std::optional<Scalar>& posinf, const std::optional<Scalar>& neginf) {
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

// One arithmetic per map, held by both the form that allocates its answer and
// the form that is handed one, so the two can never answer differently.
struct XlogyPair {
    double operator()(double x, double y) const {
        return tensorplay::special_math::calc_xlogy(x, y);
    }
};
// The pairwise maps below answer at the width of the operands they are
// handed: the flat loop over a Float32 pair reads single-precision values,
// so the answer has to be computed at that width too.  A map that always
// computed in double precision would pay for a double-precision call per
// element and then narrow the answer, which is the whole cost of the map.
struct LogAddExpPair {
    template <typename T>
    T operator()(T x, T y) const {
        T m = std::max(x, y);
        if (m == -std::numeric_limits<T>::infinity()) return m;
        if (m != m) return m;  // NaN propagates
        return m + std::log1p(std::exp(-std::fabs(x - y)));
    }
};
struct LogAddExp2Pair {
    template <typename T>
    T operator()(T x, T y) const {
        T m = std::max(x, y);
        if (m == -std::numeric_limits<T>::infinity()) return m;
        if (m != m) return m;
        return m + std::log1p(std::exp2(-std::fabs(x - y))) / static_cast<T>(M_LN2);
    }
};
struct CopysignPair {
    double operator()(double x, double y) const { return std::copysign(x, y); }
};
struct HypotPair {
    template <typename T>
    T operator()(T x, T y) const { return std::hypot(x, y); }
};
struct Atan2Pair {
    template <typename T>
    T operator()(T x, T y) const { return std::atan2(x, y); }
};

Tensor xlogy_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, XlogyPair{}, "xlogy");
}
Tensor& xlogy_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return binary_float_out_kernel(a, b, out, xlogy_cpu, XlogyPair{}, "xlogy");
}
// The pairwise maps whose answer is a library call per element answer
// several elements at a time on the vector path.  The flat loop answers
// them one element at a time, so a Float32 pair would pay for that call
// once per element; the vector path walks eight at a time instead.
//
// The path is taken when both operands carry the same floating width, have
// the same shape, are packed, and the destination is a packed tensor of that
// width and shape that no operand reads through.  Everything else -- a
// broadcast pair, a mixed pair, a reduced width -- is left to the flat loop.
static bool pair_math_takes(const Tensor& a, const Tensor& b, const Tensor& out) {
    if (!vecbinary::avx2_available() || a.dtype() != b.dtype() ||
        out.dtype() != a.dtype() ||
        (a.dtype() != DType::Float32 && a.dtype() != DType::Float64)) {
        return false;
    }
    return a.shape() == b.shape() && a.shape() == out.shape() &&
           a.is_contiguous() && b.is_contiguous() && out.is_contiguous() &&
           cpu::binary_float_takes(a, b, out);
}

template <typename F>
static Tensor pair_math_kernel(const Tensor& a, const Tensor& b, F f,
                               const char* name, vecbinary::BOp op) {
    if (vecbinary::avx2_available() && a.dtype() == b.dtype() &&
        (a.dtype() == DType::Float32 || a.dtype() == DType::Float64) &&
        a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(a.shape()), a.dtype(),
            a.device());
        const int64_t n = result.numel();
        if (n > 0) {
            if (a.dtype() == DType::Float32) {
                vecbinary::f32_chunk_avx2(op, a.data_ptr<float>(),
                                          b.data_ptr<float>(),
                                          result.data_ptr<float>(), 0, n);
            } else {
                vecbinary::f64_chunk_avx2(op, a.data_ptr<double>(),
                                          b.data_ptr<double>(),
                                          result.data_ptr<double>(), 0, n);
            }
        }
        return result;
    }
    return binary_float_kernel(a, b, f, name);
}

// The same map with the destination the caller holds: the answer is laid on
// it directly when it is packed and holds the width the map answers in.
template <typename F>
static Tensor& pair_math_out_kernel(const Tensor& a, const Tensor& b,
                                    Tensor& out, F f, const char* name,
                                    vecbinary::BOp op) {
    if (pair_math_takes(a, b, out)) {
        const int64_t n = out.numel();
        if (n > 0) {
            if (out.dtype() == DType::Float32) {
                vecbinary::f32_chunk_avx2(op, a.data_ptr<float>(),
                                          b.data_ptr<float>(),
                                          out.data_ptr<float>(), 0, n);
            } else {
                vecbinary::f64_chunk_avx2(op, a.data_ptr<double>(),
                                          b.data_ptr<double>(),
                                          out.data_ptr<double>(), 0, n);
            }
        }
        return out;
    }
    return binary_float_out_kernel(
        a, b, out,
        [=](const Tensor&, const Tensor&) { return pair_math_kernel(a, b, f, name, op); },
        f, name);
}

Tensor logaddexp_cpu(const Tensor& a, const Tensor& b) {
    return pair_math_kernel(a, b, LogAddExpPair{}, "logaddexp",
                             vecbinary::BOp::LogAddExp);
}
Tensor& logaddexp_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return pair_math_out_kernel(a, b, out, LogAddExpPair{}, "logaddexp",
                                 vecbinary::BOp::LogAddExp);
}
Tensor logaddexp2_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, LogAddExp2Pair{}, "logaddexp2");
}
Tensor& logaddexp2_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return binary_float_out_kernel(a, b, out, logaddexp2_cpu, LogAddExp2Pair{}, "logaddexp2");
}
Tensor copysign_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, CopysignPair{}, "copysign");
}
Tensor& copysign_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return binary_float_out_kernel(a, b, out, copysign_cpu, CopysignPair{}, "copysign");
}
Tensor copysign_scalar_cpu(const Tensor& self, const Scalar& other) {
    // The sign comes from the scalar alone, so the divisor width never
    // participates in promotion -- Float32 carries every sign bit exactly.
    return copysign_cpu(self, Tensor::full({}, other, DType::Float32, self.device()));
}
Tensor hypot_cpu(const Tensor& a, const Tensor& b) {
    return pair_math_kernel(a, b, HypotPair{}, "hypot", vecbinary::BOp::Hypot);
}
Tensor& hypot_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return pair_math_out_kernel(a, b, out, HypotPair{}, "hypot",
                                 vecbinary::BOp::Hypot);
}

Tensor atan2_cpu(const Tensor& a, const Tensor& b) {
    return pair_math_kernel(a, b, Atan2Pair{}, "atan2", vecbinary::BOp::Atan2);
}
Tensor& atan2_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return pair_math_out_kernel(a, b, out, Atan2Pair{}, "atan2",
                                 vecbinary::BOp::Atan2);
}
// One representable step from `from` toward `to`, taken in the 16-bit layout the
// reduced widths share.  Clearing the sign bit leaves a pattern that grows with
// the magnitude, so a step away from zero is one increment of that pattern and a
// step toward zero one decrement, with the sign carried through.  Stepping such
// a value at Float32 instead moves it by a float ulp -- a distance no half or
// bfloat16 value is even that far away from where it started.
template <typename T>
inline T step_toward16(T from, T to) {
    using Bits = uint16_t;
    static constexpr Bits kSign = static_cast<Bits>(1) << 15;
    if (std::isnan(static_cast<float>(from)) || std::isnan(static_cast<float>(to))) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    Bits f = from.x;
    const Bits t = to.x;
    if (f == t) return from;
    const Bits abs_f = static_cast<Bits>(f & ~kSign);
    const Bits abs_t = static_cast<Bits>(t & ~kSign);
    if (abs_f == 0) {
        // Two zeros differ only in sign, so the answer is the one the step heads
        // for; a zero stepping toward a value of either sign leaves by the
        // smallest subnormal carrying that sign.
        if (abs_t == 0) return to;
        return T(static_cast<unsigned short>((t & kSign) | static_cast<Bits>(1)),
                 T::from_bits());
    }
    f = (abs_f > abs_t || ((f ^ t) & kSign)) ? static_cast<Bits>(f - 1)
                                             : static_cast<Bits>(f + 1);
    return T(static_cast<unsigned short>(f), T::from_bits());
}

// The step in whatever width the buffers hold: Float32 and Float64 have one in
// the library, the reduced widths are stepped bit by bit above.
template <typename T>
inline T step_toward(T from, T to) {
    if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
        return step_toward16(from, to);
    } else {
        return std::nextafter(from, to);
    }
}

template <typename T>
void nextafter_fill(const Tensor& ac, const Tensor& bc, Tensor& out) {
    const T* ap = ac.data_ptr<T>();
    const T* bp = bc.data_ptr<T>();
    T* dp = out.data_ptr<T>();
    const int64_t n = out.numel();
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) dp[i] = step_toward<T>(ap[i], bp[i]);
    });
}

Tensor nextafter_kernel(const Tensor& a, const Tensor& b,
                        const Tensor* destination = nullptr) {
    if (a.device() != b.device()) {
        TP_THROW(DeviceMismatchError, "nextafter", ": inputs must be on the same device");
    }
    // The step happens in the element dtype: one taken at Double rounds a
    // Float32 value straight back to where it started once the answer narrows.
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a.shape()),
        static_cast<std::vector<int64_t>>(b.shape()));
    DType dt = promoteTypes(a.dtype(), b.dtype());
    if (!isFloatingType(dt)) dt = DType::Float32;
    Tensor ac = a.to(dt).expand(out_shape).contiguous();
    Tensor bc = b.to(dt).expand(out_shape).contiguous();
    Tensor out = destination != nullptr
                     ? *destination
                     : Tensor::empty(out_shape, dt, a.device());
    switch (dt) {
        case DType::Float64: nextafter_fill<double>(ac, bc, out); break;
        case DType::Float32: nextafter_fill<float>(ac, bc, out); break;
        case DType::Float16: nextafter_fill<Half>(ac, bc, out); break;
        case DType::BFloat16: nextafter_fill<BFloat16>(ac, bc, out); break;
        default: TP_THROW(TypeError, "nextafter: unsupported dtype");
    }
    return out;
}

// Whether the stepping loop can fill the caller's tensor itself instead of
// making one that is copied over afterwards.  Every float width is addressable
// here -- the reduced ones step in their own layout -- so the destination only
// has to be packed, carry the promoted width, hold the broadcast shape, and not
// be a second window over the bytes the loop reads.
static bool nextafter_takes(const Tensor& a, const Tensor& b, const Tensor& out) {
    const DType lhs = a.dtype(), rhs = b.dtype();
    if (isComplexType(lhs) || isComplexType(rhs) ||
        isQuantizedType(lhs) || isQuantizedType(rhs)) {
        return false;
    }
    DType dt = promoteTypes(lhs, rhs);
    if (isComplexType(dt) || isQuantizedType(dt)) {
        return false;
    }
    if (!isFloatingType(dt)) dt = DType::Float32;
    const std::vector<int64_t> shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a.shape()),
        static_cast<std::vector<int64_t>>(b.shape()));
    return destination_takes_flat(out, a.device(), dt, shape, a, b);
}

Tensor nextafter_cpu(const Tensor& a, const Tensor& b) {
    return nextafter_kernel(a, b);
}
Tensor& nextafter_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    if (nextafter_takes(a, b, out)) {
        nextafter_kernel(a, b, &out);
        return out;
    }
    return write_out(out, nextafter_cpu(a, b));
}
// The three integer-pair maps are held once, by both the form that allocates
// its answer and the form that is handed one.
struct GcdPair {
    template <typename T>
    inline T operator()(T x, T y) const {
        long long ux = static_cast<long long>(x < T(0) ? -x : x);
        long long uy = static_cast<long long>(y < T(0) ? -y : y);
        while (uy) { long long t = ux % uy; ux = uy; uy = t; }
        return static_cast<T>(ux);
    }
};
struct LcmPair {
    template <typename T>
    inline T operator()(T x, T y) const {
        long long ux = static_cast<long long>(static_cast<float>(x) < 0.0f ? -x : x);
        long long uy = static_cast<long long>(static_cast<float>(y) < 0.0f ? -y : y);
        long long g = ux, t2 = uy;
        while (t2) { long long t3 = g % t2; g = t2; t2 = t3; }
        if (g == 0) return static_cast<T>(0);
        return static_cast<T>(ux / g * uy);
    }
};
struct HeavisidePair {
    template <typename T>
    inline T operator()(T x, T v) const {
        double xd = static_cast<double>(x);
        if (xd < 0.0) return static_cast<T>(0);
        if (xd == 0.0) return static_cast<T>(v);
        return static_cast<T>(1);
    }
};

// gcd and lcm are defined on whole numbers only; the check is shared so both
// spellings of the answer reject the same operand pairs.
static void require_integral_pair(const Tensor& a, const Tensor& b, const char* name) {
    if (isFloatingType(promoteTypes(a.dtype(), b.dtype()))) {
        TP_THROW(TypeError, name, " only supports integral tensors");
    }
}

Tensor gcd_cpu(const Tensor& a, const Tensor& b) {
    require_integral_pair(a, b, "gcd");
    return binary_same_kernel(a, b, GcdPair{}, "gcd");
}
Tensor& gcd_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    require_integral_pair(a, b, "gcd");
    return binary_same_out_kernel(a, b, out, gcd_cpu, GcdPair{});
}
Tensor lcm_cpu(const Tensor& a, const Tensor& b) {
    require_integral_pair(a, b, "lcm");
    return binary_same_kernel(a, b, LcmPair{}, "lcm");
}
Tensor& lcm_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    require_integral_pair(a, b, "lcm");
    return binary_same_out_kernel(a, b, out, lcm_cpu, LcmPair{});
}
Tensor heaviside_cpu(const Tensor& a, const Tensor& values) {
    return binary_same_kernel(a, values, HeavisidePair{}, "heaviside");
}
Tensor& heaviside_out_cpu(const Tensor& a, const Tensor& values, Tensor& out) {
    return binary_same_out_kernel(a, values, out, heaviside_cpu, HeavisidePair{});
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

inline bool avx2_ok() {
    static const bool ok = __builtin_cpu_supports("avx2") != 0;
    return ok;
}

// AVX2 twins of the helpers above, selected when the 512-bit path is
// unavailable. The bound order matches so NaN lanes flow through the same
// way.
__attribute__((target("avx2")))
inline void f32_256(const float* in, float* out, int64_t n, float lo, float hi) {
    const __m256 vlo = _mm256_set1_ps(lo), vhi = _mm256_set1_ps(hi);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        v = _mm256_min_ps(vhi, _mm256_max_ps(vlo, v));
        _mm256_storeu_ps(out + i, v);
    }
    for (; i < n; ++i) {
        float v = in[i];
        v = v < lo ? lo : v;
        v = v > hi ? hi : v;
        out[i] = v;
    }
}

__attribute__((target("avx2")))
inline void f64_256(const double* in, double* out, int64_t n, double lo, double hi) {
    const __m256d vlo = _mm256_set1_pd(lo), vhi = _mm256_set1_pd(hi);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(in + i);
        v = _mm256_min_pd(vhi, _mm256_max_pd(vlo, v));
        _mm256_storeu_pd(out + i, v);
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
        if (clamp_row::avx2_ok()) {
            tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                clamp_row::f32_256(in + b, out + b, e - b, lo,
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
        if (clamp_row::avx2_ok()) {
            tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                clamp_row::f32_256(in + b, out + b, e - b,
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
Tensor clip_cpu(const Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max) {
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
            if (clamp_row::avx2_ok()) {
                tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                    clamp_row::f32_256(in + b, out + b, e - b, lo32, hi32);
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
            if (clamp_row::avx2_ok()) {
                tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                    clamp_row::f64_256(in + b, out + b, e - b, lo, hi);
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
Tensor& clamp__cpu(Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max) {
    Tensor r = clip_cpu(self, std::move(min), std::move(max));
    self.copy_(r);
    return self;
}

// ===========================================================================
// Activations
// ===========================================================================

// Inside the inclusive [-lambd, lambd] band the answer is zero, outside it the
// operand passes through.  The limit is read back through the element's own
// width, so a boundary value that only rounds there still compares exactly.
struct HardshrinkBand {
    double l;

    template <typename T>
    inline T operator()(T x) const {
        const double lt = static_cast<double>(static_cast<T>(l));
        double v = static_cast<double>(x);
        return (v >= -lt && v <= lt) ? static_cast<T>(0) : x;
    }
};

// One limit's distance outside the band, and zero inside it -- where zero keeps
// the sign the operand carried.
struct SoftshrinkBand {
    double l;

    template <typename T>
    inline T operator()(T x) const {
        const double lt = static_cast<double>(static_cast<T>(l));
        double v = static_cast<double>(x);
        if (v > lt) return static_cast<T>(v - lt);
        if (v < -lt) return static_cast<T>(v + lt);
        return static_cast<T>(v * 0.0);
    }
};

// Either shrink passes the incoming gradient through wherever the operand lies
// outside its band, and drops it inside.
struct ShrinkBandGrad {
    double l;

    template <typename T>
    inline T operator()(T g, T s) const {
        const double lt = static_cast<double>(static_cast<T>(l));
        double v = static_cast<double>(s);
        return (v >= -lt && v <= lt) ? static_cast<T>(0) : g;
    }
};

Tensor hardshrink_cpu(const Tensor& self, const Scalar& lambd) {
    return dtype_unary_kernel(self, HardshrinkBand{lambd.toDouble()}, "hardshrink");
}
Tensor& hardshrink_out_cpu(const Tensor& self, const Scalar& lambd, Tensor& out) {
    double l = lambd.toDouble();
    return dtype_unary_out_kernel(
        self, out, [l](const Tensor& v) { return hardshrink_cpu(v, Scalar(l)); },
        HardshrinkBand{l}, "hardshrink");
}
Tensor softshrink_cpu(const Tensor& self, const Scalar& lambd) {
    return dtype_unary_kernel(self, SoftshrinkBand{lambd.toDouble()}, "softshrink");
}
Tensor& softshrink_out_cpu(const Tensor& self, const Scalar& lambd, Tensor& out) {
    double l = lambd.toDouble();
    return dtype_unary_out_kernel(
        self, out, [l](const Tensor& v) { return softshrink_cpu(v, Scalar(l)); },
        SoftshrinkBand{l}, "softshrink");
}
Tensor hardshrink_backward_cpu(const Tensor& grad_out, const Tensor& self, const Scalar& lambd) {
    return binary_same_kernel(grad_out, self,
                              ShrinkBandGrad{lambd.toDouble()}, "hardshrink_backward");
}
Tensor& hardshrink_backward_out_cpu(const Tensor& grad_out, const Tensor& self,
                                   const Scalar& lambd, Tensor& grad_input) {
    double l = lambd.toDouble();
    return binary_same_out_kernel(
        grad_out, self, grad_input,
        [l](const Tensor& g, const Tensor& s) {
            return hardshrink_backward_cpu(g, s, Scalar(l));
        },
        ShrinkBandGrad{l});
}
Tensor softshrink_backward_cpu(const Tensor& grad_output, const Tensor& self, const Scalar& lambd) {
    return binary_same_kernel(grad_output, self,
                              ShrinkBandGrad{lambd.toDouble()}, "softshrink_backward");
}
Tensor& softshrink_backward_out_cpu(const Tensor& grad_output, const Tensor& self,
                                    const Scalar& lambd, Tensor& grad_input) {
    double l = lambd.toDouble();
    return binary_same_out_kernel(
        grad_output, self, grad_input,
        [l](const Tensor& g, const Tensor& s) {
            return softshrink_backward_cpu(g, s, Scalar(l));
        },
        ShrinkBandGrad{l});
}
// The saved forward answer of sigmoid is itself the derivative's input:
// grad * output * (1 - output).
struct SigmoidGradPair {
    template <typename T>
    inline T operator()(T g, T o) const {
        return g * o * (static_cast<T>(1) - o);
    }
};
// The saved forward answer of tanh: grad * (1 - output^2).
struct TanhGradPair {
    template <typename T>
    inline T operator()(T g, T o) const {
        return g * (static_cast<T>(1) - o * o);
    }
};
// grad / (x (1-x)) inside [0,1], NaN outside, and grad*inf at exact 0/1; with
// eps >= 0 the values outside [eps, 1-eps] -- compared at the element's own
// width, because the bound is read back through it -- are masked to zero.
struct LogitGradPair {
    double e;

    template <typename T>
    inline T operator()(T g, T s) const {
        const T zero = static_cast<T>(0);
        const T one = static_cast<T>(1);
        if (e < 0) {
            if (s < zero || s > one) return std::numeric_limits<T>::quiet_NaN();
            if (s == zero || s == one) return g * std::numeric_limits<T>::infinity();
            return g / (s * (one - s));
        }
        const T lo = static_cast<T>(e);
        const T hi = one - lo;
        if (s < lo || s > hi) return zero;
        if (s == zero || s == one) return g * std::numeric_limits<T>::infinity();
        return g / (s * (one - s));
    }
};

Tensor sigmoid_backward_cpu(const Tensor& grad_output, const Tensor& output) {
    return binary_same_kernel(grad_output, output, SigmoidGradPair{}, "sigmoid_backward");
}
Tensor& sigmoid_backward_out_cpu(const Tensor& grad_output, const Tensor& output,
                                Tensor& grad_input) {
    return binary_same_out_kernel(grad_output, output, grad_input, sigmoid_backward_cpu,
                                  SigmoidGradPair{});
}
Tensor tanh_backward_cpu(const Tensor& grad_output, const Tensor& output) {
    return binary_same_kernel(grad_output, output, TanhGradPair{}, "tanh_backward");
}
Tensor& tanh_backward_out_cpu(const Tensor& grad_output, const Tensor& output,
                              Tensor& grad_input) {
    return binary_same_out_kernel(grad_output, output, grad_input, tanh_backward_cpu,
                                  TanhGradPair{});
}
Tensor logit_backward_cpu(const Tensor& grad_output, const Tensor& self, const std::optional<Scalar>& eps) {
    double e = eps.has_value() ? eps->toDouble() : -1.0;
    return binary_same_kernel(grad_output, self, LogitGradPair{e}, "logit_backward");
}
Tensor& logit_backward_out_cpu(const Tensor& grad_output, const Tensor& self,
                               std::optional<double> eps, Tensor& grad_input) {
    const double e = eps.has_value() ? *eps : -1.0;
    const std::optional<Scalar> as_scalar =
        eps.has_value() ? std::optional<Scalar>(Scalar(*eps)) : std::nullopt;
    return binary_same_out_kernel(
        grad_output, self, grad_input,
        [as_scalar](const Tensor& g, const Tensor& s) {
            return logit_backward_cpu(g, s, as_scalar);
        },
        LogitGradPair{e});
}
// Every element at or below the limit takes the replacement value; above it the
// operand passes through unchanged.
struct ThresholdAt {
    double t;
    double val;

    template <typename T>
    inline T operator()(T x) const {
        return static_cast<double>(x) <= t ? static_cast<T>(val) : static_cast<T>(x);
    }
};

Tensor threshold_cpu(const Tensor& self, const Scalar& threshold, const Scalar& value) {
    if (isQuantizedType(self.dtype())) return quantized::activation(self, 6, threshold.toDouble(), value.toDouble());

    return dtype_unary_kernel(self, ThresholdAt{threshold.toDouble(), value.toDouble()},
                              "threshold");
}
Tensor& threshold_out_cpu(const Tensor& self, const Scalar& threshold, const Scalar& value,
                          Tensor& out) {
    if (isQuantizedType(self.dtype())) return write_out(out, threshold_cpu(self, threshold, value));

    double t = threshold.toDouble(), val = value.toDouble();
    return dtype_unary_out_kernel(
        self, out, [t, val](const Tensor& v) { return threshold_cpu(v, Scalar(t), Scalar(val)); },
        ThresholdAt{t, val}, "threshold");
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
    m.impl("negative.out", negative_out_cpu);
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
    m.impl("signbit.out", signbit_out_cpu);
    m.impl("logical_not.out", logical_not_out_cpu);
    m.impl("logical_and.out", logical_and_out_cpu);
    m.impl("logical_or.out", logical_or_out_cpu);
    m.impl("logical_xor.out", logical_xor_out_cpu);
    m.impl("isfinite", isfinite_cpu);
    m.impl("isinf", isinf_cpu);
    m.impl("isnan", isnan_cpu);
    m.impl("isneginf", isneginf_cpu);
    m.impl("isposinf", isposinf_cpu);
    m.impl("isneginf.out", isneginf_out_cpu);
    m.impl("isposinf.out", isposinf_out_cpu);
    // Math
    m.impl("reciprocal", reciprocal_cpu);
    m.impl("sgn", sgn_cpu);
    m.impl("sgn.out", sgn_out_cpu);
    m.impl("exp2", exp2_cpu);
    m.impl("sinc", sinc_cpu);
    m.impl("deg2rad", deg2rad_cpu);
    m.impl("rad2deg", rad2deg_cpu);
    m.impl("erfinv", erfinv_cpu);
    m.impl("logit", logit_cpu);
    m.impl("digamma", digamma_cpu);
    m.impl("i0", i0_cpu);
    m.impl("reciprocal.out", reciprocal_out_cpu);
    m.impl("exp2.out", exp2_out_cpu);
    m.impl("sinc.out", sinc_out_cpu);
    m.impl("deg2rad.out", deg2rad_out_cpu);
    m.impl("rad2deg.out", rad2deg_out_cpu);
    m.impl("erfinv.out", erfinv_out_cpu);
    m.impl("logit.out", logit_out_cpu);
    m.impl("digamma.out", digamma_out_cpu);
    m.impl("i0.out", i0_out_cpu);
    m.impl("nan_to_num", nan_to_num_cpu);
    m.impl("xlogy", xlogy_cpu);
    m.impl("logaddexp", logaddexp_cpu);
    m.impl("logaddexp2", logaddexp2_cpu);
    m.impl("copysign.Tensor", copysign_cpu);
    m.impl("copysign.Scalar", copysign_scalar_cpu);
    m.impl("hypot", hypot_cpu);
    m.impl("atan2", atan2_cpu);
    m.impl("nextafter", nextafter_cpu);
    m.impl("nextafter.out", nextafter_out_cpu);
    m.impl("gcd", gcd_cpu);
    m.impl("lcm", lcm_cpu);
    m.impl("heaviside", heaviside_cpu);
    m.impl("xlogy.OutTensor", xlogy_out_cpu);
    m.impl("logaddexp.out", logaddexp_out_cpu);
    m.impl("logaddexp2.out", logaddexp2_out_cpu);
    m.impl("copysign.out", copysign_out_cpu);
    m.impl("hypot.out", hypot_out_cpu);
    // The two spellings of the same angle map take the destination the same way.
    m.impl("atan2.out", atan2_out_cpu);
    m.impl("arctan2.out", atan2_out_cpu);
    m.impl("gcd.out", gcd_out_cpu);
    m.impl("lcm.out", lcm_out_cpu);
    m.impl("heaviside.out", heaviside_out_cpu);
    m.impl("floor_divide.out", floor_divide_out_cpu);
    // Clamp family
    m.impl("clamp_", clamp__cpu);
    m.impl("clamp_min.Scalar", clamp_min_scalar_cpu);
    m.impl("clamp_max.Scalar", clamp_max_scalar_cpu);
    m.impl("clamp_min.Tensor", clamp_min_tensor_cpu);
    m.impl("clamp_max.Tensor", clamp_max_tensor_cpu);
    // Activations
    m.impl("hardshrink", hardshrink_cpu);
    m.impl("hardshrink.out", hardshrink_out_cpu);
    m.impl("hardshrink_backward", hardshrink_backward_cpu);
    m.impl("hardshrink_backward.grad_input", hardshrink_backward_out_cpu);
    m.impl("softshrink", softshrink_cpu);
    m.impl("softshrink.out", softshrink_out_cpu);
    m.impl("softshrink_backward", softshrink_backward_cpu);
    m.impl("softshrink_backward.grad_input", softshrink_backward_out_cpu);
    m.impl("sigmoid_backward", sigmoid_backward_cpu);
    m.impl("sigmoid_backward.grad_input", sigmoid_backward_out_cpu);
    m.impl("tanh_backward", tanh_backward_cpu);
    m.impl("tanh_backward.grad_input", tanh_backward_out_cpu);
    m.impl("logit_backward", logit_backward_cpu);
    m.impl("logit_backward.grad_input", logit_backward_out_cpu);
    m.impl("threshold", threshold_cpu);
    m.impl("threshold.out", threshold_out_cpu);
    m.impl("prelu", prelu_cpu);
}

} // namespace cpu
} // namespace tensorplay
