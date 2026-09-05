// Bitwise operator family - CPU kernels.
//
// The family covers bitwise_not, bitwise_and/or/xor, bitwise_left_shift and
// bitwise_right_shift. All of them are defined on integral and boolean data
// only; boolean operands apply the corresponding logical operation.
//
// Variants:
//   .Tensor          elementwise with broadcasting
//   .Scalar          constant operand folded into the tensor's dtype
//   .Scalar_Tensor   scalar on the left, tensor on the right; the scalar is
//                    materialized as a 0-dim tensor in the tensor's dtype
//                    (wrapped-number semantics: the tensor dtype wins)
//   *_out            write the broadcasted result into a caller-owned tensor
//
// Tensors carrying an active transform level (vmap) hold their payload in the
// transform wrapper, so every entry point rejects them instead of touching
// storage directly; batch rules live in the transform layer.

#include "Tensor.h"
#include "TensorImpl.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "DType.h"
#include "Utils.h"
#include "Exception.h"
#include "Parallel.h"
#include "TypePromotion.h"

#include <cstdint>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace {

#define TENSORPLAY_FORALL_INT_TYPES(_) \
    _(uint8_t, UInt8)                  \
    _(int8_t, Int8)                    \
    _(int16_t, Int16)                  \
    _(int32_t, Int32)                  \
    _(int64_t, Int64)                  \
    _(uint16_t, UInt16)                \
    _(uint32_t, UInt32)                \
    _(uint64_t, UInt64)

inline void bitwise_check_cpu(const Tensor& t, const char* name) {
    if (t.unsafeGetTensorImpl() && t.unsafeGetTensorImpl()->is_batched()) {
        TP_THROW(NotImplementedError, name,
                 " is not supported for tensors inside an active transform "
                 "(vmap/grad) layer");
    }
    DType d = t.dtype();
    if (d == DType::Bool || isIntegralType(d)) return;
    TP_THROW(TypeError, name, ": only integral and boolean types are supported");
}

template <typename T, bool kLeft>
inline T bitwise_shift_value(T value, T shift) {
    using S = typename std::make_signed<T>::type;
    using U = typename std::make_unsigned<T>::type;
    constexpr U kBits = static_cast<U>(sizeof(T) * 8);
    const bool invalid = static_cast<S>(shift) < 0 || static_cast<U>(shift) >= kBits;
    if constexpr (kLeft) {
        if (invalid) return T(0);
        return static_cast<T>(static_cast<U>(value) << static_cast<U>(shift));
    }
    if (invalid) {
        if constexpr (std::is_signed_v<T>) return value < 0 ? T(-1) : T(0);
        return T(0);
    }
    return static_cast<T>(value >> static_cast<U>(shift));
}

template <typename Pred>
Tensor bitwise_binary_cpu(const Tensor& a_in, const Tensor& b_in, Pred pred, const char* name) {
    bitwise_check_cpu(a_in, name);
    bitwise_check_cpu(b_in, name);
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    if (a_in.dtype() == DType::Bool && b_in.dtype() == DType::Bool) dt = DType::Bool;
    if (dt != DType::Bool && !isIntegralType(dt)) {
        TP_THROW(TypeError, name, ": only integral and boolean types are supported");
    }
    Tensor ac = (a_in.dtype() == dt ? a_in : a_in.to(dt)).expand(out_shape).contiguous();
    Tensor bc = (b_in.dtype() == dt ? b_in : b_in.to(dt)).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, dt, a_in.device());
    int64_t n = out.numel();
    if (dt == DType::Bool) {
        const bool* ap = ac.data_ptr<bool>();
        const bool* bp = bc.data_ptr<bool>();
        bool* dp = out.data_ptr<bool>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i)
                dp[i] = pred(static_cast<uint8_t>(ap[i]), static_cast<uint8_t>(bp[i]));
        });
        return out;
    }
#define TP_BIT_BIN_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* ap = ac.data_ptr<ctype>(); \
        const ctype* bp = bc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) dp[i] = pred(ap[i], bp[i]); \
        }); \
        break; \
    }
    switch (dt) {
        TENSORPLAY_FORALL_INT_TYPES(TP_BIT_BIN_CASE)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BIT_BIN_CASE
    return out;
}

template <typename Pred>
Tensor bitwise_scalar_cpu(const Tensor& self_in, Scalar other, Pred pred, const char* name) {
    bitwise_check_cpu(self_in, name);
    Tensor sc = self_in.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self_in.shape()),
                               self_in.dtype(), self_in.device());
    int64_t n = out.numel();
    if (self_in.dtype() == DType::Bool) {
        const bool* sp = sc.data_ptr<bool>();
        uint8_t o = other.to<bool>() ? 1 : 0;
        bool* dp = out.data_ptr<bool>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i)
                dp[i] = pred(static_cast<uint8_t>(sp[i]), o);
        });
        return out;
    }
#define TP_BIT_SCALAR_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype ov = static_cast<ctype>(other.to<int64_t>()); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) dp[i] = pred(sp[i], ov); \
        }); \
        break; \
    }
    switch (self_in.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_BIT_SCALAR_CASE)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BIT_SCALAR_CASE
    return out;
}

template <bool kLeft>
Tensor bitwise_shift_scalar_cpu(const Tensor& self_in, Scalar other, const char* name) {
    bitwise_check_cpu(self_in, name);
    const int64_t shift = other.to<int64_t>();
    Tensor sc = self_in.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self_in.shape()),
                               self_in.dtype(), self_in.device());
    int64_t n = out.numel();
#define TP_SHIFT_SCALAR_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        const ctype sh = static_cast<ctype>(shift); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) \
                dp[i] = bitwise_shift_value<ctype, kLeft>(sp[i], sh); \
        }); \
        break; \
    }
    switch (self_in.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_SHIFT_SCALAR_CASE)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_SHIFT_SCALAR_CASE
    return out;
}

Tensor bitwise_not_cpu(const Tensor& self) {
    bitwise_check_cpu(self, "bitwise_not");
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                               self.dtype(), self.device());
    int64_t n = out.numel();
    if (self.dtype() == DType::Bool) {
        const bool* sp = sc.data_ptr<bool>();
        bool* dp = out.data_ptr<bool>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) dp[i] = !sp[i];
        });
        return out;
    }
#define TP_BNOT_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) dp[i] = static_cast<ctype>(~sp[i]); \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_BNOT_CASE)
        default: TP_THROW(TypeError, "bitwise_not: unsupported dtype");
    }
#undef TP_BNOT_CASE
    return out;
}

template <bool kLeft>
Tensor bitwise_shift_tensor_cpu(const Tensor& a_in, const Tensor& b_in, const char* name) {
    bitwise_check_cpu(a_in, name);
    bitwise_check_cpu(b_in, name);
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    if (a_in.dtype() == DType::Bool && b_in.dtype() == DType::Bool) dt = DType::Bool;
    if (dt != DType::Bool && !isIntegralType(dt)) {
        TP_THROW(TypeError, name, ": only integral and boolean types are supported");
    }
    Tensor ac = (a_in.dtype() == dt ? a_in : a_in.to(dt)).expand(out_shape).contiguous();
    Tensor bc = (b_in.dtype() == dt ? b_in : b_in.to(dt)).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, dt, a_in.device());
    int64_t n = out.numel();
#define TP_SHIFT_BIN_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* ap = ac.data_ptr<ctype>(); \
        const ctype* bp = bc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) { \
                dp[i] = bitwise_shift_value<ctype, kLeft>(ap[i], bp[i]); \
            } \
        }); \
        break; \
    }
    switch (dt) {
        TENSORPLAY_FORALL_INT_TYPES(TP_SHIFT_BIN_CASE)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_SHIFT_BIN_CASE
    return out;
}

// --- Named entry points registered with the dispatcher ----------------------

Tensor bitwise_and_tensor_cpu(const Tensor& a, const Tensor& b) {
    return bitwise_binary_cpu(a, b,
        [](auto x, auto y) { return static_cast<decltype(x)>(x & y); }, "bitwise_and");
}
Tensor bitwise_or_tensor_cpu(const Tensor& a, const Tensor& b) {
    return bitwise_binary_cpu(a, b,
        [](auto x, auto y) { return static_cast<decltype(x)>(x | y); }, "bitwise_or");
}
Tensor bitwise_xor_tensor_cpu(const Tensor& a, const Tensor& b) {
    return bitwise_binary_cpu(a, b,
        [](auto x, auto y) { return static_cast<decltype(x)>(x ^ y); }, "bitwise_xor");
}
Tensor bitwise_and_scalar_cpu(const Tensor& a, Scalar b) {
    return bitwise_scalar_cpu(a, b,
        [](auto x, auto y) { return static_cast<decltype(x)>(x & y); }, "bitwise_and");
}
Tensor bitwise_or_scalar_cpu(const Tensor& a, Scalar b) {
    return bitwise_scalar_cpu(a, b,
        [](auto x, auto y) { return static_cast<decltype(x)>(x | y); }, "bitwise_or");
}
Tensor bitwise_xor_scalar_cpu(const Tensor& a, Scalar b) {
    return bitwise_scalar_cpu(a, b,
        [](auto x, auto y) { return static_cast<decltype(x)>(x ^ y); }, "bitwise_xor");
}
Tensor bitwise_lshift_tensor_cpu(const Tensor& a, const Tensor& b) {
    return bitwise_shift_tensor_cpu<true>(a, b, "bitwise_left_shift");
}
Tensor bitwise_rshift_tensor_cpu(const Tensor& a, const Tensor& b) {
    return bitwise_shift_tensor_cpu<false>(a, b, "bitwise_right_shift");
}
Tensor bitwise_lshift_scalar_cpu(const Tensor& a, Scalar b) {
    return bitwise_shift_scalar_cpu<true>(a, b, "bitwise_left_shift");
}
Tensor bitwise_rshift_scalar_cpu(const Tensor& a, Scalar b) {
    return bitwise_shift_scalar_cpu<false>(a, b, "bitwise_right_shift");
}

// Scalar-first variants: materialize the scalar as a 0-dim tensor in the
// tensor's dtype, then run the plain tensor-tensor kernel.  A floating or
// complex scalar would move the result out of the integral domain, so it is
// refused up front.

inline void bitwise_scalar_check_cpu(Scalar self, const char* name) {
    if (self.isBoolean() || self.isIntegral()) return;
    TP_THROW(TypeError, name,
             ": only integral and boolean scalar operands are supported");
}

Tensor bitwise_and_scalar_tensor_cpu(Scalar self, const Tensor& other) {
    bitwise_check_cpu(other, "bitwise_and");
    bitwise_scalar_check_cpu(self, "bitwise_and");
    Tensor wrapped = Tensor::full({}, self, other.dtype(), other.device());
    return bitwise_binary_cpu(wrapped, other,
        [](auto x, auto y) { return static_cast<decltype(x)>(x & y); }, "bitwise_and");
}
Tensor bitwise_or_scalar_tensor_cpu(Scalar self, const Tensor& other) {
    bitwise_check_cpu(other, "bitwise_or");
    bitwise_scalar_check_cpu(self, "bitwise_or");
    Tensor wrapped = Tensor::full({}, self, other.dtype(), other.device());
    return bitwise_binary_cpu(wrapped, other,
        [](auto x, auto y) { return static_cast<decltype(x)>(x | y); }, "bitwise_or");
}
Tensor bitwise_xor_scalar_tensor_cpu(Scalar self, const Tensor& other) {
    bitwise_check_cpu(other, "bitwise_xor");
    bitwise_scalar_check_cpu(self, "bitwise_xor");
    Tensor wrapped = Tensor::full({}, self, other.dtype(), other.device());
    return bitwise_binary_cpu(wrapped, other,
        [](auto x, auto y) { return static_cast<decltype(x)>(x ^ y); }, "bitwise_xor");
}
Tensor bitwise_lshift_scalar_tensor_cpu(Scalar self, const Tensor& other) {
    bitwise_check_cpu(other, "bitwise_left_shift");
    bitwise_scalar_check_cpu(self, "bitwise_left_shift");
    Tensor wrapped = Tensor::full({}, self, other.dtype(), other.device());
    return bitwise_shift_tensor_cpu<true>(wrapped, other, "bitwise_left_shift");
}
Tensor bitwise_rshift_scalar_tensor_cpu(Scalar self, const Tensor& other) {
    bitwise_check_cpu(other, "bitwise_right_shift");
    bitwise_scalar_check_cpu(self, "bitwise_right_shift");
    Tensor wrapped = Tensor::full({}, self, other.dtype(), other.device());
    return bitwise_shift_tensor_cpu<false>(wrapped, other, "bitwise_right_shift");
}

// Out variants: compute into a fresh buffer, then transfer into the
// caller-owned tensor.  Matching shapes copy in place; otherwise the output
// adopts the result's metadata.

Tensor& bitwise_assign_out_cpu(Tensor& out, const Tensor& result) {
    if (static_cast<std::vector<int64_t>>(out.shape()) ==
        static_cast<std::vector<int64_t>>(result.shape())) {
        out.copy_(result);
    } else {
        out.unsafeGetTensorImpl()->copy_metadata_from(*result.unsafeGetTensorImpl());
    }
    return out;
}

Tensor& bitwise_and_tensor_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_and_tensor_cpu(a, b));
}
Tensor& bitwise_or_tensor_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_or_tensor_cpu(a, b));
}
Tensor& bitwise_xor_tensor_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_xor_tensor_cpu(a, b));
}
Tensor& bitwise_and_scalar_out_cpu(const Tensor& a, Scalar b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_and_scalar_cpu(a, b));
}
Tensor& bitwise_or_scalar_out_cpu(const Tensor& a, Scalar b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_or_scalar_cpu(a, b));
}
Tensor& bitwise_xor_scalar_out_cpu(const Tensor& a, Scalar b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_xor_scalar_cpu(a, b));
}
Tensor& bitwise_lshift_tensor_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_lshift_tensor_cpu(a, b));
}
Tensor& bitwise_rshift_tensor_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_rshift_tensor_cpu(a, b));
}
Tensor& bitwise_lshift_scalar_out_cpu(const Tensor& a, Scalar b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_lshift_scalar_cpu(a, b));
}
Tensor& bitwise_rshift_scalar_out_cpu(const Tensor& a, Scalar b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_rshift_scalar_cpu(a, b));
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(CPU, BitwiseKernels) {
    m.impl("bitwise_not", bitwise_not_cpu);
    m.impl("bitwise_and.Tensor", bitwise_and_tensor_cpu);
    m.impl("bitwise_or.Tensor", bitwise_or_tensor_cpu);
    m.impl("bitwise_xor.Tensor", bitwise_xor_tensor_cpu);
    m.impl("bitwise_and.Scalar", bitwise_and_scalar_cpu);
    m.impl("bitwise_or.Scalar", bitwise_or_scalar_cpu);
    m.impl("bitwise_xor.Scalar", bitwise_xor_scalar_cpu);
    m.impl("bitwise_left_shift.Tensor", bitwise_lshift_tensor_cpu);
    m.impl("bitwise_right_shift.Tensor", bitwise_rshift_tensor_cpu);
    m.impl("bitwise_left_shift.Tensor_Scalar", bitwise_lshift_scalar_cpu);
    m.impl("bitwise_right_shift.Tensor_Scalar", bitwise_rshift_scalar_cpu);
    // Scalar-first variants
    m.impl("bitwise_and.Scalar_Tensor", bitwise_and_scalar_tensor_cpu);
    m.impl("bitwise_or.Scalar_Tensor", bitwise_or_scalar_tensor_cpu);
    m.impl("bitwise_xor.Scalar_Tensor", bitwise_xor_scalar_tensor_cpu);
    m.impl("bitwise_left_shift.Scalar_Tensor", bitwise_lshift_scalar_tensor_cpu);
    m.impl("bitwise_right_shift.Scalar_Tensor", bitwise_rshift_scalar_tensor_cpu);
    // Out variants
    m.impl("bitwise_and.Tensor_out", bitwise_and_tensor_out_cpu);
    m.impl("bitwise_or.Tensor_out", bitwise_or_tensor_out_cpu);
    m.impl("bitwise_xor.Tensor_out", bitwise_xor_tensor_out_cpu);
    m.impl("bitwise_left_shift.Tensor_out", bitwise_lshift_tensor_out_cpu);
    m.impl("bitwise_right_shift.Tensor_out", bitwise_rshift_tensor_out_cpu);
    m.impl("bitwise_and.Scalar_out", bitwise_and_scalar_out_cpu);
    m.impl("bitwise_or.Scalar_out", bitwise_or_scalar_out_cpu);
    m.impl("bitwise_xor.Scalar_out", bitwise_xor_scalar_out_cpu);
    m.impl("bitwise_left_shift.Tensor_Scalar_out", bitwise_lshift_scalar_out_cpu);
    m.impl("bitwise_right_shift.Tensor_Scalar_out", bitwise_rshift_scalar_out_cpu);
}

} // namespace cpu
} // namespace tensorplay
